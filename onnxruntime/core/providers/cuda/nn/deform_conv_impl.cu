// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//
// CUDA implementation of DeformConv: deformable im2col kernel + bilinear interpolation.
// Reference: torchvision deform_conv2d_kernel.cu, ONNX DeformConv spec.

#include "deform_conv_impl.h"
#include "core/providers/cuda/cu_inc/common.cuh"
#include "core/providers/cuda/shared_inc/fast_divmod.h"
#include "core/common/float16.h"
#include <cstdint>
#include <type_traits>
#include <algorithm>
#include <limits>

namespace onnxruntime {
namespace cuda {

namespace {

constexpr int kDeformConvThreadsPerBlock = 256;

template <int N>
struct DeformConvKSize {
  static constexpr int value = N;
};

// Calculate grid size with a safety limit to prevent overflow.
// Since we use grid-stride loops in kernels, limiting the grid size is safe.
inline int GetGridSize(size_t n, size_t threads_per_block) {
  size_t blocks_needed = (n + threads_per_block - 1) / threads_per_block;
  return static_cast<int>(std::min(blocks_needed, static_cast<size_t>(std::numeric_limits<int>::max())));
}

// Bilinear interpolation at (h, w). Returns 0 if out of bounds (ONNX spec).
template <typename T>
__device__ __inline__ T BilinearInterpolate(
    const T* in,
    int64_t height,
    int64_t width,
    T h,
    T w) {
  if (h <= static_cast<T>(-1) || h >= height || w <= static_cast<T>(-1) || w >= width) {
    return static_cast<T>(0);
  }
  int h_low = static_cast<int>(_Floor(h));
  int w_low = static_cast<int>(_Floor(w));
  int h_high = h_low + 1;
  int w_high = w_low + 1;

  T lh = h - static_cast<T>(h_low);
  T lw = w - static_cast<T>(w_low);
  T hh = static_cast<T>(1) - lh;
  T hw = static_cast<T>(1) - lw;

  T v1 = (h_low >= 0 && w_low >= 0) ? __ldg(in + h_low * width + w_low) : static_cast<T>(0);
  T v2 = (h_low >= 0 && w_high < width) ? __ldg(in + h_low * width + w_high) : static_cast<T>(0);
  T v3 = (h_high < height && w_low >= 0) ? __ldg(in + h_high * width + w_low) : static_cast<T>(0);
  T v4 = (h_high < height && w_high < width) ? __ldg(in + h_high * width + w_high) : static_cast<T>(0);

  T w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;
  return w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4;
}

// Coalesced bilinear for SharedKernel Phase 1: one warp loads for one channel.
// 32 threads load 32 consecutive elements from rows h_low and h_high (coalesced).
// Lane 0 has (h_low,w_low),(h_low,w_high); lane 1 has (h_high,w_low),(h_high,w_high). Lane 0 computes and broadcasts.
template <typename T>
__device__ __inline__ T BilinearInterpolateWarpCoalesced(
    const T* ch_in, int64_t height, int64_t width,
    int h_low, int w_low, float lh, float lw) {
  const int h_high = h_low + 1;
  const float hh = 1.0f - lh, hw = 1.0f - lw;
  const int lane = static_cast<int>(threadIdx.x % GPU_WARP_SIZE);
  float v_row0 = 0.0f, v_row1 = 0.0f;
  if (w_low + lane < width) {
    if (h_low >= 0) v_row0 = static_cast<float>(DeformConvLdg(ch_in + h_low * width + w_low + lane));
    if (h_high < height) v_row1 = static_cast<float>(DeformConvLdg(ch_in + h_high * width + w_low + lane));
  }
  const float v1 = v_row0;
  const float v2 = WARP_SHFL(v_row0, 1);
  const float v3 = v_row1;
  const float v4 = WARP_SHFL(v_row1, 1);
  float r = (lane == 0) ? (hh * hw * v1 + hh * lw * v2 + lh * hw * v3 + lh * lw * v4) : 0.0f;
  r = WARP_SHFL(r, 0);
  return static_cast<T>(r);
}

// FP16/BF16: coordinate and weight math in float to avoid precision loss.
template <typename T>
struct DeformConvUseFloatCoords : std::false_type {};
template <>
struct DeformConvUseFloatCoords<half> : std::true_type {};
template <>
struct DeformConvUseFloatCoords<BFloat16> : std::true_type {};

// __ldg has no overload for BFloat16*; use 16-bit load + FromBits. Other types use __ldg directly.
template <typename T>
__device__ __inline__ T DeformConvLdg(const T* p) {
  return __ldg(p);
}
template <>
__device__ __inline__ BFloat16 DeformConvLdg<BFloat16>(const BFloat16* p) {
  return BFloat16::FromBits(__ldg(reinterpret_cast<const uint16_t*>(p)));
}

__device__ __inline__ half BilinearInterpolate(
    const half* in,
    int64_t height,
    int64_t width,
    float h,
    float w) {
  if (h <= -1.0f || h >= height || w <= -1.0f || w >= width) {
    return __float2half(0.0f);
  }
  int h_low = static_cast<int>(floorf(h));
  int w_low = static_cast<int>(floorf(w));
  int h_high = h_low + 1;
  int w_high = w_low + 1;

  float lh = h - static_cast<float>(h_low);
  float lw = w - static_cast<float>(w_low);
  float hh = 1.0f - lh;
  float hw = 1.0f - lw;

  float v1 = (h_low >= 0 && w_low >= 0) ? __half2float(__ldg(in + h_low * width + w_low)) : 0.0f;
  float v2 = (h_low >= 0 && w_high < width) ? __half2float(__ldg(in + h_low * width + w_high)) : 0.0f;
  float v3 = (h_high < height && w_low >= 0) ? __half2float(__ldg(in + h_high * width + w_low)) : 0.0f;
  float v4 = (h_high < height && w_high < width) ? __half2float(__ldg(in + h_high * width + w_high)) : 0.0f;

  float w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;
  return __float2half(w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
}

__device__ __inline__ BFloat16 BilinearInterpolate(
    const BFloat16* in,
    int64_t height,
    int64_t width,
    float h,
    float w) {
  if (h <= -1.0f || h >= height || w <= -1.0f || w >= width) {
    return BFloat16(0.0f);
  }
  int h_low = static_cast<int>(floorf(h));
  int w_low = static_cast<int>(floorf(w));
  int h_high = h_low + 1;
  int w_high = w_low + 1;

  float lh = h - static_cast<float>(h_low);
  float lw = w - static_cast<float>(w_low);
  float hh = 1.0f - lh;
  float hw = 1.0f - lw;

  float v1 = (h_low >= 0 && w_low >= 0) ? static_cast<float>(DeformConvLdg(in + h_low * width + w_low)) : 0.0f;
  float v2 = (h_low >= 0 && w_high < width) ? static_cast<float>(DeformConvLdg(in + h_low * width + w_high)) : 0.0f;
  float v3 = (h_high < height && w_low >= 0) ? static_cast<float>(DeformConvLdg(in + h_high * width + w_low)) : 0.0f;
  float v4 = (h_high < height && w_high < width) ? static_cast<float>(DeformConvLdg(in + h_high * width + w_high)) : 0.0f;

  float w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;
  return BFloat16(w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
}

// kH/kW = -1 means dynamic (runtime); >= 0 means compile-time constant for loop unrolling.
template <typename T, typename IndexT, int kH = -1, int kW = -1>
__global__ void DeformableIm2ColKernel(
    IndexT num_kernels,
    const T* __restrict__ input,
    const T* __restrict__ offset,
    const T* __restrict__ mask,
    int64_t height,
    int64_t width,
    int64_t weight_h,
    int64_t weight_w,
    int64_t pad_h,
    int64_t pad_w,
    int64_t stride_h,
    int64_t stride_w,
    int64_t dilation_h,
    int64_t dilation_w,
    int64_t channels,
    int64_t offset_group,
    DivMod<IndexT> out_h_div,
    DivMod<IndexT> out_w_div,
    DivMod<IndexT> parallel_imgs_div,
    DivMod<IndexT> channel_per_offset_grp_div,
    bool use_mask,
    T* __restrict__ data_col) {
  constexpr bool is_fixed = (kH >= 0 && kW >= 0);
  const int64_t h_dim = is_fixed ? kH : weight_h;
  const int64_t w_dim = is_fixed ? kW : weight_w;

  // Reconstruct dimensions from DivMod objects
  const int64_t out_h = out_h_div.d_;
  const int64_t out_w = out_w_div.d_;
  const int64_t parallel_imgs = parallel_imgs_div.d_;

  const int64_t out_size = out_h * out_w;
  // The stride for data_col is (parallel_imgs * out_h * out_w)
  const int64_t col_stride = parallel_imgs * out_size;

  using CoordT = typename std::conditional<DeformConvUseFloatCoords<T>::value, float, T>::type;

  for (IndexT index = blockIdx.x * blockDim.x + threadIdx.x; index < num_kernels; index += blockDim.x * gridDim.x) {
    IndexT val = index;
    IndexT out_x, out_y, out_b, in_c;

    // Fast division/modulo to recover coordinates
    out_w_div.divmod(val, val, out_x);
    out_h_div.divmod(val, val, out_y);
    parallel_imgs_div.divmod(val, in_c, out_b);

    // [Optimization 3] Avoid expensive division if offset_group is 1 (very common case).
    IndexT offset_grp = 0;
    if (offset_group > 1) {
      IndexT dummy;
      channel_per_offset_grp_div.divmod(in_c, offset_grp, dummy);
    }

    // [Optimization 2] Common Subexpression Elimination (CSE) & Pointer Arithmetic
    // Pre-calculate base pointers to reduce integer arithmetic inside the inner loops.

    // 1. Input pointer base for this batch and channel.
    const T* input_ptr = input + static_cast<int64_t>(out_b) * (channels * height * width) + static_cast<int64_t>(in_c) * (height * width);

    // 2. Spatial index in the output feature map.
    const int64_t spatial_idx = static_cast<int64_t>(out_y) * out_w + static_cast<int64_t>(out_x);

    // 3. Offset pointer base calculation.
    // Layout: (N, offset_groups, 2*KH*KW, OH, OW)
    // We pre-calculate the pointer to the start of the specific (n, g) block, plus spatial_idx.
    const int64_t offset_group_block_size = 2 * h_dim * w_dim * out_size;
    const T* offset_ptr_base = offset + (static_cast<int64_t>(out_b) * offset_group + static_cast<int64_t>(offset_grp)) * offset_group_block_size + spatial_idx;

    // 4. Mask pointer base calculation (if used).
    // Layout: (N, offset_groups, KH*KW, OH, OW)
    const T* mask_ptr_base = nullptr;
    if (use_mask) {
      const int64_t mask_group_block_size = h_dim * w_dim * out_size;
      mask_ptr_base = mask + (static_cast<int64_t>(out_b) * offset_group + static_cast<int64_t>(offset_grp)) * mask_group_block_size + spatial_idx;
    }

    // 5. Output pointer base calculation.
    // data_col Layout: (C * KH * KW, N * OH * OW)
    // The current thread writes to the column `c_col` = (b * OH * OW) + spatial_idx.
    // The starting row for this channel is `in_c * KH * KW`.
    const int64_t c_col = static_cast<int64_t>(out_b) * out_size + spatial_idx;
    T* data_col_ptr_base = data_col + (static_cast<int64_t>(in_c) * h_dim * w_dim) * col_stride + c_col;

    // 6. Pre-calculate invariant coordinate parts.
    // Use float for coordinate math when T is half or BFloat16 to avoid precision loss.
    const CoordT base_h_im = static_cast<CoordT>(out_y * stride_h - pad_h);
    const CoordT base_w_im = static_cast<CoordT>(out_x * stride_w - pad_w);

    auto process_kernel_point = [&](int64_t i, int64_t j) {
      const int64_t kernel_idx = i * w_dim + j;
      T mask_val = static_cast<T>(1);
      if (use_mask) {
        // Access mask using pre-calculated base and stride.
        mask_val = DeformConvLdg(mask_ptr_base + kernel_idx * out_size);

        // [Optimization 1] Early Exit / Pruning
        // If mask is 0, the contribution is 0. Skip expensive offset load and interpolation.
        // Note: casting to float for comparison is safe for standard floating point types.
        if (static_cast<float>(mask_val) == 0.0f) {
          data_col_ptr_base[kernel_idx * col_stride] = static_cast<T>(0);
          return;
        }
      }

      // Calculate offset pointers relative to the base.
      // The offset tensor stores (y_offset, x_offset) pairs for each kernel weight.
      // Stride between y_offset and x_offset is `out_size`.
      const int64_t offset_offset_idx = (2 * kernel_idx) * out_size;

      const CoordT offset_h = static_cast<CoordT>(DeformConvLdg(offset_ptr_base + offset_offset_idx));
      const CoordT offset_w = static_cast<CoordT>(DeformConvLdg(offset_ptr_base + offset_offset_idx + out_size));

      const CoordT h_im = base_h_im + static_cast<CoordT>(i * dilation_h) + offset_h;
      const CoordT w_im = base_w_im + static_cast<CoordT>(j * dilation_w) + offset_w;

      T val = BilinearInterpolate(input_ptr, height, width, h_im, w_im);

      // Write result to data_col using pre-calculated base.
      data_col_ptr_base[kernel_idx * col_stride] = val * mask_val;
    };

    if constexpr (is_fixed) {
#pragma unroll
      for (int i = 0; i < kH; ++i) {
#pragma unroll
        for (int j = 0; j < kW; ++j) {
          process_kernel_point(i, j);
        }
      }
    } else {
      for (int64_t i = 0; i < weight_h; ++i) {
        for (int64_t j = 0; j < weight_w; ++j) {
          process_kernel_point(i, j);
        }
      }
    }
  }
}

// Bias add: Y[n,m,oh,ow] += B[m]. Layout NCHW.
template <typename T>
__global__ void DeformConvAddBiasKernel(
    T* Y,
    const T* B,
    DivMod<int64_t> spatial_div,  // For dividing by (H * W)
    DivMod<int64_t> channel_div,  // For dividing by M (channel count)
    int64_t total_elements) {
  for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total_elements; idx += blockDim.x * gridDim.x) {
    int64_t val = idx;
    int64_t batch_channel_idx, pixel_idx;

    // 1. First decomposition: decompose idx into (batch_channel_idx, pixel_idx)
    // Equivalent to: batch_channel_idx = idx / (H*W); pixel_idx = idx % (H*W);
    spatial_div.divmod(val, batch_channel_idx, pixel_idx);

    int64_t batch_idx, channel_idx;

    // 2. Second decomposition: decompose batch_channel_idx into (batch_idx, channel_idx)
    // Equivalent to: channel_idx = batch_channel_idx % M;
    // We only need channel_idx (i.e. m)
    channel_div.divmod(batch_channel_idx, batch_idx, channel_idx);

    // channel_idx is what we need (i.e. m)
    Y[idx] += DeformConvLdg(B + channel_idx);
  }
}

constexpr int64_t kDeformConv1x1MaxSharedC = 4096;

// WriteOutput: ONLY place that adds bias. dot_result must NOT include bias.
template <typename T>
__device__ __forceinline__ void DeformConv1x1WriteOutput(
    T* output, int64_t b, int64_t m, int64_t spatial_idx, int64_t M, int64_t out_size,
    T dot_result, const T* bias) {
  output[b * M * out_size + m * out_size + spatial_idx] =
      bias != nullptr ? dot_result + DeformConvLdg(bias + m) : dot_result;
}

template <typename T>
__device__ __forceinline__ void DeformConv1x1WriteBiasOrZero(
    T* output, int64_t b, int64_t m, int64_t spatial_idx, int64_t M, int64_t out_size, const T* bias) {
  output[b * M * out_size + m * out_size + spatial_idx] =
      bias != nullptr ? DeformConvLdg(bias + m) : static_cast<T>(0);
}

// Warp reduce for warp-tile: 32 threads sum to lane 0. Uses WARP_SHFL_DOWN.
template <typename T>
__device__ __forceinline__ T DeformConvWarpReduceSum(T val) {
#pragma unroll
  for (int offset = GPU_WARP_SIZE / 2; offset > 0; offset /= 2) {
    val += WARP_SHFL_DOWN(val, offset);
  }
  return val;
}

constexpr int kDeformConv1x1WarpsPerBlock = 8;
constexpr int kDeformConv1x1OutputsPerBlock = kDeformConv1x1WarpsPerBlock;

static_assert(GPU_WARP_SIZE == 32, "DeformConv1x1SharedKernel assumes warp size 32");
static_assert(kDeformConvThreadsPerBlock == kDeformConv1x1WarpsPerBlock * GPU_WARP_SIZE,
              "block size must equal warps_per_block * warp_size");

// 1x1 kernel with block-level shared sampling + warp-tile. Use when offset_group=1.
// UseMask: compile-time, avoid runtime branch when mask is always 1.
template <typename T, bool UseMask>
__global__ void FusedDeformConv1x1SharedKernel(
    const T* __restrict__ input,
    const T* __restrict__ offset,
    const T* __restrict__ mask,
    const T* __restrict__ weight,
    const T* __restrict__ bias,
    T* __restrict__ output,
    int64_t N,
    int64_t C,
    int64_t H,
    int64_t W,
    int64_t M,
    int64_t out_h,
    int64_t out_w,
    int64_t pad_h,
    int64_t pad_w,
    int64_t stride_h,
    int64_t stride_w,
    int64_t group,
    int64_t K_positions) {
  using CoordT = typename std::conditional<DeformConvUseFloatCoords<T>::value, float, T>::type;
  const int64_t out_size = out_h * out_w;
  const int64_t C_per_group = C / group;
  const int64_t M_per_group = M / group;
  extern __shared__ char smem_raw[];
  T* smem = reinterpret_cast<T*>(smem_raw);

  const int64_t m_blocks = (M + kDeformConv1x1OutputsPerBlock - 1) / kDeformConv1x1OutputsPerBlock;
  const int64_t spatial_total = N * out_size;
  const int64_t spatial_blocks = (spatial_total + K_positions - 1) / K_positions;
  const int64_t block_row = blockIdx.x / m_blocks;
  const int64_t m_block = blockIdx.x % m_blocks;
  const int64_t m_start = m_block * kDeformConv1x1OutputsPerBlock;
  const int64_t m_end = (m_start + kDeformConv1x1OutputsPerBlock < M) ? (m_start + kDeformConv1x1OutputsPerBlock) : M;
  const int64_t num_outputs = m_end - m_start;

  const int64_t spatial_start = block_row * K_positions;
  const int64_t spatial_end = (spatial_start + K_positions < spatial_total) ? (spatial_start + K_positions) : spatial_total;
  const int64_t K_actual = spatial_end - spatial_start;
  const int64_t tasks_total = K_actual * num_outputs;

  const int warp_idx = static_cast<int>(threadIdx.x / GPU_WARP_SIZE);
  const int lane = static_cast<int>(threadIdx.x % GPU_WARP_SIZE);
  using AccT = typename std::conditional<std::is_same<T, double>::value, double, float>::type;

  // Phase 1: sample for all K positions. Early skip when mask==0 (sparse mask).
  for (int64_t pos_idx = 0; pos_idx < K_actual; pos_idx++) {
    const int64_t spatial_idx = spatial_start + pos_idx;
    const int64_t b = spatial_idx / out_size;
    const int64_t oh = (spatial_idx % out_size) / out_w;
    const int64_t ow = spatial_idx % out_w;
    const CoordT base_h = static_cast<CoordT>(oh * stride_h - pad_h);
    const CoordT base_w = static_cast<CoordT>(ow * stride_w - pad_w);
    const int64_t offset_base = b * 2 * out_size + spatial_idx;
    const CoordT offset_h = static_cast<CoordT>(DeformConvLdg(offset + offset_base));
    const CoordT offset_w = static_cast<CoordT>(DeformConvLdg(offset + offset_base + out_size));
    const CoordT h_im = base_h + offset_h;
    const CoordT w_im = base_w + offset_w;
    T mask_val = UseMask ? DeformConvLdg(mask + b * out_size + spatial_idx) : static_cast<T>(1);
    const bool mask_zero = UseMask && (static_cast<float>(mask_val) == 0.0f);

    T* smem_pos = smem + pos_idx * C;
    const int64_t chunk = kDeformConv1x1WarpsPerBlock * GPU_WARP_SIZE;
    if (mask_zero) {
      for (int64_t base = warp_idx * GPU_WARP_SIZE; base < C; base += chunk) {
        const int n = static_cast<int>(std::min(static_cast<int64_t>(GPU_WARP_SIZE), C - base));
        if (lane < n) smem_pos[base + lane] = static_cast<T>(0);
      }
    } else {
      const int64_t input_batch_offset = b * C * H * W;
      const T* in_base = input + input_batch_offset;
      const float h_im_f = static_cast<float>(h_im), w_im_f = static_cast<float>(w_im);
      const bool use_coalesced = (W >= 2 && h_im_f > -1.0f && h_im_f < static_cast<float>(H) &&
                                  w_im_f > -1.0f && w_im_f < static_cast<float>(W));
      const int h_low = static_cast<int>(floorf(h_im_f)), w_low = static_cast<int>(floorf(w_im_f));
      const float lh = h_im_f - h_low, lw = w_im_f - w_low;
      for (int64_t base = warp_idx * GPU_WARP_SIZE; base < C; base += chunk) {
        T my_val = static_cast<T>(0);
        const int n = static_cast<int>(std::min(static_cast<int64_t>(GPU_WARP_SIZE), C - base));
#pragma unroll
        for (int i = 0; i < GPU_WARP_SIZE; i++) {
          if (i >= n) break;
          const int64_t c = base + i;
          T val = use_coalesced
              ? (BilinearInterpolateWarpCoalesced(in_base + c * H * W, H, W, h_low, w_low, lh, lw) * mask_val)
              : ((lane == 0) ? (BilinearInterpolate(in_base + c * H * W, H, W, h_im, w_im) * mask_val) : static_cast<T>(0));
          if (!use_coalesced) val = WARP_SHFL(val, 0);
          if (lane == i) my_val = val;
        }
        if (lane < n) smem_pos[base + lane] = my_val;
      }
    }
  }
  __syncthreads();

  // Phase 2: each warp computes (pos, m) outputs
  for (int64_t task = warp_idx; task < tasks_total; task += kDeformConv1x1WarpsPerBlock) {
    const int64_t pos_idx = task / num_outputs;
    const int64_t m_offset = task % num_outputs;
    const int64_t spatial_idx = spatial_start + pos_idx;
    const int64_t b = spatial_idx / out_size;
    const int64_t m = m_start + m_offset;
    const T* smem_pos = smem + pos_idx * C;

    T mask_val = static_cast<T>(1);
    if (UseMask) {
      mask_val = DeformConvLdg(mask + b * out_size + spatial_idx);
      if (static_cast<float>(mask_val) == 0.0f) {
        if (lane == 0) DeformConv1x1WriteBiasOrZero(output, b, m, spatial_idx, M, out_size, bias);
        continue;
      }
    }

    const int64_t g = m / M_per_group;
    const T* W_row = weight + (g * M_per_group + m % M_per_group) * C_per_group;
    const T* smem_g = smem_pos + g * C_per_group;
    AccT acc = static_cast<AccT>(0);
    const bool w_aligned_f4 = (reinterpret_cast<uintptr_t>(W_row) & 15) == 0;
    const bool s_aligned_f4 = (reinterpret_cast<uintptr_t>(smem_g) & 15) == 0;
    const bool w_aligned_h2 = (reinterpret_cast<uintptr_t>(W_row) & 3) == 0;
    const bool s_aligned_h2 = (reinterpret_cast<uintptr_t>(smem_g) & 3) == 0;
    const bool c_div4 = (C_per_group & 3) == 0;
    const bool c_div2 = (C_per_group & 1) == 0;
    if (std::is_same<T, float>::value && w_aligned_f4 && s_aligned_f4 && c_div4) {
      for (int64_t k = lane * 4; k < C_per_group; k += GPU_WARP_SIZE * 4) {
        float4 w = *reinterpret_cast<const float4*>(W_row + k);
        float4 s = *reinterpret_cast<const float4*>(smem_g + k);
        acc += w.x * s.x + w.y * s.y + w.z * s.z + w.w * s.w;
      }
    } else if (std::is_same<T, half>::value && w_aligned_h2 && s_aligned_h2 && c_div2) {
      for (int64_t k = lane * 2; k < C_per_group; k += GPU_WARP_SIZE * 2) {
        half2 w = *reinterpret_cast<const half2*>(W_row + k);
        half2 s = *reinterpret_cast<const half2*>(smem_g + k);
        acc += __half2float(w.x) * __half2float(s.x) + __half2float(w.y) * __half2float(s.y);
      }
    } else {
      for (int64_t k = 0; k < C_per_group; k += GPU_WARP_SIZE) {
        const int64_t c = k + lane;
        if (c < C_per_group) {
          acc += static_cast<AccT>(DeformConvLdg(W_row + c)) * static_cast<AccT>(smem_g[c]);
        }
      }
    }
    acc = DeformConvWarpReduceSum(acc);
    if (lane == 0) {
      DeformConv1x1WriteOutput(output, b, m, spatial_idx, M, out_size, static_cast<T>(acc), bias);
    }
  }
}

// Fused 1x1 DeformConv: per-thread path. Optimizations: offset_group=1 hoists offset/mask.
template <typename T>
__global__ void FusedDeformConv1x1Kernel(
    const T* __restrict__ input,
    const T* __restrict__ offset,
    const T* __restrict__ mask,
    const T* __restrict__ weight,
    const T* __restrict__ bias,
    T* __restrict__ output,
    int64_t N,
    int64_t C,
    int64_t H,
    int64_t W,
    int64_t M,
    int64_t out_h,
    int64_t out_w,
    int64_t pad_h,
    int64_t pad_w,
    int64_t stride_h,
    int64_t stride_w,
    int64_t group,
    int64_t offset_group,
    bool use_mask) {
  using CoordT = typename std::conditional<DeformConvUseFloatCoords<T>::value, float, T>::type;
  const int64_t out_size = out_h * out_w;
  const int64_t C_per_group = C / group;
  const int64_t M_per_group = M / group;
  const int64_t C_per_offset_grp = C / offset_group;

  for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < N * M * out_size; idx += blockDim.x * gridDim.x) {
    int64_t val = idx;
    const int64_t spatial_idx = val % out_size;
    val /= out_size;
    const int64_t m = val % M;
    val /= M;
    const int64_t b = val;
    const int64_t oh = spatial_idx / out_w;
    const int64_t ow = spatial_idx % out_w;

    const int64_t g = m / M_per_group;
    const int64_t m_local = m % M_per_group;
    const CoordT base_h = static_cast<CoordT>(oh * stride_h - pad_h);
    const CoordT base_w = static_cast<CoordT>(ow * stride_w - pad_w);

    T acc = static_cast<T>(0);
    const T* W_row = weight + (g * M_per_group + m_local) * C_per_group;
    const int64_t input_batch_offset = b * C * H * W;

    // offset_group=1: load once; else load per channel
    CoordT h_im, w_im;
    T mask_val = static_cast<T>(1);
    if (offset_group == 1) {
      const int64_t ob = b * 2 * out_size + spatial_idx;
      h_im = base_h + static_cast<CoordT>(DeformConvLdg(offset + ob));
      w_im = base_w + static_cast<CoordT>(DeformConvLdg(offset + ob + out_size));
      if (use_mask) {
        mask_val = DeformConvLdg(mask + b * out_size + spatial_idx);
        if (static_cast<float>(mask_val) == 0.0f) {
          DeformConv1x1WriteBiasOrZero(output, b, m, spatial_idx, M, out_size, bias);
          continue;
        }
      }
    }

    for (int64_t c = 0; c < C_per_group; ++c) {
      const int64_t c_global = g * C_per_group + c;
      CoordT ch, cw;
      T cm = static_cast<T>(1);
      if (offset_group > 1) {
        const int64_t og = c_global / C_per_offset_grp;
        const int64_t ob = (b * offset_group + og) * 2 * out_size + spatial_idx;
        ch = base_h + static_cast<CoordT>(DeformConvLdg(offset + ob));
        cw = base_w + static_cast<CoordT>(DeformConvLdg(offset + ob + out_size));
        if (use_mask) {
          cm = DeformConvLdg(mask + (b * offset_group + og) * out_size + spatial_idx);
          if (static_cast<float>(cm) == 0.0f) continue;
        }
      } else {
        ch = h_im;
        cw = w_im;
        cm = mask_val;
      }
      T s = BilinearInterpolate(input + input_batch_offset + c_global * H * W, H, W, ch, cw);
      acc += DeformConvLdg(W_row + c) * (s * cm);
    }
    DeformConv1x1WriteOutput(output, b, m, spatial_idx, M, out_size, acc, bias);
  }
}

// Copy GEMM output (row-major [M_per_group, cur_parallel*output_image_size]) into NCHW Y_g.
// src(c, j) with j = b_idx*output_image_size + pos -> dst[b_idx*M*output_image_size + c*output_image_size + pos].
template <typename T>
__global__ void CopyGemmOutputRowMajorToNCHWKernel(
    const T* __restrict__ src,
    T* __restrict__ dst,
    int64_t M,
    int64_t M_per_group,
    int64_t output_image_size,
    int64_t cur_parallel) {
  int64_t total = cur_parallel * M_per_group * output_image_size;
  for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += blockDim.x * gridDim.x) {
    int64_t pos = idx % output_image_size;
    int64_t c = (idx / output_image_size) % M_per_group;
    int64_t b_idx = idx / (output_image_size * M_per_group);
    int64_t j = b_idx * output_image_size + pos;
    // src index for row-major: c * (cur_parallel * output_image_size) + j
    dst[b_idx * M * output_image_size + c * output_image_size + pos] = src[c * (cur_parallel * output_image_size) + j];
  }
}

}  // namespace

template <typename T>
Status DeformConvAddBiasImpl(cudaStream_t stream, T* Y, const T* B, int64_t N, int64_t M, int64_t out_h, int64_t out_w) {
  int64_t total = N * M * out_h * out_w;
  if (total <= 0) return Status::OK();

  // 1. Prepare divisor
  int64_t out_size = out_h * out_w;

  // 2. Create FastDivMod object (note: ensure int64_t version of DivMod is used here)
  DivMod<int64_t> spatial_div(out_size);
  DivMod<int64_t> channel_div(M);

  int blocks = GetGridSize(static_cast<size_t>(total), kDeformConvThreadsPerBlock);

  // 3. Pass DivMod objects
  DeformConvAddBiasKernel<T><<<blocks, kDeformConvThreadsPerBlock, 0, stream>>>(
      Y,
      B,
      spatial_div,
      channel_div,
      total);
  return CUDA_CALL(cudaGetLastError());
}

template <typename T>
Status DeformConvCopyGemmOutputRowMajorToNCHW(
    cudaStream_t stream,
    const T* gemm_output,
    T* Y_g,
    int64_t M,
    int64_t M_per_group,
    int64_t output_image_size,
    int64_t cur_parallel) {
  int64_t total = cur_parallel * M_per_group * output_image_size;
  if (total <= 0) return Status::OK();
  int blocks = GetGridSize(static_cast<size_t>(total), kDeformConvThreadsPerBlock);
  CopyGemmOutputRowMajorToNCHWKernel<T><<<blocks, kDeformConvThreadsPerBlock, 0, stream>>>(
      gemm_output, Y_g, M, M_per_group, output_image_size, cur_parallel);
  return CUDA_CALL(cudaGetLastError());
}

template <typename T>
Status DeformConvFused1x1Impl(
    cudaStream_t stream,
    const DeformConvParams& params,
    const T* input,
    const T* offset,
    const T* mask,
    const T* weight,
    const T* bias,
    T* output,
    size_t max_smem_per_block) {
  const int64_t total = params.N * params.M * params.out_h * params.out_w;
  if (total <= 0) return Status::OK();

  const int64_t out_size = params.out_h * params.out_w;
  const int64_t spatial_total = params.N * out_size;
  const size_t smem_bytes = static_cast<size_t>(params.C) * sizeof(T);
  const size_t smem_limit = (max_smem_per_block > 0) ? max_smem_per_block : 16384;

  if (params.offset_group == 1 && params.C <= kDeformConv1x1MaxSharedC && smem_bytes <= smem_limit) {
    const int64_t m_blocks = (params.M + kDeformConv1x1OutputsPerBlock - 1) / kDeformConv1x1OutputsPerBlock;
    const int64_t K_max_smem = static_cast<int64_t>(smem_limit / (params.C * sizeof(T)));
    const int64_t K_positions = (params.C < 256)
        ? std::min(std::max(static_cast<int64_t>(1), (256 + params.C - 1) / params.C), std::max(static_cast<int64_t>(1), K_max_smem))
        : 1;
    const size_t smem_k = static_cast<size_t>(K_positions) * params.C * sizeof(T);
    const int64_t spatial_blocks = (spatial_total + K_positions - 1) / K_positions;
    const size_t blocks_needed = static_cast<size_t>(spatial_blocks) * static_cast<size_t>(m_blocks);
    if (blocks_needed <= static_cast<size_t>(std::numeric_limits<int>::max())) {
      const int blocks = static_cast<int>(blocks_needed);
      if (params.use_mask) {
        FusedDeformConv1x1SharedKernel<T, true><<<blocks, kDeformConvThreadsPerBlock, smem_k, stream>>>(
            input, offset, mask, weight, bias, output,
            params.N, params.C, params.H, params.W_in, params.M,
            params.out_h, params.out_w,
            params.pad_h, params.pad_w, params.stride_h, params.stride_w,
            params.group, K_positions);
      } else {
        FusedDeformConv1x1SharedKernel<T, false><<<blocks, kDeformConvThreadsPerBlock, smem_k, stream>>>(
            input, offset, mask, weight, bias, output,
            params.N, params.C, params.H, params.W_in, params.M,
            params.out_h, params.out_w,
            params.pad_h, params.pad_w, params.stride_h, params.stride_w,
            params.group, K_positions);
      }
    } else {
      int blocks = GetGridSize(static_cast<size_t>(total), kDeformConvThreadsPerBlock);
      FusedDeformConv1x1Kernel<T><<<blocks, kDeformConvThreadsPerBlock, 0, stream>>>(
        input, offset, mask, weight, bias, output,
        params.N, params.C, params.H, params.W_in, params.M,
        params.out_h, params.out_w,
        params.pad_h, params.pad_w, params.stride_h, params.stride_w,
        params.group, params.offset_group, params.use_mask);
    }
  } else {
    int blocks = GetGridSize(static_cast<size_t>(total), kDeformConvThreadsPerBlock);
    FusedDeformConv1x1Kernel<T><<<blocks, kDeformConvThreadsPerBlock, 0, stream>>>(
        input, offset, mask, weight, bias, output,
        params.N, params.C, params.H, params.W_in, params.M,
        params.out_h, params.out_w,
        params.pad_h, params.pad_w, params.stride_h, params.stride_w,
        params.group, params.offset_group, params.use_mask);
  }
  return CUDA_CALL(cudaGetLastError());
}

template <typename T>
Status DeformConvIm2ColImpl(
    cudaStream_t stream,
    const T* input,
    const T* offset,
    const T* mask,
    T* col_buffer,
    int64_t parallel_imgs,
    int64_t C,
    int64_t H,
    int64_t W,
    int64_t kH,
    int64_t kW,
    int64_t out_h,
    int64_t out_w,
    int64_t pad_h,
    int64_t pad_w,
    int64_t stride_h,
    int64_t stride_w,
    int64_t dilation_h,
    int64_t dilation_w,
    int64_t offset_group,
    bool use_mask) {
  const int64_t num_kernels = static_cast<int64_t>(C) * out_h * out_w * parallel_imgs;
  if (num_kernels <= 0) {
    return Status::OK();
  }

  const int64_t col_numel = static_cast<int64_t>(C) * kH * kW * parallel_imgs * out_h * out_w;
  const bool use_64bit = (num_kernels > static_cast<int64_t>(std::numeric_limits<int32_t>::max())) ||
                         (col_numel > static_cast<int64_t>(std::numeric_limits<int32_t>::max()));

  int blocks = GetGridSize(static_cast<size_t>(num_kernels), kDeformConvThreadsPerBlock);

  auto launch = [&](auto kH_tag, auto kW_tag) {
    constexpr int KH = decltype(kH_tag)::value;
    constexpr int KW = decltype(kW_tag)::value;
    if (use_64bit) {
      DeformableIm2ColKernel<T, int64_t, KH, KW><<<blocks, kDeformConvThreadsPerBlock, 0, stream>>>(
          num_kernels, input, offset, mask, H, W, kH, kW, pad_h, pad_w,
          stride_h, stride_w, dilation_h, dilation_w, C, offset_group,
          DivMod<int64_t>(out_h), DivMod<int64_t>(out_w), DivMod<int64_t>(parallel_imgs),
          DivMod<int64_t>(C / offset_group), use_mask, col_buffer);
    } else {
      DeformableIm2ColKernel<T, int32_t, KH, KW><<<blocks, kDeformConvThreadsPerBlock, 0, stream>>>(
          static_cast<int32_t>(num_kernels), input, offset, mask, H, W, kH, kW, pad_h, pad_w,
          stride_h, stride_w, dilation_h, dilation_w, C, offset_group,
          DivMod<int32_t>(static_cast<int32_t>(out_h)),
          DivMod<int32_t>(static_cast<int32_t>(out_w)),
          DivMod<int32_t>(static_cast<int32_t>(parallel_imgs)),
          DivMod<int32_t>(static_cast<int32_t>(C / offset_group)),
          use_mask, col_buffer);
    }
  };

  if (kH == 3 && kW == 3) {
    launch(DeformConvKSize<3>{}, DeformConvKSize<3>{});
  } else if (kH == 5 && kW == 5) {
    launch(DeformConvKSize<5>{}, DeformConvKSize<5>{});
  } else {
    launch(DeformConvKSize<-1>{}, DeformConvKSize<-1>{});
  }
  return CUDA_CALL(cudaGetLastError());
}

#define INST_DeformConvIm2ColImpl(T) \
  template Status DeformConvIm2ColImpl<T>(cudaStream_t, const T*, const T*, const T*, T*, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, bool)

INST_DeformConvIm2ColImpl(float);
INST_DeformConvIm2ColImpl(double);
INST_DeformConvIm2ColImpl(half);
INST_DeformConvIm2ColImpl(BFloat16);

#define INST_DeformConvFused1x1Impl(T)                                                    \
  template Status DeformConvFused1x1Impl<T>(cudaStream_t, const DeformConvParams&,        \
                                           const T*, const T*, const T*, const T*,        \
                                           const T*, T*, size_t)

INST_DeformConvFused1x1Impl(float)
INST_DeformConvFused1x1Impl(double)
INST_DeformConvFused1x1Impl(half)
INST_DeformConvFused1x1Impl(BFloat16)

template Status DeformConvCopyGemmOutputRowMajorToNCHW<float>(cudaStream_t, const float*, float*, int64_t, int64_t, int64_t, int64_t);
template Status DeformConvCopyGemmOutputRowMajorToNCHW<double>(cudaStream_t, const double*, double*, int64_t, int64_t, int64_t, int64_t);
template Status DeformConvCopyGemmOutputRowMajorToNCHW<half>(cudaStream_t, const half*, half*, int64_t, int64_t, int64_t, int64_t);
template Status DeformConvCopyGemmOutputRowMajorToNCHW<BFloat16>(cudaStream_t, const BFloat16*, BFloat16*, int64_t, int64_t, int64_t, int64_t);

template Status DeformConvAddBiasImpl<float>(cudaStream_t, float*, const float*, int64_t, int64_t, int64_t, int64_t);
template Status DeformConvAddBiasImpl<double>(cudaStream_t, double*, const double*, int64_t, int64_t, int64_t, int64_t);
template Status DeformConvAddBiasImpl<half>(cudaStream_t, half*, const half*, int64_t, int64_t, int64_t, int64_t);
template Status DeformConvAddBiasImpl<BFloat16>(cudaStream_t, BFloat16*, const BFloat16*, int64_t, int64_t, int64_t, int64_t);

// Delegate ORT type to CUDA type (e.g. MLFloat16 -> half); avoids repeating three identical specializations.
#define DELEGATE_DEFORM_CONV_IMPL(ORT_T, CUDA_T)                                                                    \
  template <>                                                                                                       \
  Status DeformConvFused1x1Impl<ORT_T>(cudaStream_t stream, const DeformConvParams& params,                        \
                                        const ORT_T* input, const ORT_T* offset, const ORT_T* mask,               \
                                        const ORT_T* weight, const ORT_T* bias, ORT_T* output,                    \
                                        size_t max_smem_per_block) {                                                \
    return DeformConvFused1x1Impl<CUDA_T>(stream, params,                                                           \
                                          reinterpret_cast<const CUDA_T*>(input),                                   \
                                          reinterpret_cast<const CUDA_T*>(offset),                                 \
                                          mask ? reinterpret_cast<const CUDA_T*>(mask) : nullptr,                   \
                                          reinterpret_cast<const CUDA_T*>(weight),                                  \
                                          bias ? reinterpret_cast<const CUDA_T*>(bias) : nullptr,                   \
                                          reinterpret_cast<CUDA_T*>(output), max_smem_per_block);                    \
  }                                                                                                                 \
  template <>                                                                                                       \
  Status DeformConvIm2ColImpl<ORT_T>(cudaStream_t stream, const ORT_T* input,                                       \
                                     const ORT_T* offset, const ORT_T* mask, ORT_T* col_buffer,                     \
                                     int64_t parallel_imgs, int64_t C, int64_t H, int64_t W,                        \
                                     int64_t kH, int64_t kW, int64_t out_h, int64_t out_w,                          \
                                     int64_t pad_h, int64_t pad_w, int64_t stride_h, int64_t stride_w,              \
                                     int64_t dilation_h, int64_t dilation_w, int64_t offset_group, bool use_mask) { \
    return DeformConvIm2ColImpl<CUDA_T>(stream, reinterpret_cast<const CUDA_T*>(input),                             \
                                        reinterpret_cast<const CUDA_T*>(offset),                                    \
                                        mask ? reinterpret_cast<const CUDA_T*>(mask) : nullptr,                     \
                                        reinterpret_cast<CUDA_T*>(col_buffer),                                      \
                                        parallel_imgs, C, H, W, kH, kW, out_h, out_w,                               \
                                        pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w,                   \
                                        offset_group, use_mask);                                                    \
  }                                                                                                                 \
  template <>                                                                                                       \
  Status DeformConvCopyGemmOutputRowMajorToNCHW<ORT_T>(cudaStream_t stream,                                         \
                                                       const ORT_T* gemm_output, ORT_T* Y_g,                        \
                                                       int64_t M, int64_t M_per_group,                              \
                                                       int64_t output_image_size, int64_t cur_parallel) {           \
    return DeformConvCopyGemmOutputRowMajorToNCHW<CUDA_T>(stream,                                                   \
                                                          reinterpret_cast<const CUDA_T*>(gemm_output),             \
                                                          reinterpret_cast<CUDA_T*>(Y_g),                           \
                                                          M, M_per_group, output_image_size, cur_parallel);         \
  }                                                                                                                 \
  template <>                                                                                                       \
  Status DeformConvAddBiasImpl<ORT_T>(cudaStream_t stream, ORT_T * Y, const ORT_T* B,                               \
                                      int64_t N, int64_t M, int64_t out_h, int64_t out_w) {                         \
    return DeformConvAddBiasImpl<CUDA_T>(stream, reinterpret_cast<CUDA_T*>(Y),                                      \
                                         reinterpret_cast<const CUDA_T*>(B), N, M, out_h, out_w);                   \
  }

DELEGATE_DEFORM_CONV_IMPL(MLFloat16, half)

}  // namespace cuda
}  // namespace onnxruntime
