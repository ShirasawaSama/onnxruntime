// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//
// CUDA implementation of DeformConv: deformable im2col kernel + bilinear interpolation.
// Reference: torchvision deform_conv2d_kernel.cu, ONNX DeformConv spec.

#include "deform_conv_impl.h"
#include "core/providers/cuda/cu_inc/common.cuh"
#include "core/providers/cuda/shared_inc/fast_divmod.h"
#include "core/common/float16.h"
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

// Bilinear interpolation for NHWC input [H, W, C]. Samples at (h, w) for channel channel_idx.
// For coalesced access: when threads process same (h,w) with consecutive channel_idx, the 4 loads
// hit consecutive addresses (channels are contiguous at each spatial position).
template <typename T>
__device__ __inline__ T BilinearInterpolateNHWC(
    const T* in_base,
    int64_t height,
    int64_t width,
    int64_t channels,
    int64_t channel_idx,
    float h,
    float w) {
  if (h <= -1.0f || h >= height || w <= -1.0f || w >= width) {
    return static_cast<T>(0);
  }
  int h_low = static_cast<int>(floorf(h));
  int w_low = static_cast<int>(floorf(w));
  int h_high = h_low + 1;
  int w_high = w_low + 1;

  float lh = h - static_cast<float>(h_low);
  float lw = w - static_cast<float>(w_low);
  float hh = 1.0f - lh;
  float hw = 1.0f - lw;

  // NHWC: base addr for each spatial pos is (h*width+w)*channels; channel c is at +channel_idx
  const int64_t stride = channels;
  const int64_t p0 = (h_low * width + w_low) * stride + channel_idx;
  const int64_t p1 = (h_low * width + w_high) * stride + channel_idx;
  const int64_t p2 = (h_high * width + w_low) * stride + channel_idx;
  const int64_t p3 = (h_high * width + w_high) * stride + channel_idx;
  float v1 = (h_low >= 0 && w_low >= 0) ? static_cast<float>(DeformConvLdg(in_base + p0)) : 0.0f;
  float v2 = (h_low >= 0 && w_high < width) ? static_cast<float>(DeformConvLdg(in_base + p1)) : 0.0f;
  float v3 = (h_high < height && w_low >= 0) ? static_cast<float>(DeformConvLdg(in_base + p2)) : 0.0f;
  float v4 = (h_high < height && w_high < width) ? static_cast<float>(DeformConvLdg(in_base + p3)) : 0.0f;

  float w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;
  return static_cast<T>(w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
}

// kH/kW = -1 means dynamic (runtime); >= 0 means compile-time constant for loop unrolling.
// IsNHWC: true = input [N,H,W,C], false = input [N,C,H,W].
template <typename T, typename IndexT, bool IsNHWC, int kH = -1, int kW = -1>
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
    DivMod<IndexT> channel_div,  // For NHWC: channel varies fastest; divisor C for coalesced access
    bool use_mask,
    T* __restrict__ data_col) {
  constexpr bool is_fixed = (kH >= 0 && kW >= 0);
  const int64_t h_dim = is_fixed ? kH : weight_h;
  const int64_t w_dim = is_fixed ? kW : weight_w;

  // Reconstruct dimensions from DivMod objects
  const int64_t out_h = out_h_div.d_;
  const int64_t out_w = out_w_div.d_;

  const int64_t out_size = out_h * out_w;

  using CoordT = typename std::conditional<DeformConvUseFloatCoords<T>::value, float, T>::type;

  for (IndexT index = blockIdx.x * blockDim.x + threadIdx.x; index < num_kernels; index += blockDim.x * gridDim.x) {
    IndexT val = index;
    IndexT out_x, out_y, out_b, in_c;

    // For NHWC: channel varies fastest so threads in a warp load consecutive channels at same (h,w)
    // for coalesced bilinear interpolation. Index order: in_c + C*(out_x + out_w*(out_y + out_h*out_b))
    if constexpr (IsNHWC) {
      IndexT spatial_idx;
      channel_div.divmod(val, spatial_idx, in_c);  // spatial_idx=val/C, in_c=val%C
      out_w_div.divmod(spatial_idx, spatial_idx, out_x);
      out_h_div.divmod(spatial_idx, out_b, out_y);  // out_b=spatial/out_h, out_y=spatial%out_h
    } else {
      out_w_div.divmod(val, val, out_x);
      out_h_div.divmod(val, val, out_y);
      parallel_imgs_div.divmod(val, in_c, out_b);
    }

    // [Optimization 3] Avoid expensive division if offset_group is 1 (very common case).
    IndexT offset_grp = 0;
    if (offset_group > 1) {
      IndexT dummy;
      channel_per_offset_grp_div.divmod(in_c, offset_grp, dummy);
    }

    // [Optimization 2] Common Subexpression Elimination (CSE) & Pointer Arithmetic
    // Pre-calculate base pointers to reduce integer arithmetic inside the inner loops.

    // 1. Input pointer base for this batch and channel.
    const T* input_ptr;
    if constexpr (IsNHWC) {
      const int64_t input_image_size = height * width * channels;
      input_ptr = input + static_cast<int64_t>(out_b) * input_image_size;
    } else {
      input_ptr = input + static_cast<int64_t>(out_b) * (channels * height * width) + static_cast<int64_t>(in_c) * (height * width);
    }

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
    const int64_t c_col = static_cast<int64_t>(out_b) * out_size + spatial_idx;
    T* data_col_ptr_base;
    int64_t write_stride;
    if constexpr (IsNHWC) {
      // [L, C, kH×kW] layout: coalesced writes (stride=1), then reorder for Batched GEMM
      const int64_t kernel_dim = h_dim * w_dim;
      data_col_ptr_base = data_col + c_col * (channels * kernel_dim) + static_cast<int64_t>(in_c) * kernel_dim;
      write_stride = 1;
    } else {
      // [C*KH*KW, col_stride] row-major, matches Batched GEMM directly
      const int64_t parallel_imgs = parallel_imgs_div.d_;
      const int64_t col_stride = parallel_imgs * out_size;
      data_col_ptr_base = data_col + (static_cast<int64_t>(in_c) * h_dim * w_dim) * col_stride + c_col;
      write_stride = col_stride;
    }

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
          data_col_ptr_base[kernel_idx * write_stride] = static_cast<T>(0);
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

      T val;
      if constexpr (IsNHWC) {
        val = BilinearInterpolateNHWC(input_ptr, height, width, channels, static_cast<int64_t>(in_c), static_cast<float>(h_im), static_cast<float>(w_im));
      } else {
        val = BilinearInterpolate(input_ptr, height, width, h_im, w_im);
      }

      // Write result to data_col using pre-calculated base.
      data_col_ptr_base[kernel_idx * write_stride] = val * mask_val;
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

// Bias add: Y += B. IsNHWC: true = Y[n,oh,ow,m], false = Y[n,m,oh,ow].
template <typename T, bool IsNHWC>
__global__ void DeformConvAddBiasKernel(
    T* Y,
    const T* B,
    DivMod<int64_t> div1,  // NCHW: spatial(out_size); NHWC: channel(M)
    DivMod<int64_t> div2,  // NCHW: channel(M); NHWC: spatial(out_size)
    int64_t total_elements) {
  for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total_elements; idx += blockDim.x * gridDim.x) {
    int64_t channel_idx;
    if constexpr (IsNHWC) {
      int64_t batch_pixel_idx;
      div1.divmod(idx, batch_pixel_idx, channel_idx);
    } else {
      int64_t batch_channel_idx, pixel_idx;
      div1.divmod(idx, batch_channel_idx, pixel_idx);
      int64_t batch_idx;
      div2.divmod(batch_channel_idx, batch_idx, channel_idx);
    }
    Y[idx] += DeformConvLdg(B + channel_idx);
  }
}

// Reorder col from [L, C, kH*kW] (NHWC coalesced layout) to Batched GEMM format.
// Input:  src[l, c, k] at src + l*(C*kernel_size) + c*kernel_size + k
// Output: per-group column-major [cur_out_size, kernel_dim]; group g at dst + g*(cur_out_size*kernel_dim).
template <typename T>
__global__ void DeformConvColReorderLxCKToBatchedKernel(
    const T* __restrict__ src,
    T* __restrict__ dst,
    int64_t cur_out_size,
    int64_t C,
    int64_t kernel_size,
    int64_t kernel_dim,
    int64_t group) {
  const int64_t total = cur_out_size * C * kernel_size;
  for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += blockDim.x * gridDim.x) {
    int64_t g = idx / (cur_out_size * kernel_dim);
    int64_t rest = idx % (cur_out_size * kernel_dim);
    int64_t l = rest % cur_out_size;
    int64_t k = rest / cur_out_size;
    dst[idx] = DeformConvLdg(src + l * (C * kernel_size) + g * kernel_dim + k);
  }
}

// Copy GEMM output. IsNHWC: dst layout differs. channel_offset only used when IsNHWC.
// For NHWC: index order (b_idx, pos, c) with c fastest so consecutive threads write consecutive
// channels at same spatial pos = coalesced writes. src layout: (M_per_group, cur_parallel*output_image_size).
template <typename T, bool IsNHWC>
__global__ void CopyGemmOutputToLayoutKernel(
    const T* __restrict__ src,
    T* __restrict__ dst,
    int64_t M,
    int64_t M_per_group,
    int64_t channel_offset,
    int64_t output_image_size,
    int64_t cur_parallel) {
  int64_t total = cur_parallel * M_per_group * output_image_size;
  int64_t src_stride = cur_parallel * output_image_size;
  for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += blockDim.x * gridDim.x) {
    int64_t b_idx, pos, c;
    if constexpr (IsNHWC) {
      c = idx % M_per_group;
      int64_t spatial_idx = idx / M_per_group;
      pos = spatial_idx % output_image_size;
      b_idx = spatial_idx / output_image_size;
    } else {
      pos = idx % output_image_size;
      c = (idx / output_image_size) % M_per_group;
      b_idx = idx / (output_image_size * M_per_group);
    }
    int64_t j = b_idx * output_image_size + pos;
    T v = src[c * src_stride + j];
    if constexpr (IsNHWC) {
      dst[b_idx * output_image_size * M + pos * M + channel_offset + c] = v;
    } else {
      dst[b_idx * M * output_image_size + c * output_image_size + pos] = v;
    }
  }
}

}  // namespace

template <typename T, bool IsNHWC>
Status DeformConvAddBiasImplLayout(cudaStream_t stream, T* Y, const T* B, int64_t N, int64_t M, int64_t out_h, int64_t out_w) {
  int64_t total = N * M * out_h * out_w;
  if (total <= 0) return Status::OK();

  int64_t out_size = out_h * out_w;
  DivMod<int64_t> div1(IsNHWC ? M : out_size);
  DivMod<int64_t> div2(IsNHWC ? out_size : M);

  int blocks = GetGridSize(static_cast<size_t>(total), kDeformConvThreadsPerBlock);
  DeformConvAddBiasKernel<T, IsNHWC><<<blocks, kDeformConvThreadsPerBlock, 0, stream>>>(Y, B, div1, div2, total);
  return CUDA_CALL(cudaGetLastError());
}

template <typename T>
Status DeformConvAddBiasImpl(cudaStream_t stream, T* Y, const T* B, int64_t N, int64_t M, int64_t out_h, int64_t out_w) {
  return DeformConvAddBiasImplLayout<T, false>(stream, Y, B, N, M, out_h, out_w);
}

template <typename T>
Status DeformConvAddBiasImplNHWC(cudaStream_t stream, T* Y, const T* B, int64_t N, int64_t M, int64_t out_h, int64_t out_w) {
  return DeformConvAddBiasImplLayout<T, true>(stream, Y, B, N, M, out_h, out_w);
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
  CopyGemmOutputToLayoutKernel<T, false><<<blocks, kDeformConvThreadsPerBlock, 0, stream>>>(
      gemm_output, Y_g, M, M_per_group, 0, output_image_size, cur_parallel);
  return CUDA_CALL(cudaGetLastError());
}

template <typename T, bool IsNHWC>
Status DeformConvIm2ColImplLayout(
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
  if (num_kernels <= 0) return Status::OK();

  const int64_t col_numel = static_cast<int64_t>(C) * kH * kW * parallel_imgs * out_h * out_w;
  const bool use_64bit = (num_kernels > static_cast<int64_t>(std::numeric_limits<int32_t>::max())) ||
                         (col_numel > static_cast<int64_t>(std::numeric_limits<int32_t>::max()));

  int blocks = GetGridSize(static_cast<size_t>(num_kernels), kDeformConvThreadsPerBlock);

  auto launch = [&](auto kH_tag, auto kW_tag) {
    constexpr int KH = decltype(kH_tag)::value;
    constexpr int KW = decltype(kW_tag)::value;
    if (use_64bit) {
      DeformableIm2ColKernel<T, int64_t, IsNHWC, KH, KW><<<blocks, kDeformConvThreadsPerBlock, 0, stream>>>(
          num_kernels, input, offset, mask, H, W, kH, kW, pad_h, pad_w,
          stride_h, stride_w, dilation_h, dilation_w, C, offset_group,
          DivMod<int64_t>(out_h), DivMod<int64_t>(out_w), DivMod<int64_t>(parallel_imgs),
          DivMod<int64_t>(C / offset_group),
          DivMod<int64_t>(static_cast<int64_t>(C)),
          use_mask, col_buffer);
    } else {
      DeformableIm2ColKernel<T, int32_t, IsNHWC, KH, KW><<<blocks, kDeformConvThreadsPerBlock, 0, stream>>>(
          static_cast<int32_t>(num_kernels), input, offset, mask, H, W, kH, kW, pad_h, pad_w,
          stride_h, stride_w, dilation_h, dilation_w, C, offset_group,
          DivMod<int32_t>(static_cast<int32_t>(out_h)),
          DivMod<int32_t>(static_cast<int32_t>(out_w)),
          DivMod<int32_t>(static_cast<int32_t>(parallel_imgs)),
          DivMod<int32_t>(static_cast<int32_t>(C / offset_group)),
          DivMod<int32_t>(static_cast<int32_t>(C)),
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
  return DeformConvIm2ColImplLayout<T, false>(
      stream, input, offset, mask, col_buffer, parallel_imgs, C, H, W, kH, kW,
      out_h, out_w, pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w, offset_group, use_mask);
}

template <typename T>
Status DeformConvIm2ColImplNHWC(
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
  return DeformConvIm2ColImplLayout<T, true>(
      stream, input, offset, mask, col_buffer, parallel_imgs, C, H, W, kH, kW,
      out_h, out_w, pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w, offset_group, use_mask);
}

template <typename T>
Status DeformConvColReorderLxCKToBatched(
    cudaStream_t stream,
    const T* src,
    T* dst,
    int64_t cur_out_size,
    int64_t C,
    int64_t kernel_size,
    int64_t kernel_dim,
    int64_t group) {
  const int64_t total = cur_out_size * C * kernel_size;
  if (total <= 0) return Status::OK();
  int blocks = GetGridSize(static_cast<size_t>(total), kDeformConvThreadsPerBlock);
  DeformConvColReorderLxCKToBatchedKernel<T><<<blocks, kDeformConvThreadsPerBlock, 0, stream>>>(
      src, dst, cur_out_size, C, kernel_size, kernel_dim, group);
  return CUDA_CALL(cudaGetLastError());
}

template <typename T>
Status DeformConvCopyGemmOutputRowMajorToNHWC(
    cudaStream_t stream,
    const T* gemm_output,
    T* Y,
    int64_t M,
    int64_t M_per_group,
    int64_t channel_offset,
    int64_t output_image_size,
    int64_t cur_parallel) {
  int64_t total = cur_parallel * M_per_group * output_image_size;
  if (total <= 0) return Status::OK();
  int blocks = GetGridSize(static_cast<size_t>(total), kDeformConvThreadsPerBlock);
  CopyGemmOutputToLayoutKernel<T, true><<<blocks, kDeformConvThreadsPerBlock, 0, stream>>>(
      gemm_output, Y, M, M_per_group, channel_offset, output_image_size, cur_parallel);
  return CUDA_CALL(cudaGetLastError());
}


#define INST_DeformConvIm2ColImpl(T) \
  template Status DeformConvIm2ColImpl<T>(cudaStream_t, const T*, const T*, const T*, T*, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, bool);

INST_DeformConvIm2ColImpl(float)
INST_DeformConvIm2ColImpl(double)
INST_DeformConvIm2ColImpl(half)
INST_DeformConvIm2ColImpl(BFloat16)

                template Status DeformConvCopyGemmOutputRowMajorToNCHW<float>(cudaStream_t, const float*, float*, int64_t, int64_t, int64_t, int64_t);
template Status DeformConvCopyGemmOutputRowMajorToNCHW<double>(cudaStream_t, const double*, double*, int64_t, int64_t, int64_t, int64_t);
template Status DeformConvCopyGemmOutputRowMajorToNCHW<half>(cudaStream_t, const half*, half*, int64_t, int64_t, int64_t, int64_t);
template Status DeformConvCopyGemmOutputRowMajorToNCHW<BFloat16>(cudaStream_t, const BFloat16*, BFloat16*, int64_t, int64_t, int64_t, int64_t);

#define INST_DeformConvIm2ColImplNHWC(T) \
  template Status DeformConvIm2ColImplNHWC<T>(cudaStream_t, const T*, const T*, const T*, T*, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, bool);
INST_DeformConvIm2ColImplNHWC(float)
INST_DeformConvIm2ColImplNHWC(double)
INST_DeformConvIm2ColImplNHWC(half)
INST_DeformConvIm2ColImplNHWC(BFloat16)

template Status DeformConvColReorderLxCKToBatched<float>(cudaStream_t, const float*, float*, int64_t, int64_t, int64_t, int64_t, int64_t);
template Status DeformConvColReorderLxCKToBatched<double>(cudaStream_t, const double*, double*, int64_t, int64_t, int64_t, int64_t, int64_t);
template Status DeformConvColReorderLxCKToBatched<half>(cudaStream_t, const half*, half*, int64_t, int64_t, int64_t, int64_t, int64_t);
template Status DeformConvColReorderLxCKToBatched<BFloat16>(cudaStream_t, const BFloat16*, BFloat16*, int64_t, int64_t, int64_t, int64_t, int64_t);

template Status DeformConvCopyGemmOutputRowMajorToNHWC<float>(cudaStream_t, const float*, float*, int64_t, int64_t, int64_t, int64_t, int64_t);
template Status DeformConvCopyGemmOutputRowMajorToNHWC<double>(cudaStream_t, const double*, double*, int64_t, int64_t, int64_t, int64_t, int64_t);
template Status DeformConvCopyGemmOutputRowMajorToNHWC<half>(cudaStream_t, const half*, half*, int64_t, int64_t, int64_t, int64_t, int64_t);
template Status DeformConvCopyGemmOutputRowMajorToNHWC<BFloat16>(cudaStream_t, const BFloat16*, BFloat16*, int64_t, int64_t, int64_t, int64_t, int64_t);

template Status DeformConvAddBiasImplNHWC<float>(cudaStream_t, float*, const float*, int64_t, int64_t, int64_t, int64_t);
template Status DeformConvAddBiasImplNHWC<double>(cudaStream_t, double*, const double*, int64_t, int64_t, int64_t, int64_t);
template Status DeformConvAddBiasImplNHWC<half>(cudaStream_t, half*, const half*, int64_t, int64_t, int64_t, int64_t);
template Status DeformConvAddBiasImplNHWC<BFloat16>(cudaStream_t, BFloat16*, const BFloat16*, int64_t, int64_t, int64_t, int64_t);

template Status DeformConvAddBiasImpl<float>(cudaStream_t, float*, const float*, int64_t, int64_t, int64_t, int64_t);
template Status DeformConvAddBiasImpl<double>(cudaStream_t, double*, const double*, int64_t, int64_t, int64_t, int64_t);
template Status DeformConvAddBiasImpl<half>(cudaStream_t, half*, const half*, int64_t, int64_t, int64_t, int64_t);
template Status DeformConvAddBiasImpl<BFloat16>(cudaStream_t, BFloat16*, const BFloat16*, int64_t, int64_t, int64_t, int64_t);

// Delegate ORT type to CUDA type (e.g. MLFloat16 -> half); avoids repeating three identical specializations.
#define DELEGATE_DEFORM_CONV_IMPL(ORT_T, CUDA_T)                                                                    \
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
  Status DeformConvIm2ColImplNHWC<ORT_T>(cudaStream_t stream, const ORT_T* input,                                   \
                                         const ORT_T* offset, const ORT_T* mask, ORT_T* col_buffer,                \
                                         int64_t parallel_imgs, int64_t C, int64_t H, int64_t W,                   \
                                         int64_t kH, int64_t kW, int64_t out_h, int64_t out_w,                     \
                                         int64_t pad_h, int64_t pad_w, int64_t stride_h, int64_t stride_w,         \
                                         int64_t dilation_h, int64_t dilation_w, int64_t offset_group, bool use_mask) { \
    return DeformConvIm2ColImplNHWC<CUDA_T>(stream, reinterpret_cast<const CUDA_T*>(input),                         \
                                            reinterpret_cast<const CUDA_T*>(offset),                                \
                                            mask ? reinterpret_cast<const CUDA_T*>(mask) : nullptr,                \
                                            reinterpret_cast<CUDA_T*>(col_buffer),                                  \
                                            parallel_imgs, C, H, W, kH, kW, out_h, out_w,                           \
                                            pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w,               \
                                            offset_group, use_mask);                                                \
  }                                                                                                                 \
  template <>                                                                                                       \
  Status DeformConvColReorderLxCKToBatched<ORT_T>(cudaStream_t stream, const ORT_T* src, ORT_T* dst,                \
                                                  int64_t cur_out_size, int64_t C, int64_t kernel_size,            \
                                                  int64_t kernel_dim, int64_t group) {                             \
    return DeformConvColReorderLxCKToBatched<CUDA_T>(stream, reinterpret_cast<const CUDA_T*>(src),                 \
                                                     reinterpret_cast<CUDA_T*>(dst),                                \
                                                     cur_out_size, C, kernel_size, kernel_dim, group);             \
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
  Status DeformConvCopyGemmOutputRowMajorToNHWC<ORT_T>(cudaStream_t stream,                                         \
                                                       const ORT_T* gemm_output, ORT_T* Y,                          \
                                                       int64_t M, int64_t M_per_group, int64_t channel_offset,      \
                                                       int64_t output_image_size, int64_t cur_parallel) {            \
    return DeformConvCopyGemmOutputRowMajorToNHWC<CUDA_T>(stream,                                                  \
                                                           reinterpret_cast<const CUDA_T*>(gemm_output),            \
                                                           reinterpret_cast<CUDA_T*>(Y),                            \
                                                           M, M_per_group, channel_offset, output_image_size, cur_parallel); \
  }                                                                                                                 \
  template <>                                                                                                       \
  Status DeformConvAddBiasImpl<ORT_T>(cudaStream_t stream, ORT_T * Y, const ORT_T* B,                               \
                                      int64_t N, int64_t M, int64_t out_h, int64_t out_w) {                         \
    return DeformConvAddBiasImpl<CUDA_T>(stream, reinterpret_cast<CUDA_T*>(Y),                                      \
                                         reinterpret_cast<const CUDA_T*>(B), N, M, out_h, out_w);                   \
  }                                                                                                                 \
  template <>                                                                                                       \
  Status DeformConvAddBiasImplNHWC<ORT_T>(cudaStream_t stream, ORT_T* Y, const ORT_T* B,                            \
                                          int64_t N, int64_t M, int64_t out_h, int64_t out_w) {                     \
    return DeformConvAddBiasImplNHWC<CUDA_T>(stream, reinterpret_cast<CUDA_T*>(Y),                                   \
                                             reinterpret_cast<const CUDA_T*>(B), N, M, out_h, out_w);               \
  }

DELEGATE_DEFORM_CONV_IMPL(MLFloat16, half)

}  // namespace cuda
}  // namespace onnxruntime
