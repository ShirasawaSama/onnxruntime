// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <stdint.h>
#include "core/common/status.h"

namespace onnxruntime {
namespace cuda {

// Adds bias to output: Y[n,m,oh,ow] += B[m]. Y is [N, M, out_h, out_w], B is [M].
// T may be float, double, MLFloat16 (FP16), or BFloat16.
template <typename T>
Status DeformConvAddBiasImpl(
    cudaStream_t stream,
    T* Y,
    const T* B,
    int64_t N,
    int64_t M,
    int64_t out_h,
    int64_t out_w);

// Copies GEMM output (row-major [M_per_group, cur_parallel*output_image_size]) to NCHW slice at Y_g.
// T may be float, double, MLFloat16 (FP16), or BFloat16.
template <typename T>
Status DeformConvCopyGemmOutputRowMajorToNCHW(
    cudaStream_t stream,
    const T* gemm_output,
    T* Y_g,
    int64_t M,
    int64_t M_per_group,
    int64_t output_image_size,
    int64_t cur_parallel);

// Fills col_buffer with deformable im2col. col_buffer layout: row-major [C*kH*kW, parallel_imgs*out_h*out_w].
// Called once per batch block; caller does GEMM and bias. T may be float, double, MLFloat16 (FP16), or BFloat16.
template <typename T>
Status DeformConvIm2ColImpl(
    cudaStream_t stream,
    const T* input,   // [parallel_imgs, C, H, W]
    const T* offset,  // [parallel_imgs, offset_group*2*kH*kW, out_h, out_w]
    const T* mask,    // [parallel_imgs, offset_group*kH*kW, out_h, out_w] or nullptr
    T* col_buffer,    // [C*kH*kW, parallel_imgs*out_h*out_w]
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
    bool use_mask);

// NHWC variant: input is [parallel_imgs, H, W, C], col_buffer layout [C*kH*kW, col_stride] same as NCHW.
template <typename T>
Status DeformConvIm2ColImplNHWC(
    cudaStream_t stream,
    const T* input,   // [parallel_imgs, H, W, C]
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
    bool use_mask);

// Copies GEMM output to NHWC Y: [N, out_h, out_w, M]. channel_offset is the start channel for this group.
template <typename T>
Status DeformConvCopyGemmOutputRowMajorToNHWC(
    cudaStream_t stream,
    const T* gemm_output,
    T* Y,  // Full NHWC output [N, out_h, out_w, M]
    int64_t M,
    int64_t M_per_group,
    int64_t channel_offset,
    int64_t output_image_size,
    int64_t cur_parallel);

// Adds bias to NHWC Y: Y[n,oh,ow,m] += B[m].
template <typename T>
Status DeformConvAddBiasImplNHWC(
    cudaStream_t stream,
    T* Y,
    const T* B,
    int64_t N,
    int64_t M,
    int64_t out_h,
    int64_t out_w);

}  // namespace cuda
}  // namespace onnxruntime
