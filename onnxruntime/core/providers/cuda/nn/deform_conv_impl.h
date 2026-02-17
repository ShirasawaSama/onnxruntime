// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <stdint.h>
#include "core/providers/cuda/shared_inc/cuda_utils.h"

namespace onnxruntime {
namespace cuda {

// Adds bias to output: Y[n,m,oh,ow] += B[m]. Y is [N, M, out_h, out_w], B is [M].
template <typename T>
void DeformConvAddBiasImpl(
    cudaStream_t stream,
    T* Y,
    const T* B,
    int64_t N,
    int64_t M,
    int64_t out_h,
    int64_t out_w);

// Transposes row-major [rows, cols] to column-major [rows, cols]: dst[i+j*rows] = src[i*cols+j].
template <typename T>
void DeformConvTransposeRowMajorToColMajor(
    cudaStream_t stream,
    const T* row_major_src,
    T* col_major_dst,
    int64_t rows,
    int64_t cols);

// Copies from column-major [rows, cols] to row-major [rows, cols]. Used after GEMM.
template <typename T>
void DeformConvCopyColMajorToRowMajor(
    cudaStream_t stream,
    const T* col_major_src,
    T* row_major_dst,
    int64_t rows,
    int64_t cols);

// Fills col_buffer with deformable im2col. col_buffer layout: row-major [C*kH*kW, parallel_imgs*out_h*out_w].
// Called once per batch block; caller does GEMM and bias.
template <typename T>
void DeformConvIm2ColImpl(
    cudaStream_t stream,
    const T* input,      // [parallel_imgs, C, H, W]
    const T* offset,     // [parallel_imgs, offset_group*2*kH*kW, out_h, out_w]
    const T* mask,       // [parallel_imgs, offset_group*kH*kW, out_h, out_w] or nullptr
    T* col_buffer,       // [C*kH*kW, parallel_imgs*out_h*out_w]
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

}  // namespace cuda
}  // namespace onnxruntime
