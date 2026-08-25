/*
* Copyright(c) 2024 Intel Corporation
* SPDX - License - Identifier: BSD - 2 - Clause - Patent
*/
#include <cuda_runtime.h>
#include "NltCuda.cuh"

__global__ void svt_nlt_k_scale_8bit(const uint8_t* src, uint32_t src_pitch, int32_t* dst, uint32_t dst_pitch, uint32_t width,
                                     uint32_t height, uint8_t shift, int32_t offset) {
    uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height)
        return;
    uint8_t v = src[(size_t)y * src_pitch + x];
    dst[(size_t)y * dst_pitch + x] = (int32_t)((uint32_t)v << shift) - offset;
}

__global__ void svt_nlt_k_scale_16bit(const uint16_t* src, uint32_t src_pitch, int32_t* dst, uint32_t dst_pitch, uint32_t width,
                                      uint32_t height, uint8_t shift, int32_t offset, uint8_t bit_depth) {
    uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height)
        return;
    uint16_t mask = (uint16_t)((1u << bit_depth) - 1);
    uint16_t v = src[(size_t)y * src_pitch + x] & mask;
    dst[(size_t)y * dst_pitch + x] = (int32_t)((uint32_t)v << shift) - offset;
}

int svt_cuda_nlt_scale_component(const void* in_plane, uint32_t plane_stride, uint32_t width, uint32_t height,
                                 uint8_t input_bit_depth, uint8_t hdr_Bw, int32_t* out_scaled) {
    if (width == 0 || height == 0) {
        return 1;
    }
    size_t in_elem_size = (input_bit_depth <= 8) ? sizeof(uint8_t) : sizeof(uint16_t);
    uint8_t* d_in = NULL;
    int32_t* d_out = NULL;
    cudaError_t err = cudaSuccess;

    do {
        if ((err = cudaMalloc(&d_in, (size_t)plane_stride * height * in_elem_size)) != cudaSuccess)
            break;
        if ((err = cudaMalloc(&d_out, (size_t)width * height * sizeof(int32_t))) != cudaSuccess)
            break;
        if ((err = cudaMemcpy2D(d_in,
                                plane_stride * in_elem_size,
                                in_plane,
                                plane_stride * in_elem_size,
                                width * in_elem_size,
                                height,
                                cudaMemcpyHostToDevice)) != cudaSuccess)
            break;

        const uint8_t shift = hdr_Bw - input_bit_depth;
        const int32_t offset = 1 << (hdr_Bw - 1);
        dim3 block2d(32, 8);
        dim3 grid2d((width + block2d.x - 1) / block2d.x, (height + block2d.y - 1) / block2d.y);
        if (input_bit_depth <= 8) {
            svt_nlt_k_scale_8bit<<<grid2d, block2d>>>(d_in, plane_stride, d_out, width, width, height, shift, offset);
        }
        else {
            svt_nlt_k_scale_16bit<<<grid2d, block2d>>>(
                (const uint16_t*)d_in, plane_stride, d_out, width, width, height, shift, offset, input_bit_depth);
        }

        if ((err = cudaMemcpy(out_scaled, d_out, (size_t)width * height * sizeof(int32_t), cudaMemcpyDeviceToHost)) != cudaSuccess)
            break;
        err = cudaGetLastError();
    } while (0);

    cudaFree(d_in);
    cudaFree(d_out);
    return err == cudaSuccess ? 0 : (int)err;
}
