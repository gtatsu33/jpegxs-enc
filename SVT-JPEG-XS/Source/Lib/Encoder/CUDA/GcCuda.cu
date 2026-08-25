/*
* Copyright(c) 2024 Intel Corporation
* SPDX - License - Identifier: BSD - 2 - Clause - Patent
*/
#include <cuda_runtime.h>
#include "GcCuda.cuh"

/* Matches log2_32_c (SvtUtility.c): floor(log2(x)), x != 0. */
__device__ __forceinline__ uint8_t svt_cuda_log2_32(uint32_t x) {
    return (uint8_t)(31 - __clz(x));
}

/* One thread per group of `group_size` (GROUP_SIZE=4) coefficients, plus one
 * extra thread for the trailing partial group when width % group_size != 0
 * (thread id == full_groups_num). Mirrors gc_precinct_stage_scalar_c /
 * gc_precinct_stage_scalar_loop_c exactly. */
__global__ void k_gc_precinct_stage_scalar(uint8_t* gcli, const uint16_t* coeff, uint32_t width, uint32_t full_groups_num,
                                           uint32_t leftover) {
    uint32_t g = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t total = full_groups_num + (leftover ? 1 : 0);
    if (g >= total)
        return;

    uint32_t base = g * 4;
    uint32_t n = (g < full_groups_num) ? 4 : leftover;
    uint16_t merge_or = 0;
    for (uint32_t i = 0; i < n; i++) {
        merge_or |= coeff[base + i];
    }
    merge_or = (uint16_t)(merge_or << 1); // remove sign bit
    gcli[g] = merge_or ? svt_cuda_log2_32(merge_or) : 0;
}

/* One thread per significance group of `group_sign_size` (SIGNIFICANCE_GROUP_SIZE=8)
 * gcli values (plus one trailing partial group). Mirrors gc_precinct_sigflags_max_c. */
__global__ void k_gc_precinct_sigflags_max(uint8_t* sig_max, const uint8_t* gcli, uint32_t group_sign_size, uint32_t full_groups_num,
                                           uint32_t leftover) {
    uint32_t g = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t total = full_groups_num + (leftover ? 1 : 0);
    if (g >= total)
        return;

    uint32_t base = g * group_sign_size;
    uint32_t n = (g < full_groups_num) ? group_sign_size : leftover;
    uint8_t m = 0;
    for (uint32_t i = 0; i < n; i++) {
        uint8_t v = gcli[base + i];
        if (v > m)
            m = v;
    }
    sig_max[g] = m;
}

int svt_cuda_gc_precinct_stage_scalar(uint8_t* gcli_data_ptr, const uint16_t* coeff_data_ptr_16bit, uint32_t group_size,
                                      uint32_t width) {
    if (group_size != 4 || width == 0) {
        return 1;
    }
    uint32_t full_groups_num = width / 4;
    uint32_t leftover = width % 4;
    uint32_t total_groups = full_groups_num + (leftover ? 1 : 0);

    uint16_t* d_coeff = NULL;
    uint8_t* d_gcli = NULL;
    cudaError_t err = cudaSuccess;

    do {
        if ((err = cudaMalloc(&d_coeff, (size_t)width * sizeof(uint16_t))) != cudaSuccess)
            break;
        if ((err = cudaMalloc(&d_gcli, total_groups)) != cudaSuccess)
            break;
        if ((err = cudaMemcpy(d_coeff, coeff_data_ptr_16bit, (size_t)width * sizeof(uint16_t), cudaMemcpyHostToDevice)) !=
            cudaSuccess)
            break;

        uint32_t threads = 256;
        uint32_t blocks = (total_groups + threads - 1) / threads;
        k_gc_precinct_stage_scalar<<<blocks, threads>>>(d_gcli, d_coeff, width, full_groups_num, leftover);

        if ((err = cudaMemcpy(gcli_data_ptr, d_gcli, total_groups, cudaMemcpyDeviceToHost)) != cudaSuccess)
            break;
        err = cudaGetLastError();
    } while (0);

    cudaFree(d_coeff);
    cudaFree(d_gcli);
    return err == cudaSuccess ? 0 : (int)err;
}

int svt_cuda_gc_precinct_sigflags_max(uint8_t* significance_data_max_ptr, const uint8_t* gcli_data_ptr, uint32_t group_sign_size,
                                      uint32_t gcli_width) {
    if (group_sign_size == 0 || gcli_width == 0) {
        return 1;
    }
    uint32_t full_groups_num = gcli_width / group_sign_size;
    uint32_t leftover = gcli_width % group_sign_size;
    uint32_t total_groups = full_groups_num + (leftover ? 1 : 0);

    uint8_t *d_gcli = NULL, *d_sig = NULL;
    cudaError_t err = cudaSuccess;

    do {
        if ((err = cudaMalloc(&d_gcli, gcli_width)) != cudaSuccess)
            break;
        if ((err = cudaMalloc(&d_sig, total_groups)) != cudaSuccess)
            break;
        if ((err = cudaMemcpy(d_gcli, gcli_data_ptr, gcli_width, cudaMemcpyHostToDevice)) != cudaSuccess)
            break;

        uint32_t threads = 256;
        uint32_t blocks = (total_groups + threads - 1) / threads;
        k_gc_precinct_sigflags_max<<<blocks, threads>>>(d_sig, d_gcli, group_sign_size, full_groups_num, leftover);

        if ((err = cudaMemcpy(significance_data_max_ptr, d_sig, total_groups, cudaMemcpyDeviceToHost)) != cudaSuccess)
            break;
        err = cudaGetLastError();
    } while (0);

    cudaFree(d_gcli);
    cudaFree(d_sig);
    return err == cudaSuccess ? 0 : (int)err;
}
