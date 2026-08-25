/*
* Copyright(c) 2024 Intel Corporation
* SPDX - License - Identifier: BSD - 2 - Clause - Patent
*/
#include <cuda_runtime.h>
#include "CudaSmokeTest.cuh"

__global__ void svt_cuda_smoke_vector_add(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

int svt_cuda_smoke_test(void) {
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
        return 1;
    }

    const int n = 1 << 16;
    float *da = NULL, *db = NULL, *dc = NULL;
    if (cudaMalloc(&da, n * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&db, n * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&dc, n * sizeof(float)) != cudaSuccess) {
        cudaFree(da);
        cudaFree(db);
        cudaFree(dc);
        return 2;
    }

    float* ha = new float[n];
    float* hb = new float[n];
    float* hc = new float[n];
    for (int i = 0; i < n; i++) {
        ha[i] = (float)i;
        hb[i] = (float)(n - i);
    }
    cudaMemcpy(da, ha, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(db, hb, n * sizeof(float), cudaMemcpyHostToDevice);

    svt_cuda_smoke_vector_add<<<(n + 255) / 256, 256>>>(da, db, dc, n);
    cudaMemcpy(hc, dc, n * sizeof(float), cudaMemcpyDeviceToHost);

    int result = 0;
    if (cudaDeviceSynchronize() != cudaSuccess) {
        result = 3;
    }
    else {
        for (int i = 0; i < n; i++) {
            if (hc[i] != (float)n) {
                result = 4;
                break;
            }
        }
    }

    cudaFree(da);
    cudaFree(db);
    cudaFree(dc);
    delete[] ha;
    delete[] hb;
    delete[] hc;
    return result;
}

int svt_cuda_smoke_bench(int iterations, double* out_mean_ms, double* out_min_ms, double* out_max_ms) {
    if (iterations <= 0) {
        return 1;
    }
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
        return 1;
    }

    const int n = 1 << 20;
    float *da = NULL, *db = NULL, *dc = NULL;
    if (cudaMalloc(&da, n * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&db, n * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&dc, n * sizeof(float)) != cudaSuccess) {
        cudaFree(da);
        cudaFree(db);
        cudaFree(dc);
        return 2;
    }
    /* Content does not matter for timing purposes, buffers are left
     * uninitialized on the device (no H2D transfer counted here). */

    cudaEvent_t start_evt, stop_evt;
    cudaEventCreate(&start_evt);
    cudaEventCreate(&stop_evt);

    double sum_ms = 0.0, min_ms = -1.0, max_ms = 0.0;
    dim3 block(256);
    dim3 grid((n + block.x - 1) / block.x);
    for (int i = 0; i < iterations; i++) {
        cudaEventRecord(start_evt);
        svt_cuda_smoke_vector_add<<<grid, block>>>(da, db, dc, n);
        cudaEventRecord(stop_evt);
        cudaEventSynchronize(stop_evt);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start_evt, stop_evt);
        sum_ms += ms;
        if (min_ms < 0 || ms < min_ms)
            min_ms = ms;
        if (ms > max_ms)
            max_ms = ms;
    }

    cudaEventDestroy(start_evt);
    cudaEventDestroy(stop_evt);
    cudaFree(da);
    cudaFree(db);
    cudaFree(dc);

    if (cudaGetLastError() != cudaSuccess) {
        return 3;
    }

    *out_mean_ms = sum_ms / iterations;
    *out_min_ms = min_ms;
    *out_max_ms = max_ms;
    return 0;
}
