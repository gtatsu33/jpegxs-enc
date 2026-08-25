/*
* Copyright(c) 2024 Intel Corporation
* SPDX - License - Identifier: BSD - 2 - Clause - Patent
*/
#ifndef CudaSmokeTest_cuh
#define CudaSmokeTest_cuh

#ifdef __cplusplus
extern "C" {
#endif

/* Placeholder for Phase 0 toolchain verification.
 * Enumerates the CUDA device and runs a trivial vector-add kernel.
 * Returns 0 on success, non-zero on any CUDA error or result mismatch.
 * Will be removed once real Phase 1 kernels (NLT/DWT/GC) land. */
int svt_cuda_smoke_test(void);

/* Runs the smoke vector-add kernel `iterations` times on a fixed-size
 * pre-allocated buffer and reports pure device execution time (kernel only,
 * no H2D/D2H, measured via CUDA events). Used to prove out the CPU-vs-GPU
 * benchmarking harness before real Phase 1 kernels exist.
 * Returns 0 on success and writes the mean/min/max kernel time in
 * milliseconds to *out_mean_ms/*out_min_ms/*out_max_ms. */
int svt_cuda_smoke_bench(int iterations, double* out_mean_ms, double* out_min_ms, double* out_max_ms);

#ifdef __cplusplus
}
#endif

#endif // CudaSmokeTest_cuh
