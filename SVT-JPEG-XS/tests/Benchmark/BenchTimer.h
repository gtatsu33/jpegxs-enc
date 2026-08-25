/*
* Copyright(c) 2024 Intel Corporation
* SPDX - License - Identifier: BSD - 2 - Clause - Patent
*/
#ifndef BenchTimer_h
#define BenchTimer_h

#include <chrono>

/* Wall-clock timer for CPU (C reference) code paths. */
class CpuTimer {
  public:
    void start() {
        t0_ = std::chrono::high_resolution_clock::now();
    }
    double stop_ms() {
        auto t1 = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(t1 - t0_).count();
    }

  private:
    std::chrono::high_resolution_clock::time_point t0_;
};

#ifdef SVT_ENABLE_CUDA
#include <cuda_runtime.h>

/* GPU-side timer using CUDA events. Measures device execution time only
 * (kernel launch to completion), excluding host-side setup and H2D/D2H
 * transfers unless those are explicitly placed between start()/stop(). */
class GpuTimer {
  public:
    GpuTimer() {
        cudaEventCreate(&start_evt_);
        cudaEventCreate(&stop_evt_);
    }
    ~GpuTimer() {
        cudaEventDestroy(start_evt_);
        cudaEventDestroy(stop_evt_);
    }
    void start(cudaStream_t stream = 0) {
        cudaEventRecord(start_evt_, stream);
    }
    /* Blocks until the stop event completes, then returns elapsed ms. */
    double stop_ms(cudaStream_t stream = 0) {
        cudaEventRecord(stop_evt_, stream);
        cudaEventSynchronize(stop_evt_);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start_evt_, stop_evt_);
        return (double)ms;
    }

  private:
    cudaEvent_t start_evt_;
    cudaEvent_t stop_evt_;
};
#endif // SVT_ENABLE_CUDA

#endif // BenchTimer_h
