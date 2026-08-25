/*
* Copyright(c) 2024 Intel Corporation
* SPDX - License - Identifier: BSD - 2 - Clause - Patent
*/
#ifndef BenchReport_h
#define BenchReport_h

#include <string>
#include <vector>

#include "BenchTimer.h"

/* One benchmarking record: a module measured under a given porting phase,
 * on a given backend (CPU reference vs CUDA), across N iterations. */
struct BenchRecord {
    std::string phase;    // e.g. "Phase0-Proof", "Phase1-NLT", "Phase1-DWT", "Phase1-GC"
    std::string module;   // e.g. "GC(CPU reference)", "DWT(CUDA)"
    std::string backend;  // "CPU" or "GPU"
    int iterations;
    double mean_ms;
    double min_ms;
    double max_ms;
};

class BenchReport {
  public:
    void add(const BenchRecord& record);

    /* Prints an aligned table to stdout. */
    void print_table() const;

    /* Writes/appends records as CSV rows. Writes a header if the file
     * does not already exist. */
    void write_csv(const std::string& path) const;

  private:
    std::vector<BenchRecord> records_;
};

/* Runs `fn` `iterations` times using a CpuTimer and returns a BenchRecord.
 * `fn` must take no arguments and return void. */
template <typename Fn>
BenchRecord bench_cpu(const std::string& phase, const std::string& module, int iterations, Fn fn) {
    // Warm-up (not counted): stabilizes cache state / clock frequency.
    fn();

    double sum = 0.0, mn = -1.0, mx = 0.0;
    for (int i = 0; i < iterations; i++) {
        CpuTimer timer;
        timer.start();
        fn();
        double ms = timer.stop_ms();
        sum += ms;
        if (mn < 0 || ms < mn)
            mn = ms;
        if (ms > mx)
            mx = ms;
    }
    BenchRecord r;
    r.phase = phase;
    r.module = module;
    r.backend = "CPU";
    r.iterations = iterations;
    r.mean_ms = sum / iterations;
    r.min_ms = mn;
    r.max_ms = mx;
    return r;
}

#endif // BenchReport_h
