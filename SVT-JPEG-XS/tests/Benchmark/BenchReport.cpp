/*
* Copyright(c) 2024 Intel Corporation
* SPDX - License - Identifier: BSD - 2 - Clause - Patent
*/
#include "BenchReport.h"

#include <cstdio>
#include <fstream>
#include <sys/stat.h>

void BenchReport::add(const BenchRecord& record) {
    records_.push_back(record);
}

void BenchReport::print_table() const {
    printf("%-16s %-24s %-5s %6s %10s %10s %10s\n", "Phase", "Module", "Back", "Iters", "Mean(ms)", "Min(ms)", "Max(ms)");
    for (const auto& r : records_) {
        printf("%-16s %-24s %-5s %6d %10.4f %10.4f %10.4f\n",
               r.phase.c_str(),
               r.module.c_str(),
               r.backend.c_str(),
               r.iterations,
               r.mean_ms,
               r.min_ms,
               r.max_ms);
    }
}

void BenchReport::write_csv(const std::string& path) const {
    struct stat buf;
    bool exists = (stat(path.c_str(), &buf) == 0);

    std::ofstream ofs(path, std::ios::app);
    if (!exists) {
        ofs << "phase,module,backend,iterations,mean_ms,min_ms,max_ms\n";
    }
    for (const auto& r : records_) {
        ofs << r.phase << "," << r.module << "," << r.backend << "," << r.iterations << "," << r.mean_ms << ","
            << r.min_ms << "," << r.max_ms << "\n";
    }
}
