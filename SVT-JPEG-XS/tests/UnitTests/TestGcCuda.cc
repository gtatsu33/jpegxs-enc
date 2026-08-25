/*
* Copyright(c) 2024 Intel Corporation
* SPDX - License - Identifier: BSD - 2 - Clause - Patent
*/
#include "gtest/gtest.h"
#include "random.h"
#include "GcStageProcess.h"
#include "EncDec.h"
#include "encoder_dsp_rtcd.h"
#include "Codestream.h"
#include "SvtUtility.h"
#include "GcCuda.cuh"

/* Phase 1 bit-exact validation: gc_precinct_stage_scalar_c (CPU reference)
 * vs svt_cuda_gc_precinct_stage_scalar (CUDA), same pattern as
 * TestGcStageProcess.cc's C-vs-SIMD comparisons. */
TEST(GcCuda, gc_precinct_stage_scalar_matches_cpu_reference) {
    setup_common_rtcd_internal(CPU_FLAGS_ALL);
    setup_encoder_rtcd_internal(CPU_FLAGS_ALL);

    uint32_t group_size = 4;
    uint32_t width = 3839; // odd, exercises the leftover-group path (3839 % 4 == 3)
    uint32_t gc_width = (width + group_size - 1) / group_size;

    uint16_t* coeff = (uint16_t*)malloc(width * sizeof(uint16_t));
    uint8_t* gc_cpu = (uint8_t*)malloc(gc_width);
    uint8_t* gc_cuda = (uint8_t*)malloc(gc_width);

    svt_jxs_test_tool::SVTRandom rand16(16, false);
    for (uint32_t i = 0; i < width; i++) {
        coeff[i] = (uint16_t)rand16.random();
    }

    memset(gc_cpu, 0xcd, gc_width);
    memset(gc_cuda, 0xcd, gc_width);

    gc_precinct_stage_scalar_c(gc_cpu, coeff, group_size, width);
    int err = svt_cuda_gc_precinct_stage_scalar(gc_cuda, coeff, group_size, width);
    ASSERT_EQ(err, 0);

    EXPECT_EQ(memcmp(gc_cpu, gc_cuda, gc_width), 0);

    free(coeff);
    free(gc_cpu);
    free(gc_cuda);
}

TEST(GcCuda, gc_precinct_sigflags_max_matches_cpu_reference) {
    uint32_t gcli_width = 383; // exercises leftover path (383 % 8 == 7)
    uint32_t sig_width = DIV_ROUND_UP(gcli_width, SIGNIFICANCE_GROUP_SIZE);

    uint8_t* gcli = (uint8_t*)malloc(gcli_width);
    uint8_t* sig_cpu = (uint8_t*)calloc(sig_width, 1);
    uint8_t* sig_cuda = (uint8_t*)calloc(sig_width, 1);

    svt_jxs_test_tool::SVTRandom rand(0, TRUNCATION_MAX);
    for (uint32_t i = 0; i < gcli_width; i++) {
        gcli[i] = (uint8_t)rand.random();
    }

    gc_precinct_sigflags_max_c(sig_cpu, gcli, SIGNIFICANCE_GROUP_SIZE, gcli_width);
    int err = svt_cuda_gc_precinct_sigflags_max(sig_cuda, gcli, SIGNIFICANCE_GROUP_SIZE, gcli_width);
    ASSERT_EQ(err, 0);

    EXPECT_EQ(memcmp(sig_cpu, sig_cuda, sig_width), 0);

    free(gcli);
    free(sig_cpu);
    free(sig_cuda);
}
