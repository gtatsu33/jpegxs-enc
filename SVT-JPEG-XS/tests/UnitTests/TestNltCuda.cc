/*
* Copyright(c) 2024 Intel Corporation
* SPDX - License - Identifier: BSD - 2 - Clause - Patent
*/
#include "gtest/gtest.h"
#include "random.h"
#include "NltEnc.h"
#include "encoder_dsp_rtcd.h"
#include "NltCuda.cuh"

/* Phase 1 bit-exact validation: linear_input_scaling_line_{8,16}bit_c (CPU
 * reference) vs svt_cuda_nlt_scale_component (CUDA), for a synthetic
 * 4K-width, 10bit plane (matching the project's target workload). */
TEST(NltCuda, scale_16bit_10bpc_matches_cpu_reference) {
    setup_encoder_rtcd_internal(CPU_FLAGS_ALL);

    const uint32_t width = 3840;
    const uint32_t height = 8;
    const uint8_t input_bit_depth = 10;
    const uint8_t hdr_Bw = 20; // WAVELET_IN_DEPTH_BW_DEFAULT
    const uint8_t shift = hdr_Bw - input_bit_depth;
    const int32_t offset = 1 << (hdr_Bw - 1);

    uint16_t* src = (uint16_t*)malloc((size_t)width * height * sizeof(uint16_t));
    int32_t* dst_cpu = (int32_t*)malloc((size_t)width * height * sizeof(int32_t));
    int32_t* dst_cuda = (int32_t*)malloc((size_t)width * height * sizeof(int32_t));

    svt_jxs_test_tool::SVTRandom rand10(10, false);
    for (size_t i = 0; i < (size_t)width * height; i++) {
        src[i] = (uint16_t)rand10.random();
    }

    for (uint32_t y = 0; y < height; y++) {
        linear_input_scaling_line_16bit_c(src + (size_t)y * width, dst_cpu + (size_t)y * width, width, shift, offset, input_bit_depth);
    }

    int err = svt_cuda_nlt_scale_component(src, width, width, height, input_bit_depth, hdr_Bw, dst_cuda);
    ASSERT_EQ(err, 0);

    EXPECT_EQ(memcmp(dst_cpu, dst_cuda, (size_t)width * height * sizeof(int32_t)), 0);

    free(src);
    free(dst_cpu);
    free(dst_cuda);
}

TEST(NltCuda, scale_8bit_matches_cpu_reference) {
    setup_encoder_rtcd_internal(CPU_FLAGS_ALL);

    const uint32_t width = 1921; // odd width, exercises non-multiple-of-block-size tail
    const uint32_t height = 5;
    const uint8_t input_bit_depth = 8;
    const uint8_t hdr_Bw = 20;
    const uint8_t shift = hdr_Bw - input_bit_depth;
    const int32_t offset = 1 << (hdr_Bw - 1);

    uint8_t* src = (uint8_t*)malloc((size_t)width * height);
    int32_t* dst_cpu = (int32_t*)malloc((size_t)width * height * sizeof(int32_t));
    int32_t* dst_cuda = (int32_t*)malloc((size_t)width * height * sizeof(int32_t));

    svt_jxs_test_tool::SVTRandom rand8(8, false);
    for (size_t i = 0; i < (size_t)width * height; i++) {
        src[i] = (uint8_t)rand8.random();
    }

    for (uint32_t y = 0; y < height; y++) {
        linear_input_scaling_line_8bit_c(src + (size_t)y * width, dst_cpu + (size_t)y * width, width, shift, offset);
    }

    int err = svt_cuda_nlt_scale_component(src, width, width, height, input_bit_depth, hdr_Bw, dst_cuda);
    ASSERT_EQ(err, 0);

    EXPECT_EQ(memcmp(dst_cpu, dst_cuda, (size_t)width * height * sizeof(int32_t)), 0);

    free(src);
    free(dst_cpu);
    free(dst_cuda);
}
