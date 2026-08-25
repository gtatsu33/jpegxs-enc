/*
* Copyright(c) 2024 Intel Corporation
* SPDX - License - Identifier: BSD - 2 - Clause - Patent
*/
#include "gtest/gtest.h"
#include "random.h"
#include "Dwt.h"
#include "encoder_dsp_rtcd.h"
#include "Codestream.h"
#include "DwtCuda.cuh"
#include <vector>

/* Phase 1 bit-exact validation for the DWT kernel chain.
 *
 * The CPU reference here is NOT a reimplementation: it drives the real
 * `dwt_horizontal_line` RTCD function pointer (resolves to
 * dwt_horizontal_line_c) for BOTH the horizontal pass (applied to a row)
 * and the vertical pass (applied to a column copied into a contiguous
 * temp buffer first) -- Dwt.c's own vertical loop functions
 * (transform_vertical_loop_*_c) implement this identical 5/3 recurrence
 * along columns, so using the horizontal routine on column data is
 * mathematically equivalent and lets the test exercise the actual,
 * unmodified CPU lifting code rather than a second reimplementation of it.
 * The recursive band-tree traversal below mirrors Pi.c's pi_compute()
 * geometry (vertical decomp levels each produce a 2D 4-way split; any
 * remaining horizontal-only levels further split just the LL band), which
 * svt_cuda_dwt_component()'s host orchestration also implements. A bug in
 * either the lifting formulas or the recursion structure will show up as a
 * memcmp mismatch against this reference. */

static void cpu_horizontal_lift(const int32_t* src, uint32_t pitch, uint32_t x, uint32_t y, uint32_t w, uint32_t h,
                                int32_t* dst_lf, uint32_t lf_pitch, uint32_t lfx, uint32_t lfy, int32_t* dst_hf, uint32_t hf_pitch,
                                uint32_t hfx, uint32_t hfy) {
    std::vector<int32_t> in(w), lf(w), hf(w);
    uint32_t w2 = w / 2, w1 = w - w2;
    for (uint32_t row = 0; row < h; row++) {
        memcpy(in.data(), src + (size_t)(y + row) * pitch + x, w * sizeof(int32_t));
        dwt_horizontal_line(lf.data(), hf.data(), in.data(), w);
        memcpy(dst_lf + (size_t)(lfy + row) * lf_pitch + lfx, lf.data(), w1 * sizeof(int32_t));
        memcpy(dst_hf + (size_t)(hfy + row) * hf_pitch + hfx, hf.data(), w2 * sizeof(int32_t));
    }
}

static void cpu_vertical_lift(const int32_t* src, uint32_t pitch, uint32_t x, uint32_t y, uint32_t w, uint32_t h, int32_t* dst_lf,
                              uint32_t lf_pitch, uint32_t lfx, uint32_t lfy, int32_t* dst_hf, uint32_t hf_pitch, uint32_t hfx,
                              uint32_t hfy) {
    std::vector<int32_t> in(h), lf(h), hf(h);
    uint32_t h2 = h / 2, h1 = h - h2;
    for (uint32_t col = 0; col < w; col++) {
        for (uint32_t row = 0; row < h; row++) {
            in[row] = src[(size_t)(y + row) * pitch + x + col];
        }
        dwt_horizontal_line(lf.data(), hf.data(), in.data(), h);
        for (uint32_t row = 0; row < h1; row++) {
            dst_lf[(size_t)(lfy + row) * lf_pitch + lfx + col] = lf[row];
        }
        for (uint32_t row = 0; row < h2; row++) {
            dst_hf[(size_t)(hfy + row) * hf_pitch + hfx + col] = hf[row];
        }
    }
}

/* CPU reference: NLT + full recursive 5/3 DWT, producing the same dense
 * pyramid layout as svt_cuda_dwt_component(). */
static void cpu_dwt_component(const int32_t* nlt_scaled, uint32_t comp_width, uint32_t comp_height, uint32_t decom_h,
                              uint32_t decom_v, uint8_t hdr_Fq, uint16_t* out_pyramid_16bit) {
    size_t elems = (size_t)comp_width * comp_height;
    std::vector<int32_t> cur(elems), other(elems), vert(elems), pyramid(elems);
    memcpy(cur.data(), nlt_scaled, elems * sizeof(int32_t));

    uint32_t active_w = comp_width, active_h = comp_height;

    for (uint32_t v = 0; v < decom_v; v++) {
        uint32_t h2 = active_h / 2, h1 = active_h - h2;
        uint32_t w2 = active_w / 2, w1 = active_w - w2;

        cpu_vertical_lift(cur.data(),
                          comp_width,
                          0,
                          0,
                          active_w,
                          active_h,
                          vert.data(),
                          comp_width,
                          0,
                          0,
                          vert.data(),
                          comp_width,
                          0,
                          h1);

        cpu_horizontal_lift(vert.data(),
                            comp_width,
                            0,
                            0,
                            active_w,
                            h1,
                            other.data(),
                            comp_width,
                            0,
                            0,
                            pyramid.data(),
                            comp_width,
                            w1,
                            0);

        cpu_horizontal_lift(vert.data(),
                            comp_width,
                            0,
                            h1,
                            active_w,
                            h2,
                            pyramid.data(),
                            comp_width,
                            0,
                            h1,
                            pyramid.data(),
                            comp_width,
                            w1,
                            h1);

        active_w = w1;
        active_h = h1;
        std::swap(cur, other);
    }

    for (uint32_t hh = decom_v; hh < decom_h; hh++) {
        uint32_t w2 = active_w / 2, w1 = active_w - w2;
        cpu_horizontal_lift(
            cur.data(), comp_width, 0, 0, active_w, active_h, other.data(), comp_width, 0, 0, pyramid.data(), comp_width, w1, 0);
        active_w = w1;
        std::swap(cur, other);
    }

    for (uint32_t row = 0; row < active_h; row++) {
        memcpy(pyramid.data() + (size_t)row * comp_width, cur.data() + (size_t)row * comp_width, active_w * sizeof(int32_t));
    }

    const int32_t shift_out = hdr_Fq;
    const int32_t offset_out = 1 << (hdr_Fq - 1);
    for (size_t i = 0; i < elems; i++) {
        int32_t val = pyramid[i];
        uint16_t out_v;
        if (val >= 0) {
            out_v = (uint16_t)((val + offset_out) >> shift_out);
        }
        else {
            int32_t v = (-val + offset_out) >> shift_out;
            out_v = (uint16_t)v;
            if (v) {
                out_v |= BITSTREAM_MASK_SIGN;
            }
        }
        out_pyramid_16bit[i] = out_v;
    }
}

static void run_dwt_cuda_vs_cpu(uint32_t width, uint32_t height, uint32_t decom_h, uint32_t decom_v, uint8_t input_bit_depth) {
    setup_encoder_rtcd_internal(CPU_FLAGS_ALL);

    const uint8_t hdr_Bw = 20;
    const uint8_t hdr_Fq = 8;
    const uint8_t shift = hdr_Bw - input_bit_depth;
    const int32_t offset = 1 << (hdr_Bw - 1);
    size_t elems = (size_t)width * height;

    std::vector<uint16_t> src16;
    std::vector<uint8_t> src8;
    std::vector<int32_t> nlt_scaled(elems);

    svt_jxs_test_tool::SVTRandom rand_bd(input_bit_depth, false);
    if (input_bit_depth <= 8) {
        src8.resize(elems);
        for (size_t i = 0; i < elems; i++) {
            src8[i] = (uint8_t)rand_bd.random();
            nlt_scaled[i] = (int32_t)((uint32_t)src8[i] << shift) - offset;
        }
    }
    else {
        src16.resize(elems);
        for (size_t i = 0; i < elems; i++) {
            src16[i] = (uint16_t)rand_bd.random();
            nlt_scaled[i] = (int32_t)((uint32_t)src16[i] << shift) - offset;
        }
    }

    std::vector<uint16_t> out_cpu(elems), out_cuda(elems);
    cpu_dwt_component(nlt_scaled.data(), width, height, decom_h, decom_v, hdr_Fq, out_cpu.data());

    const void* in_plane = (input_bit_depth <= 8) ? (const void*)src8.data() : (const void*)src16.data();
    int err = svt_cuda_dwt_component(
        in_plane, width, width, height, decom_h, decom_v, input_bit_depth, hdr_Bw, hdr_Fq, out_cuda.data());
    ASSERT_EQ(err, 0);

    EXPECT_EQ(memcmp(out_cpu.data(), out_cuda.data(), elems * sizeof(uint16_t)), 0);
}

TEST(DwtCuda, decom_h5_v2_10bit_small) {
    run_dwt_cuda_vs_cpu(256, 128, 5, 2, 10);
}

TEST(DwtCuda, decom_h5_v0_horizontal_only) {
    run_dwt_cuda_vs_cpu(256, 32, 5, 0, 10);
}

TEST(DwtCuda, decom_h3_v1_8bit) {
    run_dwt_cuda_vs_cpu(200, 96, 3, 1, 8);
}

TEST(DwtCuda, decom_h5_v2_10bit_4k) {
    run_dwt_cuda_vs_cpu(3840, 2160, 5, 2, 10);
}
