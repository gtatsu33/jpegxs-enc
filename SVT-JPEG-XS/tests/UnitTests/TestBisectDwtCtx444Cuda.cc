/*
* Bisection scratch test (temporary): checks whether svt_cuda_dwt_component_ctx()
* -- the persistent-context DWT entry point actually used by
* svt_cuda_encode_frame() (unlike svt_cuda_dwt_component(), which
* TestDwtCuda.cc already validates but which is NOT what the full pipeline
* calls) -- stays correct across 3 SEQUENTIAL calls sharing the SAME
* SvtCudaFrameContext scratch buffers with 3 EQUAL-size components (444/RGB
* shape), rather than TestDwtCuda.cc's single isolated per-call scratch
* buffers.
*/
#include "gtest/gtest.h"
#include "random.h"
#include "Dwt.h"
#include "encoder_dsp_rtcd.h"
#include "Codestream.h"
#include "DwtCuda.cuh"
#include "FrameContextCuda.cuh"
#include <vector>
#include <cstdio>

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

static void cpu_dwt_component(const int32_t* nlt_scaled, uint32_t comp_width, uint32_t comp_height, uint32_t decom_h,
                              uint32_t decom_v, uint8_t hdr_Fq, uint16_t* out_pyramid_16bit) {
    size_t elems = (size_t)comp_width * comp_height;
    std::vector<int32_t> cur(elems), other(elems), vert(elems), pyramid(elems);
    memcpy(cur.data(), nlt_scaled, elems * sizeof(int32_t));

    uint32_t active_w = comp_width, active_h = comp_height;

    for (uint32_t v = 0; v < decom_v; v++) {
        uint32_t h2 = active_h / 2, h1 = active_h - h2;
        uint32_t w2 = active_w / 2, w1 = active_w - w2;

        cpu_vertical_lift(cur.data(), comp_width, 0, 0, active_w, active_h, vert.data(), comp_width, 0, 0, vert.data(), comp_width, 0, h1);
        cpu_horizontal_lift(vert.data(), comp_width, 0, 0, active_w, h1, other.data(), comp_width, 0, 0, pyramid.data(), comp_width, w1, 0);
        cpu_horizontal_lift(vert.data(), comp_width, 0, h1, active_w, h2, pyramid.data(), comp_width, 0, h1, pyramid.data(), comp_width, w1, h1);

        active_w = w1;
        active_h = h1;
        std::swap(cur, other);
    }

    for (uint32_t hh = decom_v; hh < decom_h; hh++) {
        uint32_t w2 = active_w / 2, w1 = active_w - w2;
        cpu_horizontal_lift(cur.data(), comp_width, 0, 0, active_w, active_h, other.data(), comp_width, 0, 0, pyramid.data(), comp_width, w1, 0);
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

static void run_three_equal_components(uint32_t width, uint32_t height) {
    setup_encoder_rtcd_internal(CPU_FLAGS_ALL);

    const uint32_t decom_h = 5, decom_v = 2;
    const uint8_t input_bit_depth = 10;
    const uint8_t hdr_Bw = 20;
    const uint8_t hdr_Fq = 8;
    const uint8_t shift = hdr_Bw - input_bit_depth;
    const int32_t offset = 1 << (hdr_Bw - 1);
    size_t elems = (size_t)width * height;

    // Minimal band geometry -- doesn't matter for this test, only comp_width/height do.
    SvtCudaFrameBandGeom band;
    memset(&band, 0, sizeof(band));
    band.comp_id = 0;
    band.band_id = 0;
    band.width = width;
    band.height = height;
    band.height_lines_num = height; // single "precinct" spanning the whole thing, not exercised here
    band.gcli_width = 1;
    band.significance_width = 1;

    uint32_t comp_width[3] = {width, width, width};
    uint32_t comp_height[3] = {height, height, height};

    SvtCudaFrameContext ctx;
    memset(&ctx, 0, sizeof(ctx));
    int rc = svt_cuda_frame_context_create(&ctx, 3, comp_width, comp_height, 1, &band, 1, 1, 4096, 4);
    ASSERT_EQ(rc, 0);

    svt_jxs_test_tool::SVTRandom rand_bd(input_bit_depth, false);
    std::vector<std::vector<uint16_t>> src(3, std::vector<uint16_t>(elems));
    std::vector<std::vector<int32_t>> nlt_scaled(3, std::vector<int32_t>(elems));
    std::vector<std::vector<uint16_t>> out_cpu(3, std::vector<uint16_t>(elems));
    std::vector<std::vector<uint16_t>> out_cuda(3, std::vector<uint16_t>(elems));

    for (int c = 0; c < 3; c++) {
        for (size_t i = 0; i < elems; i++) {
            src[c][i] = (uint16_t)rand_bd.random();
            nlt_scaled[c][i] = (int32_t)((uint32_t)src[c][i] << shift) - offset;
        }
        cpu_dwt_component(nlt_scaled[c].data(), width, height, decom_h, decom_v, hdr_Fq, out_cpu[c].data());
    }

    // Sequentially call the SAME entry point svt_cuda_encode_frame() uses (svt_cuda_dwt_component_ctx),
    // sharing ctx's scratch buffers across all 3 calls, exactly like efc_run_dwt()'s per-component loop.
    for (int c = 0; c < 3; c++) {
        int err = svt_cuda_dwt_component_ctx(src[c].data(),
                                             width,
                                             width,
                                             height,
                                             decom_h,
                                             decom_v,
                                             input_bit_depth,
                                             hdr_Bw,
                                             hdr_Fq,
                                             ctx.d_in_raw,
                                             ctx.d_cur,
                                             ctx.d_other,
                                             ctx.d_vert,
                                             ctx.d_pyramid32,
                                             ctx.d_pyramid16[c],
                                             ctx.stream);
        ASSERT_EQ(err, 0) << "component " << c;
    }
    cudaStreamSynchronize(ctx.stream);
    for (int c = 0; c < 3; c++) {
        cudaMemcpy(out_cuda[c].data(), ctx.d_pyramid16[c], elems * sizeof(uint16_t), cudaMemcpyDeviceToHost);
    }

    for (int c = 0; c < 3; c++) {
        int mismatches = 0;
        for (size_t i = 0; i < elems; i++) {
            if (out_cpu[c][i] != out_cuda[c][i]) {
                mismatches++;
                if (mismatches <= 5) {
                    printf("[bisect-dwt-ctx] comp %d elem %zu: cpu=%u cuda=%u\n", c, i, out_cpu[c][i], out_cuda[c][i]);
                }
            }
        }
        printf("[bisect-dwt-ctx] comp %d: %d/%zu mismatches\n", c, mismatches, elems);
        EXPECT_EQ(mismatches, 0) << "component " << c;
    }

    svt_cuda_frame_context_destroy(&ctx);
}

TEST(BisectDwtCtx444Cuda, three_equal_components_sequential_matches_cpu) {
    run_three_equal_components(256, 64);
}

TEST(BisectDwtCtx444Cuda, three_equal_components_sequential_matches_cpu_4k) {
    run_three_equal_components(3840, 2160);
}
