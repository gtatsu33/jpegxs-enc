/*
* Bisection scratch test (temporary): isolates DWT with REAL image data at
* full width (3840), since TestBisectRealCrop444Cuda proved the mismatch
* needs BOTH real (non-random) content AND width=3840 (a 256-wide real crop
* passed; a 3840-wide random-noise frame passed) -- narrowing to whether DWT
* itself (svt_cuda_dwt_component_ctx, the persistent-context entry point
* svt_cuda_encode_frame() actually uses) is where the divergence begins for
* real correlated pixel data at that width, vs the RC+Quant/Pack stages.
*/
#include "gtest/gtest.h"
#include "Dwt.h"
#include "GcStageProcess.h"
#include "encoder_dsp_rtcd.h"
#include "common_dsp_rtcd.h"
#include "Codestream.h"
#include "SvtUtility.h"
#include "DwtCuda.cuh"
#include "FrameContextCuda.cuh"
#include "GcCuda.cuh"
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cstring>

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

static bool read_ppm_component0(const char* path, uint32_t crop_w, uint32_t crop_h, std::vector<uint16_t>* out) {
    FILE* f = fopen(path, "rb");
    if (!f)
        return false;
    char magic[3] = {0};
    if (fscanf(f, "%2s", magic) != 1 || strcmp(magic, "P6") != 0) {
        fclose(f);
        return false;
    }
    uint32_t w, h, maxval;
    if (fscanf(f, "%u %u %u", &w, &h, &maxval) != 3) {
        fclose(f);
        return false;
    }
    fgetc(f);
    out->resize((size_t)crop_w * crop_h);
    std::vector<uint8_t> row(w * 3 * 2);
    for (uint32_t y = 0; y < crop_h; y++) {
        if (fread(row.data(), 1, row.size(), f) != row.size()) {
            fclose(f);
            return false;
        }
        for (uint32_t x = 0; x < crop_w; x++) {
            uint16_t r = (uint16_t)((row[x * 6 + 0] << 8) | row[x * 6 + 1]); // component 0 = R
            (*out)[(size_t)y * crop_w + x] = r;
        }
    }
    fclose(f);
    return true;
}

TEST(BisectRealDwtCuda, real_component0_full_width_matches_cpu) {
    setup_encoder_rtcd_internal(CPU_FLAGS_ALL);

    const char* path = "C:\\Users\\0000108885\\codes\\jpegxs-enc\\testdata\\SOMED_TUM\\TUM_3840x2160_204_S-BD1_01_T02_00250_gma.ppm";
    const uint32_t width = 3840, height = 64;
    const uint32_t decom_h = 5, decom_v = 2;
    const uint8_t input_bit_depth = 10;
    const uint8_t hdr_Bw = 20;
    const uint8_t hdr_Fq = 8;
    const uint8_t shift = hdr_Bw - input_bit_depth;
    const int32_t offset = 1 << (hdr_Bw - 1);
    size_t elems = (size_t)width * height;

    std::vector<uint16_t> src;
    ASSERT_TRUE(read_ppm_component0(path, width, height, &src));

    std::vector<int32_t> nlt_scaled(elems);
    for (size_t i = 0; i < elems; i++) {
        nlt_scaled[i] = (int32_t)((uint32_t)src[i] << shift) - offset;
    }

    std::vector<uint16_t> out_cpu(elems);
    cpu_dwt_component(nlt_scaled.data(), width, height, decom_h, decom_v, hdr_Fq, out_cpu.data());

    // Standalone entry point (already covered by TestDwtCuda.cc, but never with real data)
    std::vector<uint16_t> out_cuda_standalone(elems);
    int err = svt_cuda_dwt_component(src.data(), width, width, height, decom_h, decom_v, input_bit_depth, hdr_Bw, hdr_Fq, out_cuda_standalone.data());
    ASSERT_EQ(err, 0);

    int mismatches_standalone = 0;
    int first_mismatch = -1;
    for (size_t i = 0; i < elems; i++) {
        if (out_cpu[i] != out_cuda_standalone[i]) {
            mismatches_standalone++;
            if (first_mismatch == -1)
                first_mismatch = (int)i;
        }
    }
    printf("[bisect-real-dwt] standalone svt_cuda_dwt_component: %d/%zu mismatches, first at %d (row=%d col=%d)\n",
           mismatches_standalone, elems, first_mismatch,
           first_mismatch >= 0 ? first_mismatch / (int)width : -1,
           first_mismatch >= 0 ? first_mismatch % (int)width : -1);
    EXPECT_EQ(mismatches_standalone, 0);

    // Context (persistent-buffer) entry point -- what svt_cuda_encode_frame() actually calls.
    uint32_t comp_width[3] = {width, width, width};
    uint32_t comp_height[3] = {height, height, height};
    SvtCudaFrameBandGeom band;
    memset(&band, 0, sizeof(band));
    band.comp_id = 0;
    band.band_id = 0;
    band.width = width;
    band.height = height;
    band.height_lines_num = height;
    band.gcli_width = 1;
    band.significance_width = 1;
    SvtCudaFrameContext ctx;
    memset(&ctx, 0, sizeof(ctx));
    int rc = svt_cuda_frame_context_create(&ctx, 3, comp_width, comp_height, 1, &band, 1, 1, 4096, 4);
    ASSERT_EQ(rc, 0);

    int err2 = svt_cuda_dwt_component_ctx(src.data(), width, width, height, decom_h, decom_v, input_bit_depth, hdr_Bw, hdr_Fq,
                                          ctx.d_in_raw, ctx.d_cur, ctx.d_other, ctx.d_vert, ctx.d_pyramid32, ctx.d_pyramid16[0], ctx.stream);
    ASSERT_EQ(err2, 0);
    cudaStreamSynchronize(ctx.stream);
    std::vector<uint16_t> out_cuda_ctx(elems);
    cudaMemcpy(out_cuda_ctx.data(), ctx.d_pyramid16[0], elems * sizeof(uint16_t), cudaMemcpyDeviceToHost);

    int mismatches_ctx = 0;
    first_mismatch = -1;
    for (size_t i = 0; i < elems; i++) {
        if (out_cpu[i] != out_cuda_ctx[i]) {
            mismatches_ctx++;
            if (first_mismatch == -1)
                first_mismatch = (int)i;
        }
    }
    printf("[bisect-real-dwt] ctx svt_cuda_dwt_component_ctx: %d/%zu mismatches, first at %d (row=%d col=%d)\n",
           mismatches_ctx, elems, first_mismatch,
           first_mismatch >= 0 ? first_mismatch / (int)width : -1,
           first_mismatch >= 0 ? first_mismatch % (int)width : -1);
    EXPECT_EQ(mismatches_ctx, 0);

    svt_cuda_frame_context_destroy(&ctx);

    // --- GC on the (proven bit-exact) real DWT pyramid output, row by row,
    // full width -- real correlated coefficients (long runs, many zeros in
    // smooth image regions), unlike TestGcCuda.cc's uniform random uint16
    // noise. Uses the same standalone entry points TestGcCuda.cc already
    // validates against synthetic random data. ---
    setup_common_rtcd_internal(CPU_FLAGS_ALL);
    const uint32_t group_size = 4;
    const uint32_t gc_width = (width + group_size - 1) / group_size;
    std::vector<uint8_t> gcli_cpu(gc_width), gcli_cuda(gc_width);
    int gc_row_mismatches = 0;
    for (uint32_t row = 0; row < height; row++) {
        std::vector<uint16_t> coeff_row(out_cpu.begin() + (size_t)row * width, out_cpu.begin() + (size_t)(row + 1) * width);
        memset(gcli_cpu.data(), 0xcd, gc_width);
        memset(gcli_cuda.data(), 0xcd, gc_width);
        gc_precinct_stage_scalar_c(gcli_cpu.data(), coeff_row.data(), group_size, width);
        int gc_err = svt_cuda_gc_precinct_stage_scalar(gcli_cuda.data(), coeff_row.data(), group_size, width);
        ASSERT_EQ(gc_err, 0);
        if (memcmp(gcli_cpu.data(), gcli_cuda.data(), gc_width) != 0) {
            gc_row_mismatches++;
            if (gc_row_mismatches <= 3) {
                for (uint32_t g = 0; g < gc_width; g++) {
                    if (gcli_cpu[g] != gcli_cuda[g]) {
                        printf("[bisect-real-dwt] GC row %u group %u: cpu=%u cuda=%u\n", row, g, gcli_cpu[g], gcli_cuda[g]);
                    }
                }
            }
        }
    }
    printf("[bisect-real-dwt] GC standalone: %d/%u rows mismatched\n", gc_row_mismatches, height);
    EXPECT_EQ(gc_row_mismatches, 0);
}
