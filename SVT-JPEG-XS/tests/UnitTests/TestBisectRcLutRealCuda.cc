/*
* Bisection scratch test (temporary): the real-crop full-width (3840x64)
* full-pipeline mismatch (see TestBisectRealCrop444Cuda.cc,
* crop_0_0_full_width_64) survives with DWT and GC both proven bit-exact
* (TestBisectRealDwtCuda.cc). This test isolates the RC/LUT stage: after
* svt_cuda_encode_frame() on the SAME real crop, it independently recomputes
* each precinct's gtli via the real CPU reference function
* (rate_control_precinct()/precinct_quantization() from RateControl.c /
* QuantStageProcess.c) fed the SAME real gcli/significance/coefficient data
* CUDA used (extracted from ctx->d_gcli_frame/d_sig_frame/d_pyramid16, which
* are already proven bit-exact to the CPU DWT+GC reference), then diffs
* against ctx->h_gtli (CUDA's own RC decision, precinct-major).
*/
#include "gtest/gtest.h"
#include "SvtJpegxs.h"
#include "SvtJpegxsEnc.h"
#include "SvtJpegxsImageBufferTools.h"
#include "EncHandle.h"
#include "PackHeaders.h"
#include "BitstreamWriter.h"
#include "Codestream.h"
#include "RateControl.h"
#include "QuantStageProcess.h"
#include "PictureControlSet.h"
#include "PrecinctEnc.h"
#include "PackPrecinct.h"
#include "EncodeFrameCuda.cuh"
#include "encoder_dsp_rtcd.h"
#include "common_dsp_rtcd.h"
#include <vector>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <cstdlib>

namespace {
std::vector<uint32_t> compute_precinct_budgets_no_move_padding(uint32_t prec_num, uint32_t slice_budget_bytes) {
    uint32_t min_budget = slice_budget_bytes / prec_num;
    uint32_t left = slice_budget_bytes - min_budget * prec_num;
    std::vector<uint32_t> out(prec_num);
    for (uint32_t i = 0; i < prec_num; i++) {
        out[i] = min_budget + (i < left ? 1 : 0);
    }
    return out;
}

bool read_ppm(const char* path, uint32_t* out_w, uint32_t* out_h, std::vector<uint16_t>* out_rgb) {
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
    size_t n = (size_t)w * h * 3;
    std::vector<uint8_t> raw(n * 2);
    if (fread(raw.data(), 1, raw.size(), f) != raw.size()) {
        fclose(f);
        return false;
    }
    fclose(f);
    out_rgb->resize(n);
    for (size_t i = 0; i < n; i++) {
        (*out_rgb)[i] = (uint16_t)((raw[i * 2] << 8) | raw[i * 2 + 1]);
    }
    *out_w = w;
    *out_h = h;
    return true;
}
} // namespace

TEST(BisectRcLutRealCuda, real_crop_full_width_gtli_matches_cpu) {
    setup_common_rtcd_internal(CPU_FLAGS_ALL);
    setup_encoder_rtcd_internal(CPU_FLAGS_ALL);

    /* Was hardcoded to a different machine's user-specific absolute path
     * (testdata/SOMED_TUM/..., never committed to this repo); repointed at
     * the repo-relative TESTDATA_DIR (see tests/UnitTests/CMakeLists.txt). */
    const char* ppm_path = TESTDATA_DIR "/chart_color_3840x2160_R1.321_B1.975_CL50.0_00010_out_srz_u82.ppm";
    const uint32_t width = 3840, height = 64;
    const uint8_t bit_depth = 10;

    uint32_t full_w = 0, full_h = 0;
    std::vector<uint16_t> full_rgb;
    ASSERT_TRUE(read_ppm(ppm_path, &full_w, &full_h, &full_rgb));
    ASSERT_LE(width, full_w);
    ASSERT_LE(height, full_h);

    svt_jpeg_xs_encoder_api_t enc;
    ASSERT_EQ(svt_jpeg_xs_encoder_load_default_parameters(SVT_JPEGXS_API_VER_MAJOR, SVT_JPEGXS_API_VER_MINOR, &enc), SvtJxsErrorNone);
    enc.verbose = VERBOSE_NONE;
    enc.source_width = width;
    enc.source_height = height;
    enc.input_bit_depth = bit_depth;
    enc.colour_format = COLOUR_FORMAT_PLANAR_YUV444_OR_RGB;
    enc.bpp_numerator = 3;
    enc.bpp_denominator = 1;
    enc.threads_num = 1;
    enc.slice_height = height;
    enc.rate_control_mode = RC_CBR_PER_PRECINCT;

    svt_jpeg_xs_image_config_t image_config;
    uint32_t bytes_per_frame = 0;
    ASSERT_EQ(svt_jpeg_xs_encoder_get_image_config(SVT_JPEGXS_API_VER_MAJOR, SVT_JPEGXS_API_VER_MINOR, &enc, &image_config, &bytes_per_frame),
              SvtJxsErrorNone);

    svt_jpeg_xs_image_buffer_t* in_buf = svt_jpeg_xs_image_buffer_alloc(&image_config);
    ASSERT_NE(in_buf, nullptr);
    for (int32_t c = 0; c < image_config.components_num; ++c) {
        uint16_t* p = (uint16_t*)in_buf->data_yuv[c];
        for (uint32_t y = 0; y < height; y++) {
            for (uint32_t x = 0; x < width; x++) {
                size_t src_idx = ((size_t)y * full_w + x) * 3 + c;
                p[(size_t)y * in_buf->stride[c] + x] = full_rgb[src_idx];
            }
        }
    }

    ASSERT_EQ(svt_jpeg_xs_encoder_init(SVT_JPEGXS_API_VER_MAJOR, SVT_JPEGXS_API_VER_MINOR, &enc), SvtJxsErrorNone);

    svt_jpeg_xs_bitstream_buffer_t out_buf;
    out_buf.allocation_size = bytes_per_frame * 2 + 4096;
    out_buf.used_size = 0;
    out_buf.buffer = (uint8_t*)malloc(out_buf.allocation_size);
    ASSERT_NE(out_buf.buffer, nullptr);

    svt_jpeg_xs_frame_t enc_input;
    enc_input.bitstream = out_buf;
    enc_input.image = *in_buf;
    enc_input.user_prv_ctx_ptr = NULL;
    ASSERT_EQ(svt_jpeg_xs_encoder_send_picture(&enc, &enc_input, 1), SvtJxsErrorNone);
    svt_jpeg_xs_frame_t enc_output;
    memset(&enc_output, 0, sizeof(enc_output));
    ASSERT_EQ(svt_jpeg_xs_encoder_get_packet(&enc, &enc_output, 1), SvtJxsErrorNone);

    svt_jpeg_xs_encoder_api_prv_t* prv = (svt_jpeg_xs_encoder_api_prv_t*)enc.private_ptr;
    svt_jpeg_xs_encoder_common_t* enc_common = &prv->enc_common;
    pi_t* pi = &enc_common->pi;
    ASSERT_EQ(pi->slice_num, 1u);

    uint32_t slice_budget_bytes =
        enc_common->picture_header_dynamic.hdr_Lcod - enc_common->frame_header_length_bytes - SLICE_HEADER_SIZE_BYTES - CODESTREAM_SIZE_BYTES;
    std::vector<uint32_t> precinct_budgets = compute_precinct_budgets_no_move_padding(pi->precincts_line_num, slice_budget_bytes);

    SvtCudaFrameContext ctx;
    memset(&ctx, 0, sizeof(ctx));
    ASSERT_EQ(svt_cuda_frame_context_create_from_pi(&ctx, pi, &enc_common->pi_enc, slice_budget_bytes + 4096), 0);

    const void* in_planes[FCC_MAX_COMPONENTS] = {NULL, NULL, NULL, NULL};
    uint32_t in_stride[FCC_MAX_COMPONENTS] = {0, 0, 0, 0};
    for (uint32_t c = 0; c < pi->comps_num; c++) {
        in_planes[c] = in_buf->data_yuv[c];
        in_stride[c] = in_buf->stride[c];
    }

    std::vector<uint8_t> precinct_data(slice_budget_bytes + 4096, 0);
    uint32_t precinct_used_bytes = 0;
    int rc = svt_cuda_encode_frame(&ctx, in_planes, in_stride, pi->decom_h, pi->decom_v, bit_depth,
                                   enc_common->picture_header_dynamic.hdr_Bw, enc_common->picture_header_dynamic.hdr_Fq,
                                   (uint8_t)enc_common->picture_header_dynamic.hdr_Qpih, (uint8_t)pi->use_short_header,
                                   (uint8_t)enc_common->coding_significance,
                                   (uint8_t)enc_common->picture_header_dynamic.hdr_Rl, enc_common->pi_enc.max_quantization,
                                   enc_common->pi_enc.max_refinement, precinct_budgets.data(), pi->bands_num_exists,
                                   (uint32_t)pi->p_info[PRECINCT_NORMAL].packets_exist_num, 0 /* is_packed_input */,
                                   precinct_data.data(), &precinct_used_bytes);
    ASSERT_EQ(rc, 0);

    // --- Pull CUDA's frame-wide gcli/significance buffers to host (already
    // proven bit-exact to CPU DWT+GC in TestBisectRealDwtCuda.cc); use them
    // as the REAL input to an independent CPU RC computation. ---
    std::vector<uint8_t> gcli_frame_host(ctx.gcli_frame_total);
    std::vector<uint8_t> sig_frame_host(ctx.sig_frame_total);
    cudaMemcpy(gcli_frame_host.data(), ctx.d_gcli_frame, ctx.gcli_frame_total, cudaMemcpyDeviceToHost);
    cudaMemcpy(sig_frame_host.data(), ctx.d_sig_frame, ctx.sig_frame_total, cudaMemcpyDeviceToHost);

    std::vector<std::vector<uint16_t>> pyramid_host(pi->comps_num);
    for (uint32_t c = 0; c < pi->comps_num; c++) {
        size_t elems = (size_t)ctx.comp_width[c] * ctx.comp_height[c];
        pyramid_host[c].resize(elems);
        cudaMemcpy(pyramid_host[c].data(), ctx.d_pyramid16[c], elems * sizeof(uint16_t), cudaMemcpyDeviceToHost);
    }

    printf("[bisect-rc-lut] precincts_line_num=%u bands_num_all=%u\n", pi->precincts_line_num, pi->bands_num_all);

    uint32_t gtli_mismatches = 0, gtli_checked = 0;
    uint32_t quant_mismatches = 0, refine_mismatches = 0, total_bytes_mismatches = 0;

    for (uint32_t p = 0; p < pi->precincts_line_num; p++) {
        svt_jpeg_xs_encoder_common_t enc_common_shadow = *enc_common; // shallow copy, pi/pi_enc by value is fine (read-only use below)
        PictureControlSet pcs;
        memset(&pcs, 0, sizeof(pcs));
        pcs.enc_common = &enc_common_shadow;

        precinct_enc_t precinct_cpu;
        memset(&precinct_cpu, 0, sizeof(precinct_cpu));
        precinct_cpu.p_info = &pi->p_info[PRECINCT_NORMAL];

        std::vector<std::vector<uint8_t>> gcli_band(pi->bands_num_all), sig_band(pi->bands_num_all);
        std::vector<std::vector<uint16_t>> coeff_band(pi->bands_num_all);

        for (uint32_t flat = 0; flat < pi->bands_num_all; flat++) {
            uint32_t c = pi->global_band_info[flat].comp_id;
            uint32_t b = pi->global_band_info[flat].band_id;
            if (b == BAND_NOT_EXIST) {
                continue;
            }
            const precinct_band_info_t& info = pi->p_info[PRECINCT_NORMAL].b_info[c][b];
            const SvtCudaFrameBandGeom& geom = ctx.h_bands[flat];
            uint32_t height_lines = info.height;

            gcli_band[flat].resize((size_t)height_lines * info.gcli_width);
            sig_band[flat].resize((size_t)height_lines * info.significance_width);
            coeff_band[flat].resize((size_t)height_lines * info.width);

            for (uint32_t line = 0; line < height_lines; line++) {
                uint32_t global_row = p * height_lines + line;
                memcpy(&gcli_band[flat][(size_t)line * info.gcli_width],
                       &gcli_frame_host[geom.gcli_offset + (size_t)global_row * info.gcli_width],
                       info.gcli_width);
                memcpy(&sig_band[flat][(size_t)line * info.significance_width],
                       &sig_frame_host[geom.sig_offset + (size_t)global_row * info.significance_width],
                       info.significance_width);
                uint32_t pyr_row = geom.y + global_row;
                memcpy(&coeff_band[flat][(size_t)line * info.width],
                       &pyramid_host[c][(size_t)pyr_row * ctx.comp_width[c] + geom.x],
                       info.width * sizeof(uint16_t));
            }

            auto* band_cpu = &precinct_cpu.bands[c][b];
            for (uint32_t line = 0; line < height_lines; line++) {
                band_cpu->lines_common[line].gcli_data_ptr = &gcli_band[flat][(size_t)line * info.gcli_width];
                band_cpu->lines_common[line].significance_data_max_ptr = &sig_band[flat][(size_t)line * info.significance_width];
                band_cpu->lines_common[line].coeff_data_ptr_16bit = &coeff_band[flat][(size_t)line * info.width];
            }
            band_cpu->cache_buffers[0].gtli_rc_last_calculated = UINT8_MAX;
            band_cpu->cache_buffers[1].gtli_rc_last_calculated = UINT8_MAX;
            band_cpu->cache_index = 0;
            band_cpu->cache_actual = &band_cpu->cache_buffers[0];
        }
        precinct_cpu.p_info->packets_exist_num = pi->packets_num;

        rate_control_init_precinct(&pcs, &precinct_cpu, SIGN_HANDLING_STRATEGY_OFF);
        SvtJxsErrorType_t ret =
            rate_control_precinct(&pcs, &precinct_cpu, precinct_budgets[p], METHOD_PRED_DISABLE, SIGN_HANDLING_STRATEGY_OFF);
        ASSERT_EQ(ret, SvtJxsErrorNone) << "precinct " << p;
        precinct_quantization(&pcs, pi, &precinct_cpu);

        // --- Pack: CPU reference pack_precinct() on this same real,
        // now-CPU-recomputed-but-proven-identical-to-CUDA precinct data, vs
        // the actual bytes svt_cuda_encode_frame() wrote for this precinct
        // (k_pack_precinct_frame, NOT the separately-tested standalone
        // svt_cuda_pack_precinct/PackCuda.cu -- see EncodeFrameCuda.cu's own
        // EfcWriter pack helpers). ---
        std::vector<uint8_t> ref_pack_buf(precinct_cpu.pack_total_bytes + 16, 0xAB);
        bitstream_writer_t bw;
        bitstream_writer_init(&bw, ref_pack_buf.data(), ref_pack_buf.size());
        SvtJxsErrorType_t pack_ret = pack_precinct(&bw, pi, &precinct_cpu, SIGN_HANDLING_STRATEGY_OFF);
        ASSERT_EQ(pack_ret, SvtJxsErrorNone) << "precinct " << p;
        uint32_t ref_pack_used = bitstream_writer_get_used_bytes(&bw);
        ASSERT_EQ(ref_pack_used, precinct_cpu.pack_total_bytes) << "precinct " << p;

        uint32_t cuda_out_offset = ctx.h_out_offset ? ctx.h_out_offset[p] : 0xFFFFFFFF;
        if (ctx.h_out_offset && cuda_out_offset + precinct_cpu.pack_total_bytes <= precinct_data.size()) {
            int pack_diff = memcmp(ref_pack_buf.data(), precinct_data.data() + cuda_out_offset, precinct_cpu.pack_total_bytes);
            if (pack_diff != 0) {
                int first_byte_diff = -1;
                int byte_diffs = 0;
                for (uint32_t i = 0; i < precinct_cpu.pack_total_bytes; i++) {
                    if (ref_pack_buf[i] != precinct_data[cuda_out_offset + i]) {
                        if (first_byte_diff == -1)
                            first_byte_diff = (int)i;
                        byte_diffs++;
                    }
                }
                printf("[bisect-rc-lut] PACK MISMATCH precinct %u: out_offset=%u total_bytes=%u first_byte_diff=%d byte_diffs=%d/%u\n",
                       p, cuda_out_offset, precinct_cpu.pack_total_bytes, first_byte_diff, byte_diffs, precinct_cpu.pack_total_bytes);
                for (int i = first_byte_diff; i >= 0 && i < first_byte_diff + 8 && i < (int)precinct_cpu.pack_total_bytes; i++) {
                    printf("[bisect-rc-lut]   byte %d: cpu=0x%02x cuda=0x%02x\n", i, ref_pack_buf[i], precinct_data[cuda_out_offset + i]);
                }
            }
            EXPECT_EQ(pack_diff, 0) << "precinct " << p << " packed bytes differ";
        }

        for (uint32_t flat = 0; flat < pi->bands_num_all; flat++) {
            uint32_t c = pi->global_band_info[flat].comp_id;
            uint32_t b = pi->global_band_info[flat].band_id;
            if (b == BAND_NOT_EXIST) {
                continue;
            }
            uint8_t cpu_gtli = precinct_cpu.bands[c][b].gtli;
            uint8_t cuda_gtli = ctx.h_gtli[(size_t)p * pi->bands_num_all + flat];
            gtli_checked++;
            if (cpu_gtli != cuda_gtli) {
                gtli_mismatches++;
                if (gtli_mismatches <= 10) {
                    printf("[bisect-rc-lut] precinct %u band flat=%u (comp=%u band=%u): cpu_gtli=%u cuda_gtli=%u\n",
                           p, flat, c, b, cpu_gtli, cuda_gtli);
                }
            }
        }

        uint8_t cuda_quant = ctx.h_quant ? ctx.h_quant[p] : 0xff;
        uint8_t cuda_refine = ctx.h_refine ? ctx.h_refine[p] : 0xff;
        uint32_t cuda_total = ctx.h_total_bytes ? ctx.h_total_bytes[p] : 0xffffffff;
        if (ctx.h_quant && cuda_quant != precinct_cpu.pack_quantization) {
            quant_mismatches++;
            printf("[bisect-rc-lut] precinct %u quantization: cpu=%u cuda=%u\n", p, precinct_cpu.pack_quantization, cuda_quant);
        }
        if (ctx.h_refine && cuda_refine != precinct_cpu.pack_refinement) {
            refine_mismatches++;
            printf("[bisect-rc-lut] precinct %u refinement: cpu=%u cuda=%u\n", p, precinct_cpu.pack_refinement, cuda_refine);
        }
        if (ctx.h_total_bytes && cuda_total != precinct_cpu.pack_total_bytes) {
            total_bytes_mismatches++;
            printf("[bisect-rc-lut] precinct %u total_bytes: cpu=%u cuda=%u\n", p, precinct_cpu.pack_total_bytes, cuda_total);
        }
    }

    printf("[bisect-rc-lut] gtli_mismatches=%u/%u quant_mismatches=%u refine_mismatches=%u total_bytes_mismatches=%u\n",
           gtli_mismatches, gtli_checked, quant_mismatches, refine_mismatches, total_bytes_mismatches);
    EXPECT_EQ(gtli_mismatches, 0u);
    EXPECT_EQ(quant_mismatches, 0u);
    EXPECT_EQ(refine_mismatches, 0u);
    EXPECT_EQ(total_bytes_mismatches, 0u);

    svt_cuda_frame_context_destroy(&ctx);
    svt_jpeg_xs_encoder_close(&enc);
    svt_jpeg_xs_image_buffer_free(in_buf);
    free(out_buf.buffer);
}
