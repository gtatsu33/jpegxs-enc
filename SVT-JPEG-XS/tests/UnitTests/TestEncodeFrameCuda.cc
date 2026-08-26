/*
* Copyright(c) 2024 Intel Corporation
* SPDX - License - Identifier: BSD - 2 - Clause - Patent
*/
#include "gtest/gtest.h"
#include "random.h"
#include "SvtJpegxs.h"
#include "SvtJpegxsEnc.h"
#include "SvtJpegxsImageBufferTools.h"
#include "EncHandle.h"
#include "PackHeaders.h"
#include "BitstreamWriter.h"
#include "Codestream.h"
#include "EncodeFrameCuda.cuh"
#include "encoder_dsp_rtcd.h"
#include "common_dsp_rtcd.h"
#include "PixelIo.h"
#include <vector>
#include <cstdint>
#include <cstring>

/* Phase 4a end-to-end bit-exact validation: encodes ONE real image through
 * the actual, unmodified CPU pipeline (svt_jpeg_xs_encoder_send_picture() /
 * get_packet()) and separately through svt_cuda_encode_frame(), then
 * memcmp's the full output bitstream.
 *
 * Two explicit, documented deviations from the encoder's DEFAULT
 * configuration were required to keep this scope tractable (see
 * PortingStrategy.txt / plan file Phase 4a notes for the full rationale):
 *   1. slice_height is forced to source_height (single slice for the whole
 *      frame) -- the default (16) creates ~135 slices/frame, and this
 *      function's precinct budget math currently only replicates the
 *      single-slice distribution formula from PackStageProcess.c.
 *   2. rate_control_mode is forced to RC_CBR_PER_PRECINCT (not the default
 *      RC_CBR_PER_PRECINCT_MOVE_PADDING) -- move-padding carries the
 *      previous precinct's actual leftover padding forward, which is a true
 *      sequential dependency between precincts; RC_CBR_PER_PRECINCT's
 *      budget distribution has no such dependency and is exactly
 *      precomputable up front, matching this function's design.
 *
 * The underlying ported RC/Quant/Pack CUDA kernels do not know or care how
 * the budget was derived -- both deviations only affect which of the
 * encoder's two real, supported RC configurations this test exercises.
 *
 * The real encoder's internal enc_common (pi_t/pi_enc_t/frame_header_buffer/
 * hdr_Lcod/frame_header_length_bytes) is read directly via
 * svt_jpeg_xs_encoder_api_prv_t (private_ptr), instead of rebuilding a
 * parallel pi_t by hand -- this guarantees the CUDA path is compared against
 * the exact geometry/header bytes the real encoder produced, not a
 * hand-rederived approximation of it.
 */

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

} // namespace

static void run_encode_frame_cuda_vs_cpu(uint32_t width, uint32_t height, uint8_t bit_depth, ColourFormat_t colour_format,
                                         uint32_t bpp_numerator, uint32_t bpp_denominator) {
    setup_common_rtcd_internal(CPU_FLAGS_ALL);
    setup_encoder_rtcd_internal(CPU_FLAGS_ALL);

    svt_jpeg_xs_encoder_api_t enc;
    ASSERT_EQ(svt_jpeg_xs_encoder_load_default_parameters(SVT_JPEGXS_API_VER_MAJOR, SVT_JPEGXS_API_VER_MINOR, &enc),
              SvtJxsErrorNone);
    enc.verbose = VERBOSE_NONE;
    enc.source_width = width;
    enc.source_height = height;
    enc.input_bit_depth = bit_depth;
    enc.colour_format = colour_format;
    enc.bpp_numerator = bpp_numerator;
    enc.bpp_denominator = bpp_denominator;
    enc.threads_num = 1;
    enc.slice_height = height;                    /* Phase 4a scope: single slice (see file comment) */
    enc.rate_control_mode = RC_CBR_PER_PRECINCT;   /* Phase 4a scope: no move-padding (see file comment) */

    svt_jpeg_xs_image_config_t image_config;
    uint32_t bytes_per_frame = 0;
    ASSERT_EQ(svt_jpeg_xs_encoder_get_image_config(
                  SVT_JPEGXS_API_VER_MAJOR, SVT_JPEGXS_API_VER_MINOR, &enc, &image_config, &bytes_per_frame),
              SvtJxsErrorNone);

    svt_jpeg_xs_image_buffer_t* in_buf = svt_jpeg_xs_image_buffer_alloc(&image_config);
    ASSERT_NE(in_buf, nullptr);
    svt_jxs_test_tool::SVTRandom rand_bd(bit_depth, false);
    for (int32_t c = 0; c < image_config.components_num; ++c) {
        uint32_t comp_w = image_config.components[c].width, comp_h = image_config.components[c].height;
        if (bit_depth <= 8) {
            uint8_t* p = (uint8_t*)in_buf->data_yuv[c];
            for (uint32_t y = 0; y < comp_h; y++) {
                for (uint32_t x = 0; x < comp_w; x++) {
                    p[(size_t)y * in_buf->stride[c] + x] = (uint8_t)rand_bd.random();
                }
            }
        }
        else {
            uint16_t* p = (uint16_t*)in_buf->data_yuv[c];
            for (uint32_t y = 0; y < comp_h; y++) {
                for (uint32_t x = 0; x < comp_w; x++) {
                    p[(size_t)y * in_buf->stride[c] + x] = (uint16_t)rand_bd.random();
                }
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
    std::vector<uint8_t> ref_bitstream(enc_output.bitstream.buffer, enc_output.bitstream.buffer + enc_output.bitstream.used_size);

    /* --- Read the real encoder's internal state directly (see file comment). --- */
    ASSERT_NE(enc.private_ptr, nullptr);
    svt_jpeg_xs_encoder_api_prv_t* prv = (svt_jpeg_xs_encoder_api_prv_t*)enc.private_ptr;
    svt_jpeg_xs_encoder_common_t* enc_common = &prv->enc_common;
    pi_t* pi = &enc_common->pi;
    ASSERT_EQ(pi->slice_num, 1u) << "Phase 4a scope requires a single slice";

    /* Sanity: this Phase 4a port assumes uniform (NORMAL) precinct geometry
     * for every precinct row, including the last one. */
    for (uint32_t c = 0; c < pi->comps_num; c++) {
        for (uint32_t b = 0; b < pi->components[c].bands_num; b++) {
            const precinct_band_info_t& nb = pi->p_info[PRECINCT_NORMAL].b_info[c][b];
            const precinct_band_info_t& lb = pi->p_info[PRECINCT_LAST_NORMAL].b_info[c][b];
            ASSERT_EQ(nb.width, lb.width) << "comp " << c << " band " << b;
            ASSERT_EQ(nb.height, lb.height) << "comp " << c << " band " << b;
        }
    }

    uint32_t slice_budget_bytes =
        enc_common->picture_header_dynamic.hdr_Lcod - enc_common->frame_header_length_bytes - SLICE_HEADER_SIZE_BYTES - CODESTREAM_SIZE_BYTES;
    std::vector<uint32_t> precinct_budgets = compute_precinct_budgets_no_move_padding(pi->precincts_line_num, slice_budget_bytes);

    SvtCudaFrameContext ctx;
    memset(&ctx, 0, sizeof(ctx));
    int rc = svt_cuda_frame_context_create_from_pi(&ctx, pi, &enc_common->pi_enc, slice_budget_bytes + 4096);
    ASSERT_EQ(rc, 0);

    const void* in_planes[FCC_MAX_COMPONENTS] = {NULL, NULL, NULL, NULL};
    uint32_t in_stride[FCC_MAX_COMPONENTS] = {0, 0, 0, 0};
    for (uint32_t c = 0; c < pi->comps_num; c++) {
        in_planes[c] = in_buf->data_yuv[c];
        in_stride[c] = in_buf->stride[c];
    }

    std::vector<uint8_t> precinct_data(slice_budget_bytes + 4096, 0);
    uint32_t precinct_used_bytes = 0;
    rc = svt_cuda_encode_frame(&ctx, in_planes, in_stride, pi->decom_h, pi->decom_v, bit_depth,
                               enc_common->picture_header_dynamic.hdr_Bw, enc_common->picture_header_dynamic.hdr_Fq,
                               (uint8_t)enc_common->picture_header_dynamic.hdr_Qpih, (uint8_t)pi->use_short_header,
                               (uint8_t)enc_common->coding_significance,
                               (uint8_t)enc_common->picture_header_dynamic.hdr_Rl, enc_common->pi_enc.max_quantization,
                               enc_common->pi_enc.max_refinement, precinct_budgets.data(), pi->bands_num_exists,
                               (uint32_t)pi->p_info[PRECINCT_NORMAL].packets_exist_num, precinct_data.data(),
                               &precinct_used_bytes);
    ASSERT_EQ(rc, 0);

    /* --- Assemble: frame header + slice header + precinct data + tail. --- */
    std::vector<uint8_t> my_bitstream;
    my_bitstream.insert(
        my_bitstream.end(), enc_common->frame_header_buffer, enc_common->frame_header_buffer + enc_common->frame_header_length_bytes);

    uint8_t slice_hdr[SLICE_HEADER_SIZE_BYTES];
    bitstream_writer_t bw_slice;
    bitstream_writer_init(&bw_slice, slice_hdr, sizeof(slice_hdr));
    write_slice_header(&bw_slice, 0);
    my_bitstream.insert(my_bitstream.end(), slice_hdr, slice_hdr + SLICE_HEADER_SIZE_BYTES);

    my_bitstream.insert(my_bitstream.end(), precinct_data.data(), precinct_data.data() + precinct_used_bytes);

    uint8_t tail[CODESTREAM_SIZE_BYTES];
    bitstream_writer_t bw_tail;
    bitstream_writer_init(&bw_tail, tail, sizeof(tail));
    write_tail(&bw_tail);
    my_bitstream.insert(my_bitstream.end(), tail, tail + CODESTREAM_SIZE_BYTES);

    EXPECT_EQ(my_bitstream.size(), ref_bitstream.size());
    if (my_bitstream.size() == ref_bitstream.size()) {
        EXPECT_EQ(memcmp(my_bitstream.data(), ref_bitstream.data(), ref_bitstream.size()), 0);
    }

    /* Phase 4b-2: the first svt_cuda_encode_frame() call above captures two
     * CUDA graphs (see EncodeFrameCuda.cu); this second call, with the same
     * context/inputs, exercises the cudaGraphLaunch() replay path instead of
     * the capture path and must produce a bit-identical bitstream. */
    std::vector<uint8_t> precinct_data2(slice_budget_bytes + 4096, 0);
    uint32_t precinct_used_bytes2 = 0;
    rc = svt_cuda_encode_frame(&ctx, in_planes, in_stride, pi->decom_h, pi->decom_v, bit_depth,
                               enc_common->picture_header_dynamic.hdr_Bw, enc_common->picture_header_dynamic.hdr_Fq,
                               (uint8_t)enc_common->picture_header_dynamic.hdr_Qpih, (uint8_t)pi->use_short_header,
                               (uint8_t)enc_common->coding_significance,
                               (uint8_t)enc_common->picture_header_dynamic.hdr_Rl, enc_common->pi_enc.max_quantization,
                               enc_common->pi_enc.max_refinement, precinct_budgets.data(), pi->bands_num_exists,
                               (uint32_t)pi->p_info[PRECINCT_NORMAL].packets_exist_num, precinct_data2.data(),
                               &precinct_used_bytes2);
    ASSERT_EQ(rc, 0);
    EXPECT_EQ(precinct_used_bytes2, precinct_used_bytes);
    EXPECT_EQ(memcmp(precinct_data2.data(), precinct_data.data(), precinct_used_bytes), 0);

    svt_cuda_frame_context_destroy(&ctx);
    svt_jpeg_xs_encoder_close(&enc);
    svt_jpeg_xs_image_buffer_free(in_buf);
    free(out_buf.buffer);
}

TEST(EncodeFrameCuda, small_yuv422_10bit_matches_real_cpu_pipeline) {
    run_encode_frame_cuda_vs_cpu(256, 64, 10, COLOUR_FORMAT_PLANAR_YUV422, 6, 1);
}

TEST(EncodeFrameCuda, real_size_4k_yuv422_10bit_matches_real_cpu_pipeline) {
    run_encode_frame_cuda_vs_cpu(3840, 2160, 10, COLOUR_FORMAT_PLANAR_YUV422, 4, 1);
}

/* Phase 5: RAW-mode packet coding validation. The tests above use uniform
 * random pixel data, which is high-entropy enough that RAW coding is
 * (almost) never selected by RateControl.c's precinct_get_budget_bytes()
 * (see PortingStrategy.txt section 8 Phase 5 / section 12) -- that is
 * exactly why the CUDA Pack path's total lack of RAW support went
 * undetected by the 20/20 existing *Cuda* tests. This test loads a real,
 * low-entropy image (flat colour-chart patches) via PixelIo so RAW coding
 * is actually exercised, deinterleaving to planar RGB the same way
 * SampleEncoderCuda's main.c does. */
static void run_encode_frame_cuda_vs_cpu_real_ppm(const char* ppm_path, uint32_t bpp_numerator, uint32_t bpp_denominator) {
    setup_common_rtcd_internal(CPU_FLAGS_ALL);
    setup_encoder_rtcd_internal(CPU_FLAGS_ALL);

    PixelImage_t image;
    ASSERT_EQ(pixel_image_load_ppm(ppm_path, &image), 0) << "failed to load " << ppm_path;

    svt_jpeg_xs_encoder_api_t enc;
    ASSERT_EQ(svt_jpeg_xs_encoder_load_default_parameters(SVT_JPEGXS_API_VER_MAJOR, SVT_JPEGXS_API_VER_MINOR, &enc),
              SvtJxsErrorNone);
    enc.verbose = VERBOSE_NONE;
    enc.source_width = image.width;
    enc.source_height = image.height;
    enc.input_bit_depth = (uint8_t)image.bit_depth;
    enc.colour_format = COLOUR_FORMAT_PLANAR_YUV444_OR_RGB;
    enc.bpp_numerator = bpp_numerator;
    enc.bpp_denominator = bpp_denominator;
    enc.threads_num = 1;
    enc.slice_height = image.height;             /* Phase 4a scope: single slice (see file comment) */
    enc.rate_control_mode = RC_CBR_PER_PRECINCT; /* Phase 4a scope: no move-padding (see file comment) */
    /* coding_raw_disable left at its default (0, i.e. hdr_Rl enabled) -- this
     * is the whole point of this test. */

    svt_jpeg_xs_image_config_t image_config;
    uint32_t bytes_per_frame = 0;
    ASSERT_EQ(svt_jpeg_xs_encoder_get_image_config(
                  SVT_JPEGXS_API_VER_MAJOR, SVT_JPEGXS_API_VER_MINOR, &enc, &image_config, &bytes_per_frame),
              SvtJxsErrorNone);

    svt_jpeg_xs_image_buffer_t* in_buf = svt_jpeg_xs_image_buffer_alloc(&image_config);
    ASSERT_NE(in_buf, nullptr);

    /* PPM data is interleaved (R,G,B,R,G,B,...); planar RGB needs one buffer
     * per channel (mirrors SampleEncoderCuda/main.c's deinterleave_rgb()). */
    const uint64_t pixels_num = (uint64_t)image.width * image.height;
    if (image.bytes_per_sample == 1) {
        const uint8_t* src = image.data;
        uint8_t* r = (uint8_t*)in_buf->data_yuv[0];
        uint8_t* g = (uint8_t*)in_buf->data_yuv[1];
        uint8_t* b = (uint8_t*)in_buf->data_yuv[2];
        for (uint64_t i = 0; i < pixels_num; i++) {
            r[i] = src[i * 3 + 0];
            g[i] = src[i * 3 + 1];
            b[i] = src[i * 3 + 2];
        }
    }
    else {
        const uint16_t* src = (const uint16_t*)image.data;
        uint16_t* r = (uint16_t*)in_buf->data_yuv[0];
        uint16_t* g = (uint16_t*)in_buf->data_yuv[1];
        uint16_t* b = (uint16_t*)in_buf->data_yuv[2];
        for (uint64_t i = 0; i < pixels_num; i++) {
            r[i] = src[i * 3 + 0];
            g[i] = src[i * 3 + 1];
            b[i] = src[i * 3 + 2];
        }
    }
    pixel_image_free(&image);

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
    std::vector<uint8_t> ref_bitstream(enc_output.bitstream.buffer, enc_output.bitstream.buffer + enc_output.bitstream.used_size);

    ASSERT_NE(enc.private_ptr, nullptr);
    svt_jpeg_xs_encoder_api_prv_t* prv = (svt_jpeg_xs_encoder_api_prv_t*)enc.private_ptr;
    svt_jpeg_xs_encoder_common_t* enc_common = &prv->enc_common;
    pi_t* pi = &enc_common->pi;
    ASSERT_EQ(pi->slice_num, 1u) << "Phase 4a scope requires a single slice";
    ASSERT_NE(enc_common->picture_header_dynamic.hdr_Rl, 0)
        << "test expects RAW coding enabled (coding_raw_disable default), otherwise it isn't testing what it claims to";

    for (uint32_t c = 0; c < pi->comps_num; c++) {
        for (uint32_t b = 0; b < pi->components[c].bands_num; b++) {
            const precinct_band_info_t& nb = pi->p_info[PRECINCT_NORMAL].b_info[c][b];
            const precinct_band_info_t& lb = pi->p_info[PRECINCT_LAST_NORMAL].b_info[c][b];
            ASSERT_EQ(nb.width, lb.width) << "comp " << c << " band " << b;
            ASSERT_EQ(nb.height, lb.height) << "comp " << c << " band " << b;
        }
    }

    uint32_t slice_budget_bytes =
        enc_common->picture_header_dynamic.hdr_Lcod - enc_common->frame_header_length_bytes - SLICE_HEADER_SIZE_BYTES - CODESTREAM_SIZE_BYTES;
    std::vector<uint32_t> precinct_budgets = compute_precinct_budgets_no_move_padding(pi->precincts_line_num, slice_budget_bytes);

    SvtCudaFrameContext ctx;
    memset(&ctx, 0, sizeof(ctx));
    int rc = svt_cuda_frame_context_create_from_pi(&ctx, pi, &enc_common->pi_enc, slice_budget_bytes + 4096);
    ASSERT_EQ(rc, 0);

    const void* in_planes[FCC_MAX_COMPONENTS] = {NULL, NULL, NULL, NULL};
    uint32_t in_stride[FCC_MAX_COMPONENTS] = {0, 0, 0, 0};
    for (uint32_t c = 0; c < pi->comps_num; c++) {
        in_planes[c] = in_buf->data_yuv[c];
        in_stride[c] = in_buf->stride[c];
    }

    std::vector<uint8_t> precinct_data(slice_budget_bytes + 4096, 0);
    uint32_t precinct_used_bytes = 0;
    rc = svt_cuda_encode_frame(&ctx, in_planes, in_stride, pi->decom_h, pi->decom_v, (uint8_t)image.bit_depth,
                               enc_common->picture_header_dynamic.hdr_Bw, enc_common->picture_header_dynamic.hdr_Fq,
                               (uint8_t)enc_common->picture_header_dynamic.hdr_Qpih, (uint8_t)pi->use_short_header,
                               (uint8_t)enc_common->coding_significance,
                               (uint8_t)enc_common->picture_header_dynamic.hdr_Rl, enc_common->pi_enc.max_quantization,
                               enc_common->pi_enc.max_refinement, precinct_budgets.data(), pi->bands_num_exists,
                               (uint32_t)pi->p_info[PRECINCT_NORMAL].packets_exist_num, precinct_data.data(),
                               &precinct_used_bytes);
    ASSERT_EQ(rc, 0);

    /* --- Assemble: frame header + slice header + precinct data + tail. --- */
    std::vector<uint8_t> my_bitstream;
    my_bitstream.insert(
        my_bitstream.end(), enc_common->frame_header_buffer, enc_common->frame_header_buffer + enc_common->frame_header_length_bytes);

    uint8_t slice_hdr[SLICE_HEADER_SIZE_BYTES];
    bitstream_writer_t bw_slice;
    bitstream_writer_init(&bw_slice, slice_hdr, sizeof(slice_hdr));
    write_slice_header(&bw_slice, 0);
    my_bitstream.insert(my_bitstream.end(), slice_hdr, slice_hdr + SLICE_HEADER_SIZE_BYTES);

    my_bitstream.insert(my_bitstream.end(), precinct_data.data(), precinct_data.data() + precinct_used_bytes);

    uint8_t tail[CODESTREAM_SIZE_BYTES];
    bitstream_writer_t bw_tail;
    bitstream_writer_init(&bw_tail, tail, sizeof(tail));
    write_tail(&bw_tail);
    my_bitstream.insert(my_bitstream.end(), tail, tail + CODESTREAM_SIZE_BYTES);

    EXPECT_EQ(my_bitstream.size(), ref_bitstream.size());
    if (my_bitstream.size() == ref_bitstream.size()) {
        EXPECT_EQ(memcmp(my_bitstream.data(), ref_bitstream.data(), ref_bitstream.size()), 0);
    }

    svt_cuda_frame_context_destroy(&ctx);
    svt_jpeg_xs_encoder_close(&enc);
    svt_jpeg_xs_image_buffer_free(in_buf);
    free(out_buf.buffer);
}

TEST(EncodeFrameCuda, real_ppm_chart_color_raw_mode_matches_real_cpu_pipeline) {
    run_encode_frame_cuda_vs_cpu_real_ppm(TESTDATA_DIR "/chart_color_3840x2160_R1.321_B1.975_CL50.0_00010_out_srz_u82.ppm", 3, 1);
}
