/*
* Bisection scratch test (temporary): same as TestBisectSmall444Cuda's small
* test, but with a real-image crop (from testdata/) substituted for uniform
* random noise, since random-444-at-4K was proven bit-exact (see
* TestBisectSmall444Cuda.cc) -- the mismatch is data-content dependent, not
* format/size dependent. This crops a small rectangle directly out of the
* real PPM (manual minimal P6 parsing here, deliberately not using PixelIo/
* stb_image, since this is throwaway bisection code) to get a fast repro if
* real image statistics are the actual trigger.
*/
#include "gtest/gtest.h"
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

/* Minimal binary PPM (P6) reader: returns interleaved 16-bit-per-sample RGB
 * (big-endian on disk, converted to native here), full image. */
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
    fgetc(f); // single whitespace before binary data
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

static void run_real_crop444(const char* ppm_path, uint32_t crop_x, uint32_t crop_y, uint32_t width, uint32_t height) {
    setup_common_rtcd_internal(CPU_FLAGS_ALL);
    setup_encoder_rtcd_internal(CPU_FLAGS_ALL);

    uint32_t full_w = 0, full_h = 0;
    std::vector<uint16_t> full_rgb;
    ASSERT_TRUE(read_ppm(ppm_path, &full_w, &full_h, &full_rgb)) << "could not read " << ppm_path;
    ASSERT_LE(crop_x + width, full_w);
    ASSERT_LE(crop_y + height, full_h);

    const uint8_t bit_depth = 10;

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
                size_t src_idx = ((size_t)(crop_y + y) * full_w + (crop_x + x)) * 3 + c;
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
    std::vector<uint8_t> ref_bitstream(enc_output.bitstream.buffer, enc_output.bitstream.buffer + enc_output.bitstream.used_size);

    ASSERT_NE(enc.private_ptr, nullptr);
    svt_jpeg_xs_encoder_api_prv_t* prv = (svt_jpeg_xs_encoder_api_prv_t*)enc.private_ptr;
    svt_jpeg_xs_encoder_common_t* enc_common = &prv->enc_common;
    pi_t* pi = &enc_common->pi;
    ASSERT_EQ(pi->slice_num, 1u);
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
                               (uint32_t)pi->p_info[PRECINCT_NORMAL].packets_exist_num, 0 /* is_packed_input */,
                               precinct_data.data(), &precinct_used_bytes);
    ASSERT_EQ(rc, 0);

    std::vector<uint8_t> my_bitstream;
    my_bitstream.insert(my_bitstream.end(), enc_common->frame_header_buffer, enc_common->frame_header_buffer + enc_common->frame_header_length_bytes);

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

    printf("[bisect-real-crop] crop=%ux%u+%u+%u precincts_line_num=%u ref=%zu my=%zu\n",
           width, height, crop_x, crop_y, pi->precincts_line_num, ref_bitstream.size(), my_bitstream.size());
    EXPECT_EQ(my_bitstream.size(), ref_bitstream.size());
    if (my_bitstream.size() == ref_bitstream.size()) {
        int first_diff = -1;
        int diffs = 0;
        for (size_t i = 0; i < ref_bitstream.size(); i++) {
            if (ref_bitstream[i] != my_bitstream[i]) {
                if (first_diff == -1)
                    first_diff = (int)i;
                diffs++;
            }
        }
        printf("[bisect-real-crop] frame_header_length_bytes=%u first_diff_offset=%d total_diffs=%d/%zu\n",
               enc_common->frame_header_length_bytes, first_diff, diffs, ref_bitstream.size());
        EXPECT_EQ(memcmp(my_bitstream.data(), ref_bitstream.data(), ref_bitstream.size()), 0);
    }

    svt_cuda_frame_context_destroy(&ctx);
    svt_jpeg_xs_encoder_close(&enc);
    svt_jpeg_xs_image_buffer_free(in_buf);
    free(out_buf.buffer);
}

/* Was hardcoded to a different machine's user-specific absolute path
 * (testdata/SOMED_TUM/..., never committed to this repo); repointed at the
 * repo-relative TESTDATA_DIR (see tests/UnitTests/CMakeLists.txt) chart image
 * -- same 3840x2160/10-bit real-image shape this scratch test needs. */
static const char* kRealPpm = TESTDATA_DIR "/chart_color_3840x2160_R1.321_B1.975_CL50.0_00010_out_srz_u82.ppm";

TEST(BisectRealCrop444Cuda, crop_0_0_256x64) {
    run_real_crop444(kRealPpm, 0, 0, 256, 64);
}

TEST(BisectRealCrop444Cuda, crop_1000_1000_256x64) {
    run_real_crop444(kRealPpm, 1000, 1000, 256, 64);
}

TEST(BisectRealCrop444Cuda, crop_0_0_full_width_64) {
    run_real_crop444(kRealPpm, 0, 0, 3840, 64);
}
