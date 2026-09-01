/*
* Copyright(c) 2024 Intel Corporation
* SPDX - License - Identifier: BSD - 2 - Clause - Patent
*/

/* CUDA-enabled build sample: encodes one PPM image through both the real
 * CPU pipeline (public API) and the standalone GPU path (svt_cuda_encode_frame,
 * Phase 4 of the CUDA port), then verifies the two outputs are bit-exact and
 * prints a timing comparison. See PortingStrategy.txt section 12 for the
 * design and section 7/8 for why this app needs the Encoder library's
 * internal headers (svt_cuda_encode_frame() is intentionally not wired into
 * the public API). Modeled directly on
 * tests/UnitTests/TestEncodeFrameCuda.cc, which validates this same
 * CPU-vs-CUDA bit-exactness as an automated test.
 *
 * Scope (Phase 4a, inherited unchanged): VPRED disabled, Signs handling=OFF,
 * RC_CBR_PER_PRECINCT (not the default move-padding mode), and every
 * precinct row must share the same (NORMAL) band geometry -- i.e.
 * source_height must be evenly divisible by the precinct height. Inputs
 * outside this scope are rejected with an error message rather than
 * silently adjusted.
 *
 * Multi-slice (2026-08-29, see PortingStrategy.txt TODO E): slice_height
 * uses the library default (16), matching the ISO/IEC 21122-2 structural
 * requirement for any named (Main/High) profile. svt_cuda_encode_frame()
 * still processes every precinct of the whole frame in one GPU batch --
 * only this app's host-side RC-budget split and bitstream assembly are
 * slice-aware, via svt_cuda_get_precinct_layout().
 */

#include <SvtJpegxsEnc.h>
#include <SvtJpegxsImageBufferTools.h>
#include "PixelIo.h"
#include "UtilityApp.h"

#include "EncHandle.h"
#include "PackHeaders.h"
#include "BitstreamWriter.h"
#include "Codestream.h"
#include "EncodeFrameCuda.cuh"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* PPM data is interleaved (R,G,B,R,G,B,...); the encoder's planar RGB format
 * (COLOUR_FORMAT_PLANAR_YUV444_OR_RGB) needs one buffer per channel. */
static void deinterleave_rgb(const PixelImage_t *image, svt_jpeg_xs_image_buffer_t *dst) {
    const uint64_t pixels_num = (uint64_t)image->width * image->height;
    if (image->bytes_per_sample == 1) {
        const uint8_t *src = image->data;
        uint8_t *r = (uint8_t *)dst->data_yuv[0];
        uint8_t *g = (uint8_t *)dst->data_yuv[1];
        uint8_t *b = (uint8_t *)dst->data_yuv[2];
        for (uint64_t i = 0; i < pixels_num; i++) {
            r[i] = src[i * 3 + 0];
            g[i] = src[i * 3 + 1];
            b[i] = src[i * 3 + 2];
        }
    }
    else {
        const uint16_t *src = (const uint16_t *)image->data;
        uint16_t *r = (uint16_t *)dst->data_yuv[0];
        uint16_t *g = (uint16_t *)dst->data_yuv[1];
        uint16_t *b = (uint16_t *)dst->data_yuv[2];
        for (uint64_t i = 0; i < pixels_num; i++) {
            r[i] = src[i * 3 + 0];
            g[i] = src[i * 3 + 1];
            b[i] = src[i * 3 + 2];
        }
    }
}

/* Planar YUV422 8-bit input is already laid out the way the encoder's planar
 * buffers expect (Y plane, then U, then V; component strides equal component
 * widths for non-packed formats -- see ImageBuffer.c), so this is a flat
 * per-plane memcpy, unlike deinterleave_rgb() above. */
static void copy_yuv422p8_planes(const PixelImage_t *image, svt_jpeg_xs_image_buffer_t *dst) {
    const uint8_t *src = image->data;
    const size_t y_bytes = (size_t)image->width * image->height;
    const size_t c_bytes = (size_t)(image->width / 2) * image->height;
    memcpy(dst->data_yuv[0], src, y_bytes);
    memcpy(dst->data_yuv[1], src + y_bytes, c_bytes);
    memcpy(dst->data_yuv[2], src + y_bytes + c_bytes, c_bytes);
}

/* Raw YUV has no header, so width/height are parsed from the filename's
 * "_WIDTHxHEIGHT_" token (matches the naming already used by every raw file
 * in testdata/, e.g. test_1920x1080_yuv422p8.yuv). Returns 0 on success. */
static int parse_wxh_from_filename(const char *path, uint32_t *out_w, uint32_t *out_h) {
    const char *base = path;
    for (const char *p = path; *p; p++) {
        if (*p == '/' || *p == '\\') {
            base = p + 1;
        }
    }
    for (const char *p = base; *p; p++) {
        if (*p < '0' || *p > '9') {
            continue;
        }
        const char *digit_start = p;
        while (*p >= '0' && *p <= '9') {
            p++;
        }
        if (*p != 'x' && *p != 'X') {
            continue;
        }
        const char *w_end = p;
        const char *h_start = p + 1;
        const char *q = h_start;
        while (*q >= '0' && *q <= '9') {
            q++;
        }
        if (q == h_start) {
            continue;
        }
        /* Require a non-digit (typically '_') immediately before/after so a
         * bit-depth-like "10" inside a longer number isn't mistaken for one
         * half of the pair. */
        int before_ok = (digit_start == base) || (*(digit_start - 1) < '0' || *(digit_start - 1) > '9');
        if (!before_ok) {
            p = q - 1;
            continue;
        }
        *out_w = (uint32_t)strtoul(digit_start, NULL, 10);
        *out_h = (uint32_t)strtoul(h_start, NULL, 10);
        (void)w_end;
        return 0;
    }
    return -1;
}

static int has_suffix_ci(const char *s, const char *suffix) {
    size_t s_len = strlen(s), suf_len = strlen(suffix);
    if (suf_len > s_len) {
        return 0;
    }
    const char *tail = s + (s_len - suf_len);
    for (size_t i = 0; i < suf_len; i++) {
        char a = tail[i], b = suffix[i];
        if (a >= 'A' && a <= 'Z')
            a = (char)(a - 'A' + 'a');
        if (b >= 'A' && b <= 'Z')
            b = (char)(b - 'A' + 'a');
        if (a != b) {
            return 0;
        }
    }
    return 1;
}

/* Replicates RC_CBR_PER_PRECINCT's (no move-padding) per-precinct byte-budget
 * split -- an even distribution of the slice budget, remainder handed to the
 * first `left` precincts. Matches
 * compute_precinct_budgets_no_move_padding() in TestEncodeFrameCuda.cc. */
static void compute_precinct_budgets(uint32_t prec_num, uint32_t slice_budget_bytes, uint32_t *out_budgets) {
    uint32_t min_budget = slice_budget_bytes / prec_num;
    uint32_t left = slice_budget_bytes - min_budget * prec_num;
    for (uint32_t i = 0; i < prec_num; i++) {
        out_budgets[i] = min_budget + (i < left ? 1 : 0);
    }
}

int32_t main(int32_t argc, char *argv[]) {
    if (argc < 3) {
        printf("Usage: %s <input.ppm|input.yuv> <output_prefix> [bpp]\n", argv[0]);
        printf("  input.ppm      binary PPM (Netpbm \"P6\", interleaved RGB), maxval 1-65535\n");
        printf("  input.yuv      raw headerless planar YUV422 8-bit (\"yuv422p\"), one frame;\n");
        printf("                 width/height parsed from a \"_WIDTHxHEIGHT_\" token in the filename\n");
        printf("  output_prefix  bitstreams are written to <output_prefix>.cpu.jxs / .cuda.jxs\n");
        printf("  bpp            optional target bits-per-pixel, integer or decimal (e.g. 0.5, 3, 3.75). Default: 3\n");
        printf("Scope: RC_CBR_PER_PRECINCT, source_height evenly divisible by\n");
        printf("the precinct height (see PortingStrategy.txt section 12 / Phase 4a).\n");
        return -1;
    }
    const char *input_file_name = argv[1];
    const char *output_prefix = argv[2];
    const char *bpp_arg = argc > 3 ? argv[3] : "3";

    int is_yuv = has_suffix_ci(input_file_name, ".yuv");

    PixelImage_t image;
    if (is_yuv) {
        uint32_t yuv_w = 0, yuv_h = 0;
        if (parse_wxh_from_filename(input_file_name, &yuv_w, &yuv_h) != 0) {
            printf("Could not find a \"_WIDTHxHEIGHT_\" token in the filename to size the raw yuv input: %s\n",
                   input_file_name);
            return -1;
        }
        if (pixel_image_load_yuv422p8(input_file_name, yuv_w, yuv_h, &image) != 0) {
            return -1;
        }
    }
    else {
        if (pixel_image_load_ppm(input_file_name, &image) != 0) {
            return -1;
        }
    }

    svt_jpeg_xs_encoder_api_t enc;
    SvtJxsErrorType_t err = svt_jpeg_xs_encoder_load_default_parameters(SVT_JPEGXS_API_VER_MAJOR, SVT_JPEGXS_API_VER_MINOR, &enc);
    if (err != SvtJxsErrorNone) {
        pixel_image_free(&image);
        return err;
    }
    enc.verbose = VERBOSE_NONE;
    enc.source_width = image.width;
    enc.source_height = image.height;
    enc.input_bit_depth = (uint8_t)image.bit_depth;
    enc.colour_format = is_yuv ? COLOUR_FORMAT_PLANAR_YUV422 : COLOUR_FORMAT_PLANAR_YUV444_OR_RGB;
    if (is_yuv) {
        /* ISO/IEC 21122-2 defines no "High" (NLy<=2) profile for 4:2:2/4:0:0
         * sampling -- Main (NLy<=1) is the only named profile available, so
         * the actual vertical decomposition must be capped to match it (the
         * library default is ndecomp_v=2, which would make the declared
         * Main422.10 profile's Nly field a lie). See PortingStrategy.txt
         * TODO E. */
        enc.ndecomp_v = 1;
    }
    parse_bpp_arg(bpp_arg, &enc.bpp_numerator, &enc.bpp_denominator);
    enc.threads_num = 1; /* single-threaded CPU reference, matching the timing methodology in PortingStrategy.txt section 10 */
    /* slice_height left at the library default (16, set by
     * svt_jpeg_xs_encoder_load_default_parameters()) -- see PortingStrategy.txt
     * TODO E: an image-height single slice made every named (Main/High)
     * profile declaration structurally non-conformant. */
    enc.rate_control_mode = RC_CBR_PER_PRECINCT; /* Phase 4a scope: no move-padding */

    err = svt_jpeg_xs_encoder_init(SVT_JPEGXS_API_VER_MAJOR, SVT_JPEGXS_API_VER_MINOR, &enc);
    if (err != SvtJxsErrorNone) {
        printf("svt_jpeg_xs_encoder_init failed: %d\n", err);
        pixel_image_free(&image);
        return err;
    }

    svt_jpeg_xs_image_config_t image_config;
    uint32_t bytes_per_frame = 0;
    err = svt_jpeg_xs_encoder_get_image_config(
        SVT_JPEGXS_API_VER_MAJOR, SVT_JPEGXS_API_VER_MINOR, &enc, &image_config, &bytes_per_frame);
    if (err != SvtJxsErrorNone) {
        svt_jpeg_xs_encoder_close(&enc);
        pixel_image_free(&image);
        return err;
    }

    svt_jpeg_xs_image_buffer_t *in_buf = svt_jpeg_xs_image_buffer_alloc(&image_config);
    if (!in_buf) {
        svt_jpeg_xs_encoder_close(&enc);
        pixel_image_free(&image);
        return SvtJxsErrorInsufficientResources;
    }
    if (is_yuv) {
        copy_yuv422p8_planes(&image, in_buf);
    }
    else {
        deinterleave_rgb(&image, in_buf);
    }
    pixel_image_free(&image);

    svt_jpeg_xs_bitstream_buffer_t out_buf;
    out_buf.allocation_size = bytes_per_frame * 2 + 4096;
    out_buf.used_size = 0;
    out_buf.buffer = malloc(out_buf.allocation_size);
    if (!out_buf.buffer) {
        svt_jpeg_xs_image_buffer_free(in_buf);
        svt_jpeg_xs_encoder_close(&enc);
        return SvtJxsErrorInsufficientResources;
    }

    svt_jpeg_xs_frame_t enc_input;
    enc_input.bitstream = out_buf;
    enc_input.image = *in_buf;
    enc_input.user_prv_ctx_ptr = NULL;

    uint64_t t0s, t0m, t1s, t1m;
    get_current_time(&t0s, &t0m);
    err = svt_jpeg_xs_encoder_send_picture(&enc, &enc_input, 1 /*blocking*/);
    if (err == SvtJxsErrorNone) {
        svt_jpeg_xs_frame_t enc_output;
        err = svt_jpeg_xs_encoder_get_packet(&enc, &enc_output, 1 /*blocking*/);
        get_current_time(&t1s, &t1m);
        if (err != SvtJxsErrorNone) {
            printf("svt_jpeg_xs_encoder_get_packet failed: %d\n", err);
            free(out_buf.buffer);
            svt_jpeg_xs_image_buffer_free(in_buf);
            svt_jpeg_xs_encoder_close(&enc);
            return err;
        }
        out_buf.used_size = enc_output.bitstream.used_size;
    }
    else {
        printf("svt_jpeg_xs_encoder_send_picture failed: %d\n", err);
        free(out_buf.buffer);
        svt_jpeg_xs_image_buffer_free(in_buf);
        svt_jpeg_xs_encoder_close(&enc);
        return err;
    }
    double cpu_ms = compute_elapsed_time_in_ms(t0s, t0m, t1s, t1m);

    char cpu_path[1024];
    snprintf(cpu_path, sizeof(cpu_path), "%s.cpu.jxs", output_prefix);
    FILE *cpu_file = NULL;
    FOPEN(cpu_file, cpu_path, "wb");
    if (cpu_file) {
        fwrite(out_buf.buffer, 1, out_buf.used_size, cpu_file);
        fclose(cpu_file);
    }
    else {
        printf("Can not open output file: %s!\n", cpu_path);
    }

    /* --- Read the real encoder's internal state directly (svt_cuda_encode_frame()
     * is not wired into the public API -- see PortingStrategy.txt section 7). --- */
    svt_jpeg_xs_encoder_api_prv_t *prv = (svt_jpeg_xs_encoder_api_prv_t *)enc.private_ptr;
    svt_jpeg_xs_encoder_common_t *enc_common = &prv->enc_common;
    pi_t *pi = &enc_common->pi;

    int scope_ok = 1;
    for (uint32_t c = 0; c < pi->comps_num && scope_ok; c++) {
        for (uint32_t b = 0; b < pi->components[c].bands_num; b++) {
            const precinct_band_info_t *nb = &pi->p_info[PRECINCT_NORMAL].b_info[c][b];
            const precinct_band_info_t *lb = &pi->p_info[PRECINCT_LAST_NORMAL].b_info[c][b];
            if (nb->width != lb->width || nb->height != lb->height) {
                scope_ok = 0;
                break;
            }
        }
    }
    if (!scope_ok) {
        printf("CPU-only encode written to %s. CUDA comparison skipped: image geometry is\n"
               "outside the Phase 4a scope (source_height must be evenly divisible by the\n"
               "precinct height so every precinct row shares the same band geometry).\n",
               cpu_path);
        free(out_buf.buffer);
        svt_jpeg_xs_image_buffer_free(in_buf);
        svt_jpeg_xs_encoder_close(&enc);
        return 2;
    }

    /* Total precinct-data budget across the whole frame (all slices), matching
     * EncHandle.c's size_all_precincts_bytes formula. */
    uint32_t total_precinct_budget_bytes = enc_common->picture_header_dynamic.hdr_Lcod - enc_common->frame_header_length_bytes -
        SLICE_HEADER_SIZE_BYTES * pi->slice_num - CODESTREAM_SIZE_BYTES;
    uint32_t *precinct_budgets = malloc(sizeof(uint32_t) * pi->precincts_line_num);
    if (!precinct_budgets) {
        free(out_buf.buffer);
        svt_jpeg_xs_image_buffer_free(in_buf);
        svt_jpeg_xs_encoder_close(&enc);
        return SvtJxsErrorInsufficientResources;
    }
    /* Split the per-precinct budget in two stages, exactly matching
     * PreRcStageProcess.c (per-slice budget from enc_common->slice_sizes[])
     * then PackStageProcess.c:594-601 (even split within the slice, RC_CBR_
     * PER_PRECINCT branch) -- required for CPU/CUDA bit-exactness once
     * slice_num > 1. */
    for (uint32_t slice_idx = 0; slice_idx < pi->slice_num; slice_idx++) {
        uint32_t prec_first_idx = pi->precincts_per_slice * slice_idx;
        uint32_t prec_num_in_slice = (slice_idx + 1 < pi->slice_num) ? pi->precincts_per_slice
                                                                      : pi->precincts_line_num - prec_first_idx;
        uint32_t this_slice_budget_bytes = (slice_idx + 1 < pi->slice_num)
            ? enc_common->slice_sizes[slice_idx] - SLICE_HEADER_SIZE_BYTES
            : enc_common->slice_sizes[slice_idx] - SLICE_HEADER_SIZE_BYTES - CODESTREAM_SIZE_BYTES;
        compute_precinct_budgets(prec_num_in_slice, this_slice_budget_bytes, precinct_budgets + prec_first_idx);
    }

    SvtCudaFrameContext ctx;
    memset(&ctx, 0, sizeof(ctx));
    int rc = svt_cuda_frame_context_create_from_pi(&ctx, pi, &enc_common->pi_enc, total_precinct_budget_bytes + 4096);
    if (rc != 0) {
        printf("svt_cuda_frame_context_create_from_pi failed: %d\n", rc);
        free(precinct_budgets);
        free(out_buf.buffer);
        svt_jpeg_xs_image_buffer_free(in_buf);
        svt_jpeg_xs_encoder_close(&enc);
        return -1;
    }

    const void *in_planes[FCC_MAX_COMPONENTS] = {NULL, NULL, NULL, NULL};
    uint32_t in_stride[FCC_MAX_COMPONENTS] = {0, 0, 0, 0};
    for (uint32_t c = 0; c < pi->comps_num; c++) {
        in_planes[c] = in_buf->data_yuv[c];
        in_stride[c] = in_buf->stride[c];
    }

    uint32_t precinct_capacity = total_precinct_budget_bytes + 4096;
    uint8_t *precinct_data = malloc(precinct_capacity);
    uint32_t precinct_used_bytes = 0;
    if (!precinct_data) {
        svt_cuda_frame_context_destroy(&ctx);
        free(precinct_budgets);
        free(out_buf.buffer);
        svt_jpeg_xs_image_buffer_free(in_buf);
        svt_jpeg_xs_encoder_close(&enc);
        return SvtJxsErrorInsufficientResources;
    }

    /* Two calls on the SAME ctx: the 1st necessarily captures graph1+graph2
     * (hundreds of kernel-launch nodes recorded + cudaGraphInstantiate), a
     * one-time cost folded into that call's measured time; the 2nd is a pure
     * cudaGraphLaunch replay. This matches tests/Benchmark's own methodology
     * (see PortingStrategy.txt section 10, "capture excluded as warm-up") --
     * a single-call measurement here would silently report a cold-start
     * number, not the steady-state per-frame cost the 16ms/33ms targets are
     * about. The 2nd call's output is what gets written out / bit-exact
     * checked below. */
    get_current_time(&t0s, &t0m);
    rc = svt_cuda_encode_frame(&ctx,
                                in_planes,
                                in_stride,
                                pi->decom_h,
                                pi->decom_v,
                                enc.input_bit_depth,
                                enc_common->picture_header_dynamic.hdr_Bw,
                                enc_common->picture_header_dynamic.hdr_Fq,
                                (uint8_t)enc_common->picture_header_dynamic.hdr_Qpih,
                                (uint8_t)pi->use_short_header,
                                (uint8_t)enc_common->coding_significance,
                                (uint8_t)enc_common->picture_header_dynamic.hdr_Rl,
                                enc_common->pi_enc.max_quantization,
                                enc_common->pi_enc.max_refinement,
                                precinct_budgets,
                                pi->bands_num_exists,
                                (uint32_t)pi->p_info[PRECINCT_NORMAL].packets_exist_num,
                                precinct_data,
                                &precinct_used_bytes);
    get_current_time(&t1s, &t1m);
    double cuda_cold_ms = compute_elapsed_time_in_ms(t0s, t0m, t1s, t1m);

    if (rc == 0) {
        get_current_time(&t0s, &t0m);
        rc = svt_cuda_encode_frame(&ctx,
                                    in_planes,
                                    in_stride,
                                    pi->decom_h,
                                    pi->decom_v,
                                    enc.input_bit_depth,
                                    enc_common->picture_header_dynamic.hdr_Bw,
                                    enc_common->picture_header_dynamic.hdr_Fq,
                                    (uint8_t)enc_common->picture_header_dynamic.hdr_Qpih,
                                    (uint8_t)pi->use_short_header,
                                    (uint8_t)enc_common->coding_significance,
                                    (uint8_t)enc_common->picture_header_dynamic.hdr_Rl,
                                    enc_common->pi_enc.max_quantization,
                                    enc_common->pi_enc.max_refinement,
                                    precinct_budgets,
                                    pi->bands_num_exists,
                                    (uint32_t)pi->p_info[PRECINCT_NORMAL].packets_exist_num,
                                    precinct_data,
                                    &precinct_used_bytes);
        get_current_time(&t1s, &t1m);
    }
    double cuda_ms = compute_elapsed_time_in_ms(t0s, t0m, t1s, t1m);

    if (rc != 0) {
        printf("svt_cuda_encode_frame failed: %d\n", rc);
        free(precinct_data);
        svt_cuda_frame_context_destroy(&ctx);
        free(precinct_budgets);
        free(out_buf.buffer);
        svt_jpeg_xs_image_buffer_free(in_buf);
        svt_jpeg_xs_encoder_close(&enc);
        return -1;
    }

    /* --- Assemble: frame header + (slice header + precinct data) per slice +
     * tail (same layout the real encoder produces; svt_cuda_encode_frame()
     * only ports the precinct data path, see EncodeFrameCuda.cuh). Per-slice
     * byte ranges within precinct_data come from svt_cuda_get_precinct_layout()
     * -- precinct index there is 1:1 with pi_t's precinct-row index, same as
     * PackStageProcess.c's precincts_per_slice * slice_idx indexing. --- */
    uint32_t cuda_bitstream_size = enc_common->frame_header_length_bytes + SLICE_HEADER_SIZE_BYTES * pi->slice_num +
        precinct_used_bytes + CODESTREAM_SIZE_BYTES;
    uint8_t *cuda_bitstream = malloc(cuda_bitstream_size);
    if (!cuda_bitstream) {
        free(precinct_data);
        svt_cuda_frame_context_destroy(&ctx);
        free(precinct_budgets);
        free(out_buf.buffer);
        svt_jpeg_xs_image_buffer_free(in_buf);
        svt_jpeg_xs_encoder_close(&enc);
        return SvtJxsErrorInsufficientResources;
    }
    uint32_t offset = 0;
    memcpy(cuda_bitstream + offset, enc_common->frame_header_buffer, enc_common->frame_header_length_bytes);
    offset += enc_common->frame_header_length_bytes;

    const uint32_t *precinct_offsets = NULL;
    const uint32_t *precinct_sizes = NULL;
    svt_cuda_get_precinct_layout(&ctx, &precinct_offsets, &precinct_sizes);

    for (uint32_t slice_idx = 0; slice_idx < pi->slice_num; slice_idx++) {
        uint32_t prec_first_idx = pi->precincts_per_slice * slice_idx;
        uint32_t prec_num_in_slice = (slice_idx + 1 < pi->slice_num) ? pi->precincts_per_slice
                                                                      : pi->precincts_line_num - prec_first_idx;
        uint32_t prec_last_idx = prec_first_idx + prec_num_in_slice - 1;
        uint32_t slice_byte_start = precinct_offsets[prec_first_idx];
        uint32_t slice_byte_len = precinct_offsets[prec_last_idx] + precinct_sizes[prec_last_idx] - slice_byte_start;

        bitstream_writer_t bw_slice;
        bitstream_writer_init(&bw_slice, cuda_bitstream + offset, SLICE_HEADER_SIZE_BYTES);
        write_slice_header(&bw_slice, (int)slice_idx);
        offset += SLICE_HEADER_SIZE_BYTES;

        memcpy(cuda_bitstream + offset, precinct_data + slice_byte_start, slice_byte_len);
        offset += slice_byte_len;
    }

    bitstream_writer_t bw_tail;
    bitstream_writer_init(&bw_tail, cuda_bitstream + offset, CODESTREAM_SIZE_BYTES);
    write_tail(&bw_tail);
    offset += CODESTREAM_SIZE_BYTES;

    char cuda_path[1024];
    snprintf(cuda_path, sizeof(cuda_path), "%s.cuda.jxs", output_prefix);
    FILE *cuda_file = NULL;
    FOPEN(cuda_file, cuda_path, "wb");
    if (cuda_file) {
        fwrite(cuda_bitstream, 1, offset, cuda_file);
        fclose(cuda_file);
    }
    else {
        printf("Can not open output file: %s!\n", cuda_path);
    }

    int bit_exact = (offset == out_buf.used_size) && (memcmp(cuda_bitstream, out_buf.buffer, offset) == 0);

    printf("Encoded %s (%ux%u, %u-bit %s, bpp=%u/%u)\n",
           input_file_name,
           enc.source_width,
           enc.source_height,
           enc.input_bit_depth,
           is_yuv ? "YUV422" : "RGB",
           enc.bpp_numerator,
           enc.bpp_denominator);
    printf("  CPU  (%s): %u bytes, %.3f ms\n", cpu_path, out_buf.used_size, cpu_ms);
    printf("  CUDA (%s): %u bytes, %.3f ms warm (2nd call, graph replay only) / %.3f ms cold (1st call, incl. graph capture)\n",
           cuda_path,
           offset,
           cuda_ms,
           cuda_cold_ms);
    printf("  bit-exact match: %s\n", bit_exact ? "YES" : "NO (mismatch!)");
    if (cuda_ms > 0.0) {
        printf("  speedup (CPU/CUDA): %.2fx\n", cpu_ms / cuda_ms);
    }

    free(cuda_bitstream);
    free(precinct_data);
    svt_cuda_frame_context_destroy(&ctx);
    free(precinct_budgets);
    free(out_buf.buffer);
    svt_jpeg_xs_image_buffer_free(in_buf);
    svt_jpeg_xs_encoder_close(&enc);
    return bit_exact ? 0 : 1;
}
