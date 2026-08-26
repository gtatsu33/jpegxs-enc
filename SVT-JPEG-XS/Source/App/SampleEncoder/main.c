/*
* Copyright(c) 2024 Intel Corporation
* SPDX - License - Identifier: BSD - 2 - Clause - Patent
*/

#include <SvtJpegxsEnc.h>
#include <SvtJpegxsImageBufferTools.h>
#include "PixelIo.h"
#include "UtilityApp.h"
#include <stdio.h>
#include <stdlib.h>

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

int32_t main(int32_t argc, char *argv[]) {
    if (argc < 3) {
        printf("Usage: %s <input.ppm> <output.jxs> [bpp]\n", argv[0]);
        printf("  input.ppm  binary PPM (Netpbm \"P6\", interleaved RGB), maxval 1-65535\n");
        printf("  bpp        optional target bits-per-pixel, integer or decimal (e.g. 0.5, 3, 3.75). Default: 3\n");
        return -1;
    }
    const char *input_file_name = argv[1];
    const char *output_file_name = argv[2];
    const char *bpp_arg = argc > 3 ? argv[3] : "3";

    PixelImage_t image;
    if (pixel_image_load_ppm(input_file_name, &image) != 0) {
        return -1;
    }

    svt_jpeg_xs_encoder_api_t enc;
    SvtJxsErrorType_t err = svt_jpeg_xs_encoder_load_default_parameters(SVT_JPEGXS_API_VER_MAJOR, SVT_JPEGXS_API_VER_MINOR, &enc);
    if (err != SvtJxsErrorNone) {
        pixel_image_free(&image);
        return err;
    }

    enc.source_width = image.width;
    enc.source_height = image.height;
    enc.input_bit_depth = (uint8_t)image.bit_depth;
    enc.colour_format = COLOUR_FORMAT_PLANAR_YUV444_OR_RGB;
    parse_bpp_arg(bpp_arg, &enc.bpp_numerator, &enc.bpp_denominator);

    err = svt_jpeg_xs_encoder_init(SVT_JPEGXS_API_VER_MAJOR, SVT_JPEGXS_API_VER_MINOR, &enc);
    if (err != SvtJxsErrorNone) {
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
    deinterleave_rgb(&image, in_buf);
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

    err = svt_jpeg_xs_encoder_send_picture(&enc, &enc_input, 1 /*blocking*/);
    if (err != SvtJxsErrorNone) {
        free(out_buf.buffer);
        svt_jpeg_xs_image_buffer_free(in_buf);
        svt_jpeg_xs_encoder_close(&enc);
        return err;
    }

    svt_jpeg_xs_frame_t enc_output;
    err = svt_jpeg_xs_encoder_get_packet(&enc, &enc_output, 1 /*blocking*/);
    if (err != SvtJxsErrorNone) {
        free(out_buf.buffer);
        svt_jpeg_xs_image_buffer_free(in_buf);
        svt_jpeg_xs_encoder_close(&enc);
        return err;
    }

    FILE *output_file = NULL;
    FOPEN(output_file, output_file_name, "wb");
    if (!output_file) {
        printf("Can not open output file: %s!\n", output_file_name);
        free(out_buf.buffer);
        svt_jpeg_xs_image_buffer_free(in_buf);
        svt_jpeg_xs_encoder_close(&enc);
        return -1;
    }
    fwrite(enc_output.bitstream.buffer, 1, enc_output.bitstream.used_size, output_file);
    fclose(output_file);
    printf("Encoded %s (%ux%u, %u-bit RGB) -> %s: %u bytes (bpp=%u/%u)\n",
           input_file_name,
           enc.source_width,
           enc.source_height,
           enc.input_bit_depth,
           output_file_name,
           enc_output.bitstream.used_size,
           enc.bpp_numerator,
           enc.bpp_denominator);

    svt_jpeg_xs_encoder_close(&enc);
    free(out_buf.buffer);
    svt_jpeg_xs_image_buffer_free(in_buf);
    return 0;
}
