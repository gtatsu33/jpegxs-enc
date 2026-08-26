/*
* Copyright(c) 2024 Intel Corporation
* SPDX - License - Identifier: BSD - 2 - Clause - Patent
*/
#include "PixelIo.h"

#include <stdio.h>
#include <string.h>

/* Note: STBI_ONLY_PNM is deliberately NOT defined here. It also defines
 * STBI_NO_PNG/STBI_NO_PSD internally, which excludes stbi__convert_format16()
 * -- but stbi__pnm_load()'s 16-bit branch still calls it (an upstream
 * inconsistency in this ONLY_* combination), causing an unresolved external
 * symbol at link time even though that branch is never reached for a
 * same-channel-count P6 load. Building the full decoder set avoids it. */
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

/* Reads just the PPM header's maxval field. stb_image's public API only
 * exposes whether a sample decoded as 8- or 16-bit, not the original maxval
 * value, and the encoder needs the true bit depth (e.g. 10, 12), not just
 * "8 or 16". This tokenizes the three decimal header fields per the Netpbm
 * spec (whitespace/'#'-comment skipping -- the same rules stb_image itself
 * applies internally). The actual pixel decode (the part with real
 * correctness risk -- see the endianness patch note in stb_image.h) is left
 * entirely to stb_image.
 */
static int read_ppm_header(const char* path, uint32_t* out_width, uint32_t* out_height, uint32_t* out_maxval) {
    FILE* f = NULL;
#ifdef _WIN32
    fopen_s(&f, path, "rb");
#else
    f = fopen(path, "rb");
#endif
    if (!f) {
        fprintf(stderr, "PixelIo: cannot open file: %s\n", path);
        return -1;
    }

    int c = fgetc(f);
    int t = fgetc(f);
    if (c != 'P' || t != '6') {
        fprintf(stderr, "PixelIo: unsupported format (only binary PPM \"P6\" RGB is supported): %s\n", path);
        fclose(f);
        return -1;
    }

    uint32_t values[3];
    for (int field = 0; field < 3; field++) {
        c = fgetc(f);
        for (;;) {
            while (c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\v' || c == '\f')
                c = fgetc(f);
            if (c != '#')
                break;
            while (c != '\n' && c != '\r' && c != EOF)
                c = fgetc(f);
        }
        if (c < '0' || c > '9') {
            fprintf(stderr, "PixelIo: malformed PPM header: %s\n", path);
            fclose(f);
            return -1;
        }
        uint32_t value = 0;
        while (c >= '0' && c <= '9') {
            value = value * 10 + (uint32_t)(c - '0');
            c = fgetc(f);
        }
        values[field] = value;
    }
    fclose(f);

    *out_width = values[0];
    *out_height = values[1];
    *out_maxval = values[2];
    return 0;
}

static uint32_t bit_length(uint32_t v) {
    uint32_t bits = 0;
    while (v) {
        bits++;
        v >>= 1;
    }
    return bits;
}

int pixel_image_load_ppm(const char* path, PixelImage_t* out_image) {
    memset(out_image, 0, sizeof(*out_image));

    uint32_t hdr_width = 0, hdr_height = 0, maxval = 0;
    if (read_ppm_header(path, &hdr_width, &hdr_height, &maxval) != 0) {
        return -1;
    }
    if (maxval == 0 || maxval > 65535) {
        fprintf(stderr, "PixelIo: invalid maxval %u in %s\n", maxval, path);
        return -1;
    }

    uint32_t bit_depth = bit_length(maxval);
    if (bit_depth < 8) {
        bit_depth = 8; /* a maxval < 255 sample still occupies a full byte on disk */
    }
    if (bit_depth > 14) {
        fprintf(stderr,
                "PixelIo: %s has maxval=%u (bit_depth=%u), outside the encoder's supported 8-14 bit range\n",
                path,
                maxval,
                bit_depth);
        return -1;
    }

    int x = 0, y = 0, comp = 0;
    void* data;
    uint32_t bytes_per_sample;
    if (maxval > 255) {
        data = stbi_load_16(path, &x, &y, &comp, 3);
        bytes_per_sample = 2;
    }
    else {
        data = stbi_load(path, &x, &y, &comp, 3);
        bytes_per_sample = 1;
    }
    if (!data) {
        fprintf(stderr, "PixelIo: stb_image failed to decode %s: %s\n", path, stbi_failure_reason());
        return -1;
    }
    if ((uint32_t)x != hdr_width || (uint32_t)y != hdr_height) {
        fprintf(stderr, "PixelIo: internal dimension mismatch decoding %s\n", path);
        stbi_image_free(data);
        return -1;
    }

    out_image->width = (uint32_t)x;
    out_image->height = (uint32_t)y;
    out_image->channels_num = 3;
    out_image->bit_depth = bit_depth;
    out_image->bytes_per_sample = bytes_per_sample;
    out_image->layout = PIXEL_LAYOUT_RGB_INTERLEAVED;
    out_image->data = (uint8_t*)data;
    return 0;
}

int pixel_image_load_yuv422p8(const char* path, uint32_t width, uint32_t height, PixelImage_t* out_image) {
    memset(out_image, 0, sizeof(*out_image));

    if (width == 0 || height == 0 || (width % 2) != 0) {
        fprintf(stderr, "PixelIo: invalid width/height for yuv422p8 (%ux%u, width must be even)\n", width, height);
        return -1;
    }

    FILE* f = NULL;
#ifdef _WIN32
    fopen_s(&f, path, "rb");
#else
    f = fopen(path, "rb");
#endif
    if (!f) {
        fprintf(stderr, "PixelIo: cannot open file: %s\n", path);
        return -1;
    }

    const size_t y_bytes = (size_t)width * height;
    const size_t c_bytes = (size_t)(width / 2) * height;
    const size_t total_bytes = y_bytes + 2 * c_bytes;

    if (fseek(f, 0, SEEK_END) != 0) {
        fprintf(stderr, "PixelIo: seek failed on %s\n", path);
        fclose(f);
        return -1;
    }
    long file_size = ftell(f);
    if (file_size < 0 || (size_t)file_size != total_bytes) {
        fprintf(stderr,
                "PixelIo: %s size %ld doesn't match %ux%u yuv422p8 (expected %zu bytes)\n",
                path,
                file_size,
                width,
                height,
                total_bytes);
        fclose(f);
        return -1;
    }
    rewind(f);

    uint8_t* data = (uint8_t*)malloc(total_bytes);
    if (!data) {
        fprintf(stderr, "PixelIo: out of memory loading %s\n", path);
        fclose(f);
        return -1;
    }
    if (fread(data, 1, total_bytes, f) != total_bytes) {
        fprintf(stderr, "PixelIo: short read on %s\n", path);
        free(data);
        fclose(f);
        return -1;
    }
    fclose(f);

    out_image->width = width;
    out_image->height = height;
    out_image->channels_num = 3;
    out_image->bit_depth = 8;
    out_image->bytes_per_sample = 1;
    out_image->layout = PIXEL_LAYOUT_YUV422_PLANAR_8BIT;
    out_image->data = data;
    return 0;
}

void pixel_image_free(PixelImage_t* image) {
    if (image && image->data) {
        stbi_image_free(image->data);
        image->data = NULL;
    }
}
