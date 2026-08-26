/*
* Copyright(c) 2024 Intel Corporation
* SPDX - License - Identifier: BSD - 2 - Clause - Patent
*/
#ifndef PixelIo_h
#define PixelIo_h

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Generic pixel-buffer description, deliberately not PPM-specific: future
 * loaders (YUV, Bayer mosaics, other bit depths) are expected to fill the
 * same struct so callers (SampleEncoder/SampleEncoderCuda) don't need to
 * change when a new pixel_image_load_*() is added. Only pixel_image_load_ppm()
 * (P6, interleaved RGB) is implemented today -- see PortingStrategy.txt
 * section 12.
 */
typedef enum PixelLayout { PIXEL_LAYOUT_RGB_INTERLEAVED = 0 } PixelLayout_t;

typedef struct PixelImage {
    uint32_t width;
    uint32_t height;
    uint32_t channels_num;
    uint32_t bit_depth;        /* significant bits per sample, derived from the file's maxval */
    uint32_t bytes_per_sample; /* 1 or 2, native endianness */
    PixelLayout_t layout;
    uint8_t* data; /* channels_num * bytes_per_sample bytes per pixel, row-major, no row padding */
} PixelImage_t;

/* Loads a binary PPM (Netpbm "P6", interleaved RGB) file via stb_image
 * (third_party/stb_image/stb_image.h). maxval 1-65535 per the Netpbm spec;
 * bit_depth is derived from maxval's bit length (e.g. maxval=1023 ->
 * bit_depth=10), matching real capture data already in testdata/. Returns 0
 * on success, non-zero on failure (message printed to stderr) -- including
 * when the derived bit_depth falls outside the encoder's supported 8-14
 * range.
 */
int pixel_image_load_ppm(const char* path, PixelImage_t* out_image);

void pixel_image_free(PixelImage_t* image);

#ifdef __cplusplus
}
#endif

#endif // PixelIo_h
