/*
* Copyright(c) 2024 Intel Corporation
* SPDX - License - Identifier: BSD - 2 - Clause - Patent
*/
#ifndef NltCuda_cuh
#define NltCuda_cuh

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Standalone NLT (linear input scaling) kernel wrapper, matching
 * linear_input_scaling_line_8bit_c/_16bit_c (NltEnc.c), LSB-aligned
 * (hdr_input_msb_aligned==0, hdr_Tnlt==0) path. Exposed separately from
 * svt_cuda_dwt_component() (DwtCuda.cu, which has its own internal copy)
 * so Phase 1's NLT module can be benchmarked/validated on its own.
 *
 * in_plane: host pointer, uint8_t* if input_bit_depth<=8 else uint16_t*.
 * plane_stride: host row stride in samples.
 * width/height: plane dimensions.
 * input_bit_depth: 1..16.
 * hdr_Bw: picture_header_dynamic_t::hdr_Bw.
 * out_scaled: host output buffer, width*height int32_t, row-major, pitch=width.
 * Returns 0 on success, non-zero CUDA error code otherwise.
 */
int svt_cuda_nlt_scale_component(const void* in_plane, uint32_t plane_stride, uint32_t width, uint32_t height,
                                 uint8_t input_bit_depth, uint8_t hdr_Bw, int32_t* out_scaled);

#ifdef __cplusplus
}
#endif

#endif // NltCuda_cuh
