/*
* Copyright(c) 2024 Intel Corporation
* SPDX - License - Identifier: BSD - 2 - Clause - Patent
*/
#ifndef DwtCuda_cuh
#define DwtCuda_cuh

#include <stdint.h>
#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Runs NLT (linear input scaling) + full recursive 5/3 DWT (decom_h horizontal
 * levels, decom_v vertical levels) for one component entirely on the GPU.
 *
 * Produces a dense "pyramid" layout matching pi_band_t (x,y,width,height)
 * positions within a comp_width x comp_height array: band (x,y,w,h) occupies
 * rows [y,y+h) x cols [x,x+w) of out_pyramid_16bit (row-major, pitch=comp_width).
 * This is the classic in-place wavelet pyramid layout (LL/HL/LH/HH nested
 * quadrants), NOT the precinct-interleaved bitstream layout the CPU encoder
 * ultimately uses -- callers that need to compare against the CPU reference
 * must reassemble the CPU's per-precinct output into this same pyramid layout
 * (mechanical, using pi_t/pi_enc_t band offsets) before a memcmp.
 *
 * Output values are already 16-bit sign+magnitude (BITSTREAM_MASK_SIGN),
 * matching image_shift_c's post-DWT quantization -- i.e. bit-identical to
 * what the CPU reference stores in precinct_enc_t::coeff_buff_ptr_16bit.
 *
 * in_plane: host pointer to raw input samples for this component
 *           (uint8_t* if input_bit_depth<=8, else uint16_t*, LSB-aligned).
 * plane_stride: host row stride in samples (elements, not bytes).
 * comp_width/comp_height: this component's plane dimensions (post subsampling).
 * decom_h/decom_v: decomposition levels for this component
 *                  (pi_component_t::decom_h / decom_v).
 * input_bit_depth: bit depth of raw samples (e.g. 10). Must be 1..16.
 * hdr_Bw: picture_header_dynamic_t::hdr_Bw (nominal wavelet coefficient bit precision).
 * hdr_Fq: picture_header_dynamic_t::hdr_Fq (fractional bits / output quantization shift).
 * out_pyramid_16bit: host output buffer, size comp_width*comp_height uint16_t.
 *
 * Returns 0 on success, non-zero CUDA error code otherwise.
 */
int svt_cuda_dwt_component(const void* in_plane, uint32_t plane_stride, uint32_t comp_width, uint32_t comp_height,
                           uint32_t decom_h, uint32_t decom_v, uint8_t input_bit_depth, uint8_t hdr_Bw, uint8_t hdr_Fq,
                           uint16_t* out_pyramid_16bit);

/* Phase 4b-1: context-aware variant of svt_cuda_dwt_component() for use inside
 * svt_cuda_encode_frame(). Identical math/output layout, but takes
 * caller-owned persistent scratch buffers (from SvtCudaFrameContext, each
 * sized for at least comp_width*comp_height elements) instead of doing its
 * own cudaMalloc/cudaFree every call, writes the final 16-bit pyramid
 * directly into a persistent device buffer (d_out_pyramid16, no D2H/H2D
 * round-trip through the host), and enqueues everything on `stream` so it
 * composes with the rest of the frame's kernel sequence without an explicit
 * sync. NOT used by TestDwtCuda.cc -- svt_cuda_dwt_component() above is left
 * untouched for that.
 *
 * d_in_raw/d_cur/d_other/d_vert/d_pyramid32: scratch, each must be at least
 *   comp_width*comp_height elements (d_in_raw sized for the wider of
 *   uint8_t/uint16_t input, i.e. >= comp_width*comp_height*sizeof(uint16_t)
 *   bytes) -- these may be reused across sequential calls for different
 *   components of the same frame, since everything is ordered by `stream`.
 * d_out_pyramid16: persistent per-component destination, exactly
 *   comp_width*comp_height uint16_t.
 *
 * Returns 0 on success (kernels enqueued; errors surface via the caller's
 * eventual cudaStreamSynchronize/cudaGetLastError), non-zero CUDA error code
 * if an enqueue call itself fails synchronously.
 */
int svt_cuda_dwt_component_ctx(const void* in_plane, uint32_t plane_stride, uint32_t comp_width, uint32_t comp_height,
                               uint32_t decom_h, uint32_t decom_v, uint8_t input_bit_depth, uint8_t hdr_Bw, uint8_t hdr_Fq,
                               uint8_t* d_in_raw, int32_t* d_cur, int32_t* d_other, int32_t* d_vert, int32_t* d_pyramid32,
                               uint16_t* d_out_pyramid16, cudaStream_t stream);

/* Same as svt_cuda_dwt_component_ctx(), except d_in_raw is assumed already
 * filled (pitch == comp_width) by the caller -- e.g. via
 * svt_cuda_deinterleave_packed_rgb() below -- so no H2D copy is issued here.
 * See PortingStrategy.txt "channel-interleaved input" section. */
int svt_cuda_dwt_component_ctx_prefilled(uint32_t comp_width, uint32_t comp_height, uint32_t decom_h, uint32_t decom_v,
                                         uint8_t input_bit_depth, uint8_t hdr_Bw, uint8_t hdr_Fq, uint8_t* d_in_raw,
                                         int32_t* d_cur, int32_t* d_other, int32_t* d_vert, int32_t* d_pyramid32,
                                         uint16_t* d_out_pyramid16, cudaStream_t stream);

/* Deinterleaves one frame's worth of packed(AoS) RGB/444 samples
 * (d_packed, n = comp_width*comp_height pixels, 3 interleaved samples/pixel)
 * into 3 separate planar(SoA) device buffers in a single pass. d_out0/1/2
 * must each hold n samples (uint8_t or uint16_t per input_bit_depth,
 * matching d_in_raw's element type elsewhere in this file). Enqueued on
 * `stream`; returns 0 on success (kernel enqueued), non-zero CUDA error code
 * if the enqueue itself fails synchronously. */
int svt_cuda_deinterleave_packed_rgb(const void* d_packed, uint8_t* d_out0, uint8_t* d_out1, uint8_t* d_out2, uint32_t n,
                                     uint8_t input_bit_depth, cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif // DwtCuda_cuh
