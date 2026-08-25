/*
* Copyright(c) 2024 Intel Corporation
* SPDX - License - Identifier: BSD - 2 - Clause - Patent
*/
#ifndef EncodeFrameCuda_cuh
#define EncodeFrameCuda_cuh

#include <stdint.h>
#include "FrameContextCuda.cuh"
#include "Pi.h"
#include "PiEnc.h"

/* Phase 4a: end-to-end single-frame CUDA encode, standalone from the CPU's
 * threaded pipeline (EncHandle.c) per the project's "separate library"
 * integration policy -- see PortingStrategy.txt section 7/8 and the Phase 4
 * plan. Fixed scope inherited from Phase 2/3: VPRED disabled, Signs
 * handling = OFF, hdr_Rl = 0, Deadzone quantization, Significance enabled.
 * Additionally (Phase 4a scope constraint): the image height must be evenly
 * divisible by pi->precinct_height, i.e. every precinct row has the same
 * (NORMAL) band geometry -- true for the primary 4K/decom_v=2 target
 * (2160 / 4 == 540 exactly) but not handled generically for arbitrary
 * resolutions yet (see plan file Phase 4a notes / final report).
 */

#ifdef __cplusplus
extern "C" {
#endif

/* Convenience constructor: builds a SvtCudaFrameContext's band-geometry table
 * directly from an already-computed pi_t/pi_enc_t (caller builds these via
 * pi_compute()/pi_compute_encoder(), same as tests/UnitTests/Test{RcQuant,Pack}Cuda.cc).
 * pack_out_capacity_bytes: must be >= the largest bitstream the caller will
 * ever request from svt_cuda_encode_frame() for this geometry (e.g. hdr_Lcod).
 * Returns 0 on success, negative CUDA error code on failure, 1 if pi's
 * geometry doesn't satisfy the "evenly-divisible precinct rows" constraint
 * above.
 */
int svt_cuda_frame_context_create_from_pi(SvtCudaFrameContext* ctx, const pi_t* pi, const pi_enc_t* pi_enc,
                                          uint32_t pack_out_capacity_bytes);

/* Encodes one frame end-to-end on the GPU using a previously-created context
 * (which fixes width/height/decom/comps_num/band geometry -- must match the
 * pi/pi_enc the context was created from).
 *
 * in_planes[c] / in_stride[c]: host input planes, one per component (matches
 *   svt_jpeg_xs_image_buffer_t::data_yuv/stride).
 * input_bit_depth, hdr_Bw, hdr_Fq: matches NltCuda/DwtCuda's parameters.
 * quant_type: 0 = QUANT_TYPE_DEADZONE, 1 = QUANT_TYPE_UNIFORM.
 * use_short_header: matches pi_t::use_short_header.
 * precinct_budget_bytes: [precincts_num], host array, the per-precinct byte
 *   budget (headers+data) the caller has already computed (e.g. replicating
 *   RC_CBR_PER_PRECINCT_MOVE_PADDING's slice/precinct distribution formula,
 *   as tests/UnitTests/TestEncodeFrameCuda.cc does against the real encoder's
 *   internals) -- matches Phase 2's established design where budget
 *   distribution is a host-side concern separate from the GPU RC kernel.
 * precinct_header_len_bytes / bands_num_exists / packets_exist_num: matches
 *   precinct_info_t fields (frame-constant in the NORMAL-only scope above).
 *
 * out_buffer: host buffer, at least ctx->pack_out_capacity_bytes bytes.
 * out_used_bytes: total bitstream bytes written (sum of all precincts'
 *   pack_total_bytes; this function does NOT write frame/slice headers --
 *   caller concatenates those separately, matching the "separate library"
 *   scope: only the precinct data path is ported).
 *
 * Returns 0 on success, 1 if any precinct's RC search fails (budget too
 * small), negative CUDA error code on CUDA failure.
 */
int svt_cuda_encode_frame(SvtCudaFrameContext* ctx, const void* const in_planes[], const uint32_t in_stride[],
                          uint32_t decom_h, uint32_t decom_v, uint8_t input_bit_depth, uint8_t hdr_Bw, uint8_t hdr_Fq,
                          uint8_t quant_type, uint8_t use_short_header, uint8_t coding_significance, uint32_t max_quantization,
                          uint32_t max_refinement, const uint32_t* precinct_budget_bytes, uint32_t bands_num_exists,
                          uint32_t packets_exist_num, uint8_t* out_buffer, uint32_t* out_used_bytes);

#ifdef __cplusplus
}
#endif

#endif // EncodeFrameCuda_cuh
