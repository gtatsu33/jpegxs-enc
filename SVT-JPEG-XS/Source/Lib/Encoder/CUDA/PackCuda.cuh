/*
* Copyright(c) 2024 Intel Corporation
* SPDX - License - Identifier: BSD - 2 - Clause - Patent
*/
#ifndef PackCuda_cuh
#define PackCuda_cuh

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Phase 3 scope: VPRED disabled, Signs handling = OFF, hdr_Rl (RAW packet
 * mode) = 0 -- same default-configuration scope as Phase 2 (RcQuantCuda).
 * Under this scope only 2 of the 4 CodingMethodBand values ever occur
 * (METHOD_ZERO_SIGNIFICANCE_DISABLE/ENABLE) and the sign sub-packet never
 * exists, so this is a complete, bit-exact port of that path.
 *
 * One CUDA call packs ONE precinct with a single thread (see
 * PortingStrategy.txt / implementation plan: band-level bit-scatter
 * parallelism was considered and intentionally not used here -- true
 * parallelism is across precincts, deferred to Phase 4's batching). Inputs
 * are exactly Phase 2's (RcQuantCuda) outputs plus the gcli/significance/
 * quantized-coeff data Phase 1/2 already produced.
 */

/* Mirrors precinct_band_info_t + the subset of svt_cuda_rc_band_info_t /
 * RC outputs needed to pack one band, flattened across all (component,band)
 * pairs of one precinct in pi->global_band_info[] order (indices
 * [0, bands_num_all), including placeholder slots -- see `exists`). */
typedef struct svt_cuda_pack_band_info {
    uint8_t exists;               /* 0 if this global band slot has no (comp,band) mapping
                                    * (pi->global_band_info[idx].band_id == BAND_NOT_EXIST) */
    uint32_t width;                /* coefficient width */
    uint32_t gcli_width;           /* ceil(width / GROUP_SIZE) */
    uint32_t significance_width;   /* ceil(gcli_width / SIGNIFICANCE_GROUP_SIZE) */
    uint32_t height_lines;         /* lines of this band within this precinct */
    uint8_t pack_method;           /* RcQuantCuda's out_pack_method: 0=DISABLE, 1=ENABLE */
    uint8_t gtli;                  /* RcQuantCuda's out_gtli */
    uint32_t gcli_offset;          /* offset into the flat gcli_data array */
    uint32_t significance_offset;  /* offset into the flat significance_data array */
    uint32_t coeff_offset;         /* offset into the flat coeff_data array (already quantized) */
} svt_cuda_pack_band_info_t;

/* Mirrors pi_packet_inclusion_t (same shape as svt_cuda_rc_packet_t in
 * RcQuantCuda.cuh, redeclared here to keep this module self-contained). */
typedef struct svt_cuda_pack_packet {
    uint32_t band_start;
    uint32_t band_stop;
    uint32_t line_idx;
} svt_cuda_pack_packet_t;

/* Packs one precinct's bitstream, matching pack_precinct() (PackPrecinct.c)
 * bit-exactly for the VPRED-disabled / Signs=OFF / hdr_Rl=0 path.
 *
 * bands_num_all: number of global band slots (pi_t::bands_num_all), i.e.
 *   the length of band_info[].
 * bands_num_exists: pi_t::bands_num_exists (precinct header size calculation).
 * gcli_data / significance_data / coeff_data: flat host arrays as described
 *   for svt_cuda_rc_quant_precinct() (RcQuantCuda.cuh) -- coeff_data must
 *   already be quantized (Phase 2 output).
 * use_short_header: pi_t::use_short_header (packet header format selector).
 * pack_quantization / pack_refinement / pack_total_bytes / pack_padding_bytes:
 *   RcQuantCuda's out_quantization/out_refinement/out_total_bytes/out_padding_bytes.
 * packet_size_data_bytes / packet_size_gcli_bytes / packet_size_significance_bytes:
 *   [packets_num] each, matching precinct_enc_t::packet_size_{data,gcli,significance}_bytes
 *   (must be computed by the caller from RcQuantCuda's out_pack_{gcli,sig,data}_bits by
 *   summing per-band-per-line bits within each packet and byte-aligning -- see
 *   precinct_get_budget_bytes()'s packet loop, RateControl.c).
 * out_buffer: host buffer, at least pack_total_bytes long. Zero-initialized
 *   internally before packing.
 * out_used_bytes: actual bytes written (should equal pack_total_bytes on success).
 *
 * Returns 0 on success, 1 on a length-mismatch consistency error (mirrors
 * pack_precinct()'s SVT_ERROR/return-error checks), negative CUDA error
 * code on CUDA failure.
 */
int svt_cuda_pack_precinct(uint32_t bands_num_all, const svt_cuda_pack_band_info_t* band_info, uint32_t bands_num_exists,
                           const uint8_t* gcli_data, const uint8_t* significance_data, const uint16_t* coeff_data,
                           uint32_t packets_num, const svt_cuda_pack_packet_t* packets, uint8_t use_short_header,
                           uint8_t pack_quantization, uint8_t pack_refinement, uint32_t pack_total_bytes,
                           uint32_t pack_padding_bytes, const uint32_t* packet_size_data_bytes,
                           const uint32_t* packet_size_gcli_bytes, const uint32_t* packet_size_significance_bytes,
                           uint8_t* out_buffer, uint32_t* out_used_bytes);

#ifdef __cplusplus
}
#endif

#endif // PackCuda_cuh
