/*
* Copyright(c) 2024 Intel Corporation
* SPDX - License - Identifier: BSD - 2 - Clause - Patent
*/
#ifndef RcQuantCuda_cuh
#define RcQuantCuda_cuh

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Phase 2 scope (see PortingStrategy.txt section 8 / implementation plan):
 * VPRED disabled, Signs handling = OFF, hdr_Rl (RAW packet mode) = 0 --
 * i.e. the encoder's default configuration. Under these conditions
 * rate_control_precinct()'s quantization search always takes its "simple"
 * fast path (no VPRED/full-sign refinement pass) and refinement search has
 * no such split at all, so this is a complete, bit-exact port of that path
 * -- not an approximation of it.
 *
 * One CUDA call processes ONE precinct (matches Phase 1's per-component
 * granularity); batching multiple precincts across CUDA blocks is a Phase 4
 * concern. gcli/significance/coeff data for all bands of the precinct are
 * passed as single flat host arrays with per-band offsets (mirrors
 * pi_enc_t's own offset scheme), rather than arrays of pointers.
 */

/* Mirrors pi_band_t + pi_enc_band_t fields needed for RC+Quant, flattened
 * across all (component, band) pairs of one precinct into a single array
 * indexed [0, bands_num). */
typedef struct svt_cuda_rc_band_info {
    uint32_t width;              /* coefficient width (pi_t::components[c].bands[b].width) */
    uint32_t gcli_width;         /* ceil(width / GROUP_SIZE) */
    uint32_t significance_width; /* ceil(gcli_width / SIGNIFICANCE_GROUP_SIZE) */
    uint32_t height_lines;       /* lines of this band within this precinct (1..MAX_BAND_LINES=4) */
    uint8_t gain;                /* pi_t::components[c].bands[b].gain (weight table, precinct-independent) */
    uint8_t priority;            /* pi_t::components[c].bands[b].priority */
    uint32_t gcli_offset;        /* offset into the flat gcli_data array, in bytes/elements */
    uint32_t significance_offset;
    uint32_t coeff_offset;
} svt_cuda_rc_band_info_t;

/* Mirrors pi_packet_inclusion_t: one packet = a contiguous band_id range
 * [band_start, band_stop) of the flat band_info array, all read at the same
 * line_idx. */
typedef struct svt_cuda_rc_packet {
    uint32_t band_start;
    uint32_t band_stop;
    uint32_t line_idx;
} svt_cuda_rc_packet_t;

/* Runs rate_control_precinct()'s quantization+refinement binary search
 * (VPRED-disabled path) followed by precinct_quantization() (in-place),
 * for one precinct.
 *
 * bands_num: number of (component,band) entries in band_info[].
 * gcli_data / significance_data: flat host arrays, one byte per group,
 *   band b's data starts at band_info[b].gcli_offset / significance_offset,
 *   height_lines * gcli_width (resp. significance_width) bytes long.
 * coeff_data: flat host array, uint16_t sign+magnitude coefficients,
 *   band b's data starts at band_info[b].coeff_offset, height_lines*width
 *   elements long. Quantized in place on return (matches precinct_quantization()).
 * quant_type: 0 = QUANT_TYPE_DEADZONE, 1 = QUANT_TYPE_UNIFORM.
 * coding_significance: matches svt_jpeg_xs_encoder_common_t::coding_significance.
 * use_short_header: matches pi_t::use_short_header (selects packet header size).
 * bands_num_exists / packets_exist_num: matches pi_t::bands_num_exists /
 *   precinct_info_t::packets_exist_num (precinct header size calculation).
 * max_quantization / max_refinement: pi_enc_t::max_quantization / max_refinement.
 * budget_bytes: precinct's total byte budget (headers + data), matches
 *   rate_control_precinct()'s budget_bytes parameter -- precinct-to-precinct
 *   budget distribution (including "move padding" carry-over) is computed on
 *   the host by the caller, not by this function.
 *
 * out_gtli: [bands_num], the chosen Greatest Trimmed Line Index per band.
 * out_quantization / out_refinement / out_data_bytes / out_padding_bytes /
 *   out_total_bytes: match precinct_enc_t::pack_quantization/pack_refinement/
 *   (budget_to_data_bytes - data_bytes)/pack_padding_bytes/pack_total_bytes.
 * out_pack_method: [bands_num] (may be NULL to skip), 0 = METHOD_ZERO_SIGNIFICANCE_DISABLE,
 *   1 = METHOD_ZERO_SIGNIFICANCE_ENABLE (matches band_data_enc_cache::pack_method
 *   for the VPRED-disabled path this function implements -- Phase 3 needs this to
 *   pick the matching Pack subpacket writer per band).
 * out_pack_gcli_bits / out_pack_sig_bits / out_pack_data_bits: [bands_num*4] (may be
 *   NULL to skip), the final per-(band,line) bit counts used to compute *out_data_bytes
 *   (matches band_data_enc_cache::lines[line].pack_size_gcli_bits/pack_size_significance_bits/
 *   pack_size_data_bits) -- Phase 3 uses these directly as its "length pass" input,
 *   since PackPrecinct.c's own runtime asserts confirm these exactly match what it writes.
 *
 * Returns 0 on success, 1 if budget_bytes <= headers_bytes or no valid
 * (quantization,refinement) was found (matches rate_control_precinct()'s
 * error returns), negative CUDA error code on CUDA failure.
 */
int svt_cuda_rc_quant_precinct(uint32_t bands_num, const svt_cuda_rc_band_info_t* band_info, const uint8_t* gcli_data,
                               const uint8_t* significance_data, uint16_t* coeff_data, uint32_t packets_num,
                               const svt_cuda_rc_packet_t* packets, uint32_t bands_num_exists, uint32_t packets_exist_num,
                               uint8_t use_short_header, uint8_t coding_significance, uint8_t quant_type,
                               uint32_t max_quantization, uint32_t max_refinement, uint32_t budget_bytes, uint8_t* out_gtli,
                               uint8_t* out_quantization, uint8_t* out_refinement, uint32_t* out_data_bytes,
                               uint32_t* out_padding_bytes, uint32_t* out_total_bytes, uint8_t* out_pack_method,
                               uint32_t* out_pack_gcli_bits, uint32_t* out_pack_sig_bits, uint32_t* out_pack_data_bits);

#ifdef __cplusplus
}
#endif

#endif // RcQuantCuda_cuh
