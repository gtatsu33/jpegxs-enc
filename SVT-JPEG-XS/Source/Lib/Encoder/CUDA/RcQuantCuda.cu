/*
* Copyright(c) 2024 Intel Corporation
* SPDX - License - Identifier: BSD - 2 - Clause - Patent
*/
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>
#include <cstring>
#include "RcQuantCuda.cuh"

#define RCQ_GROUP_SIZE 4
#define RCQ_SIGNIFICANCE_GROUP_SIZE 8
#define RCQ_TRUNCATION_MAX 15
#define RCQ_SIGN_MASK ((uint16_t)1 << 15) /* BITSTREAM_MASK_SIGN */
#define RCQ_PRECINCT_HEADER_SIZE_BYTES 5
#define RCQ_PACKET_HEADER_LONG_SIZE_BYTES 7
#define RCQ_PACKET_HEADER_SHORT_SIZE_BYTES 5

static inline uint32_t rcq_align8(uint32_t bits) {
    return (bits + 7) & ~7u;
}
static inline uint32_t rcq_bits_to_bytes(uint32_t bits) {
    return (bits + 7) >> 3;
}

/* Per (band,line) precomputed cost tables, matches rc_cache_band_line_t's
 * derived arrays (gc_lookup_table_size_data_no_sign_handling, the
 * post-summarize cumulative gc_lookup_table, and significance_max_lookup_table)
 * for the Signs-handling-OFF path used by rate_control_lut_fill /
 * rate_control_lut_summarize_non_significance_gc_and_data_code_sum /
 * rate_control_lut_significance_elements_less_equal_gtli. */
struct RcqBandLineLut {
    uint32_t size_data_no_sign[RCQ_TRUNCATION_MAX + 1];
    uint32_t cum_count[RCQ_TRUNCATION_MAX + 1];
    uint32_t sig_cum[RCQ_TRUNCATION_MAX + 1];
    uint32_t leftover_max; /* significance value of the trailing partial group, RCQ_TRUNCATION_MAX+1 if none */
    uint32_t leftover_extra;
};

/* One thread per (band, line) pair. Mirrors rate_control_lut_fill(). */
__global__ void k_rcq_build_lut(const svt_cuda_rc_band_info_t* band_info, uint32_t bands_num, const uint8_t* gcli_data,
                                const uint8_t* significance_data, uint8_t coding_significance, RcqBandLineLut* out_lut) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t b = idx / 4; /* RCQ max band lines */
    uint32_t line = idx % 4;
    if (b >= bands_num)
        return;
    const svt_cuda_rc_band_info_t* bi = &band_info[b];
    if (line >= bi->height_lines)
        return;

    RcqBandLineLut* lut = &out_lut[idx];

    uint32_t raw_hist[RCQ_TRUNCATION_MAX + 1];
    for (int i = 0; i <= RCQ_TRUNCATION_MAX; i++) {
        raw_hist[i] = 0;
    }
    const uint8_t* gcli_ptr = gcli_data + bi->gcli_offset + (size_t)line * bi->gcli_width;
    for (uint32_t i = 0; i < bi->gcli_width; i++) {
        raw_hist[gcli_ptr[i]]++;
    }

    lut->size_data_no_sign[RCQ_TRUNCATION_MAX] = 0;
    uint32_t sum_back = 0;
    for (int gtli_index = RCQ_TRUNCATION_MAX - 1; gtli_index >= 0; gtli_index--) {
        sum_back += raw_hist[gtli_index + 1];
        lut->size_data_no_sign[gtli_index] = lut->size_data_no_sign[gtli_index + 1] + sum_back + raw_hist[gtli_index + 1];
    }

    uint32_t cum = 0;
    for (int v = 0; v <= RCQ_TRUNCATION_MAX; v++) {
        cum += raw_hist[v];
        lut->cum_count[v] = cum;
    }

    lut->leftover_max = RCQ_TRUNCATION_MAX + 1; /* sentinel: no leftover */
    lut->leftover_extra = 0;
    for (int v = 0; v <= RCQ_TRUNCATION_MAX; v++) {
        lut->sig_cum[v] = 0;
    }
    if (coding_significance) {
        uint32_t full_group = bi->gcli_width / RCQ_SIGNIFICANCE_GROUP_SIZE;
        const uint8_t* sig_ptr = significance_data + bi->significance_offset + (size_t)line * bi->significance_width;
        uint32_t sig_hist[RCQ_TRUNCATION_MAX + 1];
        for (int i = 0; i <= RCQ_TRUNCATION_MAX; i++) {
            sig_hist[i] = 0;
        }
        for (uint32_t i = 0; i < full_group; i++) {
            sig_hist[sig_ptr[i]]++;
        }
        uint32_t scum = 0;
        for (int v = 0; v <= RCQ_TRUNCATION_MAX; v++) {
            scum += sig_hist[v];
            lut->sig_cum[v] = scum;
        }
        if (full_group < bi->significance_width) {
            lut->leftover_max = sig_ptr[full_group];
            lut->leftover_extra = bi->gcli_width - full_group * RCQ_SIGNIFICANCE_GROUP_SIZE;
        }
    }
}

/* Matches compute_truncation() (EncDec.c). */
static inline uint8_t rcq_compute_truncation(uint8_t gain, uint8_t priority, uint32_t quantization, uint32_t refinement) {
    uint8_t pump_up = (priority < refinement) ? 1 : 0;
    if (quantization < (uint32_t)(gain + pump_up)) {
        return 0;
    }
    uint32_t truncation = quantization - gain - pump_up;
    if (truncation > RCQ_TRUNCATION_MAX) {
        truncation = RCQ_TRUNCATION_MAX;
    }
    return (uint8_t)truncation;
}

/* Matches rate_control_calculate_band_best_method() (Signs=OFF, VPRED
 * disabled path) + precinct_get_budget_bytes()'s packet aggregation.
 * Returns total precinct data budget in bytes for the given per-band gtli[]. */
static uint32_t rcq_compute_budget_bytes(uint32_t bands_num, const svt_cuda_rc_band_info_t* band_info,
                                         const std::vector<RcqBandLineLut>& lut, const uint8_t* gtli, uint8_t coding_significance,
                                         uint32_t packets_num, const svt_cuda_rc_packet_t* packets,
                                         std::vector<uint32_t>& pack_gcli_bits, std::vector<uint32_t>& pack_sig_bits,
                                         std::vector<uint32_t>& pack_data_bits, std::vector<uint8_t>* pack_method_out = NULL) {
    for (uint32_t b = 0; b < bands_num; b++) {
        const svt_cuda_rc_band_info_t& bi = band_info[b];
        uint8_t g = gtli[b];
        uint32_t sum_data_non_sig_gc[4];
        uint32_t best_budget = 0;
        for (uint32_t line = 0; line < bi.height_lines; line++) {
            const RcqBandLineLut& l = lut[(size_t)b * 4 + line];
            uint32_t sum_data = l.size_data_no_sign[g];
            uint32_t cum = l.cum_count[g];
            sum_data_non_sig_gc[line] = sum_data + cum;
            pack_data_bits[(size_t)b * 4 + line] = sum_data * RCQ_GROUP_SIZE;
            pack_gcli_bits[(size_t)b * 4 + line] = sum_data_non_sig_gc[line];
            pack_sig_bits[(size_t)b * 4 + line] = 0;
            best_budget += sum_data_non_sig_gc[line];
        }
        if (pack_method_out) {
            (*pack_method_out)[b] = 0; /* METHOD_ZERO_SIGNIFICANCE_DISABLE */
        }

        if (coding_significance) {
            uint32_t budget = bi.significance_width * bi.height_lines;
            uint32_t pack_gcli_enable[4];
            for (uint32_t line = 0; line < bi.height_lines; line++) {
                const RcqBandLineLut& l = lut[(size_t)b * 4 + line];
                uint32_t zeroed = l.sig_cum[g] * RCQ_SIGNIFICANCE_GROUP_SIZE;
                if (l.leftover_max <= RCQ_TRUNCATION_MAX && l.leftover_max <= g) {
                    zeroed += l.leftover_extra;
                }
                pack_gcli_enable[line] = sum_data_non_sig_gc[line] - zeroed;
                budget += pack_gcli_enable[line];
            }
            if (best_budget > budget) {
                best_budget = budget;
                for (uint32_t line = 0; line < bi.height_lines; line++) {
                    pack_sig_bits[(size_t)b * 4 + line] = bi.significance_width;
                    pack_gcli_bits[(size_t)b * 4 + line] = pack_gcli_enable[line];
                }
                if (pack_method_out) {
                    (*pack_method_out)[b] = 1; /* METHOD_ZERO_SIGNIFICANCE_ENABLE */
                }
            }
        }
    }

    uint32_t precinct_size_bytes = 0;
    for (uint32_t p = 0; p < packets_num; p++) {
        uint32_t data_bits = 0, gcli_bits = 0, sig_bits = 0;
        for (uint32_t bidx = packets[p].band_start; bidx < packets[p].band_stop; bidx++) {
            uint32_t line = packets[p].line_idx;
            if (line < band_info[bidx].height_lines) {
                data_bits += pack_data_bits[(size_t)bidx * 4 + line];
                gcli_bits += pack_gcli_bits[(size_t)bidx * 4 + line];
                sig_bits += pack_sig_bits[(size_t)bidx * 4 + line];
            }
        }
        precinct_size_bytes += rcq_bits_to_bytes(data_bits); /* signs handling OFF: 0 bits */
        precinct_size_bytes += rcq_bits_to_bytes(gcli_bits) + rcq_bits_to_bytes(sig_bits);
    }
    return precinct_size_bytes;
}

/* Binary search state machine, matches BinarySearch.c exactly. */
enum RcqStep { RCQ_STEP_BEGIN, RCQ_STEP_TOO_SMALL, RCQ_STEP_TOO_BIG, RCQ_STEP_OUT_OF_RANGE };
struct RcqBinarySearch {
    int32_t id_beg, id_end;
    uint8_t find_below_matching;
    int32_t best_idx, last_index;
    uint32_t step;
};
static void rcq_bs_init(RcqBinarySearch* b, uint32_t begin, uint32_t end, uint8_t find_below, uint32_t step) {
    b->id_beg = (int32_t)begin;
    b->id_end = (int32_t)end;
    b->find_below_matching = find_below;
    b->best_idx = -1;
    b->last_index = -1;
    b->step = step ? step : (uint32_t)((b->id_end - b->id_beg) / 2);
}
/* Returns 1 to continue (out_next_test set), 0 on FIND_CLOSE (out_next_test=best),
 * -1 on ERROR. */
static int rcq_bs_next(RcqBinarySearch* b, RcqStep result, uint32_t* out_next_test) {
    if (result == RCQ_STEP_TOO_SMALL) {
        if (b->find_below_matching) {
            if (b->best_idx == -1 || b->best_idx < b->last_index) {
                b->best_idx = b->last_index;
            }
        }
        if (b->id_end >= b->last_index + 1) {
            b->id_beg = b->last_index + 1;
        }
        else {
            if (b->best_idx == -1)
                return -1;
            *out_next_test = (uint32_t)b->best_idx;
            return 0;
        }
    }
    else if (result == RCQ_STEP_TOO_BIG || result == RCQ_STEP_OUT_OF_RANGE) {
        if (result == RCQ_STEP_TOO_BIG && !b->find_below_matching) {
            if (b->best_idx == -1 || b->best_idx > b->last_index) {
                b->best_idx = b->last_index;
            }
        }
        if (b->id_beg <= b->last_index - 1) {
            b->id_end = b->last_index - 1;
        }
        else {
            if (b->best_idx == -1)
                return -1;
            *out_next_test = (uint32_t)b->best_idx;
            return 0;
        }
    }
    if ((int32_t)b->step > b->id_end - b->id_beg) {
        b->step = (uint32_t)((b->id_end - b->id_beg + 1) / 2);
    }
    b->last_index = (int32_t)b->step + b->id_beg;
    *out_next_test = (uint32_t)b->last_index;
    return 1;
}

/* Matches precinct_encoder_compute_truncation(): fills gtli[] for all bands,
 * returns 1 if every band is fully truncated (empty precinct). */
static int rcq_compute_all_truncation(uint32_t bands_num, const svt_cuda_rc_band_info_t* band_info, uint32_t quantization,
                                      uint32_t refinement, uint8_t* gtli) {
    int empty = 1;
    for (uint32_t b = 0; b < bands_num; b++) {
        gtli[b] = rcq_compute_truncation(band_info[b].gain, band_info[b].priority, quantization, refinement);
        if (gtli[b] != RCQ_TRUNCATION_MAX) {
            empty = 0;
        }
    }
    return empty;
}

/* Matches quant_deadzone_c / quant_uniform_c (Quant.c). */
__global__ void k_rcq_quantize(const svt_cuda_rc_band_info_t* band_info, uint32_t bands_num, const uint8_t* gcli_data,
                               const uint8_t* out_gtli, uint16_t* coeff_data, uint8_t quant_type) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    /* linear index over all (band,line,coeff) triples via prefix search would need
     * extra bookkeeping; instead launch one thread per (band,line) and loop coeffs
     * inside -- band counts/widths are small enough this is not a bottleneck for
     * Phase 2's correctness-first scope. */
    uint32_t b = idx / 4;
    uint32_t line = idx % 4;
    if (b >= bands_num)
        return;
    const svt_cuda_rc_band_info_t& bi = band_info[b];
    if (line >= bi.height_lines)
        return;
    uint8_t gtli = out_gtli[b];
    if (gtli == 0)
        return; /* matches precinct_quantization()'s skip */

    const uint8_t* gcli_ptr = gcli_data + bi.gcli_offset + (size_t)line * bi.gcli_width;
    uint16_t* coeff_ptr = coeff_data + bi.coeff_offset + (size_t)line * bi.width;

    for (uint32_t i = 0; i < bi.width; i++) {
        uint8_t gcli = gcli_ptr[i / RCQ_GROUP_SIZE];
        if (gcli <= gtli) {
            coeff_ptr[i] = 0;
            continue;
        }
        uint16_t sign = coeff_ptr[i] & RCQ_SIGN_MASK;
        uint16_t mag = coeff_ptr[i] & ~RCQ_SIGN_MASK;
        uint16_t out_mag;
        if (quant_type == 0) { /* deadzone */
            out_mag = (uint16_t)((mag >> gtli) << gtli);
        }
        else { /* uniform */
            uint16_t scale_value = (uint16_t)(gcli - gtli + 1);
            uint16_t d = (uint16_t)(((mag << scale_value) - mag + (1 << gcli)) >> (gcli + 1));
            out_mag = (uint16_t)(d << gtli);
        }
        coeff_ptr[i] = out_mag ? (out_mag | sign) : 0;
    }
}

int svt_cuda_rc_quant_precinct(uint32_t bands_num, const svt_cuda_rc_band_info_t* band_info, const uint8_t* gcli_data,
                               const uint8_t* significance_data, uint16_t* coeff_data, uint32_t packets_num,
                               const svt_cuda_rc_packet_t* packets, uint32_t bands_num_exists, uint32_t packets_exist_num,
                               uint8_t use_short_header, uint8_t coding_significance, uint8_t quant_type,
                               uint32_t max_quantization, uint32_t max_refinement, uint32_t budget_bytes, uint8_t* out_gtli,
                               uint8_t* out_quantization, uint8_t* out_refinement, uint32_t* out_data_bytes,
                               uint32_t* out_padding_bytes, uint32_t* out_total_bytes, uint8_t* out_pack_method,
                               uint32_t* out_pack_gcli_bits, uint32_t* out_pack_sig_bits, uint32_t* out_pack_data_bits) {
    /* --- headers_bytes / budget_to_data_bytes, matches rate_control_get_headers_bytes() --- */
    uint32_t pack_header_bits = use_short_header ? (RCQ_PACKET_HEADER_SHORT_SIZE_BYTES * 8) : (RCQ_PACKET_HEADER_LONG_SIZE_BYTES * 8);
    uint32_t headers_bytes = rcq_bits_to_bytes(rcq_align8(RCQ_PRECINCT_HEADER_SIZE_BYTES * 8 + bands_num_exists * 2)) +
        rcq_bits_to_bytes(rcq_align8(pack_header_bits * packets_exist_num));
    if (budget_bytes <= headers_bytes) {
        return 1;
    }
    uint32_t budget_to_data_bytes = budget_bytes - headers_bytes;

    /* --- Build per-(band,line) LUTs on GPU (parallel, independent) --- */
    size_t max_bl = (size_t)bands_num * 4;
    svt_cuda_rc_band_info_t* d_band_info = NULL;
    uint8_t *d_gcli = NULL, *d_sig = NULL;
    RcqBandLineLut* d_lut = NULL;
    cudaError_t err = cudaSuccess;
    int rc_ok = 0;

    /* Total gcli/significance/coeff data sizes are derived from the last
     * band's offset + its own size (offsets are monotonically increasing,
     * matching pi_enc_t's own offset construction). */
    uint32_t gcli_total = 0, sig_total = 0, coeff_total = 0;
    for (uint32_t b = 0; b < bands_num; b++) {
        gcli_total = std::max(gcli_total, band_info[b].gcli_offset + band_info[b].height_lines * band_info[b].gcli_width);
        sig_total = std::max(sig_total,
                             band_info[b].significance_offset + band_info[b].height_lines * band_info[b].significance_width);
        coeff_total = std::max(coeff_total, band_info[b].coeff_offset + band_info[b].height_lines * band_info[b].width);
    }

    do {
        if ((err = cudaMalloc(&d_band_info, bands_num * sizeof(svt_cuda_rc_band_info_t))) != cudaSuccess)
            break;
        if ((err = cudaMalloc(&d_gcli, gcli_total)) != cudaSuccess)
            break;
        if ((err = cudaMalloc(&d_sig, sig_total ? sig_total : 1)) != cudaSuccess)
            break;
        if ((err = cudaMalloc(&d_lut, max_bl * sizeof(RcqBandLineLut))) != cudaSuccess)
            break;

        if ((err = cudaMemcpy(d_band_info, band_info, bands_num * sizeof(svt_cuda_rc_band_info_t), cudaMemcpyHostToDevice)) !=
            cudaSuccess)
            break;
        if ((err = cudaMemcpy(d_gcli, gcli_data, gcli_total, cudaMemcpyHostToDevice)) != cudaSuccess)
            break;
        if (sig_total && (err = cudaMemcpy(d_sig, significance_data, sig_total, cudaMemcpyHostToDevice)) != cudaSuccess)
            break;

        uint32_t threads = 256;
        uint32_t blocks = (uint32_t)((max_bl + threads - 1) / threads);
        k_rcq_build_lut<<<blocks, threads>>>(d_band_info, bands_num, d_gcli, d_sig, coding_significance, d_lut);

        std::vector<RcqBandLineLut> lut(max_bl);
        if ((err = cudaMemcpy(lut.data(), d_lut, max_bl * sizeof(RcqBandLineLut), cudaMemcpyDeviceToHost)) != cudaSuccess)
            break;
        if ((err = cudaGetLastError()) != cudaSuccess)
            break;

        /* --- Sequential binary search on host (cheap LUT lookups; the
         * expensive histogram build above already ran on GPU). Phase 4 can
         * move this into block-local device code once multiple precincts
         * are batched across blocks -- for a single precinct there is no
         * parallelism to gain from doing so today. --- */
        std::vector<uint8_t> gtli(bands_num);
        std::vector<uint32_t> pack_gcli_bits(max_bl), pack_sig_bits(max_bl), pack_data_bits(max_bl);

        uint32_t initial_step_q = 6;
        if (initial_step_q > max_quantization)
            initial_step_q = 0;
        RcqBinarySearch bs_q;
        rcq_bs_init(&bs_q, 0, max_quantization, 0, initial_step_q);
        RcqStep step = RCQ_STEP_BEGIN;
        uint32_t quantization = 0;
        int found_q = 0;
        for (;;) {
            uint32_t test_q;
            int r = rcq_bs_next(&bs_q, step, &test_q);
            if (r < 0)
                break;
            if (r == 0) {
                quantization = test_q;
                found_q = 1;
                break;
            }
            int empty = rcq_compute_all_truncation(bands_num, band_info, test_q, 0, gtli.data());
            if (empty) {
                step = RCQ_STEP_OUT_OF_RANGE;
                continue;
            }
            uint32_t total =
                rcq_compute_budget_bytes(bands_num, band_info, lut, gtli.data(), coding_significance, packets_num, packets,
                                         pack_gcli_bits, pack_sig_bits, pack_data_bits);
            step = (total > budget_to_data_bytes) ? RCQ_STEP_TOO_SMALL : RCQ_STEP_TOO_BIG;
        }
        if (!found_q) {
            break; /* not a CUDA error; rc_ok stays 0 */
        }

        uint32_t initial_step_r = 6;
        if (initial_step_r >= max_refinement)
            initial_step_r = 0;
        RcqBinarySearch bs_r;
        rcq_bs_init(&bs_r, 0, max_refinement, 1, initial_step_r);
        step = RCQ_STEP_BEGIN;
        uint32_t refinement = 0;
        int found_r = 0;
        uint32_t data_bytes = 0;
        for (;;) {
            uint32_t test_r;
            int r = rcq_bs_next(&bs_r, step, &test_r);
            if (r < 0)
                break;
            if (r == 0) {
                refinement = test_r;
                found_r = 1;
                break;
            }
            int empty = rcq_compute_all_truncation(bands_num, band_info, quantization, test_r, gtli.data());
            if (empty) {
                step = RCQ_STEP_OUT_OF_RANGE;
                continue;
            }
            uint32_t total =
                rcq_compute_budget_bytes(bands_num, band_info, lut, gtli.data(), coding_significance, packets_num, packets,
                                         pack_gcli_bits, pack_sig_bits, pack_data_bits);
            data_bytes = total;
            step = (total <= budget_to_data_bytes) ? RCQ_STEP_TOO_SMALL : RCQ_STEP_TOO_BIG;
        }
        if (!found_r) {
            break; /* not a CUDA error; rc_ok stays 0 */
        }

        /* Recompute final gtli[]/data_bytes for the chosen (quantization,refinement)
         * -- matches rate_control_find_best_refinement_binary_search()'s
         * "recalculate if last tested != chosen" branch, done unconditionally
         * here for simplicity (pure function of (quantization,refinement)). */
        std::vector<uint8_t> pack_method(bands_num, 0);
        rcq_compute_all_truncation(bands_num, band_info, quantization, refinement, gtli.data());
        data_bytes = rcq_compute_budget_bytes(
            bands_num, band_info, lut, gtli.data(), coding_significance, packets_num, packets, pack_gcli_bits, pack_sig_bits,
            pack_data_bits, &pack_method);

        *out_quantization = (uint8_t)quantization;
        *out_refinement = (uint8_t)refinement;
        *out_data_bytes = data_bytes;
        *out_padding_bytes = budget_to_data_bytes - data_bytes;
        *out_total_bytes = budget_bytes;
        memcpy(out_gtli, gtli.data(), bands_num);
        if (out_pack_method) {
            memcpy(out_pack_method, pack_method.data(), bands_num);
        }
        if (out_pack_gcli_bits) {
            memcpy(out_pack_gcli_bits, pack_gcli_bits.data(), max_bl * sizeof(uint32_t));
        }
        if (out_pack_sig_bits) {
            memcpy(out_pack_sig_bits, pack_sig_bits.data(), max_bl * sizeof(uint32_t));
        }
        if (out_pack_data_bits) {
            memcpy(out_pack_data_bits, pack_data_bits.data(), max_bl * sizeof(uint32_t));
        }
        rc_ok = 1;

        /* --- Quantization: parallel on GPU --- */
        uint8_t* d_gtli = NULL;
        if ((err = cudaMalloc(&d_gtli, bands_num)) != cudaSuccess)
            break;
        if ((err = cudaMemcpy(d_gtli, out_gtli, bands_num, cudaMemcpyHostToDevice)) != cudaSuccess) {
            cudaFree(d_gtli);
            break;
        }
        uint16_t* d_coeff = NULL;
        if ((err = cudaMalloc(&d_coeff, (size_t)coeff_total * sizeof(uint16_t))) != cudaSuccess) {
            cudaFree(d_gtli);
            break;
        }
        if ((err = cudaMemcpy(d_coeff, coeff_data, (size_t)coeff_total * sizeof(uint16_t), cudaMemcpyHostToDevice)) !=
            cudaSuccess) {
            cudaFree(d_gtli);
            cudaFree(d_coeff);
            break;
        }
        k_rcq_quantize<<<blocks, threads>>>(d_band_info, bands_num, d_gcli, d_gtli, d_coeff, quant_type);
        err = cudaMemcpy(coeff_data, d_coeff, (size_t)coeff_total * sizeof(uint16_t), cudaMemcpyDeviceToHost);
        cudaFree(d_gtli);
        cudaFree(d_coeff);
        if (err != cudaSuccess)
            break;
        err = cudaGetLastError();
    } while (0);

    cudaFree(d_band_info);
    cudaFree(d_gcli);
    cudaFree(d_sig);
    cudaFree(d_lut);

    if (err != cudaSuccess) {
        return -(int)err;
    }
    return rc_ok ? 0 : 1;
}
