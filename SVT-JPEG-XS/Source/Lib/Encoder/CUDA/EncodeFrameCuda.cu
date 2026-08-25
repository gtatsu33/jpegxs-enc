/*
* Copyright(c) 2024 Intel Corporation
* SPDX - License - Identifier: BSD - 2 - Clause - Patent
*/
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>
#include <cstring>
#include "EncodeFrameCuda.cuh"
#include "DwtCuda.cuh"
#include "PackCuda.cuh" /* svt_cuda_pack_packet_t */

#define EFC_GROUP_SIZE 4
#define EFC_SIGNIFICANCE_GROUP_SIZE 8
#define EFC_TRUNCATION_MAX 15
#define EFC_SIGN_MASK ((uint16_t)1 << 15)
#define EFC_SIGN_BIT_POS 15
#define EFC_PRECINCT_HEADER_SIZE_BYTES 5
#define EFC_PACKET_HEADER_LONG_SIZE_BYTES 7
#define EFC_PACKET_HEADER_SHORT_SIZE_BYTES 5
#define EFC_CODING_MODE_FLAG_SIGNIFICANCE 2u

/* =====================================================================
 * 1. Context construction from a real pi_t/pi_enc_t.
 * ===================================================================== */

int svt_cuda_frame_context_create_from_pi(SvtCudaFrameContext* ctx, const pi_t* pi, const pi_enc_t* pi_enc,
                                          uint32_t pack_out_capacity_bytes) {
    (void)pi_enc;
    uint32_t comp_width[FCC_MAX_COMPONENTS] = {0}, comp_height[FCC_MAX_COMPONENTS] = {0};
    for (uint32_t c = 0; c < pi->comps_num; c++) {
        comp_width[c] = pi->components[c].width;
        comp_height[c] = pi->components[c].height;
    }

    std::vector<SvtCudaFrameBandGeom> bands(pi->bands_num_all);
    for (uint32_t flat = 0; flat < pi->bands_num_all; flat++) {
        uint32_t c = pi->global_band_info[flat].comp_id;
        uint32_t b = pi->global_band_info[flat].band_id;
        SvtCudaFrameBandGeom& g = bands[flat];
        memset(&g, 0, sizeof(g));
        if (b == BAND_NOT_EXIST) {
            g.comp_id = c;
            g.band_id = BAND_NOT_EXIST;
            continue;
        }
        const pi_band_t& pb = pi->components[c].bands[b];
        g.comp_id = c;
        g.band_id = b;
        g.x = pb.x;
        g.y = pb.y;
        g.width = pb.width;
        g.height = pb.height;
        g.height_lines_num = pb.height_lines_num;
        g.gcli_width = (pb.width + EFC_GROUP_SIZE - 1) / EFC_GROUP_SIZE;
        g.significance_width = (g.gcli_width + EFC_SIGNIFICANCE_GROUP_SIZE - 1) / EFC_SIGNIFICANCE_GROUP_SIZE;
        g.gain = pb.gain;
        g.priority = pb.priority;

        /* Phase 4a scope constraint: uniform (NORMAL) precinct rows only. */
        if (g.height_lines_num == 0 || g.height % g.height_lines_num != 0) {
            return 1;
        }
        if (g.height / g.height_lines_num != pi->precincts_line_num) {
            return 1;
        }
    }

    int rc = svt_cuda_frame_context_create(ctx, pi->comps_num, comp_width, comp_height, pi->bands_num_all, bands.data(),
                                           pi->precincts_line_num, pi->packets_num, pack_out_capacity_bytes);
    if (rc != 0) {
        return rc;
    }

    std::vector<svt_cuda_pack_packet_t> packets(pi->packets_num);
    for (uint32_t p = 0; p < pi->packets_num; p++) {
        packets[p].band_start = pi->packets[p].band_start;
        packets[p].band_stop = pi->packets[p].band_stop;
        packets[p].line_idx = pi->packets[p].line_idx;
    }
    cudaError_t err = cudaMalloc(&ctx->d_packets, pi->packets_num * sizeof(svt_cuda_pack_packet_t));
    if (err == cudaSuccess) {
        err = cudaMemcpy(
            ctx->d_packets, packets.data(), pi->packets_num * sizeof(svt_cuda_pack_packet_t), cudaMemcpyHostToDevice);
    }
    if (err != cudaSuccess) {
        svt_cuda_frame_context_destroy(ctx);
        return -(int)err;
    }
    return 0;
}

/* =====================================================================
 * 2. NLT + DWT (per component). Phase 4b-1: uses svt_cuda_dwt_component_ctx(),
 *    which reads/writes SvtCudaFrameContext's persistent scratch/pyramid
 *    buffers directly on ctx->stream -- no per-call cudaMalloc/cudaFree, no
 *    host round-trip for the pyramid output (Phase 4a's version copied the
 *    DWT host function's D2H result back to device via an extra H2D copy).
 * ===================================================================== */

static int efc_run_dwt(SvtCudaFrameContext* ctx, uint32_t comp_id, const void* in_plane, uint32_t in_stride,
                       uint32_t decom_h, uint32_t decom_v, uint8_t input_bit_depth, uint8_t hdr_Bw, uint8_t hdr_Fq) {
    uint32_t w = ctx->comp_width[comp_id], h = ctx->comp_height[comp_id];
    int err = svt_cuda_dwt_component_ctx(in_plane,
                                         in_stride,
                                         w,
                                         h,
                                         decom_h,
                                         decom_v,
                                         input_bit_depth,
                                         hdr_Bw,
                                         hdr_Fq,
                                         ctx->d_in_raw,
                                         ctx->d_cur,
                                         ctx->d_other,
                                         ctx->d_vert,
                                         ctx->d_pyramid32,
                                         ctx->d_pyramid16[comp_id],
                                         ctx->stream);
    return err;
}

/* =====================================================================
 * 3. Batched GC (gcli) + significance kernels, one launch per band,
 *    spanning the band's FULL height (all precincts of the frame at once).
 * ===================================================================== */

__device__ __forceinline__ uint8_t efc_log2_32(uint32_t x) {
    return (uint8_t)(31 - __clz(x));
}

__global__ void k_gc_band_frame(const uint16_t* pyramid, uint32_t stride, uint32_t bx, uint32_t by, uint32_t width,
                                uint32_t height, uint32_t gcli_width, uint8_t* gcli_out) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t total = height * gcli_width;
    if (idx >= total)
        return;
    uint32_t row = idx / gcli_width;
    uint32_t g = idx % gcli_width;
    uint32_t base = g * EFC_GROUP_SIZE;
    uint32_t n = min((uint32_t)EFC_GROUP_SIZE, width - base);
    const uint16_t* p = pyramid + (size_t)(by + row) * stride + (bx + base);
    uint16_t merge_or = 0;
    for (uint32_t i = 0; i < n; i++) {
        merge_or |= p[i];
    }
    merge_or = (uint16_t)(merge_or << 1);
    gcli_out[idx] = merge_or ? efc_log2_32(merge_or) : 0;
}

__global__ void k_sig_band_frame(const uint8_t* gcli, uint32_t gcli_width, uint32_t height, uint32_t sig_width,
                                 uint8_t* sig_out) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t total = height * sig_width;
    if (idx >= total)
        return;
    uint32_t row = idx / sig_width;
    uint32_t s = idx % sig_width;
    uint32_t base = s * EFC_SIGNIFICANCE_GROUP_SIZE;
    uint32_t n = min((uint32_t)EFC_SIGNIFICANCE_GROUP_SIZE, gcli_width - base);
    const uint8_t* p = gcli + (size_t)row * gcli_width + base;
    uint8_t m = 0;
    for (uint32_t i = 0; i < n; i++) {
        if (p[i] > m)
            m = p[i];
    }
    sig_out[idx] = m;
}

/* =====================================================================
 * 4. Batched RC LUT build (one launch per band, whole frame height at once).
 * ===================================================================== */

struct EfcBandLineLut {
    uint32_t size_data_no_sign[EFC_TRUNCATION_MAX + 1];
    uint32_t cum_count[EFC_TRUNCATION_MAX + 1];
    uint32_t sig_cum[EFC_TRUNCATION_MAX + 1];
    uint32_t leftover_max;
    uint32_t leftover_extra;
};

__global__ void k_rc_build_lut_band_frame(const uint8_t* gcli, const uint8_t* sig, uint32_t gcli_width, uint32_t sig_width,
                                          uint32_t height, uint8_t coding_significance, EfcBandLineLut* out_lut) {
    uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= height)
        return;
    EfcBandLineLut* lut = &out_lut[row];
    uint32_t raw_hist[EFC_TRUNCATION_MAX + 1];
    for (int i = 0; i <= EFC_TRUNCATION_MAX; i++) {
        raw_hist[i] = 0;
    }
    const uint8_t* gcli_row = gcli + (size_t)row * gcli_width;
    for (uint32_t i = 0; i < gcli_width; i++) {
        raw_hist[gcli_row[i]]++;
    }
    lut->size_data_no_sign[EFC_TRUNCATION_MAX] = 0;
    uint32_t sum_back = 0;
    for (int gtli_index = EFC_TRUNCATION_MAX - 1; gtli_index >= 0; gtli_index--) {
        sum_back += raw_hist[gtli_index + 1];
        lut->size_data_no_sign[gtli_index] = lut->size_data_no_sign[gtli_index + 1] + sum_back + raw_hist[gtli_index + 1];
    }
    uint32_t cum = 0;
    for (int v = 0; v <= EFC_TRUNCATION_MAX; v++) {
        cum += raw_hist[v];
        lut->cum_count[v] = cum;
    }
    lut->leftover_max = EFC_TRUNCATION_MAX + 1;
    lut->leftover_extra = 0;
    for (int v = 0; v <= EFC_TRUNCATION_MAX; v++) {
        lut->sig_cum[v] = 0;
    }
    if (coding_significance) {
        uint32_t full_group = gcli_width / EFC_SIGNIFICANCE_GROUP_SIZE;
        const uint8_t* sig_row = sig + (size_t)row * sig_width;
        uint32_t sig_hist[EFC_TRUNCATION_MAX + 1];
        for (int i = 0; i <= EFC_TRUNCATION_MAX; i++) {
            sig_hist[i] = 0;
        }
        for (uint32_t i = 0; i < full_group; i++) {
            sig_hist[sig_row[i]]++;
        }
        uint32_t scum = 0;
        for (int v = 0; v <= EFC_TRUNCATION_MAX; v++) {
            scum += sig_hist[v];
            lut->sig_cum[v] = scum;
        }
        if (full_group < sig_width) {
            lut->leftover_max = sig_row[full_group];
            lut->leftover_extra = gcli_width - full_group * EFC_SIGNIFICANCE_GROUP_SIZE;
        }
    }
}

/* =====================================================================
 * 5. Host-side per-precinct binary search (matches RcQuantCuda.cu /
 *    BinarySearch.c / RateControl.c exactly -- see that file for the
 *    original derivation notes). Reused verbatim; only the LUT source
 *    (per absolute frame row, not per single-precinct call) changed.
 * ===================================================================== */

static inline uint32_t efc_align8(uint32_t bits) {
    return (bits + 7) & ~7u;
}
static inline uint32_t efc_bits_to_bytes(uint32_t bits) {
    return (bits + 7) >> 3;
}
static inline uint8_t efc_compute_truncation(uint8_t gain, uint8_t priority, uint32_t quantization, uint32_t refinement) {
    uint8_t pump_up = (priority < refinement) ? 1 : 0;
    if (quantization < (uint32_t)(gain + pump_up)) {
        return 0;
    }
    uint32_t truncation = quantization - gain - pump_up;
    if (truncation > EFC_TRUNCATION_MAX) {
        truncation = EFC_TRUNCATION_MAX;
    }
    return (uint8_t)truncation;
}

enum EfcStep { EFC_STEP_BEGIN, EFC_STEP_TOO_SMALL, EFC_STEP_TOO_BIG, EFC_STEP_OUT_OF_RANGE };
struct EfcBinarySearch {
    int32_t id_beg, id_end;
    uint8_t find_below_matching;
    int32_t best_idx, last_index;
    uint32_t step;
};
static void efc_bs_init(EfcBinarySearch* b, uint32_t begin, uint32_t end, uint8_t find_below, uint32_t step) {
    b->id_beg = (int32_t)begin;
    b->id_end = (int32_t)end;
    b->find_below_matching = find_below;
    b->best_idx = -1;
    b->last_index = -1;
    b->step = step ? step : (uint32_t)((b->id_end - b->id_beg) / 2);
}
static int efc_bs_next(EfcBinarySearch* b, EfcStep result, uint32_t* out_next_test) {
    if (result == EFC_STEP_TOO_SMALL) {
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
    else if (result == EFC_STEP_TOO_BIG || result == EFC_STEP_OUT_OF_RANGE) {
        if (result == EFC_STEP_TOO_BIG && !b->find_below_matching) {
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

/* Fills gtli[] for all bands of ONE precinct; returns 1 if empty. */
static int efc_compute_all_truncation(uint32_t bands_num, const SvtCudaFrameBandGeom* bands, uint32_t quantization,
                                      uint32_t refinement, uint8_t* gtli) {
    int empty = 1;
    for (uint32_t b = 0; b < bands_num; b++) {
        if (bands[b].band_id == BAND_NOT_EXIST) {
            gtli[b] = EFC_TRUNCATION_MAX;
            continue;
        }
        gtli[b] = efc_compute_truncation(bands[b].gain, bands[b].priority, quantization, refinement);
        if (gtli[b] != EFC_TRUNCATION_MAX) {
            empty = 0;
        }
    }
    return empty;
}

/* Computes total precinct data-budget bytes for the given gtli[], using each
 * band's LUT slice for THIS precinct's rows (row_base..row_base+lines). */
static uint32_t efc_compute_budget_bytes(uint32_t bands_num, const SvtCudaFrameBandGeom* bands,
                                         const std::vector<EfcBandLineLut>& lut, const std::vector<uint32_t>& lut_row_offset,
                                         uint32_t precinct_idx, const uint8_t* gtli, uint8_t coding_significance,
                                         uint32_t packets_num, const svt_cuda_pack_packet_t* packets,
                                         std::vector<uint32_t>& pack_gcli_bits, std::vector<uint32_t>& pack_sig_bits,
                                         std::vector<uint32_t>& pack_data_bits, std::vector<uint8_t>* pack_method_out) {
    for (uint32_t b = 0; b < bands_num; b++) {
        const SvtCudaFrameBandGeom& bi = bands[b];
        if (bi.band_id == BAND_NOT_EXIST) {
            if (pack_method_out)
                (*pack_method_out)[b] = 0;
            continue;
        }
        uint8_t g = gtli[b];
        uint32_t lines = bi.height_lines_num;
        uint32_t row_base = lut_row_offset[b] + precinct_idx * lines;
        uint32_t sum_data_non_sig_gc[4];
        uint32_t best_budget = 0;
        for (uint32_t line = 0; line < lines; line++) {
            const EfcBandLineLut& l = lut[row_base + line];
            uint32_t sum_data = l.size_data_no_sign[g];
            uint32_t cum = l.cum_count[g];
            sum_data_non_sig_gc[line] = sum_data + cum;
            pack_data_bits[(size_t)b * 4 + line] = sum_data * EFC_GROUP_SIZE;
            pack_gcli_bits[(size_t)b * 4 + line] = sum_data_non_sig_gc[line];
            pack_sig_bits[(size_t)b * 4 + line] = 0;
            best_budget += sum_data_non_sig_gc[line];
        }
        if (pack_method_out) {
            (*pack_method_out)[b] = 0;
        }
        if (coding_significance) {
            uint32_t budget = bi.significance_width * lines;
            uint32_t pack_gcli_enable[4];
            for (uint32_t line = 0; line < lines; line++) {
                const EfcBandLineLut& l = lut[row_base + line];
                uint32_t zeroed = l.sig_cum[g] * EFC_SIGNIFICANCE_GROUP_SIZE;
                if (l.leftover_max <= EFC_TRUNCATION_MAX && l.leftover_max <= g) {
                    zeroed += l.leftover_extra;
                }
                pack_gcli_enable[line] = sum_data_non_sig_gc[line] - zeroed;
                budget += pack_gcli_enable[line];
            }
            if (best_budget > budget) {
                best_budget = budget;
                for (uint32_t line = 0; line < lines; line++) {
                    pack_sig_bits[(size_t)b * 4 + line] = bi.significance_width;
                    pack_gcli_bits[(size_t)b * 4 + line] = pack_gcli_enable[line];
                }
                if (pack_method_out) {
                    (*pack_method_out)[b] = 1;
                }
            }
        }
    }

    uint32_t precinct_size_bytes = 0;
    for (uint32_t p = 0; p < packets_num; p++) {
        uint32_t data_bits = 0, gcli_bits = 0, sig_bits = 0;
        for (uint32_t bidx = packets[p].band_start; bidx < packets[p].band_stop; bidx++) {
            uint32_t line = packets[p].line_idx;
            if (bands[bidx].band_id == BAND_NOT_EXIST)
                continue;
            if (line < bands[bidx].height_lines_num) {
                data_bits += pack_data_bits[(size_t)bidx * 4 + line];
                gcli_bits += pack_gcli_bits[(size_t)bidx * 4 + line];
                sig_bits += pack_sig_bits[(size_t)bidx * 4 + line];
            }
        }
        precinct_size_bytes += efc_bits_to_bytes(data_bits);
        precinct_size_bytes += efc_bits_to_bytes(gcli_bits) + efc_bits_to_bytes(sig_bits);
    }
    return precinct_size_bytes;
}

/* =====================================================================
 * 6. Batched quantization (one launch per band, whole frame height).
 * ===================================================================== */

__global__ void k_quantize_band_frame(uint16_t* pyramid, uint32_t stride, uint32_t bx, uint32_t by, uint32_t width,
                                      uint32_t height, uint32_t gcli_width, const uint8_t* gcli, uint32_t height_lines_num,
                                      const uint8_t* gtli_per_precinct /* [precincts_num] for this band */,
                                      uint8_t quant_type) {
    uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= height)
        return;
    uint32_t precinct_idx = row / height_lines_num;
    uint8_t gtli = gtli_per_precinct[precinct_idx];
    if (gtli == 0)
        return;

    const uint8_t* gcli_row = gcli + (size_t)row * gcli_width;
    uint16_t* coeff_row = pyramid + (size_t)(by + row) * stride + bx;

    for (uint32_t i = 0; i < width; i++) {
        uint8_t g = gcli_row[i / EFC_GROUP_SIZE];
        if (g <= gtli) {
            coeff_row[i] = 0;
            continue;
        }
        uint16_t sign = coeff_row[i] & EFC_SIGN_MASK;
        uint16_t mag = coeff_row[i] & ~EFC_SIGN_MASK;
        uint16_t out_mag;
        if (quant_type == 0) {
            out_mag = (uint16_t)((mag >> gtli) << gtli);
        }
        else {
            uint16_t scale_value = (uint16_t)(g - gtli + 1);
            uint16_t d = (uint16_t)(((mag << scale_value) - mag + (1 << g)) >> (g + 1));
            out_mag = (uint16_t)(d << gtli);
        }
        coeff_row[i] = out_mag ? (out_mag | sign) : 0;
    }
}

/* =====================================================================
 * 7. Batched pack (one block/thread per precinct).
 * ===================================================================== */

struct EfcWriter {
    uint8_t* mem;
    uint32_t offset;
    uint32_t bits_used;
};
__device__ __forceinline__ void efc_write_1_bit(EfcWriter* bw, uint8_t input) {
    uint8_t* mem = bw->mem + bw->offset;
    if (bw->bits_used == 0) {
        mem[0] = (uint8_t)((input & 1) << (7 - bw->bits_used));
    }
    else {
        mem[0] |= (uint8_t)((input & 1) << (7 - bw->bits_used));
    }
    bw->bits_used++;
    if (bw->bits_used == 8) {
        bw->bits_used = 0;
        bw->offset++;
    }
}
__device__ __forceinline__ void efc_write_n_bits(EfcWriter* bw, uint32_t input, uint8_t bits) {
    uint8_t* mem = bw->mem + bw->offset;
    if (bw->bits_used) {
        uint32_t left = 8 - bw->bits_used;
        uint8_t bits_to_copy = bits;
        if (bits_to_copy > left)
            bits_to_copy = (uint8_t)left;
        *mem |= (uint8_t)((input >> (bits - bits_to_copy)) << (left - bits_to_copy));
        if (left > bits_to_copy) {
            bw->bits_used += bits_to_copy;
            return;
        }
        bits -= bits_to_copy;
        bw->offset++;
        bw->bits_used = 0;
        mem++;
    }
    while (bits > 7) {
        *mem = (uint8_t)((input >> (bits - 8)) & 0xFF);
        bits -= 8;
        bw->offset++;
        mem++;
    }
    if (bits) {
        *mem = (uint8_t)((input & ((1u << bits) - 1)) << (8 - bits));
        bw->bits_used = bits;
    }
}
__device__ __forceinline__ void efc_write_4_bits_align4(EfcWriter* bw, uint8_t input) {
    uint8_t* mem = bw->mem + bw->offset;
    if (bw->bits_used == 4) {
        mem[0] |= input;
        bw->bits_used = 0;
        bw->offset++;
    }
    else {
        mem[0] = (uint8_t)(input << 4);
        bw->bits_used = 4;
    }
}
__device__ __forceinline__ uint32_t efc_used_bits(const EfcWriter* bw) {
    return bw->offset * 8 + bw->bits_used;
}
__device__ __forceinline__ void efc_align_byte(EfcWriter* bw) {
    if (bw->bits_used) {
        bw->offset++;
        bw->bits_used = 0;
    }
}
__device__ __forceinline__ void efc_add_padding_bytes(EfcWriter* bw, uint32_t nbytes) {
    for (uint32_t i = 0; i < nbytes; i++) {
        bw->mem[bw->offset + i] = 0;
    }
    bw->offset += nbytes;
}
__device__ __forceinline__ void efc_write_packet_header(EfcWriter* bw, uint32_t long_hdr, uint64_t data_size_bytes,
                                                         uint64_t bitplane_count_size_bytes) {
    uint8_t* mem = bw->mem + bw->offset;
    const uint64_t sign_size_bytes = 0;
    mem[0] = 0;
    if (long_hdr) {
        mem[0] |= (uint8_t)((data_size_bytes >> 13) & 0x7F);
        mem[1] = (uint8_t)((data_size_bytes >> 5) & 0xFF);
        mem[2] = (uint8_t)(((data_size_bytes & 0x1F) << 3) | ((bitplane_count_size_bytes >> 17) & 0x07));
        mem[3] = (uint8_t)((bitplane_count_size_bytes >> 9) & 0xFF);
        mem[4] = (uint8_t)((bitplane_count_size_bytes >> 1) & 0xFF);
        mem[5] = (uint8_t)(((bitplane_count_size_bytes & 0x1) << 7) | ((sign_size_bytes >> 8) & 0x7F));
        mem[6] = (uint8_t)(sign_size_bytes & 0xFF);
        bw->offset += 7;
    }
    else {
        mem[0] |= (uint8_t)((data_size_bytes >> 8) & 0x7F);
        mem[1] = (uint8_t)(data_size_bytes & 0xFF);
        mem[2] = (uint8_t)((bitplane_count_size_bytes >> 5) & 0xFF);
        mem[3] = (uint8_t)(((bitplane_count_size_bytes & 0x1F) << 3) | ((sign_size_bytes >> 8) & 0x07));
        mem[4] = (uint8_t)(sign_size_bytes & 0xFF);
        bw->offset += 5;
    }
}
__device__ __forceinline__ void efc_vlc_encode_simple(EfcWriter* bw, int32_t nbits) {
    if (nbits > 1) {
        uint32_t vlc_bits = ((1u << nbits) - 1) << 1;
        efc_write_n_bits(bw, vlc_bits, (uint8_t)(nbits + 1));
    }
    else if (nbits == 1) {
        efc_write_n_bits(bw, 2, 2);
    }
    else {
        efc_write_1_bit(bw, 0);
    }
}
__device__ void efc_pack_significance(EfcWriter* bw, uint8_t gtli, const uint8_t* sig_max, uint32_t width) {
    for (uint32_t i = 0; i < width; i++) {
        efc_write_1_bit(bw, sig_max[i] <= gtli ? 1 : 0);
    }
}
__device__ void efc_pack_bitplane_count_no_significance(EfcWriter* bw, const uint8_t* bitplane, uint32_t width, int8_t gtli) {
    for (uint32_t i = 0; i < width; i++) {
        efc_vlc_encode_simple(bw, bitplane[i] > gtli ? (int32_t)bitplane[i] - gtli : 0);
    }
}
__device__ void efc_pack_bitplane_count_significance(EfcWriter* bw, const uint8_t* bitplane, uint32_t width, int8_t gtli,
                                                      const uint8_t* sig_max, uint32_t group_size) {
    uint32_t groups = width / group_size;
    uint32_t leftover = width % group_size;
    uint32_t g = 0;
    for (; g < groups; g++) {
        if (sig_max[g] > gtli) {
            for (uint32_t i = 0; i < group_size; i++) {
                efc_vlc_encode_simple(bw, bitplane[i] > gtli ? (int32_t)bitplane[i] - gtli : 0);
            }
        }
        bitplane += group_size;
    }
    if (leftover) {
        if (sig_max[g] > gtli) {
            for (uint32_t i = 0; i < leftover; i++) {
                efc_vlc_encode_simple(bw, bitplane[i] > gtli ? (int32_t)bitplane[i] - gtli : 0);
            }
        }
    }
}
__device__ void efc_pack_data_single_group(EfcWriter* bw, const uint16_t* buf, uint8_t gcli, uint8_t gtli) {
    uint16_t tmp[4];
    for (int i = 0; i < 4; i++) {
        tmp[i] = (uint16_t)((unsigned)buf[i] << ((EFC_SIGN_BIT_POS + 1) - gcli));
    }
    for (int32_t bits = (int32_t)gcli - gtli - 1; bits >= 0; bits--) {
        uint16_t val = (uint16_t)(tmp[0] & EFC_SIGN_MASK);
        tmp[0] = (uint16_t)(tmp[0] << 1);
        val = (uint16_t)(val | ((tmp[1] & EFC_SIGN_MASK) >> 1));
        tmp[1] = (uint16_t)(tmp[1] << 1);
        val = (uint16_t)(val | ((tmp[2] & EFC_SIGN_MASK) >> 2));
        tmp[2] = (uint16_t)(tmp[2] << 1);
        val = (uint16_t)(val | ((tmp[3] & EFC_SIGN_MASK) >> 3));
        tmp[3] = (uint16_t)(tmp[3] << 1);
        val = (uint16_t)(val >> (EFC_SIGN_BIT_POS - 3));
        efc_write_4_bits_align4(bw, (uint8_t)val);
    }
}
__device__ void efc_pack_data(EfcWriter* bw, const uint16_t* buf, uint32_t width, const uint8_t* gclis, uint8_t gtli) {
    uint32_t groups = width / EFC_GROUP_SIZE;
    uint32_t leftover = width % EFC_GROUP_SIZE;
    uint32_t group = 0;
    for (; group < groups; group++) {
        if (gclis[group] > gtli) {
            uint8_t signs = (uint8_t)((buf[0] & EFC_SIGN_MASK) >> (EFC_SIGN_BIT_POS - 3));
            signs |= (uint8_t)((buf[1] & EFC_SIGN_MASK) >> (EFC_SIGN_BIT_POS - 2));
            signs |= (uint8_t)((buf[2] & EFC_SIGN_MASK) >> (EFC_SIGN_BIT_POS - 1));
            signs |= (uint8_t)((buf[3] & EFC_SIGN_MASK) >> (EFC_SIGN_BIT_POS - 0));
            efc_write_4_bits_align4(bw, signs);
            efc_pack_data_single_group(bw, buf, gclis[group], gtli);
        }
        buf += EFC_GROUP_SIZE;
    }
    if (leftover) {
        if (gclis[group] > gtli) {
            for (uint32_t i = 0; i < EFC_GROUP_SIZE; i++) {
                efc_write_1_bit(bw, (i < leftover) ? (uint8_t)(buf[i] >> EFC_SIGN_BIT_POS) : 0);
            }
            for (int32_t bits = (int32_t)gclis[group] - 1; bits >= gtli; bits--) {
                for (uint32_t i = 0; i < EFC_GROUP_SIZE; i++) {
                    efc_write_1_bit(bw, (i < leftover) ? (uint8_t)(buf[i] >> bits) : 0);
                }
            }
        }
    }
}

__global__ void k_pack_precinct_frame(const SvtCudaFrameBandGeom* bands, uint32_t bands_num_all, uint32_t bands_num_exists,
                                      uint16_t* const* pyramid_ptrs, const uint32_t* comp_stride, const uint8_t* gcli_frame,
                                      const uint8_t* sig_frame, const uint8_t* gtli_all, const uint8_t* pack_method_all,
                                      uint32_t packets_num, const svt_cuda_pack_packet_t* packets, uint8_t use_short_header,
                                      const uint8_t* quantization_all, const uint8_t* refinement_all,
                                      const uint32_t* total_bytes_all, const uint32_t* padding_bytes_all,
                                      const uint32_t* out_offset_all, const uint32_t* psd_all, const uint32_t* psg_all,
                                      const uint32_t* pss_all, uint8_t* out_buffer, int* out_error) {
    uint32_t precinct_idx = blockIdx.x;
    const uint8_t* gtli = gtli_all + (size_t)precinct_idx * bands_num_all;
    const uint8_t* pack_method = pack_method_all + (size_t)precinct_idx * bands_num_all;
    const uint32_t* psd = psd_all + (size_t)precinct_idx * packets_num;
    const uint32_t* psg = psg_all + (size_t)precinct_idx * packets_num;
    const uint32_t* pss = pss_all + (size_t)precinct_idx * packets_num;
    uint32_t pack_total_bytes = total_bytes_all[precinct_idx];
    uint32_t pack_padding_bytes = padding_bytes_all[precinct_idx];

    EfcWriter bw;
    bw.mem = out_buffer + out_offset_all[precinct_idx];
    bw.offset = 0;
    bw.bits_used = 0;

    uint32_t header_len_bytes = (EFC_PRECINCT_HEADER_SIZE_BYTES * 8 + 2 * bands_num_exists + 7) / 8;
    uint32_t packet_bytes_size = pack_total_bytes - header_len_bytes;
    efc_write_n_bits(&bw, packet_bytes_size, 24);
    efc_write_n_bits(&bw, quantization_all[precinct_idx], 8);
    efc_write_n_bits(&bw, refinement_all[precinct_idx], 8);

    for (uint32_t band = 0; band < bands_num_all; band++) {
        if (bands[band].band_id == BAND_NOT_EXIST) {
            continue;
        }
        uint8_t type_Dpb = pack_method[band] ? EFC_CODING_MODE_FLAG_SIGNIFICANCE : 0;
        efc_write_n_bits(&bw, type_Dpb, 2);
    }
    efc_align_byte(&bw);

    for (uint32_t p = 0; p < packets_num; p++) {
        int has_band = 0;
        for (uint32_t bidx = packets[p].band_start; bidx < packets[p].band_stop; bidx++) {
            if (bands[bidx].band_id != BAND_NOT_EXIST && packets[p].line_idx < bands[bidx].height_lines_num) {
                has_band = 1;
                break;
            }
        }
        if (!has_band) {
            continue;
        }

        efc_write_packet_header(&bw, !use_short_header, psd[p], psg[p]);
        efc_align_byte(&bw);

        uint32_t bits_last = efc_used_bits(&bw);
        for (uint32_t bidx = packets[p].band_start; bidx < packets[p].band_stop; bidx++) {
            const SvtCudaFrameBandGeom& bi = bands[bidx];
            uint32_t line = packets[p].line_idx;
            if (bi.band_id == BAND_NOT_EXIST || line >= bi.height_lines_num)
                continue;
            if (pack_method[bidx] == 1) {
                uint32_t abs_row = precinct_idx * bi.height_lines_num + line;
                const uint8_t* sig = sig_frame + bi.sig_offset + (size_t)abs_row * bi.significance_width;
                efc_pack_significance(&bw, gtli[bidx], sig, bi.significance_width);
            }
        }
        efc_align_byte(&bw);
        if (efc_used_bits(&bw) - bits_last != pss[p] * 8) {
            *out_error = 1;
            return;
        }

        bits_last = efc_used_bits(&bw);
        for (uint32_t bidx = packets[p].band_start; bidx < packets[p].band_stop; bidx++) {
            const SvtCudaFrameBandGeom& bi = bands[bidx];
            uint32_t line = packets[p].line_idx;
            if (bi.band_id == BAND_NOT_EXIST || line >= bi.height_lines_num)
                continue;
            uint32_t abs_row = precinct_idx * bi.height_lines_num + line;
            const uint8_t* gcli = gcli_frame + bi.gcli_offset + (size_t)abs_row * bi.gcli_width;
            if (pack_method[bidx] == 1) {
                const uint8_t* sig = sig_frame + bi.sig_offset + (size_t)abs_row * bi.significance_width;
                efc_pack_bitplane_count_significance(&bw, gcli, bi.gcli_width, gtli[bidx], sig, EFC_SIGNIFICANCE_GROUP_SIZE);
            }
            else {
                efc_pack_bitplane_count_no_significance(&bw, gcli, bi.gcli_width, gtli[bidx]);
            }
        }
        efc_align_byte(&bw);
        if (efc_used_bits(&bw) - bits_last != psg[p] * 8) {
            *out_error = 2;
            return;
        }

        bits_last = efc_used_bits(&bw);
        for (uint32_t bidx = packets[p].band_start; bidx < packets[p].band_stop; bidx++) {
            const SvtCudaFrameBandGeom& bi = bands[bidx];
            uint32_t line = packets[p].line_idx;
            if (bi.band_id == BAND_NOT_EXIST || line >= bi.height_lines_num)
                continue;
            uint32_t abs_row = precinct_idx * bi.height_lines_num + line;
            const uint8_t* gcli = gcli_frame + bi.gcli_offset + (size_t)abs_row * bi.gcli_width;
            const uint16_t* coeff = pyramid_ptrs[bi.comp_id] + (size_t)(bi.y + abs_row) * comp_stride[bi.comp_id] + bi.x;
            efc_pack_data(&bw, coeff, bi.width, gcli, gtli[bidx]);
        }
        efc_align_byte(&bw);
        if (efc_used_bits(&bw) - bits_last != psd[p] * 8) {
            *out_error = 3;
            return;
        }
    }

    if (pack_padding_bytes) {
        efc_add_padding_bytes(&bw, pack_padding_bytes);
    }
}

/* =====================================================================
 * 8. Top-level orchestration.
 * ===================================================================== */

int svt_cuda_encode_frame(SvtCudaFrameContext* ctx, const void* const in_planes[], const uint32_t in_stride[],
                          uint32_t decom_h, uint32_t decom_v, uint8_t input_bit_depth, uint8_t hdr_Bw, uint8_t hdr_Fq,
                          uint8_t quant_type, uint8_t use_short_header, uint8_t coding_significance, uint32_t max_quantization,
                          uint32_t max_refinement, const uint32_t* precinct_budget_bytes, uint32_t bands_num_exists,
                          uint32_t packets_exist_num, uint8_t* out_buffer, uint32_t* out_used_bytes) {
    uint32_t bands_num_all = ctx->bands_num_all;
    uint32_t precincts_num = ctx->precincts_num;
    uint32_t packets_num = ctx->packets_num;
    cudaError_t cerr = cudaSuccess;

    /* --- Step 1: NLT + DWT per component --- */
    for (uint32_t c = 0; c < ctx->comps_num; c++) {
        int err = efc_run_dwt(ctx, c, in_planes[c], in_stride[c], decom_h, decom_v, input_bit_depth, hdr_Bw, hdr_Fq);
        if (err != 0) {
            return err;
        }
    }

    /* --- Step 2: batched GC + significance, one launch per band, whole frame height --- */
    for (uint32_t b = 0; b < bands_num_all; b++) {
        const SvtCudaFrameBandGeom& g = ctx->h_bands[b];
        if (g.band_id == BAND_NOT_EXIST || g.height == 0)
            continue;
        uint32_t total_gc = g.height * g.gcli_width;
        uint32_t threads = 256, blocks = (total_gc + threads - 1) / threads;
        k_gc_band_frame<<<blocks, threads, 0, ctx->stream>>>(ctx->d_pyramid16[g.comp_id], ctx->comp_width[g.comp_id], g.x, g.y,
                                                              g.width, g.height, g.gcli_width, ctx->d_gcli_frame + g.gcli_offset);
        uint32_t total_sig = g.height * g.significance_width;
        blocks = (total_sig + threads - 1) / threads;
        k_sig_band_frame<<<blocks, threads, 0, ctx->stream>>>(
            ctx->d_gcli_frame + g.gcli_offset, g.gcli_width, g.height, g.significance_width, ctx->d_sig_frame + g.sig_offset);
    }

    /* --- Step 3: batched RC LUT build, one launch per band, whole frame height --- */
    std::vector<uint32_t> lut_row_offset(bands_num_all, 0);
    uint32_t lut_total_rows = 0;
    for (uint32_t b = 0; b < bands_num_all; b++) {
        lut_row_offset[b] = lut_total_rows;
        if (ctx->h_bands[b].band_id != BAND_NOT_EXIST) {
            lut_total_rows += ctx->h_bands[b].height;
        }
    }
    EfcBandLineLut* d_lut = NULL;
    if ((cerr = cudaMalloc(&d_lut, (lut_total_rows ? lut_total_rows : 1) * sizeof(EfcBandLineLut))) != cudaSuccess) {
        return -(int)cerr;
    }
    for (uint32_t b = 0; b < bands_num_all; b++) {
        const SvtCudaFrameBandGeom& g = ctx->h_bands[b];
        if (g.band_id == BAND_NOT_EXIST || g.height == 0)
            continue;
        uint32_t threads = 256, blocks = (g.height + threads - 1) / threads;
        k_rc_build_lut_band_frame<<<blocks, threads, 0, ctx->stream>>>(ctx->d_gcli_frame + g.gcli_offset,
                                                                       ctx->d_sig_frame + g.sig_offset, g.gcli_width,
                                                                       g.significance_width, g.height, coding_significance,
                                                                       d_lut + lut_row_offset[b]);
    }
    std::vector<EfcBandLineLut> lut(lut_total_rows);
    if (lut_total_rows) {
        cerr = cudaMemcpyAsync(
            lut.data(), d_lut, lut_total_rows * sizeof(EfcBandLineLut), cudaMemcpyDeviceToHost, ctx->stream);
    }
    if (cerr == cudaSuccess) {
        cerr = cudaStreamSynchronize(ctx->stream);
    }
    cudaFree(d_lut);
    if (cerr != cudaSuccess) {
        return -(int)cerr;
    }

    /* --- Step 4: host-side per-precinct binary search (RC), sequential across
     * precincts but each precinct only does cheap LUT lookups -- the expensive
     * histogram build already ran batched on GPU in step 3. --- */
    uint32_t pack_header_bits =
        use_short_header ? (EFC_PACKET_HEADER_SHORT_SIZE_BYTES * 8) : (EFC_PACKET_HEADER_LONG_SIZE_BYTES * 8);
    uint32_t headers_bytes = efc_bits_to_bytes(efc_align8(EFC_PRECINCT_HEADER_SIZE_BYTES * 8 + bands_num_exists * 2)) +
        efc_bits_to_bytes(efc_align8(pack_header_bits * packets_exist_num));

    std::vector<uint8_t> h_gtli((size_t)precincts_num * bands_num_all);
    std::vector<uint8_t> h_pack_method((size_t)precincts_num * bands_num_all, 0);
    std::vector<uint32_t> h_psd((size_t)precincts_num * packets_num);
    std::vector<uint32_t> h_psg((size_t)precincts_num * packets_num);
    std::vector<uint32_t> h_pss((size_t)precincts_num * packets_num);
    std::vector<uint8_t> h_quant(precincts_num), h_refine(precincts_num);
    std::vector<uint32_t> h_total_bytes(precincts_num), h_padding_bytes(precincts_num), h_out_offset(precincts_num);

    std::vector<svt_cuda_pack_packet_t> h_packets(packets_num);
    /* d_packets was built once at context-creation time; also keep a host
     * mirror here purely for the CPU-side aggregation loop. */
    cerr = cudaMemcpy(h_packets.data(), ctx->d_packets, packets_num * sizeof(svt_cuda_pack_packet_t), cudaMemcpyDeviceToHost);
    if (cerr != cudaSuccess) {
        return -(int)cerr;
    }

    std::vector<uint8_t> gtli(bands_num_all);
    std::vector<uint32_t> pgb(bands_num_all * 4), psb(bands_num_all * 4), pdb(bands_num_all * 4);

    for (uint32_t pr = 0; pr < precincts_num; pr++) {
        uint32_t budget_bytes = precinct_budget_bytes[pr];
        if (budget_bytes <= headers_bytes) {
            return 1;
        }
        uint32_t budget_to_data_bytes = budget_bytes - headers_bytes;

        uint32_t initial_step_q = 6;
        if (initial_step_q > max_quantization)
            initial_step_q = 0;
        EfcBinarySearch bs_q;
        efc_bs_init(&bs_q, 0, max_quantization, 0, initial_step_q);
        EfcStep step = EFC_STEP_BEGIN;
        uint32_t quantization = 0;
        int found_q = 0;
        for (;;) {
            uint32_t test_q;
            int r = efc_bs_next(&bs_q, step, &test_q);
            if (r < 0)
                break;
            if (r == 0) {
                quantization = test_q;
                found_q = 1;
                break;
            }
            int empty = efc_compute_all_truncation(bands_num_all, ctx->h_bands, test_q, 0, gtli.data());
            if (empty) {
                step = EFC_STEP_OUT_OF_RANGE;
                continue;
            }
            uint32_t total = efc_compute_budget_bytes(bands_num_all, ctx->h_bands, lut, lut_row_offset, pr, gtli.data(),
                                                      coding_significance, packets_num, h_packets.data(), pgb, psb, pdb, NULL);
            step = (total > budget_to_data_bytes) ? EFC_STEP_TOO_SMALL : EFC_STEP_TOO_BIG;
        }
        if (!found_q) {
            return 1;
        }

        uint32_t initial_step_r = 6;
        if (initial_step_r >= max_refinement)
            initial_step_r = 0;
        EfcBinarySearch bs_r;
        efc_bs_init(&bs_r, 0, max_refinement, 1, initial_step_r);
        step = EFC_STEP_BEGIN;
        uint32_t refinement = 0;
        int found_r = 0;
        for (;;) {
            uint32_t test_r;
            int r = efc_bs_next(&bs_r, step, &test_r);
            if (r < 0)
                break;
            if (r == 0) {
                refinement = test_r;
                found_r = 1;
                break;
            }
            int empty = efc_compute_all_truncation(bands_num_all, ctx->h_bands, quantization, test_r, gtli.data());
            if (empty) {
                step = EFC_STEP_OUT_OF_RANGE;
                continue;
            }
            uint32_t total = efc_compute_budget_bytes(bands_num_all, ctx->h_bands, lut, lut_row_offset, pr, gtli.data(),
                                                      coding_significance, packets_num, h_packets.data(), pgb, psb, pdb, NULL);
            step = (total <= budget_to_data_bytes) ? EFC_STEP_TOO_SMALL : EFC_STEP_TOO_BIG;
        }
        if (!found_r) {
            return 1;
        }

        std::vector<uint8_t> pack_method(bands_num_all, 0);
        efc_compute_all_truncation(bands_num_all, ctx->h_bands, quantization, refinement, gtli.data());
        uint32_t data_bytes = efc_compute_budget_bytes(bands_num_all, ctx->h_bands, lut, lut_row_offset, pr, gtli.data(),
                                                       coding_significance, packets_num, h_packets.data(), pgb, psb, pdb,
                                                       &pack_method);

        memcpy(&h_gtli[(size_t)pr * bands_num_all], gtli.data(), bands_num_all);
        memcpy(&h_pack_method[(size_t)pr * bands_num_all], pack_method.data(), bands_num_all);
        h_quant[pr] = (uint8_t)quantization;
        h_refine[pr] = (uint8_t)refinement;
        h_total_bytes[pr] = budget_bytes;
        h_padding_bytes[pr] = budget_to_data_bytes - data_bytes;

        /* Per-packet byte sizes for THIS precinct, matches precinct_get_budget_bytes(). */
        for (uint32_t p = 0; p < packets_num; p++) {
            uint32_t data_bits = 0, gcli_bits = 0, sig_bits = 0;
            for (uint32_t bidx = h_packets[p].band_start; bidx < h_packets[p].band_stop; bidx++) {
                uint32_t line = h_packets[p].line_idx;
                if (ctx->h_bands[bidx].band_id == BAND_NOT_EXIST)
                    continue;
                if (line < ctx->h_bands[bidx].height_lines_num) {
                    data_bits += pdb[(size_t)bidx * 4 + line];
                    gcli_bits += pgb[(size_t)bidx * 4 + line];
                    sig_bits += psb[(size_t)bidx * 4 + line];
                }
            }
            h_psd[(size_t)pr * packets_num + p] = efc_bits_to_bytes(data_bits);
            h_psg[(size_t)pr * packets_num + p] = efc_bits_to_bytes(gcli_bits);
            h_pss[(size_t)pr * packets_num + p] = efc_bits_to_bytes(sig_bits);
        }
    }

    uint32_t running = 0;
    for (uint32_t pr = 0; pr < precincts_num; pr++) {
        h_out_offset[pr] = running;
        running += h_total_bytes[pr];
    }
    if (running > ctx->pack_out_capacity_bytes) {
        return 1;
    }

    /* --- Step 5: upload RC results, quantize (batched per band), pack (batched per precinct) --- */
    cerr = cudaMemcpyAsync(ctx->d_gtli, h_gtli.data(), h_gtli.size(), cudaMemcpyHostToDevice, ctx->stream);
    if (cerr == cudaSuccess)
        cerr = cudaMemcpyAsync(ctx->d_pack_method, h_pack_method.data(), h_pack_method.size(), cudaMemcpyHostToDevice, ctx->stream);
    if (cerr == cudaSuccess)
        cerr = cudaMemcpyAsync(
            ctx->d_packet_size_data_bytes, h_psd.data(), h_psd.size() * sizeof(uint32_t), cudaMemcpyHostToDevice, ctx->stream);
    if (cerr == cudaSuccess)
        cerr = cudaMemcpyAsync(
            ctx->d_packet_size_gcli_bytes, h_psg.data(), h_psg.size() * sizeof(uint32_t), cudaMemcpyHostToDevice, ctx->stream);
    if (cerr == cudaSuccess)
        cerr = cudaMemcpyAsync(ctx->d_packet_size_significance_bytes, h_pss.data(), h_pss.size() * sizeof(uint32_t),
                               cudaMemcpyHostToDevice, ctx->stream);
    if (cerr == cudaSuccess)
        cerr = cudaMemcpyAsync(ctx->d_precinct_quantization, h_quant.data(), h_quant.size(), cudaMemcpyHostToDevice, ctx->stream);
    if (cerr == cudaSuccess)
        cerr =
            cudaMemcpyAsync(ctx->d_precinct_refinement, h_refine.data(), h_refine.size(), cudaMemcpyHostToDevice, ctx->stream);
    if (cerr == cudaSuccess)
        cerr = cudaMemcpyAsync(ctx->d_precinct_total_bytes, h_total_bytes.data(), h_total_bytes.size() * sizeof(uint32_t),
                               cudaMemcpyHostToDevice, ctx->stream);
    if (cerr == cudaSuccess)
        cerr = cudaMemcpyAsync(ctx->d_precinct_padding_bytes, h_padding_bytes.data(), h_padding_bytes.size() * sizeof(uint32_t),
                               cudaMemcpyHostToDevice, ctx->stream);
    if (cerr == cudaSuccess)
        cerr = cudaMemcpyAsync(ctx->d_precinct_out_offset, h_out_offset.data(), h_out_offset.size() * sizeof(uint32_t),
                               cudaMemcpyHostToDevice, ctx->stream);
    if (cerr != cudaSuccess) {
        return -(int)cerr;
    }

    uint32_t comp_stride_h[FCC_MAX_COMPONENTS] = {
        ctx->comp_width[0], ctx->comp_width[1], ctx->comp_width[2], ctx->comp_width[3]};
    uint32_t* d_comp_stride = NULL;
    if ((cerr = cudaMalloc(&d_comp_stride, sizeof(comp_stride_h))) != cudaSuccess) {
        return -(int)cerr;
    }
    cerr = cudaMemcpyAsync(d_comp_stride, comp_stride_h, sizeof(comp_stride_h), cudaMemcpyHostToDevice, ctx->stream);
    if (cerr != cudaSuccess) {
        cudaFree(d_comp_stride);
        return -(int)cerr;
    }

    /* Gather per-band contiguous gtli[precincts_num] slices (strided source,
     * cheap: bands_num_all * precincts_num bytes total) before the
     * quantize kernels, which expect contiguous per-band arrays. */
    std::vector<uint8_t> h_gtli_per_band(precincts_num);
    uint8_t* d_gtli_per_band = NULL;
    if ((cerr = cudaMalloc(&d_gtli_per_band, precincts_num * bands_num_all)) != cudaSuccess) {
        cudaFree(d_comp_stride);
        return -(int)cerr;
    }
    for (uint32_t b = 0; b < bands_num_all; b++) {
        for (uint32_t pr = 0; pr < precincts_num; pr++) {
            h_gtli_per_band[pr] = h_gtli[(size_t)pr * bands_num_all + b];
        }
        cerr = cudaMemcpyAsync(d_gtli_per_band + (size_t)b * precincts_num, h_gtli_per_band.data(), precincts_num,
                               cudaMemcpyHostToDevice, ctx->stream);
        if (cerr != cudaSuccess)
            break;
    }
    if (cerr != cudaSuccess) {
        cudaFree(d_comp_stride);
        cudaFree(d_gtli_per_band);
        return -(int)cerr;
    }

    for (uint32_t b = 0; b < bands_num_all; b++) {
        const SvtCudaFrameBandGeom& g = ctx->h_bands[b];
        if (g.band_id == BAND_NOT_EXIST || g.height == 0)
            continue;
        uint32_t threads = 256, blocks = (g.height + threads - 1) / threads;
        k_quantize_band_frame<<<blocks, threads, 0, ctx->stream>>>(ctx->d_pyramid16[g.comp_id], ctx->comp_width[g.comp_id], g.x,
                                                                    g.y, g.width, g.height, g.gcli_width,
                                                                    ctx->d_gcli_frame + g.gcli_offset, g.height_lines_num,
                                                                    d_gtli_per_band + (size_t)b * precincts_num, quant_type);
    }

    int* d_error = NULL;
    if ((cerr = cudaMalloc(&d_error, sizeof(int))) == cudaSuccess) {
        cerr = cudaMemsetAsync(d_error, 0, sizeof(int), ctx->stream);
    }
    if (cerr != cudaSuccess) {
        cudaFree(d_comp_stride);
        cudaFree(d_gtli_per_band);
        cudaFree(d_error);
        return -(int)cerr;
    }

    k_pack_precinct_frame<<<precincts_num, 1, 0, ctx->stream>>>(
        ctx->d_bands, bands_num_all, bands_num_exists, ctx->d_pyramid_ptrs, d_comp_stride, ctx->d_gcli_frame, ctx->d_sig_frame,
        ctx->d_gtli, ctx->d_pack_method, packets_num, (const svt_cuda_pack_packet_t*)ctx->d_packets, use_short_header,
        ctx->d_precinct_quantization, ctx->d_precinct_refinement, ctx->d_precinct_total_bytes, ctx->d_precinct_padding_bytes,
        ctx->d_precinct_out_offset, ctx->d_packet_size_data_bytes, ctx->d_packet_size_gcli_bytes,
        ctx->d_packet_size_significance_bytes, ctx->d_pack_out, d_error);

    int h_error = 0;
    cerr = cudaMemcpyAsync(&h_error, d_error, sizeof(int), cudaMemcpyDeviceToHost, ctx->stream);
    if (cerr == cudaSuccess)
        cerr = cudaMemcpyAsync(out_buffer, ctx->d_pack_out, running, cudaMemcpyDeviceToHost, ctx->stream);
    if (cerr == cudaSuccess)
        cerr = cudaStreamSynchronize(ctx->stream);

    cudaFree(d_comp_stride);
    cudaFree(d_gtli_per_band);
    cudaFree(d_error);

    if (cerr != cudaSuccess) {
        return -(int)cerr;
    }
    if (h_error != 0) {
        return 1;
    }
    *out_used_bytes = running;
    return 0;
}
