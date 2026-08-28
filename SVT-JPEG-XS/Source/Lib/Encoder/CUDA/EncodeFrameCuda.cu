/*
* Copyright(c) 2024 Intel Corporation
* SPDX - License - Identifier: BSD - 2 - Clause - Patent
*/
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>
#include <cstring>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include "EncodeFrameCuda.cuh"
#include "DwtCuda.cuh"

/* Opt-in sub-phase timing (SVT_CUDA_PROFILE=1 in the environment), added
 * while investigating why real-image svt_cuda_encode_frame() latency (~22ms
 * warm on 4K chart_color/eval_fantom1) is far above the Phase4c benchmark
 * figure (~7ms, measured on a single memset(0x55)-constant frame -- see
 * PortingStrategy.txt section 10 correction). Off by default: zero overhead
 * for normal callers/tests, and does not change any captured graph content
 * (all timer calls sit strictly outside the two cudaStreamBeginCapture/
 * cudaStreamEndCapture regions). */
static inline bool efc_profile_enabled() {
    static const bool enabled = (std::getenv("SVT_CUDA_PROFILE") != nullptr);
    return enabled;
}
struct EfcProfileTimer {
    std::chrono::steady_clock::time_point t0;
    void start() {
        t0 = std::chrono::steady_clock::now();
    }
    double stop_ms() {
        return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    }
};
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

/* RC LUT row (moved above section 1 so svt_cuda_frame_context_create_from_pi()
 * can pass sizeof(EfcBandLineLut) to svt_cuda_frame_context_create(), which
 * sizes ctx->d_lut/h_lut without needing to know this layout itself). */
struct EfcBandLineLut {
    uint32_t size_data_no_sign[EFC_TRUNCATION_MAX + 1];
    uint32_t cum_count[EFC_TRUNCATION_MAX + 1];
    uint32_t sig_cum[EFC_TRUNCATION_MAX + 1];
    uint32_t leftover_max;
    uint32_t leftover_extra;
};

/* Forward declaration: svt_cuda_frame_context_create_from_pi() (section 1,
 * below) needs to pass this kernel to cudaFuncSetAttribute() (to opt into a
 * larger dynamic shared-memory limit for the Phase 6 gcli-subpacket scatter
 * pass -- see that function's body) before the kernel's full definition
 * (section 7) appears later in this file. Must stay byte-for-byte identical
 * to the definition's signature. */
__global__ void k_pack_precinct_frame(const SvtCudaFrameBandGeom* bands, uint32_t bands_num_all, uint32_t bands_num_exists,
                                      uint16_t* const* pyramid_ptrs, const uint32_t* comp_stride, const uint8_t* gcli_frame,
                                      const uint8_t* sig_frame, const uint8_t* gtli_all, const uint8_t* pack_method_all,
                                      uint32_t packets_num, const svt_cuda_pack_packet_t* packets, uint8_t use_short_header,
                                      const uint8_t* quantization_all, const uint8_t* refinement_all,
                                      const uint32_t* total_bytes_all, const uint32_t* padding_bytes_all,
                                      const uint32_t* out_offset_all, const uint32_t* psd_all, const uint32_t* psg_all,
                                      const uint32_t* pss_all, const uint8_t* packet_methods_raw_all,
                                      const uint32_t* packet_offset_all, const uint32_t* packet_group_base,
                                      uint8_t* out_buffer, int* out_error);
/* NOTE: keep this declaration's parameter list identical to the definition (section 7). */

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
                                           pi->precincts_line_num, pi->packets_num, pack_out_capacity_bytes,
                                           (uint32_t)sizeof(EfcBandLineLut));
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
    /* Persistent host mirror: packets[] never changes for this context's
     * geometry, so cache it once here instead of downloading it from the
     * device on every svt_cuda_encode_frame() call. */
    ctx->h_packets = malloc(pi->packets_num * sizeof(svt_cuda_pack_packet_t));
    memcpy(ctx->h_packets, packets.data(), pi->packets_num * sizeof(svt_cuda_pack_packet_t));

    /* Phase 5: packet_size_gcli_raw_bytes[packets_num], frame-constant RAW
     * GCLI sub-packet size, matches Pi.c's precinct_info_t::packet_size_gcli_raw_bytes
     * computation (gcli_width*4 bits per band in the packet, byte-aligned).
     * Uses the NORMAL-row geometry already enforced above. */
    ctx->h_packet_size_gcli_raw_bytes = (uint32_t*)malloc((pi->packets_num ? pi->packets_num : 1) * sizeof(uint32_t));
    ctx->h_packets_exist = (uint8_t*)malloc(pi->packets_num ? pi->packets_num : 1);
    /* packets_num+1 entries: packet_group_base[p+1]-packet_group_base[p] gives
     * k_pack_precinct_frame()'s efc_pack_gcli_parallel() the group COUNT for
     * packet p (the trailing sentinel entry holds the grand total) without a
     * separate array. */
    std::vector<uint32_t> packet_group_base(pi->packets_num + 1);
    uint32_t group_running = 0;
    for (uint32_t p = 0; p < pi->packets_num; p++) {
        uint32_t raw_bits = 0;
        uint8_t exists = 0;
        packet_group_base[p] = group_running; /* Phase 6: this packet's starting slot in
                                                * k_pack_precinct_frame()'s per-block shared-
                                                * memory gcli-offset table -- purely geometric
                                                * (band widths + this packet's fixed line_idx),
                                                * so safe to compute once here. */
        for (uint32_t bidx = pi->packets[p].band_start; bidx < pi->packets[p].band_stop; bidx++) {
            const SvtCudaFrameBandGeom& g = bands[bidx];
            if (g.band_id == BAND_NOT_EXIST)
                continue;
            if (pi->packets[p].line_idx < g.height_lines_num) {
                raw_bits += g.gcli_width * 4;
                exists = 1;
                group_running += g.gcli_width;
            }
        }
        ctx->h_packet_size_gcli_raw_bytes[p] = (raw_bits + 7) >> 3;
        /* Frame-constant: packet existence depends only on packet/band
         * geometry (line_idx vs. height_lines_num), which is precinct-
         * independent under this context's NORMAL-only scope -- see
         * k_pack_precinct_frame()'s has_band check, which this replicates. */
        ctx->h_packets_exist[p] = exists;
    }
    packet_group_base[pi->packets_num] = group_running; /* sentinel: grand total group count */
    /* efc_pack_gcli_parallel() reuses ONE small shared-memory offset table
     * across packets (processed one at a time), sized to the WIDEST single
     * packet's group count rather than the sum over all packets -- an
     * earlier version sized it to the grand total (up to ~45KB/block on a
     * real 4K image) and that shared-memory footprint alone crushed
     * occupancy enough to make graph2 measurably SLOWER than before this
     * feature existed, even with the actual scatter writes disabled for
     * diagnosis (see PortingStrategy.txt Phase 6 notes) -- confirming the
     * regression was about the size of the reservation, not what ran inside
     * it. */
    uint32_t max_groups_per_packet = 0;
    for (uint32_t p = 0; p < pi->packets_num; p++) {
        uint32_t count = packet_group_base[p + 1] - packet_group_base[p];
        if (count > max_groups_per_packet) {
            max_groups_per_packet = count;
        }
    }
    ctx->gcli_scan_shared_bytes = max_groups_per_packet * (uint32_t)sizeof(uint32_t);
    err = cudaMalloc(&ctx->d_packet_group_base, (pi->packets_num + 1) * sizeof(uint32_t));
    if (err == cudaSuccess) {
        err = cudaMemcpy(ctx->d_packet_group_base, packet_group_base.data(), (pi->packets_num + 1) * sizeof(uint32_t),
                         cudaMemcpyHostToDevice);
    }
    if (err != cudaSuccess) {
        svt_cuda_frame_context_destroy(ctx);
        return -(int)err;
    }

    /* Phase 6: k_pack_precinct_frame() now needs ctx->gcli_scan_shared_bytes of
     * dynamic shared memory per block (one uint32_t slot per coefficient-group
     * across all bands of one precinct -- see FrameContextCuda.cuh). Confirm
     * this device can actually provide that much per block, and opt into the
     * larger-than-default (48KB static) limit if needed, once here rather than
     * on every svt_cuda_encode_frame() call. */
    if (ctx->gcli_scan_shared_bytes > 0) {
        int device = 0;
        int max_shared_optin = 0;
        if ((err = cudaGetDevice(&device)) != cudaSuccess ||
            (err = cudaDeviceGetAttribute(&max_shared_optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, device)) !=
                cudaSuccess) {
            svt_cuda_frame_context_destroy(ctx);
            return -(int)err;
        }
        if (ctx->gcli_scan_shared_bytes > (uint32_t)max_shared_optin) {
            /* Out of scope: this geometry needs more per-block shared memory
             * than the device can provide for the gcli-subpacket scatter
             * pass. Matches the project's existing pattern of erroring out
             * cleanly on unsupported configurations rather than silently
             * corrupting output. */
            svt_cuda_frame_context_destroy(ctx);
            return 1;
        }
        if (ctx->gcli_scan_shared_bytes > 49152 /* default static/dynamic shared mem limit on all supported archs */) {
            if ((err = cudaFuncSetAttribute(k_pack_precinct_frame, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                            (int)ctx->gcli_scan_shared_bytes)) != cudaSuccess) {
                svt_cuda_frame_context_destroy(ctx);
                return -(int)err;
            }
        }
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
static uint32_t efc_compute_budget_bytes(uint32_t bands_num, const SvtCudaFrameBandGeom* bands, const EfcBandLineLut* lut,
                                         const uint32_t* lut_row_offset, uint32_t precinct_idx, const uint8_t* gtli,
                                         uint8_t coding_significance,
                                         uint32_t packets_num, const svt_cuda_pack_packet_t* packets,
                                         std::vector<uint32_t>& pack_gcli_bits, std::vector<uint32_t>& pack_sig_bits,
                                         std::vector<uint32_t>& pack_data_bits, std::vector<uint8_t>* pack_method_out,
                                         uint8_t coding_raw_enable, const uint32_t* packet_size_gcli_raw_bytes,
                                         std::vector<uint8_t>* packet_raw_out) {
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

        uint32_t gcli_bytes = efc_bits_to_bytes(gcli_bits);
        uint32_t sig_bytes = efc_bits_to_bytes(sig_bits);
        /* Matches RateControl.c:766-780 precinct_get_budget_bytes(): RAW
         * coding replaces the significance+gcli sub-packets whenever it is
         * enabled AND smaller. This decision feeds the byte total that the
         * RC binary search itself compares against budget_bytes, not just
         * the final packing pass -- see plan file Phase 5 notes. */
        if (coding_raw_enable && (sig_bytes + gcli_bytes > packet_size_gcli_raw_bytes[p])) {
            precinct_size_bytes += packet_size_gcli_raw_bytes[p];
            if (packet_raw_out)
                (*packet_raw_out)[p] = 1;
        }
        else {
            precinct_size_bytes += gcli_bytes + sig_bytes;
            if (packet_raw_out)
                (*packet_raw_out)[p] = 0;
        }
    }
    return precinct_size_bytes;
}

/* =====================================================================
 * 6. Batched quantization (one launch per band, whole frame height).
 * ===================================================================== */

/* [2026-08-28 graph2 optimization] Was "1 thread = 1 row, serial for loop
 * over width" -- the same naive shape DWT's k_horizontal_lift had before
 * Phase 4c's tiling. Unlike DWT lifting, each coefficient here is fully
 * independent (no cross-element dependency), so this needs none of DWT's
 * shared-memory/multi-phase machinery -- a straight 1-thread-per-element 2D
 * grid (same dim3 block2d(32,8) convention as k_nlt_scale_8bit/_16bit and
 * k_image_shift in DwtCuda.cu) parallelizes the width dimension too. Math/
 * branches below are byte-for-byte unchanged from the pre-optimization
 * version; only the index computation and loop structure changed. */
__global__ void k_quantize_band_frame(uint16_t* pyramid, uint32_t stride, uint32_t bx, uint32_t by, uint32_t width,
                                      uint32_t height, uint32_t gcli_width, const uint8_t* gcli, uint32_t height_lines_num,
                                      const uint8_t* gtli_per_precinct /* [precincts_num] for this band */,
                                      uint8_t quant_type) {
    uint32_t col = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col >= width || row >= height)
        return;
    uint32_t precinct_idx = row / height_lines_num;
    uint8_t gtli = gtli_per_precinct[precinct_idx];
    if (gtli == 0)
        return;

    uint8_t g = gcli[(size_t)row * gcli_width + col / EFC_GROUP_SIZE];
    uint16_t* coeff = pyramid + (size_t)(by + row) * stride + (bx + col);
    if (g <= gtli) {
        *coeff = 0;
        return;
    }
    uint16_t sign = *coeff & EFC_SIGN_MASK;
    uint16_t mag = *coeff & ~EFC_SIGN_MASK;
    uint16_t out_mag;
    if (quant_type == 0) {
        out_mag = (uint16_t)((mag >> gtli) << gtli);
    }
    else {
        uint16_t scale_value = (uint16_t)(g - gtli + 1);
        uint16_t d = (uint16_t)(((mag << scale_value) - mag + (1 << g)) >> (g + 1));
        out_mag = (uint16_t)(d << gtli);
    }
    *coeff = out_mag ? (out_mag | sign) : 0;
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
__device__ __forceinline__ void efc_write_packet_header(EfcWriter* bw, uint32_t long_hdr, uint8_t raw_coding,
                                                         uint64_t data_size_bytes, uint64_t bitplane_count_size_bytes) {
    uint8_t* mem = bw->mem + bw->offset;
    const uint64_t sign_size_bytes = 0;
    mem[0] = (uint8_t)(raw_coding << 7);
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
/* Phase 5: RAW GCLI sub-packet, matches PackPrecinct.c's pack_bitplane_count_raw()
 * -- each band's raw (un-truncated) per-group bitplane count is written as a
 * fixed 4-bit nibble, reusing the existing efc_write_4_bits_align4() writer. */
__device__ void efc_pack_bitplane_count_raw(EfcWriter* bw, const uint8_t* bitplane, uint32_t width) {
    for (uint32_t i = 0; i < width; i++) {
        efc_write_4_bits_align4(bw, bitplane[i]);
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
/* Writes ONE packet's header into `bw` (which the caller has already
 * positioned at this packet's precomputed byte offset) and validates/skips
 * past its three sub-packets' byte spans. Split out of k_pack_precinct_frame()
 * so it can be called independently per packet-thread (see that kernel's
 * comment). The actual significance/gcli/data CONTENT is written separately
 * by efc_pack_{significance,gcli,data}_parallel() (coefficient-group-parallel
 * scatter passes, see PortingStrategy.txt Phase 6 notes) -- this function no
 * longer touches band/coefficient data at all, only the header and the
 * byte-length bookkeeping that determines where those passes' regions start.
 * Returns 0 on success, matching the *out_error codes the caller sets inline
 * (1=significance length mismatch, 2=GCLI length mismatch, 3=data length
 * mismatch). */
__device__ int efc_pack_one_packet(EfcWriter* bw, uint8_t use_short_header, uint8_t is_raw, uint32_t psd_p,
                                   uint32_t psg_p, uint32_t pss_p) {
    efc_write_packet_header(bw, !use_short_header, is_raw, psd_p, psg_p);
    efc_align_byte(bw);

    uint32_t bits_last = efc_used_bits(bw);
    if (!is_raw) {
        /* RAW packets have no significance sub-packet at all (PackPrecinct.c:316-343).
         * The actual bit content is written separately by efc_pack_significance_parallel()
         * (coefficient-group-parallel scatter pass, see PortingStrategy.txt Phase 6 pilot
         * notes) -- here we only need to advance the writer past this sub-packet's
         * (already byte-aligned) span so the gcli/data sub-packets below start at the
         * right offset. */
        bw->offset += pss_p;
    }
    efc_align_byte(bw);
    if (efc_used_bits(bw) - bits_last != pss_p * 8) {
        return 1;
    }

    bits_last = efc_used_bits(bw);
    /* The actual bit content is written separately by efc_pack_gcli_parallel()
     * (coefficient-group-parallel scatter pass, see PortingStrategy.txt
     * Phase 6 notes) -- here we only need to advance the writer past this
     * sub-packet's (already byte-aligned) span so the data sub-packet below
     * starts at the right offset. */
    bw->offset += psg_p;
    efc_align_byte(bw);
    if (efc_used_bits(bw) - bits_last != psg_p * 8) {
        return 2;
    }

    bits_last = efc_used_bits(bw);
    /* The actual bit content is written separately by efc_pack_data_parallel()
     * (coefficient-group-parallel scatter pass, see PortingStrategy.txt
     * Phase 6 notes) -- here we only need to advance the writer past this
     * (already byte-aligned) sub-packet's span. */
    bw->offset += psd_p;
    efc_align_byte(bw);
    if (efc_used_bits(bw) - bits_last != psd_p * 8) {
        return 3;
    }
    return 0;
}

/* 2026-08-28 Phase 6 pilot: coefficient-group-parallel scatter write of the
 * significance sub-packet (see PortingStrategy.txt Phase 6 pilot notes).
 * Unlike the gcli/data sub-packets (still packet-thread-serial for now, see
 * efc_pack_one_packet() above), each significance bit's value depends only
 * on its own coefficient-group (`sig_max[i] <= gtli`, fixed 1 bit/group, no
 * VLC and no inter-group state) -- so any thread can compute and write it
 * independently given the group's absolute bit offset within the
 * sub-packet. Cooperatively run by ALL threads in the block (called once per
 * precinct-block from k_pack_precinct_frame, after the per-packet loop);
 * covers every packet's significance span in the precinct. Band ownership of
 * a given bit index is resolved by a small linear scan over the packet's
 * band range (typically ~3 bands/packet) rather than a precomputed offset
 * table, since that scan is cheap at this scale and avoids adding new
 * host-computed buffers for this pilot. */
/* Sets a single bit to 1 at absolute bit position `bit_idx` (MSB-first within
 * each byte, matching efc_write_1_bit()'s convention) within `region`. CUDA
 * has no native 8-bit atomicOr, so this ORs into the containing 32-bit-
 * aligned word instead -- the other 3 byte lanes are OR'd with 0 (no-op), so
 * concurrent calls from unrelated threads/regions cannot corrupt each other;
 * it only requires the 4-byte-aligned window to stay within the output
 * allocation (see the +4 byte safety margin on d_pack_out in
 * FrameContextCuda.cu). Caller must ensure the target byte was already
 * zeroed (this only ever sets bits, never clears). Shared by
 * efc_pack_significance_parallel() and efc_pack_gcli_parallel(). */
__device__ __forceinline__ void efc_atomic_write_bit(uint8_t* region, uint32_t bit_idx) {
    uint8_t* target = region + (bit_idx >> 3);
    uint8_t mask = (uint8_t)(1u << (7 - (bit_idx & 7)));
    uint32_t* word = (uint32_t*)((uintptr_t)target & ~(uintptr_t)3);
    uint32_t byte_in_word = (uint32_t)((uintptr_t)target & 3);
    atomicOr(word, ((uint32_t)mask) << (byte_in_word * 8));
}

__device__ void efc_pack_significance_parallel(uint8_t* precinct_mem, uint32_t header_len_bytes,
                                                const svt_cuda_pack_packet_t* packets, uint32_t packets_num,
                                                const SvtCudaFrameBandGeom* bands, const uint8_t* gtli,
                                                const uint8_t* pack_method, const uint8_t* sig_frame,
                                                uint32_t precinct_idx, const uint32_t* packet_offset,
                                                const uint32_t* pss, uint8_t use_short_header) {
    uint32_t pack_header_bytes = use_short_header ? EFC_PACKET_HEADER_SHORT_SIZE_BYTES : EFC_PACKET_HEADER_LONG_SIZE_BYTES;

    /* Zero-init pass: the atomicOr scatter below only ever sets bits, so the
     * region must start at a known all-zero baseline (unlike the sequential
     * EfcWriter path, which relies on a byte's first write being a plain
     * assignment rather than an OR -- not available here since write order
     * across threads is unspecified). */
    for (uint32_t p = 0; p < packets_num; p++) {
        uint32_t sig_bytes = pss[p];
        if (sig_bytes == 0) {
            continue;
        }
        uint8_t* sig_region = precinct_mem + header_len_bytes + packet_offset[p] + pack_header_bytes;
        for (uint32_t b = threadIdx.x; b < sig_bytes; b += blockDim.x) {
            sig_region[b] = 0;
        }
    }
    __syncthreads();

    for (uint32_t p = 0; p < packets_num; p++) {
        uint32_t sig_bytes = pss[p];
        if (sig_bytes == 0) {
            continue;
        }
        uint8_t* sig_region = precinct_mem + header_len_bytes + packet_offset[p] + pack_header_bytes;
        uint32_t band_start = packets[p].band_start;
        uint32_t band_stop = packets[p].band_stop;
        uint32_t line_idx = packets[p].line_idx;
        uint32_t total_bits = sig_bytes * 8;

        for (uint32_t k = threadIdx.x; k < total_bits; k += blockDim.x) {
            uint32_t cur = 0;
            uint32_t local_group = 0;
            uint32_t owner = 0;
            int found = 0;
            for (uint32_t bidx = band_start; bidx < band_stop; bidx++) {
                const SvtCudaFrameBandGeom& bi = bands[bidx];
                if (bi.band_id == BAND_NOT_EXIST || line_idx >= bi.height_lines_num || pack_method[bidx] != 1) {
                    continue;
                }
                uint32_t width = bi.significance_width;
                if (k < cur + width) {
                    owner = bidx;
                    local_group = k - cur;
                    found = 1;
                    break;
                }
                cur += width;
            }
            if (!found) {
                continue; /* trailing byte-alignment padding bits -- stay 0 */
            }

            const SvtCudaFrameBandGeom& bi = bands[owner];
            uint32_t abs_row = precinct_idx * bi.height_lines_num + line_idx;
            const uint8_t* sig = sig_frame + bi.sig_offset + (size_t)abs_row * bi.significance_width;
            if (sig[local_group] > gtli[owner]) {
                continue; /* bit is 0 -- region already zeroed */
            }
            efc_atomic_write_bit(sig_region, k);
        }
    }
}

/* Sets a contiguous run of `nbits` one-bits starting at absolute bit position
 * `bit_off` (MSB-first, matching efc_write_1_bit()'s convention) -- one
 * atomicOr per BYTE the run touches (at most ceil(nbits/8)+1) rather than one
 * atomicOr per BIT, since efc_pack_gcli_parallel()'s VLC runs (see that
 * function's comment) can span several bytes and per-bit atomics dominated
 * measured cost when this was first tried. The inner mask-building loop is
 * pure register arithmetic (no memory access), so only the atomicOr calls
 * themselves are the expensive part being reduced. */
__device__ __forceinline__ void efc_atomic_write_run_of_ones(uint8_t* region, uint32_t bit_off, uint32_t nbits) {
    uint32_t bit_end = bit_off + nbits;
    uint32_t byte_start = bit_off >> 3;
    uint32_t byte_end = (bit_end + 7) >> 3;
    for (uint32_t byte_idx = byte_start; byte_idx < byte_end; byte_idx++) {
        uint32_t byte_bit_lo = byte_idx * 8;
        uint32_t byte_bit_hi = byte_bit_lo + 8;
        uint32_t lo = bit_off > byte_bit_lo ? bit_off : byte_bit_lo;
        uint32_t hi = bit_end < byte_bit_hi ? bit_end : byte_bit_hi;
        uint8_t mask = 0;
        for (uint32_t p = lo; p < hi; p++) {
            mask |= (uint8_t)(1u << (7 - (p & 7)));
        }
        uint8_t* target = region + byte_idx;
        uint32_t* word = (uint32_t*)((uintptr_t)target & ~(uintptr_t)3);
        uint32_t byte_in_word = (uint32_t)((uintptr_t)target & 3);
        atomicOr(word, ((uint32_t)mask) << (byte_in_word * 8));
    }
}

__device__ __forceinline__ void efc_atomic_write_nibble(uint8_t* region, uint32_t nibble_idx, uint8_t value) {
    if (value == 0) {
        return; /* region already zeroed */
    }
    uint8_t* target = region + (nibble_idx >> 1);
    uint32_t shift_in_byte = (nibble_idx & 1) ? 0u : 4u; /* efc_write_4_bits_align4(): group0 -> high nibble first */
    uint32_t* word = (uint32_t*)((uintptr_t)target & ~(uintptr_t)3);
    uint32_t byte_in_word = (uint32_t)((uintptr_t)target & 3);
    atomicOr(word, (((uint32_t)value & 0xFu) << shift_in_byte) << (byte_in_word * 8));
}

/* Phase 6: coefficient-group-parallel scatter write of the gcli(bitplane-count)
 * sub-packet. Unlike significance (fixed 1 bit/group), this sub-packet is a
 * VLC (variable-length code): efc_vlc_encode_simple()'s three branches all
 * reduce to the same rule -- write `nbits` one-bits followed by a single
 * terminating zero-bit (total length nbits+1), where
 * nbits = bitplane[i] > gtli ? bitplane[i]-gtli : 0 (0 when pack_method==1
 * and the group's 8-group significance-group is gated off, matching
 * PackPrecinct.c's pack_bitplane_count_significance()). Because the length is
 * runtime-data-dependent (unlike significance's static band geometry), each
 * group's absolute bit offset needs a real prefix-sum scan.
 *
 * [2026-08-28] This scan is now a flat GROUP-parallel exclusive scan across
 * the whole packet (not band-parallel -- an earlier band-parallel version
 * left ~5% of graph2 time on the table because a packet has only ~3 active
 * bands, so only ~3/128 threads ever did real work; see PortingStrategy.txt
 * Phase 6 notes and the plan file referenced there for the full diagnosis):
 *   Phase 1 (blocked partition, 128-way parallel): this packet's `count`
 *   groups are split into contiguous per-thread chunks. Each thread walks
 *   its own chunk (crossing band boundaries via a small cursor, since a
 *   chunk isn't guaranteed to stay inside one band), computing each group's
 *   nbits and a THREAD-LOCAL running bit offset into s_gcli_offsets, ending
 *   with its chunk's total bit length in a register.
 *   Warp scan + combine: each warp turns its 32 threads' chunk totals into
 *   an exclusive scan via __shfl_up_sync (no shared memory, no barrier);
 *   each warp's total goes into a tiny s_warp_total[] (one syncthreads to
 *   publish it); every thread then adds its warp's base (a few registers,
 *   no barrier) to fold the packet-global offset into the entries it wrote
 *   in Phase 1 -- same "<<5 into the upper bits, low 5 length bits
 *   untouched" trick the old band-parallel version used.
 *   Pass C (parallel, grid-stride over every group of every non-RAW packet):
 *   look up the precomputed offset+length and atomicOr-write the `nbits`
 *   one-bits via efc_atomic_write_bit() (the terminating zero-bit needs no
 *   write -- the region was zeroed first).
 * RAW packets (packet_methods_raw[p]==1) skip this VLC path entirely:
 * pack_bitplane_count_raw() is a fixed 4-bits/group nibble write, so -- like
 * significance -- its offset is directly derivable from band geometry alone,
 * no scan needed (handled in its own branch below, sharing this function's
 * zero-init pass since both write into the same byte region). */
__device__ void efc_pack_gcli_parallel(uint8_t* precinct_mem, uint32_t header_len_bytes,
                                        const svt_cuda_pack_packet_t* packets, uint32_t packets_num,
                                        const SvtCudaFrameBandGeom* bands, const uint8_t* gtli, const uint8_t* pack_method,
                                        const uint8_t* gcli_frame, const uint8_t* sig_frame, uint32_t precinct_idx,
                                        const uint32_t* packet_offset, const uint32_t* pss, const uint32_t* psg,
                                        const uint8_t* packet_methods_raw, const uint32_t* packet_group_base,
                                        uint8_t use_short_header, uint32_t* s_gcli_offsets) {
    uint32_t pack_header_bytes = use_short_header ? EFC_PACKET_HEADER_SHORT_SIZE_BYTES : EFC_PACKET_HEADER_LONG_SIZE_BYTES;
    /* Small STATIC shared scratch for the per-WARP chunk totals used by the
     * group-parallel scan below (one slot per warp in the block, e.g. 4 for
     * the current 128-thread launch config -- sized to 32 so this stays
     * correct if PACK_THREADS_PER_BLOCK is ever retuned up to 1024).
     * Declared here (rather than as a parameter) since it's purely an
     * implementation detail of this function, unlike s_gcli_offsets which
     * is sized per-context and must be passed in from the caller. */
    __shared__ uint32_t s_warp_total[32];
    const unsigned lane = threadIdx.x & 31u;
    const unsigned warp_id = threadIdx.x >> 5;

    /* Zero-init pass (covers both RAW and non-RAW packets' gcli regions --
     * the gcli sub-packet always starts right after the (possibly empty,
     * pss[p]==0 for RAW) significance sub-packet). */
    for (uint32_t p = 0; p < packets_num; p++) {
        uint32_t gcli_bytes = psg[p];
        if (gcli_bytes == 0) {
            continue;
        }
        uint8_t* region = precinct_mem + header_len_bytes + packet_offset[p] + pack_header_bytes + pss[p];
        for (uint32_t b = threadIdx.x; b < gcli_bytes; b += blockDim.x) {
            region[b] = 0;
        }
    }
    __syncthreads();

    /* RAW packets: fixed 4-bit/group nibble write, direct offset (no scan). */
    for (uint32_t p = 0; p < packets_num; p++) {
        if (!packet_methods_raw[p]) {
            continue;
        }
        uint32_t gcli_bytes = psg[p];
        if (gcli_bytes == 0) {
            continue;
        }
        uint8_t* region = precinct_mem + header_len_bytes + packet_offset[p] + pack_header_bytes + pss[p];
        uint32_t band_start = packets[p].band_start;
        uint32_t band_stop = packets[p].band_stop;
        uint32_t line_idx = packets[p].line_idx;
        uint32_t total_nibbles = gcli_bytes * 2;

        for (uint32_t k = threadIdx.x; k < total_nibbles; k += blockDim.x) {
            uint32_t cur = 0;
            uint32_t local_group = 0;
            uint32_t owner = 0;
            int found = 0;
            for (uint32_t bidx = band_start; bidx < band_stop; bidx++) {
                const SvtCudaFrameBandGeom& bi = bands[bidx];
                if (bi.band_id == BAND_NOT_EXIST || line_idx >= bi.height_lines_num) {
                    continue;
                }
                uint32_t width = bi.gcli_width;
                if (k < cur + width) {
                    owner = bidx;
                    local_group = k - cur;
                    found = 1;
                    break;
                }
                cur += width;
            }
            if (!found) {
                continue; /* trailing byte-alignment padding nibble -- stays 0 */
            }
            const SvtCudaFrameBandGeom& bi = bands[owner];
            uint32_t abs_row = precinct_idx * bi.height_lines_num + line_idx;
            const uint8_t* gcli = gcli_frame + bi.gcli_offset + (size_t)abs_row * bi.gcli_width;
            efc_atomic_write_nibble(region, k, gcli[local_group]);
        }
    }

    /* Non-RAW packets: processed ONE PACKET AT A TIME, reusing the same
     * small shared-memory offset table (sized to the WIDEST single packet's
     * group count, ctx->gcli_scan_shared_bytes -- see
     * svt_cuda_frame_context_create_from_pi()) rather than holding every
     * packet's groups simultaneously. An earlier version sized the table to
     * the grand total across all packets (up to ~45KB/block on a real 4K
     * image) -- that shared-memory RESERVATION alone was enough to crush
     * occupancy and make graph2 measurably slower than before this feature
     * existed, independent of what ran inside it (confirmed by disabling the
     * actual scatter writes and seeing no change -- see PortingStrategy.txt
     * Phase 6 notes).
     *
     * Phase 1+scan: a flat, GROUP-parallel exclusive scan over this packet's
     * `count` groups (see the function-level comment above for why this
     * replaced an earlier band-parallel version). nbits is bounded by
     * EFC_TRUNCATION_MAX (15) so it fits in the low 5 bits of each
     * uint32_t slot -- this lets Pass C skip re-deriving band ownership/
     * gtli/significance entirely. */
    for (uint32_t p = 0; p < packets_num; p++) {
        if (packet_methods_raw[p] || psg[p] == 0) {
            continue;
        }
        uint32_t band_start = packets[p].band_start;
        uint32_t band_stop = packets[p].band_stop;
        uint32_t line_idx = packets[p].line_idx;
        uint32_t count = packet_group_base[p + 1] - packet_group_base[p];

        /* Blocked (not round-robin) partition of [0, count) across the
         * block's threads: thread t owns the contiguous range
         * [start, end). Blocked assignment is what makes "threads before me
         * in threadIdx.x order" equal "groups before my chunk in group
         * order", so the warp-scan-and-combine below is a valid exclusive
         * prefix sum over the whole packet. */
        uint32_t elems_per_thread = (count + blockDim.x - 1) / blockDim.x;
        uint32_t start = threadIdx.x * elems_per_thread;
        uint32_t end = start + elems_per_thread;
        if (end > count) {
            end = count;
        }

        /* Locate the band (and this thread's local index within it) that
         * owns group `start`, by walking active bands accumulating their
         * gcli_width -- a packet has only a handful of bands (~3 typical),
         * so this is cheap and done once per thread, not once per group. */
        uint32_t bidx = band_start;
        uint32_t slot = 0;
        if (start < count) {
            for (;;) {
                const SvtCudaFrameBandGeom& bi = bands[bidx];
                bool active = (bi.band_id != BAND_NOT_EXIST) && (line_idx < bi.height_lines_num);
                if (active) {
                    if (slot + bi.gcli_width > start) {
                        break;
                    }
                    slot += bi.gcli_width;
                }
                bidx++;
            }
        }
        uint32_t band_local = start - slot;

        /* Walk this thread's [start, end) chunk, crossing band boundaries
         * via the cursor above as needed. Band-dependent pointers/values
         * are reloaded only when the cursor advances to a new band, not
         * per group. bit_running is THREAD-LOCAL (starts at 0); the
         * packet-global offset is folded in below after the warp scan. */
        uint32_t bit_running = 0;
        uint32_t g = start;
        while (g < end) {
            const SvtCudaFrameBandGeom& bi = bands[bidx];
            uint8_t gt = gtli[bidx];
            uint32_t abs_row = precinct_idx * bi.height_lines_num + line_idx;
            const uint8_t* gcli = gcli_frame + bi.gcli_offset + (size_t)abs_row * bi.gcli_width;
            bool method1 = (pack_method[bidx] == 1);
            const uint8_t* sig = method1 ? (sig_frame + bi.sig_offset + (size_t)abs_row * bi.significance_width) : nullptr;
            uint32_t band_remaining = bi.gcli_width - band_local;
            uint32_t chunk = end - g;
            if (chunk > band_remaining) {
                chunk = band_remaining;
            }
            for (uint32_t k = 0; k < chunk; k++) {
                uint32_t li = band_local + k;
                if (method1) {
                    if (sig[li / EFC_SIGNIFICANCE_GROUP_SIZE] > gt) {
                        uint32_t nbits = gcli[li] > gt ? (uint32_t)(gcli[li] - gt) : 0;
                        s_gcli_offsets[g + k] = (bit_running << 5) | (nbits + 1u);
                        bit_running += nbits + 1;
                    }
                    else {
                        s_gcli_offsets[g + k] = 0u; /* gated off: 0 bits contributed, nothing to write */
                    }
                }
                else {
                    uint32_t nbits = gcli[li] > gt ? (uint32_t)(gcli[li] - gt) : 0;
                    s_gcli_offsets[g + k] = (bit_running << 5) | (nbits + 1u);
                    bit_running += nbits + 1;
                }
            }
            g += chunk;
            band_local += chunk;
            if (band_local == bi.gcli_width) {
                bidx++;
                while (bidx < band_stop) {
                    const SvtCudaFrameBandGeom& bi2 = bands[bidx];
                    if (bi2.band_id != BAND_NOT_EXIST && line_idx < bi2.height_lines_num) {
                        break;
                    }
                    bidx++;
                }
                band_local = 0;
            }
        }

        /* Warp-level exclusive scan of each thread's chunk total
         * (Kogge-Stone via __shfl_up_sync, no shared memory/barrier), then
         * combine across the block's warps via the tiny s_warp_total[]. */
        uint32_t scan = bit_running;
#pragma unroll
        for (int offset = 1; offset < 32; offset <<= 1) {
            uint32_t n = __shfl_up_sync(0xFFFFFFFFu, scan, offset);
            if (lane >= (unsigned)offset) {
                scan += n;
            }
        }
        uint32_t warp_total = __shfl_sync(0xFFFFFFFFu, scan, 31);
        uint32_t warp_exclusive = scan - bit_running;
        if (lane == 0) {
            s_warp_total[warp_id] = warp_total;
        }
        __syncthreads();

        uint32_t warp_base = 0;
        for (unsigned w = 0; w < warp_id; w++) {
            warp_base += s_warp_total[w];
        }
        uint32_t delta = (warp_base + warp_exclusive) << 5;
        /* Fold the packet-global offset into only the entries this thread
         * itself wrote above -- no other thread touches [start, end), so
         * no synchronization is needed for this step beyond the barrier
         * above (which published s_warp_total). Same "<<5 into the upper
         * bits, low 5 length bits untouched" trick as before. */
        for (uint32_t gi = start; gi < end; gi++) {
            s_gcli_offsets[gi] += delta;
        }
        __syncthreads();

        /* Pass C: parallel scatter over this packet's groups only. Encoded
         * length is (nbits+1) with 0/1 meaning "nothing to write". */
        uint8_t* region = precinct_mem + header_len_bytes + packet_offset[p] + pack_header_bytes + pss[p];
        for (uint32_t local = threadIdx.x; local < count; local += blockDim.x) {
            uint32_t packed = s_gcli_offsets[local];
            uint32_t length = packed & 0x1Fu;
            if (length <= 1) {
                continue;
            }
            uint32_t bit_off = packed >> 5;
            efc_atomic_write_run_of_ones(region, bit_off, length - 1u);
        }
        __syncthreads(); /* before next packet's phase overwrites s_gcli_offsets */
    }
}

/* Phase 6: coefficient-group-parallel scatter write of the data sub-packet
 * (see PortingStrategy.txt Phase 6 notes). Unlike gcli, this doesn't depend
 * on is_raw or pack_method at all (efc_pack_one_packet() called
 * efc_pack_data() identically in every case) -- one uniform code path
 * covers everything.
 *
 * Key simplification found while porting efc_pack_data()/
 * efc_pack_data_single_group(): the trailing "leftover" group (the last
 * group of a band whose width isn't a multiple of EFC_GROUP_SIZE, written
 * bit-by-bit in the original) produces bit-for-bit the same layout as a
 * full group's nibble writes, just with `width % EFC_GROUP_SIZE` "valid
 * lanes" instead of 4 (the other lanes contribute 0, same as if they were
 * simply absent) -- so both cases share the same formulas below with a
 * single `valid_lanes` parameter, no separate leftover path needed.
 * Per active group (gclis[group] > gtli): nibble_count = gcli-gtli+1
 * (1 sign nibble + one nibble per bitplane from gcli-1 down to gtli). The
 * original's `tmp[]` left-shift state machine is just repeated "extract bit
 * at position X" -- so nibble k (0 = sign, k=1..nibble_count-1 = bitplane
 * bit_pos = gcli-k) can be computed directly, with no iteration state, from
 * the original coefficients -- ideal for Pass C's independent per-group
 * threads. Reuses the exact same shared-memory offset table as
 * efc_pack_gcli_parallel() (same group geometry, and gcli's use of it is
 * done by the time this runs) -- nibble_count fits in the low 5 bits like
 * gcli's length did (max 16). gtli[owner]+nibble_count-1 recovers gcli in
 * Pass C without re-reading gcli_frame.
 *
 * [2026-08-28] Pass A+B uses the same flat GROUP-parallel exclusive scan as
 * efc_pack_gcli_parallel() (see that function's comment for the full
 * diagnosis/rationale -- a band-parallel version left only ~3/128 threads
 * doing real work per packet). This function's version is simpler than
 * gcli's since nibble_count has no significance/pack_method branch (data
 * doesn't depend on either -- see above). Pass C (below) is unchanged: it
 * still needs its own per-group band-ownership lookup (it reads real
 * coefficient values via pyramid_ptrs, unlike gcli's Pass C), a separate,
 * out-of-scope inefficiency left for a future pass. */
__device__ void efc_pack_data_parallel(uint8_t* precinct_mem, uint32_t header_len_bytes,
                                       const svt_cuda_pack_packet_t* packets, uint32_t packets_num,
                                       const SvtCudaFrameBandGeom* bands, const uint8_t* gtli, const uint8_t* gcli_frame,
                                       uint16_t* const* pyramid_ptrs, const uint32_t* comp_stride, uint32_t precinct_idx,
                                       const uint32_t* packet_offset, const uint32_t* pss, const uint32_t* psg,
                                       const uint32_t* psd, const uint32_t* packet_group_base, uint8_t use_short_header,
                                       uint32_t* s_gcli_offsets) {
    /* Per-warp chunk totals for the group-parallel scan below -- see
     * efc_pack_gcli_parallel()'s matching declaration for the full
     * rationale. A separate instance from gcli's (not shared): the two
     * functions never run concurrently within a block (gcli's per-packet
     * loop, syncthreads included, fully completes before this one starts),
     * but each still gets its own small static shared allocation. */
    __shared__ uint32_t s_warp_total[32];
    const unsigned lane = threadIdx.x & 31u;
    const unsigned warp_id = threadIdx.x >> 5;
    uint32_t pack_header_bytes = use_short_header ? EFC_PACKET_HEADER_SHORT_SIZE_BYTES : EFC_PACKET_HEADER_LONG_SIZE_BYTES;

    for (uint32_t p = 0; p < packets_num; p++) {
        uint32_t data_bytes = psd[p];
        if (data_bytes == 0) {
            continue;
        }
        uint8_t* region = precinct_mem + header_len_bytes + packet_offset[p] + pack_header_bytes + pss[p] + psg[p];
        for (uint32_t b = threadIdx.x; b < data_bytes; b += blockDim.x) {
            region[b] = 0;
        }
    }
    __syncthreads();

    for (uint32_t p = 0; p < packets_num; p++) {
        if (psd[p] == 0) {
            continue;
        }
        uint32_t band_start = packets[p].band_start;
        uint32_t band_stop = packets[p].band_stop;
        uint32_t line_idx = packets[p].line_idx;
        uint32_t count = packet_group_base[p + 1] - packet_group_base[p];

        /* Blocked partition of [0, count) across the block's threads --
         * see efc_pack_gcli_parallel() for why blocked (not round-robin)
         * assignment is required for the warp-scan-and-combine below to be
         * a valid exclusive prefix sum. */
        uint32_t elems_per_thread = (count + blockDim.x - 1) / blockDim.x;
        uint32_t start = threadIdx.x * elems_per_thread;
        uint32_t end = start + elems_per_thread;
        if (end > count) {
            end = count;
        }

        /* Locate the band (and this thread's local index within it) that
         * owns group `start` -- see efc_pack_gcli_parallel() for the same
         * pattern. */
        uint32_t bidx = band_start;
        uint32_t slot = 0;
        if (start < count) {
            for (;;) {
                const SvtCudaFrameBandGeom& bi = bands[bidx];
                bool active = (bi.band_id != BAND_NOT_EXIST) && (line_idx < bi.height_lines_num);
                if (active) {
                    if (slot + bi.gcli_width > start) {
                        break;
                    }
                    slot += bi.gcli_width;
                }
                bidx++;
            }
        }
        uint32_t band_local = start - slot;

        /* Walk this thread's [start, end) chunk, crossing band boundaries
         * via the cursor above as needed (same structure as
         * efc_pack_gcli_parallel(), minus the significance/pack_method
         * branch -- data uses a single unconditional formula). */
        uint32_t nibble_running = 0;
        uint32_t g = start;
        while (g < end) {
            const SvtCudaFrameBandGeom& bi = bands[bidx];
            uint8_t gt = gtli[bidx];
            uint32_t abs_row = precinct_idx * bi.height_lines_num + line_idx;
            const uint8_t* gcli = gcli_frame + bi.gcli_offset + (size_t)abs_row * bi.gcli_width;
            uint32_t band_remaining = bi.gcli_width - band_local;
            uint32_t chunk = end - g;
            if (chunk > band_remaining) {
                chunk = band_remaining;
            }
            for (uint32_t k = 0; k < chunk; k++) {
                uint32_t li = band_local + k;
                uint32_t nibble_count = gcli[li] > gt ? (uint32_t)(gcli[li] - gt + 1) : 0;
                s_gcli_offsets[g + k] = (nibble_running << 5) | nibble_count;
                nibble_running += nibble_count;
            }
            g += chunk;
            band_local += chunk;
            if (band_local == bi.gcli_width) {
                bidx++;
                while (bidx < band_stop) {
                    const SvtCudaFrameBandGeom& bi2 = bands[bidx];
                    if (bi2.band_id != BAND_NOT_EXIST && line_idx < bi2.height_lines_num) {
                        break;
                    }
                    bidx++;
                }
                band_local = 0;
            }
        }

        /* Warp-level exclusive scan of each thread's chunk total, then
         * combine across the block's warps -- identical technique to
         * efc_pack_gcli_parallel(). */
        uint32_t scan = nibble_running;
#pragma unroll
        for (int offset = 1; offset < 32; offset <<= 1) {
            uint32_t n = __shfl_up_sync(0xFFFFFFFFu, scan, offset);
            if (lane >= (unsigned)offset) {
                scan += n;
            }
        }
        uint32_t warp_total = __shfl_sync(0xFFFFFFFFu, scan, 31);
        uint32_t warp_exclusive = scan - nibble_running;
        if (lane == 0) {
            s_warp_total[warp_id] = warp_total;
        }
        __syncthreads();

        uint32_t warp_base = 0;
        for (unsigned w = 0; w < warp_id; w++) {
            warp_base += s_warp_total[w];
        }
        uint32_t delta = (warp_base + warp_exclusive) << 5;
        for (uint32_t gi = start; gi < end; gi++) {
            s_gcli_offsets[gi] += delta;
        }
        __syncthreads();

        /* Pass C: parallel scatter over this packet's groups. Each thread
         * still needs band ownership (unlike gcli's Pass C) since it must
         * read the real coefficient values, not just emit a fixed bit
         * pattern. */
        uint8_t* region = precinct_mem + header_len_bytes + packet_offset[p] + pack_header_bytes + pss[p] + psg[p];
        for (uint32_t local = threadIdx.x; local < count; local += blockDim.x) {
            uint32_t packed = s_gcli_offsets[local];
            uint32_t nibble_count = packed & 0x1Fu;
            if (nibble_count == 0) {
                continue;
            }
            uint32_t nibble_off = packed >> 5;

            uint32_t cur = 0;
            uint32_t local_group = 0;
            uint32_t owner = 0;
            int found = 0;
            for (uint32_t bidx = band_start; bidx < band_stop; bidx++) {
                const SvtCudaFrameBandGeom& bi = bands[bidx];
                if (bi.band_id == BAND_NOT_EXIST || line_idx >= bi.height_lines_num) {
                    continue;
                }
                uint32_t width = bi.gcli_width;
                if (local < cur + width) {
                    owner = bidx;
                    local_group = local - cur;
                    found = 1;
                    break;
                }
                cur += width;
            }
            if (!found) {
                continue; /* defensive -- count accounting guarantees this doesn't happen */
            }

            const SvtCudaFrameBandGeom& bi = bands[owner];
            uint8_t gt = gtli[owner];
            uint32_t gcli_val = gt + nibble_count - 1; /* recovered, avoids re-reading gcli_frame */
            uint32_t valid_lanes = 4;
            if ((bi.width & 3u) != 0 && local_group == bi.gcli_width - 1) {
                valid_lanes = bi.width & 3u; /* trailing leftover group */
            }
            uint32_t abs_row = precinct_idx * bi.height_lines_num + line_idx;
            const uint16_t* coeff = pyramid_ptrs[bi.comp_id] + (size_t)(bi.y + abs_row) * comp_stride[bi.comp_id] + bi.x +
                (size_t)local_group * EFC_GROUP_SIZE;

            for (uint32_t k = 0; k < nibble_count; k++) {
                uint8_t nibble = 0;
                if (k == 0) {
                    for (uint32_t lane = 0; lane < valid_lanes; lane++) {
                        if (coeff[lane] & EFC_SIGN_MASK) {
                            nibble |= (uint8_t)(1u << (3 - lane));
                        }
                    }
                }
                else {
                    uint32_t bit_pos = gcli_val - k;
                    for (uint32_t lane = 0; lane < valid_lanes; lane++) {
                        if ((coeff[lane] >> bit_pos) & 1u) {
                            nibble |= (uint8_t)(1u << (3 - lane));
                        }
                    }
                }
                efc_atomic_write_nibble(region, nibble_off + k, nibble);
            }
        }
        __syncthreads(); /* before next packet's Pass A+B overwrites s_gcli_offsets */
    }
}

/* 2026-08-26 Pack-parallelization trial: was "1 precinct = 1 block, 1
 * thread" (fully serial within a precinct); SVT_CUDA_PROFILE-based profiling
 * on real 4K images confirmed this to be the dominant cost (~13ms/22ms warm)
 * -- see PortingStrategy.txt section 12 correction. Each packet within a
 * precinct is now independent (sub-packets are already byte-aligned, and
 * each packet's byte size/offset is precomputed on the host before this
 * kernel launches -- see FrameContextCuda.cuh's h_packet_offset comment), so
 * packets are distributed across the block's threads with no cross-thread
 * synchronization needed. Thread 0 additionally writes the (small,
 * fixed-size) precinct header/band-flags and the trailing padding, since
 * those occupy byte ranges the packet threads never touch. */
__global__ void k_pack_precinct_frame(const SvtCudaFrameBandGeom* bands, uint32_t bands_num_all, uint32_t bands_num_exists,
                                      uint16_t* const* pyramid_ptrs, const uint32_t* comp_stride, const uint8_t* gcli_frame,
                                      const uint8_t* sig_frame, const uint8_t* gtli_all, const uint8_t* pack_method_all,
                                      uint32_t packets_num, const svt_cuda_pack_packet_t* packets, uint8_t use_short_header,
                                      const uint8_t* quantization_all, const uint8_t* refinement_all,
                                      const uint32_t* total_bytes_all, const uint32_t* padding_bytes_all,
                                      const uint32_t* out_offset_all, const uint32_t* psd_all, const uint32_t* psg_all,
                                      const uint32_t* pss_all, const uint8_t* packet_methods_raw_all,
                                      const uint32_t* packet_offset_all, const uint32_t* packet_group_base,
                                      uint8_t* out_buffer, int* out_error) {
    /* Phase 6: dynamic shared memory for efc_pack_gcli_parallel()'s per-group
     * offset table -- sized by the launch site to ctx->gcli_scan_shared_bytes
     * (see svt_cuda_frame_context_create_from_pi()). Frame-constant geometry
     * (packet_group_base is the same array for every precinct-block), so no
     * per-precinct indexing is needed on it, unlike gtli/psd/psg/pss below. */
    extern __shared__ uint32_t s_gcli_offsets[];

    uint32_t precinct_idx = blockIdx.x;
    const uint8_t* gtli = gtli_all + (size_t)precinct_idx * bands_num_all;
    const uint8_t* pack_method = pack_method_all + (size_t)precinct_idx * bands_num_all;
    const uint32_t* psd = psd_all + (size_t)precinct_idx * packets_num;
    const uint32_t* psg = psg_all + (size_t)precinct_idx * packets_num;
    const uint32_t* pss = pss_all + (size_t)precinct_idx * packets_num;
    const uint8_t* packet_methods_raw = packet_methods_raw_all + (size_t)precinct_idx * packets_num;
    const uint32_t* packet_offset = packet_offset_all + (size_t)precinct_idx * packets_num;
    uint8_t* precinct_mem = out_buffer + out_offset_all[precinct_idx];

    uint32_t header_len_bytes = (EFC_PRECINCT_HEADER_SIZE_BYTES * 8 + 2 * bands_num_exists + 7) / 8;

    if (threadIdx.x == 0) {
        uint32_t pack_total_bytes = total_bytes_all[precinct_idx];
        uint32_t pack_padding_bytes = padding_bytes_all[precinct_idx];
        uint32_t packet_bytes_size = pack_total_bytes - header_len_bytes;

        EfcWriter bw;
        bw.mem = precinct_mem;
        bw.offset = 0;
        bw.bits_used = 0;
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
        /* bw.offset must now equal header_len_bytes -- guaranteed by
         * construction (EFC_PRECINCT_HEADER_SIZE_BYTES*8 + 2 bits/existing-band,
         * byte-aligned), not re-checked here since packet threads below don't
         * depend on this write completing (they use the precomputed offset). */

        if (pack_padding_bytes) {
            /* packet_bytes_size (the 24-bit header field) covers everything
             * after the precinct header INCLUDING the trailing padding
             * (that's what makes the precinct's on-disk size match its CBR
             * budget) -- so padding starts packet_bytes_size - pack_padding_bytes
             * bytes in, i.e. right after the last actual packet's data, NOT
             * at packet_bytes_size itself. */
            EfcWriter pad_bw;
            pad_bw.mem = precinct_mem + header_len_bytes + (packet_bytes_size - pack_padding_bytes);
            pad_bw.offset = 0;
            pad_bw.bits_used = 0;
            efc_add_padding_bytes(&pad_bw, pack_padding_bytes);
        }
    }

    for (uint32_t p = threadIdx.x; p < packets_num; p += blockDim.x) {
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

        EfcWriter bw;
        bw.mem = precinct_mem + header_len_bytes + packet_offset[p];
        bw.offset = 0;
        bw.bits_used = 0;

        int err = efc_pack_one_packet(&bw, use_short_header, packet_methods_raw[p], psd[p], psg[p], pss[p]);
        if (err != 0) {
            atomicExch(out_error, err);
        }
    }

    /* Phase 6 pilot: significance sub-packet content is written separately,
     * coefficient-group-parallel across the whole block (see function
     * comment above). Disjoint byte range from the packet loop above (that
     * loop no longer writes any significance content, only advances past
     * it -- see efc_pack_one_packet()), so no __syncthreads() is needed
     * between the two loops. RAW packets have pss[p]==0 and are skipped
     * inside efc_pack_significance_parallel() itself. */
    efc_pack_significance_parallel(
        precinct_mem, header_len_bytes, packets, packets_num, bands, gtli, pack_method, sig_frame, precinct_idx,
        packet_offset, pss, use_short_header);

    /* Phase 6: gcli(bitplane-count) sub-packet content, also written
     * separately (see efc_pack_gcli_parallel()'s comment). Disjoint byte
     * range from both the packet loop (advances past it only) and the
     * significance pass above, so no extra __syncthreads() is needed here --
     * efc_pack_gcli_parallel() has its own internal sync between its Pass A+B
     * and Pass C. */
    efc_pack_gcli_parallel(precinct_mem, header_len_bytes, packets, packets_num, bands, gtli, pack_method, gcli_frame,
                           sig_frame, precinct_idx, packet_offset, pss, psg, packet_methods_raw, packet_group_base,
                           use_short_header, s_gcli_offsets);

    /* Phase 6: data sub-packet content, also written separately (see
     * efc_pack_data_parallel()'s comment). Disjoint byte range from
     * everything above (packet loop only advances past it; significance and
     * gcli passes finish before this starts, both via __syncthreads()
     * inside efc_pack_gcli_parallel() and this call ordering), and reuses
     * the same s_gcli_offsets shared table now that gcli's use of it is
     * done. */
    efc_pack_data_parallel(precinct_mem, header_len_bytes, packets, packets_num, bands, gtli, gcli_frame, pyramid_ptrs,
                           comp_stride, precinct_idx, packet_offset, pss, psg, psd, packet_group_base, use_short_header,
                           s_gcli_offsets);
}

/* =====================================================================
 * 8. Top-level orchestration.
 *
 * Phase 4b-2: the GPU-only launch sequences are captured once as two
 * cudaGraphs and replayed via cudaGraphLaunch() on every call after that
 * (see FrameContextCuda.cuh's graph1/graph2 fields for the exact boundary
 * rationale). Recapture only happens when a value baked into a captured
 * node's arguments actually changes (input pointers/strides/decom for
 * graph1, quant_type/use_short_header for graph2) -- in the realistic
 * "repeated frames of the same resolution" scenario this is a one-time
 * cost on the first call.
 * ===================================================================== */

static int efc_graph1_needs_recapture(const SvtCudaFrameContext* ctx, const void* const in_planes[],
                                      const uint32_t in_stride[], uint32_t decom_h, uint32_t decom_v,
                                      uint8_t input_bit_depth, uint8_t hdr_Bw, uint8_t hdr_Fq, uint8_t coding_significance) {
    if (!ctx->graph1_captured)
        return 1;
    if (ctx->cap_decom_h != decom_h || ctx->cap_decom_v != decom_v || ctx->cap_input_bit_depth != input_bit_depth ||
        ctx->cap_hdr_Bw != hdr_Bw || ctx->cap_hdr_Fq != hdr_Fq || ctx->cap_coding_significance != coding_significance) {
        return 1;
    }
    for (uint32_t c = 0; c < ctx->comps_num; c++) {
        if (ctx->cap_in_planes[c] != in_planes[c] || ctx->cap_in_stride[c] != in_stride[c]) {
            return 1;
        }
    }
    return 0;
}

static int efc_graph2_needs_recapture(const SvtCudaFrameContext* ctx, uint8_t quant_type, uint8_t use_short_header) {
    if (!ctx->graph2_captured)
        return 1;
    return ctx->cap_quant_type != quant_type || ctx->cap_use_short_header != use_short_header;
}

int svt_cuda_encode_frame(SvtCudaFrameContext* ctx, const void* const in_planes[], const uint32_t in_stride[],
                          uint32_t decom_h, uint32_t decom_v, uint8_t input_bit_depth, uint8_t hdr_Bw, uint8_t hdr_Fq,
                          uint8_t quant_type, uint8_t use_short_header, uint8_t coding_significance, uint8_t coding_raw_enable,
                          uint32_t max_quantization, uint32_t max_refinement, const uint32_t* precinct_budget_bytes,
                          uint32_t bands_num_exists, uint32_t packets_exist_num, uint8_t* out_buffer, uint32_t* out_used_bytes) {
    uint32_t bands_num_all = ctx->bands_num_all;
    uint32_t precincts_num = ctx->precincts_num;
    uint32_t packets_num = ctx->packets_num;
    cudaError_t cerr = cudaSuccess;
    bool efc_prof = efc_profile_enabled();

    /* [2026-08-28] graph1 sub-phase measurement (SVT_CUDA_PROFILE=1 only).
     * Replaces an earlier "isolated, uncaptured" NLT+DWT probe that turned
     * out to be measuring the wrong thing: it issued ~39 individual kernel
     * launches outside any graph capture, so its number was inflated by
     * per-launch overhead that the real, graph-replayed path never pays --
     * see PortingStrategy.txt Phase 6 notes for the full correction (that
     * probe made DWT look like ~87-90% of graph1; the true share, measured
     * the way below, may be substantially lower).
     *
     * This instead captures NLT+DWT and GC+RC-LUT+D2H as two SEPARATE,
     * temporary CUDA Graphs -- each captured/instantiated/launched/synced
     * once, right before graph1's own (real) capture below -- so both
     * halves pay the same graph-replay characteristics as the real path,
     * using the exact same ctx-based (persistent-buffer), all-components,
     * real-image code paths graph1 itself uses. Only runs on the (rare)
     * recapture call, since this is diagnostic and doesn't need to repeat
     * every frame. These temporary graphs are local to this call and
     * destroyed immediately after use; ctx and its persistent
     * graph1/graph1_exec are completely untouched by this block -- the
     * real graph1 capture right after re-runs the same kernels for real
     * (redundant but deterministic, so harmless; only happens while
     * profiling). cudaEvent_t-based intra-graph timing was tried first and
     * abandoned: elapsed-time queries on events recorded via
     * cudaGraphLaunch replay returned cudaErrorInvalidValue on this
     * driver/toolkit combination for reasons not further investigated. */
    if (efc_prof && efc_graph1_needs_recapture(ctx, in_planes, in_stride, decom_h, decom_v, input_bit_depth, hdr_Bw,
                                               hdr_Fq, coding_significance)) {
        cudaGraph_t diag_g = NULL;
        cudaGraphExec_t diag_exec = NULL;
        EfcProfileTimer diag_timer;

        /* graph1a: NLT+DWT only (all components). */
        if (cudaStreamBeginCapture(ctx->stream, cudaStreamCaptureModeThreadLocal) == cudaSuccess) {
            int derr = 0;
            for (uint32_t c = 0; c < ctx->comps_num; c++) {
                derr = efc_run_dwt(ctx, c, in_planes[c], in_stride[c], decom_h, decom_v, input_bit_depth, hdr_Bw, hdr_Fq);
                if (derr != 0) {
                    break;
                }
            }
            cudaStreamEndCapture(ctx->stream, &diag_g);
            if (derr == 0 && diag_g && cudaGraphInstantiate(&diag_exec, diag_g, 0) == cudaSuccess) {
                diag_timer.start();
                if (cudaGraphLaunch(diag_exec, ctx->stream) == cudaSuccess) {
                    cudaStreamSynchronize(ctx->stream);
                }
                fprintf(stderr, "[svt_cuda_profile]   graph1 sub-measure: NLT+DWT (own graph) = %.3f ms\n",
                        diag_timer.stop_ms());
                cudaGraphExecDestroy(diag_exec);
            }
            if (diag_g) {
                cudaGraphDestroy(diag_g);
            }
        }

        /* graph1b: GC+sig + RC-LUT build + D2H copy only (reads the DWT
         * output graph1a just wrote for real above, so this operates on
         * valid data, matching what the real graph1 would compute). */
        diag_g = NULL;
        diag_exec = NULL;
        if (cudaStreamBeginCapture(ctx->stream, cudaStreamCaptureModeThreadLocal) == cudaSuccess) {
            for (uint32_t b = 0; b < bands_num_all; b++) {
                const SvtCudaFrameBandGeom& g = ctx->h_bands[b];
                if (g.band_id == BAND_NOT_EXIST || g.height == 0)
                    continue;
                uint32_t total_gc = g.height * g.gcli_width;
                uint32_t threads = 256, blocks = (total_gc + threads - 1) / threads;
                k_gc_band_frame<<<blocks, threads, 0, ctx->stream>>>(ctx->d_pyramid16[g.comp_id], ctx->comp_width[g.comp_id],
                                                                      g.x, g.y, g.width, g.height, g.gcli_width,
                                                                      ctx->d_gcli_frame + g.gcli_offset);
                uint32_t total_sig = g.height * g.significance_width;
                blocks = (total_sig + threads - 1) / threads;
                k_sig_band_frame<<<blocks, threads, 0, ctx->stream>>>(ctx->d_gcli_frame + g.gcli_offset, g.gcli_width, g.height,
                                                                       g.significance_width, ctx->d_sig_frame + g.sig_offset);
            }
            for (uint32_t b = 0; b < bands_num_all; b++) {
                const SvtCudaFrameBandGeom& g = ctx->h_bands[b];
                if (g.band_id == BAND_NOT_EXIST || g.height == 0)
                    continue;
                uint32_t threads = 256, blocks = (g.height + threads - 1) / threads;
                k_rc_build_lut_band_frame<<<blocks, threads, 0, ctx->stream>>>(
                    ctx->d_gcli_frame + g.gcli_offset, ctx->d_sig_frame + g.sig_offset, g.gcli_width, g.significance_width,
                    g.height, coding_significance, (EfcBandLineLut*)ctx->d_lut + ctx->lut_row_offset[b]);
            }
            if (ctx->lut_total_rows) {
                cudaMemcpyAsync(ctx->h_lut, ctx->d_lut, (size_t)ctx->lut_total_rows * sizeof(EfcBandLineLut),
                               cudaMemcpyDeviceToHost, ctx->stream);
            }
            cudaStreamEndCapture(ctx->stream, &diag_g);
            if (diag_g && cudaGraphInstantiate(&diag_exec, diag_g, 0) == cudaSuccess) {
                diag_timer.start();
                if (cudaGraphLaunch(diag_exec, ctx->stream) == cudaSuccess) {
                    cudaStreamSynchronize(ctx->stream);
                }
                fprintf(stderr, "[svt_cuda_profile]   graph1 sub-measure: GC+sig+RC-LUT+D2H (own graph) = %.3f ms\n",
                        diag_timer.stop_ms());
                cudaGraphExecDestroy(diag_exec);
            }
            if (diag_g) {
                cudaGraphDestroy(diag_g);
            }
        }
    }

    /* --- Graph 1: NLT+DWT (all components) + batched GC/significance (all
     * bands) + batched RC LUT build (all bands) + D2H copy of the LUT into
     * ctx->h_lut. Captured once, replayed on every call after that. --- */
    if (efc_graph1_needs_recapture(ctx, in_planes, in_stride, decom_h, decom_v, input_bit_depth, hdr_Bw, hdr_Fq,
                                   coding_significance)) {
        if (ctx->graph1_exec) {
            cudaGraphExecDestroy(ctx->graph1_exec);
            ctx->graph1_exec = NULL;
        }
        if (ctx->graph1) {
            cudaGraphDestroy(ctx->graph1);
            ctx->graph1 = NULL;
        }

        if ((cerr = cudaStreamBeginCapture(ctx->stream, cudaStreamCaptureModeThreadLocal)) != cudaSuccess) {
            return -(int)cerr;
        }

        int dwt_err = 0;
        for (uint32_t c = 0; c < ctx->comps_num; c++) {
            dwt_err = efc_run_dwt(ctx, c, in_planes[c], in_stride[c], decom_h, decom_v, input_bit_depth, hdr_Bw, hdr_Fq);
            if (dwt_err != 0) {
                break;
            }
        }

        if (dwt_err == 0) {
            for (uint32_t b = 0; b < bands_num_all; b++) {
                const SvtCudaFrameBandGeom& g = ctx->h_bands[b];
                if (g.band_id == BAND_NOT_EXIST || g.height == 0)
                    continue;
                uint32_t total_gc = g.height * g.gcli_width;
                uint32_t threads = 256, blocks = (total_gc + threads - 1) / threads;
                k_gc_band_frame<<<blocks, threads, 0, ctx->stream>>>(ctx->d_pyramid16[g.comp_id], ctx->comp_width[g.comp_id],
                                                                      g.x, g.y, g.width, g.height, g.gcli_width,
                                                                      ctx->d_gcli_frame + g.gcli_offset);
                uint32_t total_sig = g.height * g.significance_width;
                blocks = (total_sig + threads - 1) / threads;
                k_sig_band_frame<<<blocks, threads, 0, ctx->stream>>>(ctx->d_gcli_frame + g.gcli_offset, g.gcli_width, g.height,
                                                                       g.significance_width, ctx->d_sig_frame + g.sig_offset);
            }

            for (uint32_t b = 0; b < bands_num_all; b++) {
                const SvtCudaFrameBandGeom& g = ctx->h_bands[b];
                if (g.band_id == BAND_NOT_EXIST || g.height == 0)
                    continue;
                uint32_t threads = 256, blocks = (g.height + threads - 1) / threads;
                k_rc_build_lut_band_frame<<<blocks, threads, 0, ctx->stream>>>(
                    ctx->d_gcli_frame + g.gcli_offset, ctx->d_sig_frame + g.sig_offset, g.gcli_width, g.significance_width,
                    g.height, coding_significance, (EfcBandLineLut*)ctx->d_lut + ctx->lut_row_offset[b]);
            }

            if (ctx->lut_total_rows) {
                cerr = cudaMemcpyAsync(ctx->h_lut, ctx->d_lut, (size_t)ctx->lut_total_rows * sizeof(EfcBandLineLut),
                                       cudaMemcpyDeviceToHost, ctx->stream);
            }
        }

        cudaGraph_t g1 = NULL;
        cudaError_t capend = cudaStreamEndCapture(ctx->stream, &g1);
        if (dwt_err != 0) {
            if (g1) {
                cudaGraphDestroy(g1);
            }
            return dwt_err;
        }
        if (capend != cudaSuccess || cerr != cudaSuccess) {
            if (g1) {
                cudaGraphDestroy(g1);
            }
            return capend != cudaSuccess ? -(int)capend : -(int)cerr;
        }
        if ((cerr = cudaGraphInstantiate(&ctx->graph1_exec, g1, 0)) != cudaSuccess) {
            cudaGraphDestroy(g1);
            return -(int)cerr;
        }
        ctx->graph1 = g1;
        ctx->graph1_captured = 1;
        ctx->cap_decom_h = decom_h;
        ctx->cap_decom_v = decom_v;
        ctx->cap_input_bit_depth = input_bit_depth;
        ctx->cap_hdr_Bw = hdr_Bw;
        ctx->cap_hdr_Fq = hdr_Fq;
        ctx->cap_coding_significance = coding_significance;
        for (uint32_t c = 0; c < ctx->comps_num; c++) {
            ctx->cap_in_planes[c] = in_planes[c];
            ctx->cap_in_stride[c] = in_stride[c];
        }
    }

    EfcProfileTimer efc_prof_timer;
    if (efc_prof)
        efc_prof_timer.start();
    if ((cerr = cudaGraphLaunch(ctx->graph1_exec, ctx->stream)) == cudaSuccess) {
        cerr = cudaStreamSynchronize(ctx->stream); /* ctx->h_lut must be ready before the host RC loop reads it */
    }
    if (efc_prof)
        fprintf(stderr, "[svt_cuda_profile] graph1 (DWT+NLT+GC+RC-LUT) launch+sync: %.3f ms\n", efc_prof_timer.stop_ms());
    if (cerr != cudaSuccess) {
        return -(int)cerr;
    }

    /* --- Step 4: host-side per-precinct binary search (RC), sequential across
     * precincts but each precinct only does cheap LUT lookups -- the expensive
     * histogram build already ran batched on GPU in graph1. Reads/writes
     * ctx's persistent pinned buffers directly (fixed addresses, required
     * since they are also H2D sources captured inside graph2 below). --- */
    uint32_t pack_header_bits =
        use_short_header ? (EFC_PACKET_HEADER_SHORT_SIZE_BYTES * 8) : (EFC_PACKET_HEADER_LONG_SIZE_BYTES * 8);
    uint32_t headers_bytes = efc_bits_to_bytes(efc_align8(EFC_PRECINCT_HEADER_SIZE_BYTES * 8 + bands_num_exists * 2)) +
        efc_bits_to_bytes(efc_align8(pack_header_bits * packets_exist_num));

    const EfcBandLineLut* lut = (const EfcBandLineLut*)ctx->h_lut;
    const svt_cuda_pack_packet_t* h_packets = (const svt_cuda_pack_packet_t*)ctx->h_packets;

    std::vector<uint8_t> gtli(bands_num_all);
    std::vector<uint32_t> pgb(bands_num_all * 4), psb(bands_num_all * 4), pdb(bands_num_all * 4);

    if (efc_prof)
        efc_prof_timer.start();
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
            uint32_t total = efc_compute_budget_bytes(bands_num_all, ctx->h_bands, lut, ctx->lut_row_offset, pr, gtli.data(),
                                                      coding_significance, packets_num, h_packets, pgb, psb, pdb, NULL,
                                                      coding_raw_enable, ctx->h_packet_size_gcli_raw_bytes, NULL);
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
            uint32_t total = efc_compute_budget_bytes(bands_num_all, ctx->h_bands, lut, ctx->lut_row_offset, pr, gtli.data(),
                                                      coding_significance, packets_num, h_packets, pgb, psb, pdb, NULL,
                                                      coding_raw_enable, ctx->h_packet_size_gcli_raw_bytes, NULL);
            step = (total <= budget_to_data_bytes) ? EFC_STEP_TOO_SMALL : EFC_STEP_TOO_BIG;
        }
        if (!found_r) {
            return 1;
        }

        std::vector<uint8_t> pack_method(bands_num_all, 0);
        std::vector<uint8_t> packet_raw(packets_num, 0);
        efc_compute_all_truncation(bands_num_all, ctx->h_bands, quantization, refinement, gtli.data());
        uint32_t data_bytes = efc_compute_budget_bytes(bands_num_all, ctx->h_bands, lut, ctx->lut_row_offset, pr, gtli.data(),
                                                       coding_significance, packets_num, h_packets, pgb, psb, pdb, &pack_method,
                                                       coding_raw_enable, ctx->h_packet_size_gcli_raw_bytes, &packet_raw);

        memcpy(&ctx->h_gtli[(size_t)pr * bands_num_all], gtli.data(), bands_num_all);
        memcpy(&ctx->h_pack_method[(size_t)pr * bands_num_all], pack_method.data(), bands_num_all);
        for (uint32_t b = 0; b < bands_num_all; b++) {
            ctx->h_gtli_per_band[(size_t)b * precincts_num + pr] = gtli[b];
        }
        ctx->h_quant[pr] = (uint8_t)quantization;
        ctx->h_refine[pr] = (uint8_t)refinement;
        ctx->h_total_bytes[pr] = budget_bytes;
        ctx->h_padding_bytes[pr] = budget_to_data_bytes - data_bytes;

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
            ctx->h_psd[(size_t)pr * packets_num + p] = efc_bits_to_bytes(data_bits);
            /* Apply the same RAW-vs-normal decision efc_compute_budget_bytes()
             * already made for this packet (packet_raw[p]), so the per-packet
             * byte breakdown used by the header writer / pack kernel stays
             * consistent with the totals used for the RC budget check above. */
            ctx->h_packet_methods_raw[(size_t)pr * packets_num + p] = packet_raw[p];
            if (packet_raw[p]) {
                ctx->h_psg[(size_t)pr * packets_num + p] = ctx->h_packet_size_gcli_raw_bytes[p];
                ctx->h_pss[(size_t)pr * packets_num + p] = 0;
            }
            else {
                ctx->h_psg[(size_t)pr * packets_num + p] = efc_bits_to_bytes(gcli_bits);
                ctx->h_pss[(size_t)pr * packets_num + p] = efc_bits_to_bytes(sig_bits);
            }
        }

        /* Pack-parallelization: each existing packet's byte offset relative
         * to the end of the (fixed-size) precinct header, so k_pack_precinct_frame()
         * can hand each packet to an independent thread with no cross-thread
         * synchronization (see FrameContextCuda.cuh's h_packet_offset comment). */
        {
            uint32_t running_offset = 0;
            uint32_t pack_header_bytes = pack_header_bits >> 3;
            for (uint32_t p = 0; p < packets_num; p++) {
                ctx->h_packet_offset[(size_t)pr * packets_num + p] = running_offset;
                if (ctx->h_packets_exist[p]) {
                    running_offset += pack_header_bytes + ctx->h_psd[(size_t)pr * packets_num + p] +
                        ctx->h_psg[(size_t)pr * packets_num + p] + ctx->h_pss[(size_t)pr * packets_num + p];
                }
            }
        }
    }
    if (efc_prof)
        fprintf(stderr, "[svt_cuda_profile] host RC binary search (%u precincts, %u packets/precinct, %u bands/precinct): %.3f ms\n",
                precincts_num, packets_num, bands_num_all, efc_prof_timer.stop_ms());

    uint32_t running = 0;
    for (uint32_t pr = 0; pr < precincts_num; pr++) {
        ctx->h_out_offset[pr] = running;
        running += ctx->h_total_bytes[pr];
    }
    if (running > ctx->pack_out_capacity_bytes) {
        return 1;
    }

    /* [2026-08-28] graph2 sub-phase measurement (SVT_CUDA_PROFILE=1 only),
     * same technique as the graph1 sub-measure above: two SEPARATE, temporary
     * CUDA Graphs (H2D+quantize, then pack) captured/instantiated/launched/
     * synced once right before graph2's own (real) capture below, so both
     * halves pay the same graph-replay characteristics as the real path.
     * Quantize (k_quantize_band_frame) zeroes below-threshold coefficients
     * in d_pyramid16 in place; running it here for real is idempotent (the
     * real graph2 capture below re-runs the identical, deterministic
     * zeroing), matching the graph1a/b "redundant but harmless" precedent.
     * Only runs on the (rare) recapture call. */
    if (efc_prof && efc_graph2_needs_recapture(ctx, quant_type, use_short_header)) {
        cudaGraph_t diag_g = NULL;
        cudaGraphExec_t diag_exec = NULL;
        EfcProfileTimer diag_timer;
        size_t pb_bytes = (size_t)precincts_num * bands_num_all;
        size_t pp_elems = (size_t)precincts_num * packets_num;

        /* graph2a: H2D upload of RC results (all of it -- pack needs most of
         * these buffers too, and uploading them here mirrors what the real
         * graph2 capture does) + batched quantize (all bands) only. */
        if (cudaStreamBeginCapture(ctx->stream, cudaStreamCaptureModeThreadLocal) == cudaSuccess) {
            cudaError_t derr = cudaMemcpyAsync(ctx->d_gtli, ctx->h_gtli, pb_bytes, cudaMemcpyHostToDevice, ctx->stream);
            if (derr == cudaSuccess)
                derr = cudaMemcpyAsync(ctx->d_pack_method, ctx->h_pack_method, pb_bytes, cudaMemcpyHostToDevice, ctx->stream);
            if (derr == cudaSuccess)
                derr = cudaMemcpyAsync(
                    ctx->d_packet_methods_raw, ctx->h_packet_methods_raw, pp_elems, cudaMemcpyHostToDevice, ctx->stream);
            if (derr == cudaSuccess)
                derr = cudaMemcpyAsync(ctx->d_packet_offset, ctx->h_packet_offset, pp_elems * sizeof(uint32_t),
                                       cudaMemcpyHostToDevice, ctx->stream);
            if (derr == cudaSuccess)
                derr = cudaMemcpyAsync(
                    ctx->d_packet_size_data_bytes, ctx->h_psd, pp_elems * sizeof(uint32_t), cudaMemcpyHostToDevice, ctx->stream);
            if (derr == cudaSuccess)
                derr = cudaMemcpyAsync(
                    ctx->d_packet_size_gcli_bytes, ctx->h_psg, pp_elems * sizeof(uint32_t), cudaMemcpyHostToDevice, ctx->stream);
            if (derr == cudaSuccess)
                derr = cudaMemcpyAsync(ctx->d_packet_size_significance_bytes, ctx->h_pss, pp_elems * sizeof(uint32_t),
                                       cudaMemcpyHostToDevice, ctx->stream);
            if (derr == cudaSuccess)
                derr = cudaMemcpyAsync(ctx->d_precinct_quantization, ctx->h_quant, precincts_num, cudaMemcpyHostToDevice, ctx->stream);
            if (derr == cudaSuccess)
                derr = cudaMemcpyAsync(ctx->d_precinct_refinement, ctx->h_refine, precincts_num, cudaMemcpyHostToDevice, ctx->stream);
            if (derr == cudaSuccess)
                derr = cudaMemcpyAsync(ctx->d_precinct_total_bytes, ctx->h_total_bytes, precincts_num * sizeof(uint32_t),
                                       cudaMemcpyHostToDevice, ctx->stream);
            if (derr == cudaSuccess)
                derr = cudaMemcpyAsync(ctx->d_precinct_padding_bytes, ctx->h_padding_bytes, precincts_num * sizeof(uint32_t),
                                       cudaMemcpyHostToDevice, ctx->stream);
            if (derr == cudaSuccess)
                derr = cudaMemcpyAsync(ctx->d_precinct_out_offset, ctx->h_out_offset, precincts_num * sizeof(uint32_t),
                                       cudaMemcpyHostToDevice, ctx->stream);
            if (derr == cudaSuccess)
                derr = cudaMemcpyAsync(ctx->d_gtli_per_band, ctx->h_gtli_per_band, pb_bytes, cudaMemcpyHostToDevice, ctx->stream);
            if (derr == cudaSuccess) {
                for (uint32_t b = 0; b < bands_num_all; b++) {
                    const SvtCudaFrameBandGeom& g = ctx->h_bands[b];
                    if (g.band_id == BAND_NOT_EXIST || g.height == 0)
                        continue;
                    dim3 quant_block2d(32, 8);
                    dim3 quant_grid2d((g.width + quant_block2d.x - 1) / quant_block2d.x,
                                      (g.height + quant_block2d.y - 1) / quant_block2d.y);
                    k_quantize_band_frame<<<quant_grid2d, quant_block2d, 0, ctx->stream>>>(
                        ctx->d_pyramid16[g.comp_id], ctx->comp_width[g.comp_id], g.x, g.y, g.width, g.height, g.gcli_width,
                        ctx->d_gcli_frame + g.gcli_offset, g.height_lines_num, ctx->d_gtli_per_band + (size_t)b * precincts_num,
                        quant_type);
                }
            }
            cudaStreamEndCapture(ctx->stream, &diag_g);
            if (derr == cudaSuccess && diag_g && cudaGraphInstantiate(&diag_exec, diag_g, 0) == cudaSuccess) {
                diag_timer.start();
                if (cudaGraphLaunch(diag_exec, ctx->stream) == cudaSuccess) {
                    cudaStreamSynchronize(ctx->stream);
                }
                fprintf(stderr, "[svt_cuda_profile]   graph2 sub-measure: H2D+quantize (own graph) = %.3f ms\n",
                        diag_timer.stop_ms());
                cudaGraphExecDestroy(diag_exec);
            }
            if (diag_g) {
                cudaGraphDestroy(diag_g);
            }
        }

        /* graph2a2: quantize only, further isolating graph2a's H2D-vs-kernel
         * split (reuses the buffers graph2a already uploaded for real just
         * above -- re-running quantize is idempotent, see the block comment
         * at the top of this diagnostic section). */
        diag_g = NULL;
        diag_exec = NULL;
        if (cudaStreamBeginCapture(ctx->stream, cudaStreamCaptureModeThreadLocal) == cudaSuccess) {
            for (uint32_t b = 0; b < bands_num_all; b++) {
                const SvtCudaFrameBandGeom& g = ctx->h_bands[b];
                if (g.band_id == BAND_NOT_EXIST || g.height == 0)
                    continue;
                dim3 quant_block2d(32, 8);
                dim3 quant_grid2d((g.width + quant_block2d.x - 1) / quant_block2d.x,
                                  (g.height + quant_block2d.y - 1) / quant_block2d.y);
                k_quantize_band_frame<<<quant_grid2d, quant_block2d, 0, ctx->stream>>>(
                    ctx->d_pyramid16[g.comp_id], ctx->comp_width[g.comp_id], g.x, g.y, g.width, g.height, g.gcli_width,
                    ctx->d_gcli_frame + g.gcli_offset, g.height_lines_num, ctx->d_gtli_per_band + (size_t)b * precincts_num,
                    quant_type);
            }
            cudaStreamEndCapture(ctx->stream, &diag_g);
            if (diag_g && cudaGraphInstantiate(&diag_exec, diag_g, 0) == cudaSuccess) {
                diag_timer.start();
                if (cudaGraphLaunch(diag_exec, ctx->stream) == cudaSuccess) {
                    cudaStreamSynchronize(ctx->stream);
                }
                fprintf(stderr, "[svt_cuda_profile]   graph2 sub-measure: quantize only (own graph, %u band launches) = %.3f ms\n",
                        bands_num_all, diag_timer.stop_ms());
                cudaGraphExecDestroy(diag_exec);
            }
            if (diag_g) {
                cudaGraphDestroy(diag_g);
            }
        }

        /* graph2b: pack only (reads the now-quantized d_pyramid16 and the
         * RC-result buffers graph2a just uploaded for real above). */
        diag_g = NULL;
        diag_exec = NULL;
        if (cudaStreamBeginCapture(ctx->stream, cudaStreamCaptureModeThreadLocal) == cudaSuccess) {
            cudaMemsetAsync(ctx->d_error, 0, sizeof(int), ctx->stream);
            const uint32_t PACK_THREADS_PER_BLOCK = 128;
            k_pack_precinct_frame<<<precincts_num, PACK_THREADS_PER_BLOCK, ctx->gcli_scan_shared_bytes, ctx->stream>>>(
                ctx->d_bands, bands_num_all, bands_num_exists, ctx->d_pyramid_ptrs, ctx->d_comp_stride, ctx->d_gcli_frame,
                ctx->d_sig_frame, ctx->d_gtli, ctx->d_pack_method, packets_num, (const svt_cuda_pack_packet_t*)ctx->d_packets,
                use_short_header, ctx->d_precinct_quantization, ctx->d_precinct_refinement, ctx->d_precinct_total_bytes,
                ctx->d_precinct_padding_bytes, ctx->d_precinct_out_offset, ctx->d_packet_size_data_bytes,
                ctx->d_packet_size_gcli_bytes, ctx->d_packet_size_significance_bytes, ctx->d_packet_methods_raw,
                ctx->d_packet_offset, ctx->d_packet_group_base, ctx->d_pack_out, ctx->d_error);
            cudaStreamEndCapture(ctx->stream, &diag_g);
            if (diag_g && cudaGraphInstantiate(&diag_exec, diag_g, 0) == cudaSuccess) {
                diag_timer.start();
                if (cudaGraphLaunch(diag_exec, ctx->stream) == cudaSuccess) {
                    cudaStreamSynchronize(ctx->stream);
                }
                fprintf(stderr, "[svt_cuda_profile]   graph2 sub-measure: pack (own graph) = %.3f ms\n", diag_timer.stop_ms());
                cudaGraphExecDestroy(diag_exec);
            }
            if (diag_g) {
                cudaGraphDestroy(diag_g);
            }
        }
    }

    /* --- Graph 2: H2D upload of RC results, batched quantize (all bands),
     * batched pack (all precincts), D2H copy of the pack error flag.
     * Captured once, replayed on every call after that. --- */
    if (efc_graph2_needs_recapture(ctx, quant_type, use_short_header)) {
        if (ctx->graph2_exec) {
            cudaGraphExecDestroy(ctx->graph2_exec);
            ctx->graph2_exec = NULL;
        }
        if (ctx->graph2) {
            cudaGraphDestroy(ctx->graph2);
            ctx->graph2 = NULL;
        }

        if ((cerr = cudaStreamBeginCapture(ctx->stream, cudaStreamCaptureModeThreadLocal)) != cudaSuccess) {
            return -(int)cerr;
        }

        size_t pb_bytes = (size_t)precincts_num * bands_num_all;
        size_t pp_elems = (size_t)precincts_num * packets_num;
        cerr = cudaMemcpyAsync(ctx->d_gtli, ctx->h_gtli, pb_bytes, cudaMemcpyHostToDevice, ctx->stream);
        if (cerr == cudaSuccess)
            cerr = cudaMemcpyAsync(ctx->d_pack_method, ctx->h_pack_method, pb_bytes, cudaMemcpyHostToDevice, ctx->stream);
        if (cerr == cudaSuccess)
            cerr = cudaMemcpyAsync(
                ctx->d_packet_methods_raw, ctx->h_packet_methods_raw, pp_elems, cudaMemcpyHostToDevice, ctx->stream);
        if (cerr == cudaSuccess)
            cerr = cudaMemcpyAsync(ctx->d_packet_offset, ctx->h_packet_offset, pp_elems * sizeof(uint32_t),
                                   cudaMemcpyHostToDevice, ctx->stream);
        if (cerr == cudaSuccess)
            cerr = cudaMemcpyAsync(
                ctx->d_packet_size_data_bytes, ctx->h_psd, pp_elems * sizeof(uint32_t), cudaMemcpyHostToDevice, ctx->stream);
        if (cerr == cudaSuccess)
            cerr = cudaMemcpyAsync(
                ctx->d_packet_size_gcli_bytes, ctx->h_psg, pp_elems * sizeof(uint32_t), cudaMemcpyHostToDevice, ctx->stream);
        if (cerr == cudaSuccess)
            cerr = cudaMemcpyAsync(ctx->d_packet_size_significance_bytes, ctx->h_pss, pp_elems * sizeof(uint32_t),
                                   cudaMemcpyHostToDevice, ctx->stream);
        if (cerr == cudaSuccess)
            cerr = cudaMemcpyAsync(ctx->d_precinct_quantization, ctx->h_quant, precincts_num, cudaMemcpyHostToDevice, ctx->stream);
        if (cerr == cudaSuccess)
            cerr = cudaMemcpyAsync(ctx->d_precinct_refinement, ctx->h_refine, precincts_num, cudaMemcpyHostToDevice, ctx->stream);
        if (cerr == cudaSuccess)
            cerr = cudaMemcpyAsync(ctx->d_precinct_total_bytes, ctx->h_total_bytes, precincts_num * sizeof(uint32_t),
                                   cudaMemcpyHostToDevice, ctx->stream);
        if (cerr == cudaSuccess)
            cerr = cudaMemcpyAsync(ctx->d_precinct_padding_bytes, ctx->h_padding_bytes, precincts_num * sizeof(uint32_t),
                                   cudaMemcpyHostToDevice, ctx->stream);
        if (cerr == cudaSuccess)
            cerr = cudaMemcpyAsync(ctx->d_precinct_out_offset, ctx->h_out_offset, precincts_num * sizeof(uint32_t),
                                   cudaMemcpyHostToDevice, ctx->stream);
        if (cerr == cudaSuccess)
            cerr = cudaMemcpyAsync(ctx->d_gtli_per_band, ctx->h_gtli_per_band, pb_bytes, cudaMemcpyHostToDevice, ctx->stream);

        if (cerr == cudaSuccess) {
            for (uint32_t b = 0; b < bands_num_all; b++) {
                const SvtCudaFrameBandGeom& g = ctx->h_bands[b];
                if (g.band_id == BAND_NOT_EXIST || g.height == 0)
                    continue;
                dim3 quant_block2d(32, 8);
                dim3 quant_grid2d((g.width + quant_block2d.x - 1) / quant_block2d.x,
                                  (g.height + quant_block2d.y - 1) / quant_block2d.y);
                k_quantize_band_frame<<<quant_grid2d, quant_block2d, 0, ctx->stream>>>(
                    ctx->d_pyramid16[g.comp_id], ctx->comp_width[g.comp_id], g.x, g.y, g.width, g.height, g.gcli_width,
                    ctx->d_gcli_frame + g.gcli_offset, g.height_lines_num, ctx->d_gtli_per_band + (size_t)b * precincts_num,
                    quant_type);
            }

            cerr = cudaMemsetAsync(ctx->d_error, 0, sizeof(int), ctx->stream);
        }

        if (cerr == cudaSuccess) {
            /* Pack-parallelization trial: one warp per precinct block, each
             * thread taking packets round-robin (packets_num is typically
             * well under 32 -- see plan file -- so most threads handle at
             * most one packet). */
            const uint32_t PACK_THREADS_PER_BLOCK = 128;
            k_pack_precinct_frame<<<precincts_num, PACK_THREADS_PER_BLOCK, ctx->gcli_scan_shared_bytes, ctx->stream>>>(
                ctx->d_bands, bands_num_all, bands_num_exists, ctx->d_pyramid_ptrs, ctx->d_comp_stride, ctx->d_gcli_frame,
                ctx->d_sig_frame, ctx->d_gtli, ctx->d_pack_method, packets_num, (const svt_cuda_pack_packet_t*)ctx->d_packets,
                use_short_header, ctx->d_precinct_quantization, ctx->d_precinct_refinement, ctx->d_precinct_total_bytes,
                ctx->d_precinct_padding_bytes, ctx->d_precinct_out_offset, ctx->d_packet_size_data_bytes,
                ctx->d_packet_size_gcli_bytes, ctx->d_packet_size_significance_bytes, ctx->d_packet_methods_raw,
                ctx->d_packet_offset, ctx->d_packet_group_base, ctx->d_pack_out, ctx->d_error);

            cerr = cudaMemcpyAsync(ctx->h_error, ctx->d_error, sizeof(int), cudaMemcpyDeviceToHost, ctx->stream);
        }

        cudaGraph_t g2 = NULL;
        cudaError_t capend2 = cudaStreamEndCapture(ctx->stream, &g2);
        if (capend2 != cudaSuccess || cerr != cudaSuccess) {
            if (g2) {
                cudaGraphDestroy(g2);
            }
            return capend2 != cudaSuccess ? -(int)capend2 : -(int)cerr;
        }
        if ((cerr = cudaGraphInstantiate(&ctx->graph2_exec, g2, 0)) != cudaSuccess) {
            cudaGraphDestroy(g2);
            return -(int)cerr;
        }
        ctx->graph2 = g2;
        ctx->graph2_captured = 1;
        ctx->cap_quant_type = quant_type;
        ctx->cap_use_short_header = use_short_header;
    }

    if (efc_prof)
        efc_prof_timer.start();
    if ((cerr = cudaGraphLaunch(ctx->graph2_exec, ctx->stream)) == cudaSuccess) {
        /* Final bitstream size varies with frame content, so this D2H copy
         * (unlike everything above) is not captured -- its destination is
         * also caller-supplied and may differ across calls. */
        cerr = cudaMemcpyAsync(out_buffer, ctx->d_pack_out, running, cudaMemcpyDeviceToHost, ctx->stream);
    }
    if (cerr == cudaSuccess) {
        cerr = cudaStreamSynchronize(ctx->stream);
    }
    if (efc_prof)
        fprintf(stderr, "[svt_cuda_profile] graph2 (quantize+pack) launch+sync: %.3f ms (gcli_scan_shared_bytes=%u)\n",
               efc_prof_timer.stop_ms(), ctx->gcli_scan_shared_bytes);
    if (cerr != cudaSuccess) {
        return -(int)cerr;
    }
    if (*ctx->h_error != 0) {
        return 1;
    }
    *out_used_bytes = running;
    return 0;
}
