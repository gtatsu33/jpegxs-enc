/*
* Copyright(c) 2024 Intel Corporation
* SPDX - License - Identifier: BSD - 2 - Clause - Patent
*/
/*
* CPU (C reference) vs CUDA benchmarking harness.
*
* This tool exists to track, phase by phase, how each ported module's
* execution time compares between the original scalar C implementation and
* its CUDA port (see PortingStrategy.txt).
*
* GPU entries below use CpuTimer (wall-clock) to wrap the whole
* svt_cuda_*() call, i.e. host-visible latency including cudaMalloc/H2D/
* kernel/D2H -- not pure kernel-only time (no cudaEvent breakdown yet,
* since svt_cuda_dwt_component()/svt_cuda_nlt_scale_component() don't
* expose that internally). This is intentionally the more pessimistic of
* the two numbers PortingStrategy.txt commits to tracking; the pure-kernel
* breakdown is deferred to when the CUDA Graph / persistent-buffer design
* lands (Phase 4), at which point host-visible latency becomes the
* meaningful number anyway.
*/
#ifndef NOMINMAX
#define NOMINMAX /* EncHandle.h (Phase 4a) transitively pulls in windows.h; without this its min/max
                  * macros break std::min/std::max calls elsewhere in this file. */
#endif
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "BenchReport.h"
#include "GcStageProcess.h"
#include "NltEnc.h"
#include "common_dsp_rtcd.h"
#include "encoder_dsp_rtcd.h"
#include "Pi.h"
#include "PiEnc.h"
#include "WeightTable.h"
#include "PrecinctEnc.h"
#include "PictureControlSet.h"
#include "RateControl.h"
#include "QuantStageProcess.h"
#include "PackPrecinct.h"
#include "BitstreamWriter.h"
#include "SvtType.h"
#include "Codestream.h"

#ifdef SVT_ENABLE_CUDA
#include "CudaSmokeTest.cuh"
#include "NltCuda.cuh"
#include "DwtCuda.cuh"
#include "GcCuda.cuh"
#include "RcQuantCuda.cuh"
#include "PackCuda.cuh"
#include "EncodeFrameCuda.cuh"
#include "SvtJpegxs.h"
#include "SvtJpegxsEnc.h"
#include "SvtJpegxsImageBufferTools.h"
#include "EncHandle.h"
#endif

static void bench_gc_cpu_reference(BenchReport& report, const std::string& phase, uint32_t width, int iterations) {
    std::vector<uint16_t> coeff(width);
    std::vector<uint8_t> gcli((width + GROUP_SIZE - 1) / GROUP_SIZE);
    for (uint32_t i = 0; i < width; i++) {
        coeff[i] = (uint16_t)((i * 2654435761u) & 0x7fff);
    }

    BenchRecord r = bench_cpu(phase, "GC(CPU reference)", iterations, [&]() {
        gc_precinct_stage_scalar_c(gcli.data(), coeff.data(), GROUP_SIZE, width);
    });
    report.add(r);
}

#ifdef SVT_ENABLE_CUDA
template <typename Fn>
static BenchRecord bench_gpu_hostlatency(const std::string& phase, const std::string& module, int iterations, Fn fn) {
    fn(); // warm-up (not counted)
    double sum = 0.0, mn = -1.0, mx = 0.0;
    for (int i = 0; i < iterations; i++) {
        CpuTimer timer;
        timer.start();
        fn();
        double ms = timer.stop_ms();
        sum += ms;
        if (mn < 0 || ms < mn)
            mn = ms;
        if (ms > mx)
            mx = ms;
    }
    BenchRecord r;
    r.phase = phase;
    r.module = module;
    r.backend = "GPU";
    r.iterations = iterations;
    r.mean_ms = sum / iterations;
    r.min_ms = mn;
    r.max_ms = mx;
    return r;
}
#endif

static void bench_phase1_nlt_dwt_gc(BenchReport& report, uint32_t width, uint32_t height, int iterations) {
    const uint8_t input_bit_depth = 10;
    const uint8_t hdr_Bw = 20;
    const uint8_t hdr_Fq = 8;
    const uint8_t shift = hdr_Bw - input_bit_depth;
    const int32_t offset = 1 << (hdr_Bw - 1);
    size_t elems = (size_t)width * height;

    std::vector<uint16_t> src(elems);
    for (size_t i = 0; i < elems; i++) {
        src[i] = (uint16_t)((i * 2654435761u) & 0x3ff);
    }

    // NLT: CPU reference, one row at a time (matches the real pipeline's call pattern).
    std::vector<int32_t> nlt_out(elems);
    report.add(bench_cpu("Phase1-NLT", "NLT(CPU reference)", iterations, [&]() {
        for (uint32_t y = 0; y < height; y++) {
            linear_input_scaling_line_16bit_c(
                src.data() + (size_t)y * width, nlt_out.data() + (size_t)y * width, width, shift, offset, input_bit_depth);
        }
    }));

    // GC: CPU reference, one band line at a time (mirrors the Phase0-Proof workload,
    // reusing NLT's output magnitude range as synthetic coefficient data).
    std::vector<uint16_t> coeff(width);
    for (uint32_t i = 0; i < width; i++) {
        coeff[i] = (uint16_t)((i * 2654435761u) & 0x7fff);
    }
    std::vector<uint8_t> gcli((width + GROUP_SIZE - 1) / GROUP_SIZE);
    report.add(bench_cpu("Phase1-GC", "GC(CPU reference)", iterations, [&]() {
        gc_precinct_stage_scalar_c(gcli.data(), coeff.data(), GROUP_SIZE, width);
    }));

#ifdef SVT_ENABLE_CUDA
    std::vector<int32_t> nlt_out_cuda(elems);
    report.add(bench_gpu_hostlatency("Phase1-NLT", "NLT(CUDA)", iterations, [&]() {
        svt_cuda_nlt_scale_component(src.data(), width, width, height, input_bit_depth, hdr_Bw, nlt_out_cuda.data());
    }));

    std::vector<uint8_t> gcli_cuda((width + GROUP_SIZE - 1) / GROUP_SIZE);
    report.add(bench_gpu_hostlatency("Phase1-GC", "GC(CUDA)", iterations, [&]() {
        svt_cuda_gc_precinct_stage_scalar(gcli_cuda.data(), coeff.data(), GROUP_SIZE, width);
    }));

    std::vector<uint16_t> dwt_out_cuda(elems);
    report.add(bench_gpu_hostlatency("Phase1-DWT", "NLT+DWT(CUDA)", iterations, [&]() {
        svt_cuda_dwt_component(src.data(), width, width, height, /*decom_h*/ 5, /*decom_v*/ 2, input_bit_depth, hdr_Bw, hdr_Fq,
                               dwt_out_cuda.data());
    }));
#endif
}

/* Phase 2: RC + Quant for ONE representative precinct (4K-wide luma band
 * geometry, decom_h=5/decom_v=2). This is a per-precinct cost, not a
 * per-frame one -- a 4K/10bit frame has ~540 precincts (decom_v=2), so a
 * rough full-frame estimate is roughly mean_ms * 540, not measured directly
 * here given time constraints on this benchmark harness. */
static void bench_phase2_rc_quant(BenchReport& report, int iterations) {
    pi_t pi;
    pi_enc_t pi_enc;
    memset(&pi, 0, sizeof(pi));
    memset(&pi_enc, 0, sizeof(pi_enc));
    uint32_t sx[MAX_COMPONENTS_NUM] = {1, 2, 2, 0};
    uint32_t sy[MAX_COMPONENTS_NUM] = {1, 1, 1, 0};
    if (pi_compute(&pi, 1, 3, GROUP_SIZE, SIGNIFICANCE_GROUP_SIZE, 3840, 4, 5, 2, 0, sx, sy, 0, 4) != SvtJxsErrorNone) {
        fprintf(stderr, "bench_phase2_rc_quant: pi_compute failed\n");
        return;
    }
    weight_table_calculate(&pi, 0, COLOUR_FORMAT_PLANAR_YUV422);
    pi_compute_encoder(&pi, &pi_enc, 1, 0, 0);

    svt_jpeg_xs_encoder_common_t enc_common;
    memset(&enc_common, 0, sizeof(enc_common));
    enc_common.pi = pi;
    enc_common.pi_enc = pi_enc;
    enc_common.coding_significance = 1;
    enc_common.coding_vertical_prediction_mode = METHOD_PRED_DISABLE;
    enc_common.coding_signs_handling = SIGN_HANDLING_STRATEGY_OFF;
    enc_common.picture_header_dynamic.hdr_Qpih = QUANT_TYPE_DEADZONE;

    PictureControlSet pcs;
    memset(&pcs, 0, sizeof(pcs));
    pcs.enc_common = &enc_common;

    precinct_enc_t precinct;
    memset(&precinct, 0, sizeof(precinct));
    precinct.p_info = &enc_common.pi.p_info[PRECINCT_NORMAL];

    uint32_t bands_num = pi.bands_num_all;
    std::vector<std::vector<uint8_t>> gcli(bands_num), sig(bands_num);
    std::vector<std::vector<uint16_t>> coeff_cpu(bands_num), coeff_cuda(bands_num);
#ifdef SVT_ENABLE_CUDA
    std::vector<svt_cuda_rc_band_info_t> band_info(bands_num);
#endif
    std::vector<uint8_t> gcli_flat, sig_flat;
    std::vector<uint16_t> coeff_flat;

    for (uint32_t flat = 0; flat < bands_num; flat++) {
        uint32_t c = pi.global_band_info[flat].comp_id;
        uint32_t b = pi.global_band_info[flat].band_id;
#ifdef SVT_ENABLE_CUDA
        band_info[flat].gcli_offset = (uint32_t)gcli_flat.size();
        band_info[flat].significance_offset = (uint32_t)sig_flat.size();
        band_info[flat].coeff_offset = (uint32_t)coeff_flat.size();
#endif
        if (b == BAND_NOT_EXIST)
            continue;
        const precinct_band_info_t& info = precinct.p_info->b_info[c][b];
        uint32_t hl = info.height;
#ifdef SVT_ENABLE_CUDA
        band_info[flat].width = info.width;
        band_info[flat].gcli_width = info.gcli_width;
        band_info[flat].significance_width = info.significance_width;
        band_info[flat].height_lines = hl;
        band_info[flat].gain = pi.components[c].bands[b].gain;
        band_info[flat].priority = pi.components[c].bands[b].priority;
#endif

        gcli[flat].resize((size_t)hl * info.gcli_width);
        sig[flat].resize((size_t)hl * info.significance_width, 0);
        coeff_cpu[flat].resize((size_t)hl * info.width);
        for (size_t i = 0; i < gcli[flat].size(); i++) {
            gcli[flat][i] = (uint8_t)((i * 2654435761u) % 16);
        }
        for (size_t i = 0; i < coeff_cpu[flat].size(); i++) {
            uint8_t msb = gcli[flat][i / GROUP_SIZE];
            coeff_cpu[flat][i] = msb ? (uint16_t)(1u << (msb - 1)) : 0;
        }
        for (size_t s = 0; s < sig[flat].size(); s++) {
            uint32_t end = (uint32_t)std::min((s + 1) * SIGNIFICANCE_GROUP_SIZE, (size_t)info.gcli_width);
            uint8_t m = 0;
            for (uint32_t g = (uint32_t)(s * SIGNIFICANCE_GROUP_SIZE); g < end; g++) {
                if (gcli[flat][g] > m)
                    m = gcli[flat][g];
            }
            sig[flat][s] = m;
        }
        coeff_cuda[flat] = coeff_cpu[flat];

        for (uint32_t line = 0; line < hl; line++) {
            precinct.bands[c][b].lines_common[line].gcli_data_ptr = &gcli[flat][(size_t)line * info.gcli_width];
            precinct.bands[c][b].lines_common[line].significance_data_max_ptr = &sig[flat][(size_t)line * info.significance_width];
            precinct.bands[c][b].lines_common[line].coeff_data_ptr_16bit = &coeff_cpu[flat][(size_t)line * info.width];
        }
        precinct.bands[c][b].cache_buffers[0].gtli_rc_last_calculated = UINT8_MAX;
        precinct.bands[c][b].cache_buffers[1].gtli_rc_last_calculated = UINT8_MAX;
        precinct.bands[c][b].cache_actual = &precinct.bands[c][b].cache_buffers[0];

        gcli_flat.insert(gcli_flat.end(), gcli[flat].begin(), gcli[flat].end());
        sig_flat.insert(sig_flat.end(), sig[flat].begin(), sig[flat].end());
        coeff_flat.insert(coeff_flat.end(), coeff_cuda[flat].begin(), coeff_cuda[flat].end());
    }
    precinct.p_info->packets_exist_num = pi.packets_num;

    uint32_t budget_bytes = 3840u * 4u * 3 / 8; // ~3bpp proxy budget for this precinct's row count

    report.add(bench_cpu("Phase2-RC+Quant", "RC+Quant(CPU reference, per-precinct)", iterations, [&]() {
        rate_control_init_precinct(&pcs, &precinct, SIGN_HANDLING_STRATEGY_OFF);
        rate_control_precinct(&pcs, &precinct, budget_bytes, METHOD_PRED_DISABLE, SIGN_HANDLING_STRATEGY_OFF);
        precinct_quantization(&pcs, &enc_common.pi, &precinct);
    }));

#ifdef SVT_ENABLE_CUDA
    std::vector<svt_cuda_rc_packet_t> packets(pi.packets_num);
    for (uint32_t p = 0; p < pi.packets_num; p++) {
        packets[p].band_start = pi.packets[p].band_start;
        packets[p].band_stop = pi.packets[p].band_stop;
        packets[p].line_idx = pi.packets[p].line_idx;
    }
    std::vector<uint8_t> out_gtli(bands_num);
    uint8_t out_q = 0, out_r = 0;
    uint32_t out_data = 0, out_pad = 0, out_total = 0;
    report.add(bench_gpu_hostlatency("Phase2-RC+Quant", "RC+Quant(CUDA, per-precinct)", iterations, [&]() {
        svt_cuda_rc_quant_precinct(bands_num, band_info.data(), gcli_flat.data(), sig_flat.data(), coeff_flat.data(),
                                   pi.packets_num, packets.data(), pi.bands_num_exists, precinct.p_info->packets_exist_num,
                                   (uint8_t)pi.use_short_header, (uint8_t)enc_common.coding_significance,
                                   (uint8_t)enc_common.picture_header_dynamic.hdr_Qpih, pi_enc.max_quantization,
                                   pi_enc.max_refinement, budget_bytes, out_gtli.data(), &out_q, &out_r, &out_data, &out_pad,
                                   &out_total, NULL, NULL, NULL, NULL);
    }));
#endif
}

/* Phase 3: Pack for ONE representative precinct (same 4K-wide geometry as
 * Phase 2's bench). Per-precinct cost, not per-frame -- see the note on
 * bench_phase2_rc_quant above. Runs the real CPU RC+Quant pipeline first
 * (so gtli/pack_method/packet sizes are genuine, not synthetic) before
 * timing just the packing step itself. */
static void bench_phase3_pack(BenchReport& report, int iterations) {
    pi_t pi;
    pi_enc_t pi_enc;
    memset(&pi, 0, sizeof(pi));
    memset(&pi_enc, 0, sizeof(pi_enc));
    uint32_t sx[MAX_COMPONENTS_NUM] = {1, 2, 2, 0};
    uint32_t sy[MAX_COMPONENTS_NUM] = {1, 1, 1, 0};
    if (pi_compute(&pi, 1, 3, GROUP_SIZE, SIGNIFICANCE_GROUP_SIZE, 3840, 4, 5, 2, 0, sx, sy, 0, 4) != SvtJxsErrorNone) {
        fprintf(stderr, "bench_phase3_pack: pi_compute failed\n");
        return;
    }
    weight_table_calculate(&pi, 0, COLOUR_FORMAT_PLANAR_YUV422);
    pi_compute_encoder(&pi, &pi_enc, 1, 0, 0);

    svt_jpeg_xs_encoder_common_t enc_common;
    memset(&enc_common, 0, sizeof(enc_common));
    enc_common.pi = pi;
    enc_common.pi_enc = pi_enc;
    enc_common.coding_significance = 1;
    enc_common.coding_vertical_prediction_mode = METHOD_PRED_DISABLE;
    enc_common.coding_signs_handling = SIGN_HANDLING_STRATEGY_OFF;
    enc_common.picture_header_dynamic.hdr_Qpih = QUANT_TYPE_DEADZONE;

    PictureControlSet pcs;
    memset(&pcs, 0, sizeof(pcs));
    pcs.enc_common = &enc_common;

    precinct_enc_t precinct;
    memset(&precinct, 0, sizeof(precinct));
    precinct.p_info = &enc_common.pi.p_info[PRECINCT_NORMAL];

    uint32_t bands_num = pi.bands_num_all;
    std::vector<std::vector<uint8_t>> gcli(bands_num), sig(bands_num);
    std::vector<std::vector<uint16_t>> coeff(bands_num);
#ifdef SVT_ENABLE_CUDA
    std::vector<svt_cuda_pack_band_info_t> band_info(bands_num);
#endif
    std::vector<uint8_t> gcli_flat, sig_flat;
    std::vector<uint16_t> coeff_flat;

    for (uint32_t flat = 0; flat < bands_num; flat++) {
        uint32_t c = pi.global_band_info[flat].comp_id;
        uint32_t b = pi.global_band_info[flat].band_id;
#ifdef SVT_ENABLE_CUDA
        memset(&band_info[flat], 0, sizeof(band_info[flat]));
        band_info[flat].gcli_offset = (uint32_t)gcli_flat.size();
        band_info[flat].significance_offset = (uint32_t)sig_flat.size();
        band_info[flat].coeff_offset = (uint32_t)coeff_flat.size();
#endif
        if (b == BAND_NOT_EXIST)
            continue;
#ifdef SVT_ENABLE_CUDA
        band_info[flat].exists = 1;
#endif
        const precinct_band_info_t& info = precinct.p_info->b_info[c][b];
        uint32_t hl = info.height;
#ifdef SVT_ENABLE_CUDA
        band_info[flat].width = info.width;
        band_info[flat].gcli_width = info.gcli_width;
        band_info[flat].significance_width = info.significance_width;
        band_info[flat].height_lines = hl;
#endif

        gcli[flat].resize((size_t)hl * info.gcli_width);
        sig[flat].resize((size_t)hl * info.significance_width, 0);
        coeff[flat].resize((size_t)hl * info.width);
        // Mix `flat` into the gcli sequence: indexing purely by the
        // band-local index i (resetting to 0 per band) makes every band
        // produce the identical gcli pattern, which correlates significance
        // groups across bands within a packet and can trigger a data-dependent
        // edge case in pack_precinct()'s internal consistency checks.
        for (size_t i = 0; i < gcli[flat].size(); i++) {
            gcli[flat][i] = (uint8_t)(((i + flat * 6151u + 1) * 2654435761u) % 16);
        }
        // Realistic (non-degenerate) magnitudes/signs: an exact power-of-two,
        // always-positive coefficient per group (no noise, no sign) hits an
        // edge case pack_precinct()'s internal length-consistency checks
        // reject for some (budget, geometry) combinations; add pseudo-random
        // noise below the MSB and an occasional sign bit, matching
        // TestPackCuda.cc's data generation.
        uint32_t prng_state = (uint32_t)(flat * 2246822519u + 1);
        for (size_t i = 0; i < coeff[flat].size(); i++) {
            uint8_t msb = gcli[flat][i / GROUP_SIZE];
            prng_state ^= prng_state << 13;
            prng_state ^= prng_state >> 17;
            prng_state ^= prng_state << 5;
            uint16_t mag = msb ? (uint16_t)((1u << (msb - 1)) | (prng_state & ((1u << (msb - 1)) - 1))) : 0;
            uint16_t sign = (prng_state & 0x10000) ? BITSTREAM_MASK_SIGN : 0;
            coeff[flat][i] = mag ? (uint16_t)(mag | sign) : 0;
        }
        for (size_t s = 0; s < sig[flat].size(); s++) {
            uint32_t end = (uint32_t)std::min((s + 1) * SIGNIFICANCE_GROUP_SIZE, (size_t)info.gcli_width);
            uint8_t m = 0;
            for (uint32_t g = (uint32_t)(s * SIGNIFICANCE_GROUP_SIZE); g < end; g++) {
                if (gcli[flat][g] > m)
                    m = gcli[flat][g];
            }
            sig[flat][s] = m;
        }

        for (uint32_t line = 0; line < hl; line++) {
            precinct.bands[c][b].lines_common[line].gcli_data_ptr = &gcli[flat][(size_t)line * info.gcli_width];
            precinct.bands[c][b].lines_common[line].significance_data_max_ptr = &sig[flat][(size_t)line * info.significance_width];
            precinct.bands[c][b].lines_common[line].coeff_data_ptr_16bit = &coeff[flat][(size_t)line * info.width];
        }
        precinct.bands[c][b].cache_buffers[0].gtli_rc_last_calculated = UINT8_MAX;
        precinct.bands[c][b].cache_buffers[1].gtli_rc_last_calculated = UINT8_MAX;
        precinct.bands[c][b].cache_actual = &precinct.bands[c][b].cache_buffers[0];

        gcli_flat.insert(gcli_flat.end(), gcli[flat].begin(), gcli[flat].end());
        sig_flat.insert(sig_flat.end(), sig[flat].begin(), sig[flat].end());
        coeff_flat.insert(coeff_flat.end(), coeff[flat].begin(), coeff[flat].end());
    }
    precinct.p_info->packets_exist_num = pi.packets_num;

    // Looser than Phase2's ~3bpp budget: with this synthetic (non-image-derived)
    // gcli/significance data, a tight budget can drive some band's significance
    // groups to be entirely zeroed (a legitimate but rare real-world case) which
    // pack_precinct()'s own internal length-consistency check rejects for this
    // fixture's specific random draw. A looser budget avoids that without
    // affecting what's being measured (packing cost is dominated by precinct
    // geometry, not by how close to the wire the budget is).
    uint32_t budget_bytes = 3840u * 4u * 8 / 8;
    rate_control_init_precinct(&pcs, &precinct, SIGN_HANDLING_STRATEGY_OFF);
    if (rate_control_precinct(&pcs, &precinct, budget_bytes, METHOD_PRED_DISABLE, SIGN_HANDLING_STRATEGY_OFF) !=
        SvtJxsErrorNone) {
        fprintf(stderr, "bench_phase3_pack: rate_control_precinct failed\n");
        return;
    }
    precinct_quantization(&pcs, &enc_common.pi, &precinct);

#ifdef SVT_ENABLE_CUDA
    for (uint32_t flat = 0; flat < bands_num; flat++) {
        uint32_t c = pi.global_band_info[flat].comp_id;
        uint32_t b = pi.global_band_info[flat].band_id;
        if (b == BAND_NOT_EXIST)
            continue;
        band_info[flat].pack_method =
            (precinct.bands[c][b].cache_actual->pack_method == METHOD_ZERO_SIGNIFICANCE_ENABLE) ? 1 : 0;
        band_info[flat].gtli = precinct.bands[c][b].gtli;
    }
#endif

    std::vector<uint8_t> ref_buf(precinct.pack_total_bytes, 0);
    {
        // Verify once before timing: pack_precinct() itself. This synthetic
        // (non-image-derived) fixture can land on a data-dependent edge case
        // pack_precinct()'s internal length-consistency checks reject; when it
        // does, skip Phase3-Pack timing entirely rather than report numbers
        // from a code path that errored out on every iteration.
        bitstream_writer_t bw;
        bitstream_writer_init(&bw, ref_buf.data(), ref_buf.size());
        if (pack_precinct(&bw, &pi, &precinct, SIGN_HANDLING_STRATEGY_OFF) != SvtJxsErrorNone) {
            fprintf(stderr,
                    "bench_phase3_pack: pack_precinct() failed on this synthetic fixture (data-dependent edge case,\n"
                    "not a Phase 3 port issue -- see TestPackCuda.cc for proven bit-exact correctness); "
                    "skipping Phase3-Pack benchmark entry.\n");
            return;
        }
    }
    report.add(bench_cpu("Phase3-Pack", "Pack(CPU reference, per-precinct)", iterations, [&]() {
        bitstream_writer_t bw;
        bitstream_writer_init(&bw, ref_buf.data(), ref_buf.size());
        pack_precinct(&bw, &pi, &precinct, SIGN_HANDLING_STRATEGY_OFF);
    }));

#ifdef SVT_ENABLE_CUDA
    std::vector<svt_cuda_pack_packet_t> packets(pi.packets_num);
    std::vector<uint32_t> psd(pi.packets_num), psg(pi.packets_num), pss(pi.packets_num);
    for (uint32_t p = 0; p < pi.packets_num; p++) {
        packets[p].band_start = pi.packets[p].band_start;
        packets[p].band_stop = pi.packets[p].band_stop;
        packets[p].line_idx = pi.packets[p].line_idx;
        psd[p] = precinct.packet_size_data_bytes[p];
        psg[p] = precinct.packet_size_gcli_bytes[p];
        pss[p] = precinct.packet_size_significance_bytes[p];
    }
    std::vector<uint8_t> cuda_buf(precinct.pack_total_bytes, 0);
    uint32_t cuda_used_bytes = 0;
    report.add(bench_gpu_hostlatency("Phase3-Pack", "Pack(CUDA, per-precinct)", iterations, [&]() {
        svt_cuda_pack_precinct(bands_num, band_info.data(), pi.bands_num_exists, gcli_flat.data(), sig_flat.data(),
                               coeff_flat.data(), pi.packets_num, packets.data(), (uint8_t)pi.use_short_header,
                               precinct.pack_quantization, precinct.pack_refinement, precinct.pack_total_bytes,
                               precinct.pack_padding_bytes, psd.data(), psg.data(), pss.data(), cuda_buf.data(),
                               &cuda_used_bytes);
    }));
#endif
}

#ifdef SVT_ENABLE_CUDA
/* Phase 4a: end-to-end single-frame CUDA encode using a persistent
 * SvtCudaFrameContext reused across `iterations` calls (the realistic
 * "repeated frames of the same resolution" scenario), vs a full real CPU
 * encode of the same image via the actual encoder API (send_picture/
 * get_packet, reusing one already-initialized encoder instance). Both are
 * timed with CpuTimer (host-visible wall latency). See
 * tests/UnitTests/TestEncodeFrameCuda.cc for the single-slice /
 * no-move-padding scope constraints this shares. */
static void bench_phase4_full_frame(BenchReport& report, uint32_t width, uint32_t height, int iterations) {
    svt_jpeg_xs_encoder_api_t enc;
    if (svt_jpeg_xs_encoder_load_default_parameters(SVT_JPEGXS_API_VER_MAJOR, SVT_JPEGXS_API_VER_MINOR, &enc) != SvtJxsErrorNone)
        return;
    enc.verbose = VERBOSE_NONE;
    enc.source_width = width;
    enc.source_height = height;
    enc.input_bit_depth = 10;
    enc.colour_format = COLOUR_FORMAT_PLANAR_YUV422;
    enc.bpp_numerator = 4;
    enc.bpp_denominator = 1;
    enc.threads_num = 1;
    enc.slice_height = height;
    enc.rate_control_mode = RC_CBR_PER_PRECINCT;

    svt_jpeg_xs_image_config_t image_config;
    uint32_t bytes_per_frame = 0;
    if (svt_jpeg_xs_encoder_get_image_config(SVT_JPEGXS_API_VER_MAJOR, SVT_JPEGXS_API_VER_MINOR, &enc, &image_config,
                                             &bytes_per_frame) != SvtJxsErrorNone)
        return;

    svt_jpeg_xs_image_buffer_t* in_buf = svt_jpeg_xs_image_buffer_alloc(&image_config);
    if (!in_buf)
        return;
    for (int32_t c = 0; c < image_config.components_num; ++c) {
        memset(in_buf->data_yuv[c], 0x55, in_buf->alloc_size[c]);
    }

    if (svt_jpeg_xs_encoder_init(SVT_JPEGXS_API_VER_MAJOR, SVT_JPEGXS_API_VER_MINOR, &enc) != SvtJxsErrorNone) {
        svt_jpeg_xs_image_buffer_free(in_buf);
        return;
    }

    svt_jpeg_xs_encoder_api_prv_t* prv = (svt_jpeg_xs_encoder_api_prv_t*)enc.private_ptr;
    svt_jpeg_xs_encoder_common_t* enc_common = &prv->enc_common;
    pi_t* pi = &enc_common->pi;
    if (pi->slice_num != 1) {
        svt_jpeg_xs_encoder_close(&enc);
        svt_jpeg_xs_image_buffer_free(in_buf);
        printf("Phase4-FullFrame: SKIPPED (unexpected slice_num=%u)\n", pi->slice_num);
        return;
    }

    uint32_t slice_budget_bytes = enc_common->picture_header_dynamic.hdr_Lcod - enc_common->frame_header_length_bytes -
        SLICE_HEADER_SIZE_BYTES - CODESTREAM_SIZE_BYTES;
    std::vector<uint32_t> precinct_budgets(pi->precincts_line_num);
    {
        uint32_t min_budget = slice_budget_bytes / pi->precincts_line_num;
        uint32_t left = slice_budget_bytes - min_budget * pi->precincts_line_num;
        for (uint32_t i = 0; i < pi->precincts_line_num; i++) {
            precinct_budgets[i] = min_budget + (i < left ? 1 : 0);
        }
    }

    SvtCudaFrameContext ctx;
    memset(&ctx, 0, sizeof(ctx));
    if (svt_cuda_frame_context_create_from_pi(&ctx, pi, &enc_common->pi_enc, slice_budget_bytes + 4096) != 0) {
        svt_jpeg_xs_encoder_close(&enc);
        svt_jpeg_xs_image_buffer_free(in_buf);
        printf("Phase4-FullFrame: SKIPPED (context creation failed -- geometry not evenly-divisible NORMAL-only?)\n");
        return;
    }

    const void* in_planes[FCC_MAX_COMPONENTS] = {NULL, NULL, NULL, NULL};
    uint32_t in_stride[FCC_MAX_COMPONENTS] = {0, 0, 0, 0};
    for (uint32_t c = 0; c < pi->comps_num; c++) {
        in_planes[c] = in_buf->data_yuv[c];
        in_stride[c] = in_buf->stride[c];
    }
    std::vector<uint8_t> precinct_data(slice_budget_bytes + 4096, 0);

    BenchRecord r = bench_gpu_hostlatency("Phase4-FullFrame", "Encode(CUDA, full frame)", iterations, [&]() {
        uint32_t used = 0;
        svt_cuda_encode_frame(&ctx, in_planes, in_stride, pi->decom_h, pi->decom_v, enc.input_bit_depth,
                              enc_common->picture_header_dynamic.hdr_Bw, enc_common->picture_header_dynamic.hdr_Fq,
                              (uint8_t)enc_common->picture_header_dynamic.hdr_Qpih, (uint8_t)pi->use_short_header,
                              (uint8_t)enc_common->coding_significance, enc_common->pi_enc.max_quantization,
                              enc_common->pi_enc.max_refinement, precinct_budgets.data(), pi->bands_num_exists,
                              (uint32_t)pi->p_info[PRECINCT_NORMAL].packets_exist_num, precinct_data.data(), &used);
    });
    report.add(r);

    svt_jpeg_xs_bitstream_buffer_t out_buf;
    out_buf.allocation_size = bytes_per_frame * 2 + 4096;
    out_buf.used_size = 0;
    out_buf.buffer = (uint8_t*)malloc(out_buf.allocation_size);
    if (out_buf.buffer) {
        BenchRecord cpu_r = bench_cpu("Phase4-FullFrame", "Encode(CPU reference, full frame)", iterations, [&]() {
            svt_jpeg_xs_frame_t enc_input;
            enc_input.bitstream = out_buf;
            enc_input.image = *in_buf;
            enc_input.user_prv_ctx_ptr = NULL;
            svt_jpeg_xs_encoder_send_picture(&enc, &enc_input, 1);
            svt_jpeg_xs_frame_t enc_output;
            memset(&enc_output, 0, sizeof(enc_output));
            svt_jpeg_xs_encoder_get_packet(&enc, &enc_output, 1);
        });
        report.add(cpu_r);
        free(out_buf.buffer);
    }

    svt_cuda_frame_context_destroy(&ctx);
    svt_jpeg_xs_encoder_close(&enc);
    svt_jpeg_xs_image_buffer_free(in_buf);
}
#endif

int main(int argc, char** argv) {
    // The C reference functions dispatch through RTCD (runtime CPU
    // detection) function pointers that are otherwise left uninitialized
    // (NULL) outside of the normal svt_jpeg_xs_encoder_init() path.
    const CPU_FLAGS cpu_flags = get_cpu_flags();
    setup_common_rtcd_internal(cpu_flags);
    setup_encoder_rtcd_internal(cpu_flags);

    const int iterations = 200;
    const uint32_t width = 3840; // one 4K-wide band line, proxy workload for the harness proof

    BenchReport report;

    bench_gc_cpu_reference(report, "Phase0-Proof", width, iterations);

    // Phase 1: NLT/DWT/GC over a full 4K, 10bit frame (this project's target
    // workload). Fewer iterations than the Phase0-Proof micro-benchmarks
    // since each one now covers the whole frame, not a single line.
    bench_phase1_nlt_dwt_gc(report, 3840, 2160, 20);

    // Phase 2: RC+Quant for one representative 4K-wide precinct.
    bench_phase2_rc_quant(report, 50);

    // Phase 3: Pack for one representative 4K-wide precinct.
    bench_phase3_pack(report, 50);

#ifdef SVT_ENABLE_CUDA
    // Phase 4a: full single-frame CUDA encode (persistent context reused
    // across iterations) vs a full real CPU encode of the same 4K/10bit image.
    bench_phase4_full_frame(report, 3840, 2160, 20);

    double mean_ms = 0.0, min_ms = 0.0, max_ms = 0.0;
    int err = svt_cuda_smoke_bench(iterations, &mean_ms, &min_ms, &max_ms);
    if (err != 0) {
        fprintf(stderr, "svt_cuda_smoke_bench failed with code %d\n", err);
        return 1;
    }
    BenchRecord gpu_r;
    gpu_r.phase = "Phase0-Proof";
    gpu_r.module = "VectorAdd(CUDA smoke)";
    gpu_r.backend = "GPU";
    gpu_r.iterations = iterations;
    gpu_r.mean_ms = mean_ms;
    gpu_r.min_ms = min_ms;
    gpu_r.max_ms = max_ms;
    report.add(gpu_r);
#else
    printf("Built without ENABLE_CUDA: skipping GPU-side benchmarks.\n");
#endif

    report.print_table();
    report.write_csv("benchmark_results.csv");
    printf("\nResults appended to benchmark_results.csv\n");
    return 0;
}
