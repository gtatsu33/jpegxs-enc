/*
* Copyright(c) 2024 Intel Corporation
* SPDX - License - Identifier: BSD - 2 - Clause - Patent
*/
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>
#include <cstring>
#include "FrameContextCuda.cuh"

static void fcc_free_all(SvtCudaFrameContext* ctx) {
    if (ctx->graph1_exec) {
        cudaGraphExecDestroy(ctx->graph1_exec);
    }
    if (ctx->graph1) {
        cudaGraphDestroy(ctx->graph1);
    }
    if (ctx->graph2_exec) {
        cudaGraphExecDestroy(ctx->graph2_exec);
    }
    if (ctx->graph2) {
        cudaGraphDestroy(ctx->graph2);
    }

    cudaFree(ctx->d_in_raw);
    cudaFree(ctx->d_cur);
    cudaFree(ctx->d_other);
    cudaFree(ctx->d_vert);
    cudaFree(ctx->d_pyramid32);
    for (int c = 0; c < FCC_MAX_COMPONENTS; c++) {
        cudaFree(ctx->d_pyramid16[c]);
    }
    cudaFree(ctx->d_bands);
    cudaFree(ctx->d_pyramid_ptrs);
    cudaFree(ctx->d_gcli_frame);
    cudaFree(ctx->d_sig_frame);
    cudaFree(ctx->d_gtli);
    cudaFree(ctx->d_pack_method);
    cudaFree(ctx->d_packet_size_data_bytes);
    cudaFree(ctx->d_packet_size_gcli_bytes);
    cudaFree(ctx->d_packet_size_significance_bytes);
    cudaFree(ctx->d_precinct_out_offset);
    cudaFree(ctx->d_precinct_quantization);
    cudaFree(ctx->d_precinct_refinement);
    cudaFree(ctx->d_precinct_total_bytes);
    cudaFree(ctx->d_precinct_padding_bytes);
    cudaFree(ctx->d_pack_out);
    cudaFree(ctx->d_packets);
    free(ctx->h_packets);
    free(ctx->h_bands);
    free(ctx->h_packet_size_gcli_raw_bytes);
    cudaFree(ctx->d_packet_methods_raw);
    cudaFreeHost(ctx->h_packet_methods_raw);
    free(ctx->h_packets_exist);
    cudaFree(ctx->d_packet_offset);
    cudaFreeHost(ctx->h_packet_offset);
    cudaFree(ctx->d_packet_group_base);

    cudaFree(ctx->d_comp_stride);
    cudaFree(ctx->d_lut);
    free(ctx->lut_row_offset);
    cudaFreeHost(ctx->h_lut);
    cudaFree(ctx->d_gtli_per_band);
    cudaFreeHost(ctx->h_gtli_per_band);
    cudaFree(ctx->d_error);
    cudaFreeHost(ctx->h_error);
    cudaFreeHost(ctx->h_gtli);
    cudaFreeHost(ctx->h_pack_method);
    cudaFreeHost(ctx->h_psd);
    cudaFreeHost(ctx->h_psg);
    cudaFreeHost(ctx->h_pss);
    cudaFreeHost(ctx->h_quant);
    cudaFreeHost(ctx->h_refine);
    cudaFreeHost(ctx->h_total_bytes);
    cudaFreeHost(ctx->h_padding_bytes);
    cudaFreeHost(ctx->h_out_offset);

    if (ctx->stream) {
        cudaStreamDestroy(ctx->stream);
    }
}

int svt_cuda_frame_context_create(SvtCudaFrameContext* ctx, uint32_t comps_num, const uint32_t* comp_width,
                                  const uint32_t* comp_height, uint32_t bands_num_all, const SvtCudaFrameBandGeom* bands,
                                  uint32_t precincts_num, uint32_t packets_num, uint32_t pack_out_capacity_bytes,
                                  uint32_t lut_row_size_bytes) {
    memset(ctx, 0, sizeof(*ctx));
    ctx->comps_num = comps_num;
    ctx->bands_num_all = bands_num_all;
    ctx->precincts_num = precincts_num;
    ctx->packets_num = packets_num;
    ctx->pack_out_capacity_bytes = pack_out_capacity_bytes;

    uint32_t max_w = 0, max_h = 0;
    for (uint32_t c = 0; c < comps_num; c++) {
        ctx->comp_width[c] = comp_width[c];
        ctx->comp_height[c] = comp_height[c];
        max_w = std::max(max_w, comp_width[c]);
        max_h = std::max(max_h, comp_height[c]);
    }
    size_t max_elems = (size_t)max_w * max_h;

    /* Frame-wide gcli/significance offsets: accumulate exactly (bands partition
     * each component's pyramid area without overlap, so the running sum equals
     * the true total, not an upper-bound guess). */
    std::vector<SvtCudaFrameBandGeom> hb(bands, bands + bands_num_all);
    uint32_t gcli_off = 0, sig_off = 0;
    ctx->lut_row_offset = (uint32_t*)malloc((bands_num_all ? bands_num_all : 1) * sizeof(uint32_t));
    uint32_t lut_total_rows = 0;
    for (uint32_t b = 0; b < bands_num_all; b++) {
        hb[b].gcli_offset = gcli_off;
        hb[b].sig_offset = sig_off;
        gcli_off += hb[b].height * hb[b].gcli_width;
        sig_off += hb[b].height * hb[b].significance_width;
        ctx->lut_row_offset[b] = lut_total_rows;
        lut_total_rows += hb[b].height; /* 0 for BAND_NOT_EXIST entries, per svt_cuda_frame_context_create_from_pi() */
    }
    ctx->gcli_frame_total = gcli_off;
    ctx->sig_frame_total = sig_off;
    ctx->lut_total_rows = lut_total_rows;
    /* ctx->gcli_scan_shared_bytes is NOT computed here: a single band can
     * appear in MULTIPLE packets (one packet per line_idx for bands with
     * height_lines_num > 1 -- see Pi.c's packet construction), so the
     * correct total is a sum over (packet, active band) pairs, not over
     * bands alone. svt_cuda_frame_context_create_from_pi() -- which has the
     * packet band_start/band_stop ranges this needs -- computes and
     * overwrites it once it knows that count (0/unused for contexts created
     * via this function directly, e.g. the DWT-only tests, which never
     * launch k_pack_precinct_frame()). */

    ctx->h_bands = (SvtCudaFrameBandGeom*)malloc(bands_num_all * sizeof(SvtCudaFrameBandGeom));
    memcpy(ctx->h_bands, hb.data(), bands_num_all * sizeof(SvtCudaFrameBandGeom));

    cudaError_t err = cudaSuccess;
    do {
        if ((err = cudaStreamCreate(&ctx->stream)) != cudaSuccess)
            break;

        if ((err = cudaMalloc(&ctx->d_in_raw, max_elems * sizeof(uint16_t))) != cudaSuccess)
            break;
        if ((err = cudaMalloc(&ctx->d_cur, max_elems * sizeof(int32_t))) != cudaSuccess)
            break;
        if ((err = cudaMalloc(&ctx->d_other, max_elems * sizeof(int32_t))) != cudaSuccess)
            break;
        if ((err = cudaMalloc(&ctx->d_vert, max_elems * sizeof(int32_t))) != cudaSuccess)
            break;
        if ((err = cudaMalloc(&ctx->d_pyramid32, max_elems * sizeof(int32_t))) != cudaSuccess)
            break;

        for (uint32_t c = 0; c < comps_num; c++) {
            size_t elems_c = (size_t)comp_width[c] * comp_height[c];
            if ((err = cudaMalloc(&ctx->d_pyramid16[c], elems_c * sizeof(uint16_t))) != cudaSuccess)
                break;
        }
        if (err != cudaSuccess)
            break;

        if ((err = cudaMalloc(&ctx->d_bands, bands_num_all * sizeof(SvtCudaFrameBandGeom))) != cudaSuccess)
            break;
        if ((err = cudaMemcpy(
                 ctx->d_bands, hb.data(), bands_num_all * sizeof(SvtCudaFrameBandGeom), cudaMemcpyHostToDevice)) != cudaSuccess)
            break;

        if ((err = cudaMalloc(&ctx->d_pyramid_ptrs, FCC_MAX_COMPONENTS * sizeof(uint16_t*))) != cudaSuccess)
            break;
        uint16_t* ptrs[FCC_MAX_COMPONENTS] = {
            ctx->d_pyramid16[0], ctx->d_pyramid16[1], ctx->d_pyramid16[2], ctx->d_pyramid16[3]};
        if ((err = cudaMemcpy(ctx->d_pyramid_ptrs, ptrs, sizeof(ptrs), cudaMemcpyHostToDevice)) != cudaSuccess)
            break;

        if ((err = cudaMalloc(&ctx->d_gcli_frame, ctx->gcli_frame_total ? ctx->gcli_frame_total : 1)) != cudaSuccess)
            break;
        if ((err = cudaMalloc(&ctx->d_sig_frame, ctx->sig_frame_total ? ctx->sig_frame_total : 1)) != cudaSuccess)
            break;

        size_t pb = (size_t)precincts_num * bands_num_all;
        if ((err = cudaMalloc(&ctx->d_gtli, pb ? pb : 1)) != cudaSuccess)
            break;
        if ((err = cudaMalloc(&ctx->d_pack_method, pb ? pb : 1)) != cudaSuccess)
            break;

        size_t pp = (size_t)precincts_num * packets_num;
        if ((err = cudaMalloc(&ctx->d_packet_size_data_bytes, (pp ? pp : 1) * sizeof(uint32_t))) != cudaSuccess)
            break;
        if ((err = cudaMalloc(&ctx->d_packet_size_gcli_bytes, (pp ? pp : 1) * sizeof(uint32_t))) != cudaSuccess)
            break;
        if ((err = cudaMalloc(&ctx->d_packet_size_significance_bytes, (pp ? pp : 1) * sizeof(uint32_t))) != cudaSuccess)
            break;

        if ((err = cudaMalloc(&ctx->d_precinct_out_offset, (precincts_num ? precincts_num : 1) * sizeof(uint32_t))) !=
            cudaSuccess)
            break;
        if ((err = cudaMalloc(&ctx->d_precinct_quantization, precincts_num ? precincts_num : 1)) != cudaSuccess)
            break;
        if ((err = cudaMalloc(&ctx->d_precinct_refinement, precincts_num ? precincts_num : 1)) != cudaSuccess)
            break;
        if ((err = cudaMalloc(&ctx->d_precinct_total_bytes, (precincts_num ? precincts_num : 1) * sizeof(uint32_t))) !=
            cudaSuccess)
            break;
        if ((err = cudaMalloc(&ctx->d_precinct_padding_bytes, (precincts_num ? precincts_num : 1) * sizeof(uint32_t))) !=
            cudaSuccess)
            break;

        /* +4 bytes of tail padding: Phase 6 pilot's significance-sub-packet
         * scatter write (EncodeFrameCuda.cu's efc_pack_significance_parallel())
         * ORs into the 32-bit-aligned word containing each target byte (no
         * native 8-bit atomicOr on CUDA), which can touch up to 3 bytes past
         * the last significance byte of the last precinct -- structurally
         * guarantee that stays in-bounds regardless of caller-supplied
         * capacity slack. */
        if ((err = cudaMalloc(&ctx->d_pack_out, (pack_out_capacity_bytes ? pack_out_capacity_bytes : 1) + 4)) !=
            cudaSuccess)
            break;

        /* --- Phase 4b-2: persistent scratch + pinned host mirrors, replacing
         * what used to be cudaMalloc'd/freed (and, for the LUT/gtli-per-band
         * gather, downloaded/uploaded piecemeal) on every encode call. Pinned
         * (page-locked) host memory is required here, not just an optimization:
         * these addresses are baked into cudaGraph nodes at capture time and
         * must stay valid/stable across every later cudaGraphLaunch replay. */
        if ((err = cudaMalloc(&ctx->d_comp_stride, FCC_MAX_COMPONENTS * sizeof(uint32_t))) != cudaSuccess)
            break;
        {
            uint32_t stride_h[FCC_MAX_COMPONENTS] = {0, 0, 0, 0};
            for (uint32_t c = 0; c < comps_num; c++) {
                stride_h[c] = comp_width[c];
            }
            if ((err = cudaMemcpy(ctx->d_comp_stride, stride_h, sizeof(stride_h), cudaMemcpyHostToDevice)) != cudaSuccess)
                break;
        }

        size_t lut_bytes = (size_t)(lut_total_rows ? lut_total_rows : 1) * lut_row_size_bytes;
        if ((err = cudaMalloc(&ctx->d_lut, lut_bytes)) != cudaSuccess)
            break;
        if ((err = cudaHostAlloc(&ctx->h_lut, lut_bytes, cudaHostAllocDefault)) != cudaSuccess)
            break;

        size_t pb2 = (size_t)precincts_num * bands_num_all;
        if ((err = cudaMalloc(&ctx->d_gtli_per_band, pb2 ? pb2 : 1)) != cudaSuccess)
            break;
        if ((err = cudaHostAlloc((void**)&ctx->h_gtli_per_band, pb2 ? pb2 : 1, cudaHostAllocDefault)) != cudaSuccess)
            break;

        if ((err = cudaMalloc(&ctx->d_error, sizeof(int))) != cudaSuccess)
            break;
        if ((err = cudaHostAlloc((void**)&ctx->h_error, sizeof(int), cudaHostAllocDefault)) != cudaSuccess)
            break;

        if ((err = cudaHostAlloc((void**)&ctx->h_gtli, pb ? pb : 1, cudaHostAllocDefault)) != cudaSuccess)
            break;
        if ((err = cudaHostAlloc((void**)&ctx->h_pack_method, pb ? pb : 1, cudaHostAllocDefault)) != cudaSuccess)
            break;
        if ((err = cudaMalloc(&ctx->d_packet_methods_raw, pp ? pp : 1)) != cudaSuccess)
            break;
        if ((err = cudaHostAlloc((void**)&ctx->h_packet_methods_raw, pp ? pp : 1, cudaHostAllocDefault)) != cudaSuccess)
            break;
        if ((err = cudaMalloc(&ctx->d_packet_offset, (pp ? pp : 1) * sizeof(uint32_t))) != cudaSuccess)
            break;
        if ((err = cudaHostAlloc((void**)&ctx->h_packet_offset, (pp ? pp : 1) * sizeof(uint32_t), cudaHostAllocDefault)) !=
            cudaSuccess)
            break;
        if ((err = cudaHostAlloc((void**)&ctx->h_psd, (pp ? pp : 1) * sizeof(uint32_t), cudaHostAllocDefault)) != cudaSuccess)
            break;
        if ((err = cudaHostAlloc((void**)&ctx->h_psg, (pp ? pp : 1) * sizeof(uint32_t), cudaHostAllocDefault)) != cudaSuccess)
            break;
        if ((err = cudaHostAlloc((void**)&ctx->h_pss, (pp ? pp : 1) * sizeof(uint32_t), cudaHostAllocDefault)) != cudaSuccess)
            break;
        if ((err = cudaHostAlloc((void**)&ctx->h_quant, precincts_num ? precincts_num : 1, cudaHostAllocDefault)) != cudaSuccess)
            break;
        if ((err = cudaHostAlloc((void**)&ctx->h_refine, precincts_num ? precincts_num : 1, cudaHostAllocDefault)) != cudaSuccess)
            break;
        if ((err = cudaHostAlloc((void**)&ctx->h_total_bytes, (precincts_num ? precincts_num : 1) * sizeof(uint32_t),
                                 cudaHostAllocDefault)) != cudaSuccess)
            break;
        if ((err = cudaHostAlloc((void**)&ctx->h_padding_bytes, (precincts_num ? precincts_num : 1) * sizeof(uint32_t),
                                 cudaHostAllocDefault)) != cudaSuccess)
            break;
        if ((err = cudaHostAlloc((void**)&ctx->h_out_offset, (precincts_num ? precincts_num : 1) * sizeof(uint32_t),
                                 cudaHostAllocDefault)) != cudaSuccess)
            break;
    } while (0);

    if (err != cudaSuccess) {
        fcc_free_all(ctx);
        memset(ctx, 0, sizeof(*ctx));
        return -(int)err;
    }
    return 0;
}

void svt_cuda_frame_context_destroy(SvtCudaFrameContext* ctx) {
    fcc_free_all(ctx);
    memset(ctx, 0, sizeof(*ctx));
}
