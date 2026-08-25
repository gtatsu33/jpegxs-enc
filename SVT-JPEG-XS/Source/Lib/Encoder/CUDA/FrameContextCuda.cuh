/*
* Copyright(c) 2024 Intel Corporation
* SPDX - License - Identifier: BSD - 2 - Clause - Patent
*/
#ifndef FrameContextCuda_cuh
#define FrameContextCuda_cuh

#include <stdint.h>
#include <cuda_runtime.h>

/* Phase 4a: persistent per-frame device buffer pool + stream, sized exactly
 * for ONE fixed (width,height,decom_h,decom_v,comps_num) geometry -- created
 * once via svt_cuda_frame_context_create() and reused across many
 * svt_cuda_encode_frame() calls for the same geometry (the realistic
 * "repeated video-rate frames of the same resolution" scenario), instead of
 * the per-call cudaMalloc/cudaFree pattern used by Phase 1-3's individual
 * svt_cuda_* functions (see PortingStrategy.txt section 8/10 -- that pattern
 * was identified as the dominant cost in every Phase 1-3 benchmark).
 *
 * All fields are considered private to FrameContextCuda.cu / EncodeFrameCuda.cu;
 * callers only construct/destroy the context and pass it to
 * svt_cuda_encode_frame().
 */
#define FCC_MAX_COMPONENTS 4

#ifdef __cplusplus
extern "C" {
#endif

typedef struct SvtCudaFrameBandGeom {
    uint32_t comp_id;
    uint32_t band_id;
    uint32_t x, y;                /* absolute position of this band within its component's dense pyramid buffer */
    uint32_t width;                /* band width in coefficients (constant across all rows of the band) */
    uint32_t height;                /* total band height in coefficient rows, for the WHOLE frame (not one precinct) */
    uint32_t height_lines_num;    /* rows per precinct for this band (pi_band_t::height_lines_num) */
    uint32_t gcli_width;            /* ceil(width / GROUP_SIZE) */
    uint32_t significance_width;    /* ceil(gcli_width / SIGNIFICANCE_GROUP_SIZE) */
    uint8_t gain;
    uint8_t priority;
    uint32_t gcli_offset;            /* row 0 offset into the frame-wide flat gcli buffer */
    uint32_t sig_offset;            /* row 0 offset into the frame-wide flat significance buffer */
} SvtCudaFrameBandGeom;

typedef struct SvtCudaFrameContext {
    cudaStream_t stream;

    uint32_t comps_num;
    uint32_t comp_width[FCC_MAX_COMPONENTS];
    uint32_t comp_height[FCC_MAX_COMPONENTS];

    uint32_t bands_num_all;    /* pi->bands_num_all, flat band-interleaved order (matches pi->global_band_info[]) */
    uint32_t precincts_num;    /* pi->precincts_line_num */
    uint32_t packets_num;        /* pi->packets_num (frame-constant packet geometry) */

    /* --- DWT/NLT scratch (reused sequentially across the comps_num DWT calls of one frame; sized to the largest component) --- */
    uint8_t* d_in_raw;
    int32_t* d_cur;
    int32_t* d_other;
    int32_t* d_vert;
    int32_t* d_pyramid32; /* int32 scratch pyramid, reused per component before the final 16-bit shift */

    /* --- Persistent per-component 16-bit coefficient pyramids (DWT output; GC reads from here, RC/Quant modify in place) --- */
    uint16_t* d_pyramid16[FCC_MAX_COMPONENTS];

    /* --- Frame-wide band geometry table (host-computed once at context creation, uploaded once) --- */
    SvtCudaFrameBandGeom* h_bands; /* [bands_num_all], host copy (kept for host-side aggregation loops) */
    SvtCudaFrameBandGeom* d_bands; /* [bands_num_all], device copy */
    uint16_t* d_pyramid_ptrs_storage[FCC_MAX_COMPONENTS]; /* mirrors d_pyramid16, kept so we can build a small device array of pointers */
    uint16_t** d_pyramid_ptrs;                            /* device array of FCC_MAX_COMPONENTS pointers, for kernels indexed by comp_id */

    /* --- Frame-wide GC outputs (flat, one buffer covering all bands/components of the whole frame) --- */
    uint32_t gcli_frame_total;
    uint32_t sig_frame_total;
    uint8_t* d_gcli_frame;
    uint8_t* d_sig_frame;

    /* --- Per-(precinct,band) RC outputs --- */
    uint8_t* d_gtli;           /* [precincts_num * bands_num_all] */
    uint8_t* d_pack_method;    /* [precincts_num * bands_num_all] */
    uint32_t* d_packet_size_data_bytes;  /* [precincts_num * packets_num] */
    uint32_t* d_packet_size_gcli_bytes;  /* [precincts_num * packets_num] */
    uint32_t* d_packet_size_significance_bytes; /* [precincts_num * packets_num] */
    uint32_t* d_precinct_out_offset;  /* [precincts_num], byte offset of each precinct within d_pack_out */
    uint8_t* d_precinct_quantization;  /* [precincts_num] */
    uint8_t* d_precinct_refinement;    /* [precincts_num] */
    uint32_t* d_precinct_total_bytes;  /* [precincts_num] */
    uint32_t* d_precinct_padding_bytes; /* [precincts_num] */

    /* --- Final packed bitstream (sized generously; actual per-frame size is <= this) --- */
    uint32_t pack_out_capacity_bytes;
    uint8_t* d_pack_out;

    /* --- Frame-constant packet inclusion geometry (pi->packets[]) --- */
    void* d_packets;  /* svt_cuda_pack_packet_t[packets_num], see PackCuda.cuh */
    void* h_packets;  /* persistent host mirror (avoids a D2H download every encode call) */

    /* =====================================================================
     * Phase 4b-2: CUDA Graph support. Two graphs bracket the host-side RC
     * binary search (which must stay off-graph -- see plan file Phase 4b-2):
     *   graph1 = DWT+NLT (all components) + GC/significance (all bands) +
     *            RC LUT build (all bands) + D2H copy of the LUT into h_lut.
     *   graph2 = H2D upload of RC results + quantize (all bands) + pack
     *            (all precincts) + D2H copy of the pack error flag.
     * Both graphs are captured lazily on first use (or whenever a captured
     * argument changes) and replayed via cudaGraphLaunch on every call after
     * that. See EncodeFrameCuda.cu's svt_cuda_encode_frame() for the capture/
     * replay logic and cap_* invalidation checks.
     * ===================================================================== */
    cudaGraph_t graph1;
    cudaGraphExec_t graph1_exec;
    uint8_t graph1_captured;
    const void* cap_in_planes[FCC_MAX_COMPONENTS];
    uint32_t cap_in_stride[FCC_MAX_COMPONENTS];
    uint32_t cap_decom_h, cap_decom_v;
    uint8_t cap_input_bit_depth, cap_hdr_Bw, cap_hdr_Fq, cap_coding_significance;

    cudaGraph_t graph2;
    cudaGraphExec_t graph2_exec;
    uint8_t graph2_captured;
    uint8_t cap_quant_type, cap_use_short_header;

    /* --- Phase 4b-2: buffers that used to be cudaMalloc'd/freed on every
     * svt_cuda_encode_frame() call, now persistent (also required for graph
     * capture -- captured node arguments bake in fixed addresses). --- */
    uint32_t* d_comp_stride; /* [FCC_MAX_COMPONENTS], uploaded once at creation (comp_width never changes) */

    void* d_lut;              /* opaque RC LUT rows, lut_total_rows * lut_row_size_bytes (see create()) */
    uint32_t lut_total_rows;  /* sum of h_bands[].height across all bands */
    uint32_t* lut_row_offset; /* host [bands_num_all], row offset of band b within d_lut/h_lut */
    void* h_lut;              /* pinned host mirror of d_lut (D2H target inside graph1) */

    uint8_t* d_gtli_per_band; /* [bands_num_all * precincts_num], band-major (quantize kernel input) */
    uint8_t* h_gtli_per_band; /* pinned host mirror, band-major (H2D source inside graph2) */

    int* d_error;
    int* h_error; /* pinned, D2H target inside graph2 */

    /* --- Phase 4b-2: pinned host mirrors of the per-precinct RC outputs
     * (filled by the host-side binary search between graph1 and graph2, then
     * uploaded as fixed-address H2D copies captured inside graph2). --- */
    uint8_t* h_gtli;        /* [precincts_num * bands_num_all], precinct-major (pack kernel input) */
    uint8_t* h_pack_method; /* [precincts_num * bands_num_all], precinct-major */
    uint32_t* h_psd;        /* [precincts_num * packets_num] */
    uint32_t* h_psg;        /* [precincts_num * packets_num] */
    uint32_t* h_pss;        /* [precincts_num * packets_num] */
    uint8_t* h_quant;       /* [precincts_num] */
    uint8_t* h_refine;      /* [precincts_num] */
    uint32_t* h_total_bytes;    /* [precincts_num] */
    uint32_t* h_padding_bytes;  /* [precincts_num] */
    uint32_t* h_out_offset;     /* [precincts_num] */
} SvtCudaFrameContext;

/* Creates a persistent context sized exactly for the given, already-computed
 * pi_t/pi_enc_t (caller builds these via pi_compute()/pi_compute_encoder(),
 * exactly as tests/UnitTests/Test{RcQuant,Pack}Cuda.cc already do -- this
 * function does not reach into Pi.h itself so it stays a pure CUDA/C++ TU).
 * comp_width/comp_height: per-component pixel dimensions ([0..comps_num)).
 * band geometry (x,y,width,height,height_lines_num,gain,priority,comp_id,band_id):
 * caller-supplied flat array of bands_num_all entries in pi->global_band_info[] order.
 * Returns 0 on success, negative CUDA error code on failure.
 */
/* lut_row_size_bytes: sizeof of the caller's per-(band,line) RC LUT row
 * struct (opaque to this file -- see EncodeFrameCuda.cu's EfcBandLineLut).
 * Used only to size d_lut/h_lut; this function does not interpret the LUT
 * contents itself.
 */
int svt_cuda_frame_context_create(SvtCudaFrameContext* ctx, uint32_t comps_num, const uint32_t* comp_width,
                                  const uint32_t* comp_height, uint32_t bands_num_all, const SvtCudaFrameBandGeom* bands,
                                  uint32_t precincts_num, uint32_t packets_num, uint32_t pack_out_capacity_bytes,
                                  uint32_t lut_row_size_bytes);

void svt_cuda_frame_context_destroy(SvtCudaFrameContext* ctx);

#ifdef __cplusplus
}
#endif

#endif // FrameContextCuda_cuh
