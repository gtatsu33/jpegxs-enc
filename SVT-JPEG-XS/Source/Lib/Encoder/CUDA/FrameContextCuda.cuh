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
    void* d_packets; /* svt_cuda_pack_packet_t[packets_num], see PackCuda.cuh */
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
int svt_cuda_frame_context_create(SvtCudaFrameContext* ctx, uint32_t comps_num, const uint32_t* comp_width,
                                  const uint32_t* comp_height, uint32_t bands_num_all, const SvtCudaFrameBandGeom* bands,
                                  uint32_t precincts_num, uint32_t packets_num, uint32_t pack_out_capacity_bytes);

void svt_cuda_frame_context_destroy(SvtCudaFrameContext* ctx);

#ifdef __cplusplus
}
#endif

#endif // FrameContextCuda_cuh
