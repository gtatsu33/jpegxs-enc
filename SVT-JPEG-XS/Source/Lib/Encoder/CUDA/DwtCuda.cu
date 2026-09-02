/*
* Copyright(c) 2024 Intel Corporation
* SPDX - License - Identifier: BSD - 2 - Clause - Patent
*/
#include <cuda_runtime.h>
#include "DwtCuda.cuh"

#define CUDA_DWT_SIGN_MASK ((uint16_t)1 << 15) /* matches BITSTREAM_MASK_SIGN, Codestream.h */

/* Shared by k_image_shift (whole-buffer pass) and k_horizontal_lift's
 * per-band finalize path (Phase B, Step B2): converts one raw lifting-scheme
 * int32 coefficient to 16-bit sign+magnitude, matching image_shift_c
 * (NltEnc.c) exactly. Factored out so both call sites are provably using the
 * identical formula -- no duplicated/divergent logic. */
__device__ __forceinline__ uint16_t dwt_shift_to_u16(int32_t val, int32_t shift, int32_t offset) {
    if (val >= 0) {
        return (uint16_t)((val + offset) >> shift);
    }
    int32_t v = (-val + offset) >> shift;
    uint16_t out_v = (uint16_t)v;
    if (v) {
        out_v |= CUDA_DWT_SIGN_MASK;
    }
    return out_v;
}

/* Generic 5/3 reversible lifting for one 1D line of `len` samples, matching
 * dwt_horizontal_line_c (Dwt.c) exactly -- including its boundary formulas
 * for len==2 and even/odd trailing samples. Used for BOTH the horizontal
 * pass (stride=1 along a row) and the vertical pass (stride=pitch along a
 * column): the CPU's vertical lifting functions (transform_vertical_loop_*_c)
 * implement this identical recurrence, just applied along columns. */
/* [2026-08-28 Phase B, Step B3] read(i) abstracts the sample source so this
 * core can be reused by both the existing raw-int32 path (DwtRawReader,
 * below) and the NLT-fused vertical lift's on-the-fly-scaled reads
 * (DwtNlt8Reader/DwtNlt16Reader) without duplicating the lifting recurrence
 * itself. Body is byte-for-byte the same arithmetic as the pre-Step-B3
 * dwt_lift53(), with every in[X*in_stride] replaced by read((uint32_t)X). */
template <typename Reader>
__device__ __forceinline__ void dwt_lift53_r(Reader read, int32_t* out_lf, int lf_stride, int32_t* out_hf, int hf_stride,
                                              uint32_t len) {
    if (len == 2) {
        int32_t hf0 = read(1) - read(0);
        out_hf[0] = hf0;
        out_lf[0] = read(0) + ((hf0 + 1) >> 1);
        return;
    }

    uint32_t count = (len - 1) / 2;
    int32_t hf_i = read(1) - ((read(0) + read(2)) >> 1);
    out_hf[0] = hf_i;
    out_lf[0] = read(0) + ((hf_i + 1) >> 1);
    int32_t hf_prev = hf_i;

    for (uint32_t id = 1; id < count; id++) {
        uint32_t k = id * 2;
        hf_i = read(k + 1) - ((read(k) + read(k + 2)) >> 1);
        out_hf[id * hf_stride] = hf_i;
        out_lf[id * lf_stride] = read(k) + ((hf_prev + hf_i + 2) >> 2);
        hf_prev = hf_i;
    }

    if (!(len & 1)) {
        uint32_t last = len / 2 - 1;
        hf_i = read(len - 1) - read(len - 2);
        out_hf[last * hf_stride] = hf_i;
        out_lf[last * lf_stride] = read(len - 2) + ((hf_prev + hf_i + 2) >> 2);
    }
    else {
        out_lf[(len / 2) * lf_stride] = read(len - 1) + ((hf_prev + 1) >> 1);
    }
}

/* Reader for the existing raw-int32 path -- makes dwt_lift53() a thin
 * wrapper around dwt_lift53_r() with zero change to its signature or
 * behavior (existing callers, e.g. k_vertical_lift, are untouched). */
struct DwtRawReader {
    const int32_t* in;
    int stride;
    __device__ __forceinline__ int32_t operator()(uint32_t i) const {
        return in[(int)i * stride];
    }
};

__device__ __forceinline__ void dwt_lift53(const int32_t* in, int in_stride, int32_t* out_lf, int lf_stride, int32_t* out_hf,
                                            int hf_stride, uint32_t len) {
    dwt_lift53_r(DwtRawReader{in, in_stride}, out_lf, lf_stride, out_hf, hf_stride, len);
}

/* [2026-08-28 Phase B, Step B3] Readers for the NLT-fused first vertical
 * lift (decom_v>0's v==0 level): apply linear_input_scaling_line_*_c's
 * (NltEnc.c) shift/offset conversion on the fly instead of reading an
 * already-scaled int32 from d_cur. Formulas match k_nlt_scale_8bit/_16bit
 * exactly. `col` is the fixed column this reader's thread lifts; `i` is the
 * row index within that column (same convention as k_vertical_lift's use of
 * dwt_lift53 today). */
struct DwtNlt8Reader {
    const uint8_t* src;
    uint32_t src_pitch;
    uint32_t col;
    uint8_t shift;
    int32_t offset;
    __device__ __forceinline__ int32_t operator()(uint32_t i) const {
        return (int32_t)((uint32_t)src[(size_t)i * src_pitch + col] << shift) - offset;
    }
};

struct DwtNlt16Reader {
    const uint16_t* src;
    uint32_t src_pitch;
    uint32_t col;
    uint8_t shift;
    int32_t offset;
    uint16_t mask;
    __device__ __forceinline__ int32_t operator()(uint32_t i) const {
        return (int32_t)((uint32_t)(src[(size_t)i * src_pitch + col] & mask) << shift) - offset;
    }
};

/* Phase 4c: shared-memory tiled horizontal lift. One BLOCK per row (was: one
 * thread per row). This is a pure parallelization/memory-access rewrite --
 * the arithmetic below is algebraically identical to dwt_lift53() (hf_stride
 * and lf_stride are always 1 for the horizontal pass, matching the original
 * dwt_lift53(in_row, 1, lf_row, 1, hf_row, 1, width) call), NOT an
 * approximation of it.
 *
 * Key insight enabling this: hf[id] (id in [0,count)) depends only on the
 * *input* row (hf[id] = in[2id+1] - ((in[2id]+in[2id+2])>>1)), never on
 * another hf value -- so all hf entries are embarrassingly parallel. lf[id]
 * depends only on hf[id-1] and hf[id] (plus the input row), so once hf[] is
 * known, all lf entries are embarrassingly parallel too. This turns the
 * CPU-style O(width) *serial* recurrence into two O(width/blockDim) parallel
 * passes separated by one __syncthreads(), with the whole row cached in
 * shared memory (loaded once, coalesced across the block, and reused instead
 * of re-reading global memory for every neighbor access).
 *
 * Dynamic shared memory layout: [0,width) = input row samples,
 * [width, width+width/2) = hf scratch (so lf's pass can read neighboring hf
 * values without a global-memory round trip). Caller must launch with
 * dynamic shared bytes = (width + width/2) * sizeof(int32_t) and
 * grid.x == height (one block per row); see dwt_h_tile_shmem_bytes() below. */
/* [2026-08-28 Phase B, Step B2] lf_finalize/hf_finalize: when true, the
 * corresponding output is a finalized (non-recursing) band -- instead of the
 * usual raw int32 write to dst_lf/dst_hf, this writes the shift/offset+sign
 * converted 16-bit value directly to out_lf/out_hf (same absolute pyramid
 * coordinates, via dwt_shift_to_u16()). This fuses what used to be a
 * separate whole-buffer k_image_shift pass into the band write itself,
 * eliminating the int32 round trip through d_pyramid for finalized bands.
 * dst_lf/dst_hf (or out_lf/out_hf) are expected to be NULL on the side that
 * isn't used for a given call -- an accidental branch mismatch then
 * segfaults immediately instead of silently producing wrong output. */
__global__ void k_horizontal_lift(const int32_t* src, uint32_t pitch, uint32_t sx, uint32_t sy, uint32_t width, uint32_t height,
                                  int32_t* dst_lf, uint32_t lf_pitch, uint32_t lfx, uint32_t lfy, int32_t* dst_hf,
                                  uint32_t hf_pitch, uint32_t hfx, uint32_t hfy, bool lf_finalize, uint16_t* out_lf,
                                  uint32_t out_lf_pitch, uint32_t out_lfx, uint32_t out_lfy, bool hf_finalize, uint16_t* out_hf,
                                  uint32_t out_hf_pitch, uint32_t out_hfx, uint32_t out_hfy, int32_t shift_out, int32_t offset_out) {
    extern __shared__ int32_t s_mem[];
    uint32_t row = blockIdx.x;
    if (row >= height)
        return;

    const int32_t* in_row = src + (size_t)(sy + row) * pitch + sx;
    int32_t* lf_row = lf_finalize ? NULL : dst_lf + (size_t)(lfy + row) * lf_pitch + lfx;
    int32_t* hf_row = hf_finalize ? NULL : dst_hf + (size_t)(hfy + row) * hf_pitch + hfx;
    uint16_t* lf_row16 = lf_finalize ? out_lf + (size_t)(out_lfy + row) * out_lf_pitch + out_lfx : NULL;
    uint16_t* hf_row16 = hf_finalize ? out_hf + (size_t)(out_hfy + row) * out_hf_pitch + out_hfx : NULL;

    int32_t* s_in = s_mem;         /* width elements */
    int32_t* s_hf = s_mem + width; /* width/2 elements */

    for (uint32_t i = threadIdx.x; i < width; i += blockDim.x) {
        s_in[i] = in_row[i];
    }
    __syncthreads();

    if (width == 2) {
        if (threadIdx.x == 0) {
            int32_t hf0 = s_in[1] - s_in[0];
            if (hf_finalize) {
                hf_row16[0] = dwt_shift_to_u16(hf0, shift_out, offset_out);
            }
            else {
                hf_row[0] = hf0;
            }
            int32_t lf0 = s_in[0] + ((hf0 + 1) >> 1);
            if (lf_finalize) {
                lf_row16[0] = dwt_shift_to_u16(lf0, shift_out, offset_out);
            }
            else {
                lf_row[0] = lf0;
            }
        }
        return;
    }

    uint32_t count = (width - 1) / 2;

    /* Phase 1: hf[0..count) in parallel -- identical formula for every id,
     * matching the general-case branch of dwt_lift53() (id==0 uses the same
     * hf formula there too, it's only lf[0] that's special-cased). s_hf[id]
     * is always cached (Phase 2/3 depend on it) regardless of finalize. */
    for (uint32_t id = threadIdx.x; id < count; id += blockDim.x) {
        uint32_t k = id * 2;
        int32_t hf_i = s_in[k + 1] - ((s_in[k] + s_in[k + 2]) >> 1);
        s_hf[id] = hf_i;
        if (hf_finalize) {
            hf_row16[id] = dwt_shift_to_u16(hf_i, shift_out, offset_out);
        }
        else {
            hf_row[id] = hf_i;
        }
    }
    __syncthreads();

    /* Phase 2: lf[0..count) in parallel, reading only already-finalized hf
     * values from shared memory. */
    for (uint32_t id = threadIdx.x; id < count; id += blockDim.x) {
        int32_t hf_i = s_hf[id];
        int32_t lf_i;
        if (id == 0) {
            lf_i = s_in[0] + ((hf_i + 1) >> 1);
        }
        else {
            uint32_t k = id * 2;
            int32_t hf_prev = s_hf[id - 1];
            lf_i = s_in[k] + ((hf_prev + hf_i + 2) >> 2);
        }
        if (lf_finalize) {
            lf_row16[id] = dwt_shift_to_u16(lf_i, shift_out, offset_out);
        }
        else {
            lf_row[id] = lf_i;
        }
    }

    /* Phase 3: trailing boundary sample (matches dwt_lift53()'s tail branch
     * exactly -- single thread, negligible cost, needs only s_hf[count-1]
     * which phase 1's __syncthreads() above already made visible). */
    if (threadIdx.x == 0) {
        int32_t hf_prev = s_hf[count - 1];
        if (!(width & 1)) {
            uint32_t last = width / 2 - 1;
            int32_t hf_i = s_in[width - 1] - s_in[width - 2];
            if (hf_finalize) {
                hf_row16[last] = dwt_shift_to_u16(hf_i, shift_out, offset_out);
            }
            else {
                hf_row[last] = hf_i;
            }
            int32_t lf_i = s_in[width - 2] + ((hf_prev + hf_i + 2) >> 2);
            if (lf_finalize) {
                lf_row16[last] = dwt_shift_to_u16(lf_i, shift_out, offset_out);
            }
            else {
                lf_row[last] = lf_i;
            }
        }
        else {
            int32_t lf_i = s_in[width - 1] + ((hf_prev + 1) >> 1);
            if (lf_finalize) {
                lf_row16[width / 2] = dwt_shift_to_u16(lf_i, shift_out, offset_out);
            }
            else {
                lf_row[width / 2] = lf_i;
            }
        }
    }
}

/* Launch config helpers for the tiled k_horizontal_lift above. */
#define DWT_H_TILE_THREADS 256
static inline size_t dwt_h_tile_shmem_bytes(uint32_t width) {
    return (size_t)(width + width / 2) * sizeof(int32_t);
}

/* One thread per column: splits `height` samples of each column into
 * height1=height-height/2 LF samples and height2=height/2 HF samples. */
__global__ void k_vertical_lift(const int32_t* src, uint32_t pitch, uint32_t sx, uint32_t sy, uint32_t width, uint32_t height,
                                int32_t* dst_lf, uint32_t lf_pitch, uint32_t lfx, uint32_t lfy, int32_t* dst_hf,
                                uint32_t hf_pitch, uint32_t hfx, uint32_t hfy) {
    uint32_t col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= width)
        return;
    const int32_t* in_col = src + (size_t)sy * pitch + (sx + col);
    int32_t* lf_col = dst_lf + (size_t)lfy * lf_pitch + (lfx + col);
    int32_t* hf_col = dst_hf + (size_t)hfy * hf_pitch + (hfx + col);
    dwt_lift53(in_col, (int)pitch, lf_col, (int)lf_pitch, hf_col, (int)hf_pitch, height);
}

/* [2026-08-28 Phase B, Step B3] NLT-fused first vertical lift: used only for
 * the v==0 level when decom_v>0, replacing a separate k_nlt_scale_8bit/_16bit
 * pass (which would write the whole comp_width x comp_height buffer to
 * d_cur) plus this level's k_vertical_lift read of that same data right
 * back. Reads raw source samples directly and applies the NLT shift/offset
 * on the fly via DwtNlt8Reader/DwtNlt16Reader, eliminating that round trip.
 * v==0 always has sx=sy=lfx=lfy=hfx=0 at the only call site, so those
 * offsets are dropped from the signature (unlike the general k_vertical_lift
 * above); only hfy (=h1, the HF region's row offset within d_vert) is kept. */
__global__ void k_vertical_lift_nlt_8bit(const uint8_t* src, uint32_t src_pitch, uint32_t width, uint32_t height, uint8_t shift,
                                         int32_t offset, int32_t* dst_lf, uint32_t lf_pitch, int32_t* dst_hf, uint32_t hf_pitch,
                                         uint32_t hfy) {
    uint32_t col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= width)
        return;
    DwtNlt8Reader reader{src, src_pitch, col, shift, offset};
    dwt_lift53_r(reader, dst_lf + col, (int)lf_pitch, dst_hf + (size_t)hfy * hf_pitch + col, (int)hf_pitch, height);
}

__global__ void k_vertical_lift_nlt_16bit(const uint16_t* src, uint32_t src_pitch, uint32_t width, uint32_t height, uint8_t shift,
                                          int32_t offset, uint8_t bit_depth, int32_t* dst_lf, uint32_t lf_pitch, int32_t* dst_hf,
                                          uint32_t hf_pitch, uint32_t hfy) {
    uint32_t col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= width)
        return;
    uint16_t mask = (uint16_t)((1u << bit_depth) - 1);
    DwtNlt16Reader reader{src, src_pitch, col, shift, offset, mask};
    dwt_lift53_r(reader, dst_lf + col, (int)lf_pitch, dst_hf + (size_t)hfy * hf_pitch + col, (int)hf_pitch, height);
}

/* Matches linear_input_scaling_line_8bit_c/_16bit_c (NltEnc.c), LSB-aligned
 * (hdr_input_msb_aligned==0) path only. */
__global__ void k_nlt_scale_8bit(const uint8_t* src, uint32_t src_pitch, int32_t* dst, uint32_t dst_pitch, uint32_t width,
                                 uint32_t height, uint8_t shift, int32_t offset) {
    uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height)
        return;
    uint8_t v = src[(size_t)y * src_pitch + x];
    dst[(size_t)y * dst_pitch + x] = (int32_t)((uint32_t)v << shift) - offset;
}

__global__ void k_nlt_scale_16bit(const uint16_t* src, uint32_t src_pitch, int32_t* dst, uint32_t dst_pitch, uint32_t width,
                                  uint32_t height, uint8_t shift, int32_t offset, uint8_t bit_depth) {
    uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height)
        return;
    uint16_t mask = (uint16_t)((1u << bit_depth) - 1);
    uint16_t v = src[(size_t)y * src_pitch + x] & mask;
    dst[(size_t)y * dst_pitch + x] = (int32_t)((uint32_t)v << shift) - offset;
}

/* Matches image_shift_c (NltEnc.c): converts raw lifting-scheme int32
 * coefficients to 16-bit sign+magnitude, applied uniformly over the whole
 * pyramid (shift/offset are the same constant for every band on the CPU
 * side too, so applying this once at the end is bit-identical to applying
 * it per-band as the CPU streaming code does). */
__global__ void k_image_shift(const int32_t* src, uint32_t pitch, uint32_t width, uint32_t height, uint16_t* dst,
                              uint32_t dst_pitch, int32_t shift, int32_t offset) {
    uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height)
        return;
    dst[(size_t)y * dst_pitch + x] = dwt_shift_to_u16(src[(size_t)y * pitch + x], shift, offset);
}

int svt_cuda_dwt_component(const void* in_plane, uint32_t plane_stride, uint32_t comp_width, uint32_t comp_height,
                           uint32_t decom_h, uint32_t decom_v, uint8_t input_bit_depth, uint8_t hdr_Bw, uint8_t hdr_Fq,
                           uint16_t* out_pyramid_16bit) {
    if (comp_width < 2 || comp_height < 2 || decom_h < decom_v) {
        return 1;
    }

    const size_t elems = (size_t)comp_width * comp_height;
    const size_t bytes32 = elems * sizeof(int32_t);
    /* [2026-08-28 Phase B, Step B1 -- tried, reverted] k_vertical_lift's
     * grid size (=ceil(width/THREADS)) is small at the real 4K geometry's
     * widest levels (15 blocks at L0, 8 at L1 with THREADS=256), which
     * theoretically starves this GPU's SMs (worse on wider-SM-count GPUs).
     * Swept THREADS=128/64/32 (all should proportionally raise grid size
     * with zero change to the arithmetic path, since this kernel has no
     * shared memory/per-block setup) and measured no change in NLT+DWT time
     * within noise on this dev machine (RTX 2000 Ada) -- see
     * PortingStrategy.txt Phase 6 notes for the full measurement record and
     * discussion. Reverted to 256; kept as a documented negative result
     * rather than silently dropped. */
    const uint32_t THREADS = 256;

    uint8_t* d_in_raw = NULL;
    /* d_cur: active-region input for the level about to run.
     * d_other: where the next level's active region (new LL) gets written.
     * d_vert: scratch holding the vertical-only lift result within a
     *         vertical level (never used as a cross-level active buffer).
     * d_pyramid: final output; finalized (non-recursing) bands are written
     *            directly to their absolute pyramid position and never
     *            touched again. */
    int32_t *d_cur = NULL, *d_other = NULL, *d_vert = NULL, *d_pyramid = NULL;
    uint16_t* d_out = NULL;
    cudaError_t err = cudaSuccess;

    do {
        size_t in_elem_size = (input_bit_depth <= 8) ? sizeof(uint8_t) : sizeof(uint16_t);
        if ((err = cudaMalloc(&d_in_raw, (size_t)plane_stride * comp_height * in_elem_size)) != cudaSuccess)
            break;
        if ((err = cudaMalloc(&d_cur, bytes32)) != cudaSuccess)
            break;
        if ((err = cudaMalloc(&d_other, bytes32)) != cudaSuccess)
            break;
        if ((err = cudaMalloc(&d_vert, bytes32)) != cudaSuccess)
            break;
        if ((err = cudaMalloc(&d_pyramid, bytes32)) != cudaSuccess)
            break;
        if ((err = cudaMalloc(&d_out, elems * sizeof(uint16_t))) != cudaSuccess)
            break;

        if ((err = cudaMemcpy2D(d_in_raw,
                                plane_stride * in_elem_size,
                                in_plane,
                                plane_stride * in_elem_size,
                                comp_width * in_elem_size,
                                comp_height,
                                cudaMemcpyHostToDevice)) != cudaSuccess)
            break;

        const uint8_t shift = hdr_Bw - input_bit_depth;
        const int32_t offset = 1 << (hdr_Bw - 1);
        dim3 block2d(32, 8);
        dim3 grid2d((comp_width + block2d.x - 1) / block2d.x, (comp_height + block2d.y - 1) / block2d.y);
        /* [2026-08-28 Phase B, Step B3] When decom_v>0, the v==0 iteration's
         * k_vertical_lift_nlt_* below reads d_in_raw directly and applies
         * this NLT scale on the fly -- skip the separate whole-buffer NLT
         * pass entirely in that case (d_cur's level-0 content would never be
         * read otherwise). decom_v==0 is untouched: NLT must still populate
         * d_cur here since the h-only loop's first k_horizontal_lift reads
         * it directly. */
        if (decom_v == 0) {
            if (input_bit_depth <= 8) {
                k_nlt_scale_8bit<<<grid2d, block2d>>>(d_in_raw, plane_stride, d_cur, comp_width, comp_width, comp_height, shift, offset);
            }
            else {
                k_nlt_scale_16bit<<<grid2d, block2d>>>((const uint16_t*)d_in_raw,
                                                       plane_stride,
                                                       d_cur,
                                                       comp_width,
                                                       comp_width,
                                                       comp_height,
                                                       shift,
                                                       offset,
                                                       input_bit_depth);
            }
        }

        uint32_t active_w = comp_width, active_h = comp_height;
        const int32_t shift_out = hdr_Fq;
        const int32_t offset_out = 1 << (hdr_Fq - 1);

        /* Vertical decomposition levels: each level performs one full 2D
         * split (vertical lift, then horizontal lift on each vertical half)
         * producing 4 quadrants {LL,HL,LH,HH}; only LL continues recursion.
         * Band positions match Pi.c's pi_compute() geometry exactly.
         * [2026-08-28 Phase B, Step B2] Finalized bands (HL/LH/HH) are
         * written directly as shift-converted uint16 to d_out via
         * k_horizontal_lift's finalize path, instead of raw int32 to
         * d_pyramid followed by a separate whole-buffer k_image_shift pass.
         * d_pyramid is intentionally passed as NULL wherever a band is
         * finalized (guards against a stale int32 write bug). */
        for (uint32_t v = 0; v < decom_v; v++) {
            uint32_t h2 = active_h / 2, h1 = active_h - h2;
            uint32_t w2 = active_w / 2, w1 = active_w - w2;

            uint32_t gcol = (active_w + THREADS - 1) / THREADS;
            /* [2026-08-28 Phase B, Step B3] Level 0 (only, when decom_v>0)
             * reads raw source samples directly via the NLT-fused kernel
             * instead of the pre-scaled d_cur (which was never populated for
             * this case, see the decom_v==0 guard above). Levels >=1 use the
             * unmodified k_vertical_lift on d_cur exactly as before. */
            if (v == 0) {
                if (input_bit_depth <= 8) {
                    k_vertical_lift_nlt_8bit<<<gcol, THREADS>>>(
                        d_in_raw, plane_stride, active_w, active_h, shift, offset, d_vert, comp_width, d_vert, comp_width, h1);
                }
                else {
                    k_vertical_lift_nlt_16bit<<<gcol, THREADS>>>((const uint16_t*)d_in_raw,
                                                                 plane_stride,
                                                                 active_w,
                                                                 active_h,
                                                                 shift,
                                                                 offset,
                                                                 input_bit_depth,
                                                                 d_vert,
                                                                 comp_width,
                                                                 d_vert,
                                                                 comp_width,
                                                                 h1);
                }
            }
            else {
                k_vertical_lift<<<gcol, THREADS>>>(d_cur,
                                                   comp_width,
                                                   0,
                                                   0,
                                                   active_w,
                                                   active_h,
                                                   d_vert,
                                                   comp_width,
                                                   0,
                                                   0, // V-LF -> rows [0,h1)
                                                   d_vert,
                                                   comp_width,
                                                   0,
                                                   h1); // V-HF -> rows [h1,h1+h2)
            }

            k_horizontal_lift<<<h1, DWT_H_TILE_THREADS, dwt_h_tile_shmem_bytes(active_w)>>>(d_vert,
                                                    comp_width,
                                                    0,
                                                    0,
                                                    active_w,
                                                    h1, // source: V-LF rows
                                                    d_other,
                                                    comp_width,
                                                    0,
                                                    0, // LL -> next active region
                                                    NULL,
                                                    0,
                                                    0,
                                                    0,
                                                    false,
                                                    NULL,
                                                    0,
                                                    0,
                                                    0,
                                                    true,
                                                    d_out,
                                                    comp_width,
                                                    w1,
                                                    0, // HL -> finalized directly to d_out
                                                    shift_out,
                                                    offset_out);

            k_horizontal_lift<<<h2, DWT_H_TILE_THREADS, dwt_h_tile_shmem_bytes(active_w)>>>(d_vert,
                                                      comp_width,
                                                      0,
                                                      h1,
                                                      active_w,
                                                      h2, // source: V-HF rows
                                                      NULL,
                                                      0,
                                                      0,
                                                      0,
                                                      NULL,
                                                      0,
                                                      0,
                                                      0,
                                                      true,
                                                      d_out,
                                                      comp_width,
                                                      0,
                                                      h1, // LH -> finalized directly to d_out
                                                      true,
                                                      d_out,
                                                      comp_width,
                                                      w1,
                                                      h1, // HH -> finalized directly to d_out
                                                      shift_out,
                                                      offset_out);

            active_w = w1;
            active_h = h1;
            int32_t* tmp = d_cur;
            d_cur = d_other;
            d_other = tmp;
        }

        /* Remaining horizontal-only decomposition levels (decom_h > decom_v),
         * applied only to the LL path. */
        for (uint32_t hh = decom_v; hh < decom_h; hh++) {
            uint32_t w2 = active_w / 2, w1 = active_w - w2;
            k_horizontal_lift<<<active_h, DWT_H_TILE_THREADS, dwt_h_tile_shmem_bytes(active_w)>>>(d_cur,
                                                 comp_width,
                                                 0,
                                                 0,
                                                 active_w,
                                                 active_h,
                                                 d_other,
                                                 comp_width,
                                                 0,
                                                 0, // LF -> next active region
                                                 NULL,
                                                 0,
                                                 0,
                                                 0,
                                                 false,
                                                 NULL,
                                                 0,
                                                 0,
                                                 0,
                                                 true,
                                                 d_out,
                                                 comp_width,
                                                 w1,
                                                 0, // HF -> finalized directly to d_out
                                                 shift_out,
                                                 offset_out);
            active_w = w1;
            int32_t* tmp = d_cur;
            d_cur = d_other;
            d_other = tmp;
        }

        /* Final remaining LL band (band 0) sits in d_cur at (0,0,active_w,active_h).
         * [2026-08-28 Phase B, Step B2] Previously copied into d_pyramid then
         * shift-converted by a whole-buffer k_image_shift call; now shift
         * converted directly from d_cur into d_out's (0,0) corner, scoped to
         * just the LL band's actual size instead of the whole comp_width x
         * comp_height buffer. */
        dim3 grid2d_ll((active_w + block2d.x - 1) / block2d.x, (active_h + block2d.y - 1) / block2d.y);
        k_image_shift<<<grid2d_ll, block2d>>>(d_cur, comp_width, active_w, active_h, d_out, comp_width, shift_out, offset_out);

        if ((err = cudaMemcpy(out_pyramid_16bit, d_out, elems * sizeof(uint16_t), cudaMemcpyDeviceToHost)) != cudaSuccess)
            break;

        err = cudaGetLastError();
    } while (0);

    cudaFree(d_in_raw);
    cudaFree(d_cur);
    cudaFree(d_other);
    cudaFree(d_vert);
    cudaFree(d_pyramid);
    cudaFree(d_out);
    return err == cudaSuccess ? 0 : (int)err;
}

/* Shared tail of svt_cuda_dwt_component_ctx()/svt_cuda_dwt_component_ctx_prefilled():
 * both validate their own inputs and get raw sample data into d_in_raw (one
 * via its own H2D copy, the other via a caller-supplied, already-filled
 * buffer -- e.g. a packed-input deinterleave kernel, see EncodeFrameCuda.cu),
 * then run the identical NLT-scale/vertical-lift/DWT pipeline on it. Kept as
 * one body so the two entry points can never drift out of sync. */
static int dwt_component_ctx_body(uint32_t plane_stride, uint32_t comp_width, uint32_t comp_height, uint32_t decom_h,
                                  uint32_t decom_v, uint8_t input_bit_depth, uint8_t hdr_Bw, uint8_t hdr_Fq,
                                  uint8_t* d_in_raw, int32_t* d_cur, int32_t* d_other, int32_t* d_vert, int32_t* d_pyramid,
                                  uint16_t* d_out_pyramid16, cudaStream_t stream) {
    /* [2026-08-28 Phase B, Step B1 -- tried, reverted] Must stay in lockstep
     * with the identical constant in svt_cuda_dwt_component() above -- see
     * that copy's comment for the full rationale and measurement record. */
    const uint32_t THREADS = 256;

    const uint8_t shift = hdr_Bw - input_bit_depth;
    const int32_t offset = 1 << (hdr_Bw - 1);
    dim3 block2d(32, 8);
    dim3 grid2d((comp_width + block2d.x - 1) / block2d.x, (comp_height + block2d.y - 1) / block2d.y);
    /* [2026-08-28 Phase B, Step B3] Same NLT-into-level-0 fusion as
     * svt_cuda_dwt_component() above -- see that copy's comment for the
     * full rationale. */
    if (decom_v == 0) {
        if (input_bit_depth <= 8) {
            k_nlt_scale_8bit<<<grid2d, block2d, 0, stream>>>(
                d_in_raw, plane_stride, d_cur, comp_width, comp_width, comp_height, shift, offset);
        }
        else {
            k_nlt_scale_16bit<<<grid2d, block2d, 0, stream>>>(
                (const uint16_t*)d_in_raw, plane_stride, d_cur, comp_width, comp_width, comp_height, shift, offset, input_bit_depth);
        }
    }

    uint32_t active_w = comp_width, active_h = comp_height;
    const int32_t shift_out = hdr_Fq;
    const int32_t offset_out = 1 << (hdr_Fq - 1);

    /* [2026-08-28 Phase B, Step B2] Same band-write fusion as
     * svt_cuda_dwt_component() above -- see that copy's comment for the
     * full rationale. d_pyramid is unused by this path now (kept only for
     * signature/API compatibility with FrameContextCuda's persistent
     * buffers; removing it is out of scope for this step). */
    for (uint32_t v = 0; v < decom_v; v++) {
        uint32_t h2 = active_h / 2, h1 = active_h - h2;
        uint32_t w2 = active_w / 2, w1 = active_w - w2;

        uint32_t gcol = (active_w + THREADS - 1) / THREADS;
        /* [2026-08-28 Phase B, Step B3] Same v==0 NLT-fused branch as
         * svt_cuda_dwt_component() above. */
        if (v == 0) {
            if (input_bit_depth <= 8) {
                k_vertical_lift_nlt_8bit<<<gcol, THREADS, 0, stream>>>(
                    d_in_raw, plane_stride, active_w, active_h, shift, offset, d_vert, comp_width, d_vert, comp_width, h1);
            }
            else {
                k_vertical_lift_nlt_16bit<<<gcol, THREADS, 0, stream>>>((const uint16_t*)d_in_raw,
                                                                        plane_stride,
                                                                        active_w,
                                                                        active_h,
                                                                        shift,
                                                                        offset,
                                                                        input_bit_depth,
                                                                        d_vert,
                                                                        comp_width,
                                                                        d_vert,
                                                                        comp_width,
                                                                        h1);
            }
        }
        else {
            k_vertical_lift<<<gcol, THREADS, 0, stream>>>(d_cur,
                                                           comp_width,
                                                           0,
                                                           0,
                                                           active_w,
                                                           active_h,
                                                           d_vert,
                                                           comp_width,
                                                           0,
                                                           0,
                                                           d_vert,
                                                           comp_width,
                                                           0,
                                                           h1);
        }

        k_horizontal_lift<<<h1, DWT_H_TILE_THREADS, dwt_h_tile_shmem_bytes(active_w), stream>>>(d_vert,
                                                            comp_width,
                                                            0,
                                                            0,
                                                            active_w,
                                                            h1,
                                                            d_other,
                                                            comp_width,
                                                            0,
                                                            0,
                                                            NULL,
                                                            0,
                                                            0,
                                                            0,
                                                            false,
                                                            NULL,
                                                            0,
                                                            0,
                                                            0,
                                                            true,
                                                            d_out_pyramid16,
                                                            comp_width,
                                                            w1,
                                                            0,
                                                            shift_out,
                                                            offset_out);

        k_horizontal_lift<<<h2, DWT_H_TILE_THREADS, dwt_h_tile_shmem_bytes(active_w), stream>>>(d_vert,
                                                              comp_width,
                                                              0,
                                                              h1,
                                                              active_w,
                                                              h2,
                                                              NULL,
                                                              0,
                                                              0,
                                                              0,
                                                              NULL,
                                                              0,
                                                              0,
                                                              0,
                                                              true,
                                                              d_out_pyramid16,
                                                              comp_width,
                                                              0,
                                                              h1,
                                                              true,
                                                              d_out_pyramid16,
                                                              comp_width,
                                                              w1,
                                                              h1,
                                                              shift_out,
                                                              offset_out);

        active_w = w1;
        active_h = h1;
        int32_t* tmp = d_cur;
        d_cur = d_other;
        d_other = tmp;
    }

    for (uint32_t hh = decom_v; hh < decom_h; hh++) {
        uint32_t w2 = active_w / 2, w1 = active_w - w2;
        k_horizontal_lift<<<active_h, DWT_H_TILE_THREADS, dwt_h_tile_shmem_bytes(active_w), stream>>>(d_cur,
                                                         comp_width,
                                                         0,
                                                         0,
                                                         active_w,
                                                         active_h,
                                                         d_other,
                                                         comp_width,
                                                         0,
                                                         0,
                                                         NULL,
                                                         0,
                                                         0,
                                                         0,
                                                         false,
                                                         NULL,
                                                         0,
                                                         0,
                                                         0,
                                                         true,
                                                         d_out_pyramid16,
                                                         comp_width,
                                                         w1,
                                                         0,
                                                         shift_out,
                                                         offset_out);
        active_w = w1;
        int32_t* tmp = d_cur;
        d_cur = d_other;
        d_other = tmp;
    }

    /* [2026-08-28 Phase B, Step B2] LL band shift-converted directly from
     * d_cur into d_out_pyramid16's (0,0) corner, scoped to active_w x
     * active_h instead of the whole-buffer cudaMemcpy2DAsync+k_image_shift
     * that used to run here. */
    dim3 grid2d_ll((active_w + block2d.x - 1) / block2d.x, (active_h + block2d.y - 1) / block2d.y);
    k_image_shift<<<grid2d_ll, block2d, 0, stream>>>(
        d_cur, comp_width, active_w, active_h, d_out_pyramid16, comp_width, shift_out, offset_out);

    return 0;
}

int svt_cuda_dwt_component_ctx(const void* in_plane, uint32_t plane_stride, uint32_t comp_width, uint32_t comp_height,
                               uint32_t decom_h, uint32_t decom_v, uint8_t input_bit_depth, uint8_t hdr_Bw, uint8_t hdr_Fq,
                               uint8_t* d_in_raw, int32_t* d_cur, int32_t* d_other, int32_t* d_vert, int32_t* d_pyramid,
                               uint16_t* d_out_pyramid16, cudaStream_t stream) {
    if (comp_width < 2 || comp_height < 2 || decom_h < decom_v) {
        return 1;
    }

    size_t in_elem_size = (input_bit_depth <= 8) ? sizeof(uint8_t) : sizeof(uint16_t);
    cudaError_t err = cudaMemcpy2DAsync(d_in_raw,
                                        plane_stride * in_elem_size,
                                        in_plane,
                                        plane_stride * in_elem_size,
                                        comp_width * in_elem_size,
                                        comp_height,
                                        cudaMemcpyHostToDevice,
                                        stream);
    if (err != cudaSuccess) {
        return (int)err;
    }

    return dwt_component_ctx_body(plane_stride, comp_width, comp_height, decom_h, decom_v, input_bit_depth, hdr_Bw, hdr_Fq,
                                  d_in_raw, d_cur, d_other, d_vert, d_pyramid, d_out_pyramid16, stream);
}

/* Packed-input variant (2026-09-02, see PortingStrategy.txt "channel-interleaved
 * input" section): d_in_raw is assumed ALREADY filled with this component's
 * planar samples (pitch == comp_width, i.e. no padding) by the caller's
 * deinterleave kernel -- see EncodeFrameCuda.cu's efc_run_deinterleave_packed()
 * -- so this skips the H2D copy entirely and reuses the identical downstream
 * pipeline via dwt_component_ctx_body(). */
int svt_cuda_dwt_component_ctx_prefilled(uint32_t comp_width, uint32_t comp_height, uint32_t decom_h, uint32_t decom_v,
                                         uint8_t input_bit_depth, uint8_t hdr_Bw, uint8_t hdr_Fq, uint8_t* d_in_raw,
                                         int32_t* d_cur, int32_t* d_other, int32_t* d_vert, int32_t* d_pyramid,
                                         uint16_t* d_out_pyramid16, cudaStream_t stream) {
    if (comp_width < 2 || comp_height < 2 || decom_h < decom_v) {
        return 1;
    }

    return dwt_component_ctx_body(comp_width /* plane_stride == comp_width, no padding */, comp_width, comp_height, decom_h,
                                  decom_v, input_bit_depth, hdr_Bw, hdr_Fq, d_in_raw, d_cur, d_other, d_vert, d_pyramid,
                                  d_out_pyramid16, stream);
}

/* Deinterleaves one frame's worth of packed(AoS) RGB/444 samples into 3
 * separate planar(SoA) buffers in a single pass (one read of `packed` per
 * pixel, one write per output plane) -- see PortingStrategy.txt's prototype
 * benchmark ("[B] 1x fused launch" beat 3 separate per-channel launches by
 * ~2x, since the latter re-reads the whole packed buffer once per channel). */
__global__ void k_deinterleave_packed_to_planar_8bit(const uint8_t* __restrict__ packed, uint8_t* __restrict__ out0,
                                                      uint8_t* __restrict__ out1, uint8_t* __restrict__ out2, uint32_t n) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    uint32_t base = i * 3;
    out0[i] = packed[base + 0];
    out1[i] = packed[base + 1];
    out2[i] = packed[base + 2];
}

__global__ void k_deinterleave_packed_to_planar_16bit(const uint16_t* __restrict__ packed, uint16_t* __restrict__ out0,
                                                       uint16_t* __restrict__ out1, uint16_t* __restrict__ out2, uint32_t n) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    uint32_t base = i * 3;
    out0[i] = packed[base + 0];
    out1[i] = packed[base + 1];
    out2[i] = packed[base + 2];
}

int svt_cuda_deinterleave_packed_rgb(const void* d_packed, uint8_t* d_out0, uint8_t* d_out1, uint8_t* d_out2, uint32_t n,
                                     uint8_t input_bit_depth, cudaStream_t stream) {
    const uint32_t threads = 256, blocks = (n + threads - 1) / threads;
    if (input_bit_depth <= 8) {
        k_deinterleave_packed_to_planar_8bit<<<blocks, threads, 0, stream>>>((const uint8_t*)d_packed, d_out0, d_out1, d_out2, n);
    }
    else {
        k_deinterleave_packed_to_planar_16bit<<<blocks, threads, 0, stream>>>(
            (const uint16_t*)d_packed, (uint16_t*)d_out0, (uint16_t*)d_out1, (uint16_t*)d_out2, n);
    }
    return (int)cudaPeekAtLastError();
}
