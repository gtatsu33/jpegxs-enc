/*
* Copyright(c) 2024 Intel Corporation
* SPDX - License - Identifier: BSD - 2 - Clause - Patent
*/
#include <cuda_runtime.h>
#include "DwtCuda.cuh"

#define CUDA_DWT_SIGN_MASK ((uint16_t)1 << 15) /* matches BITSTREAM_MASK_SIGN, Codestream.h */

/* Generic 5/3 reversible lifting for one 1D line of `len` samples, matching
 * dwt_horizontal_line_c (Dwt.c) exactly -- including its boundary formulas
 * for len==2 and even/odd trailing samples. Used for BOTH the horizontal
 * pass (stride=1 along a row) and the vertical pass (stride=pitch along a
 * column): the CPU's vertical lifting functions (transform_vertical_loop_*_c)
 * implement this identical recurrence, just applied along columns. */
__device__ __forceinline__ void dwt_lift53(const int32_t* in, int in_stride, int32_t* out_lf, int lf_stride, int32_t* out_hf,
                                            int hf_stride, uint32_t len) {
    if (len == 2) {
        int32_t hf0 = in[1 * in_stride] - in[0 * in_stride];
        out_hf[0] = hf0;
        out_lf[0] = in[0 * in_stride] + ((hf0 + 1) >> 1);
        return;
    }

    uint32_t count = (len - 1) / 2;
    int32_t hf_i = in[1 * in_stride] - ((in[0 * in_stride] + in[2 * in_stride]) >> 1);
    out_hf[0] = hf_i;
    out_lf[0] = in[0 * in_stride] + ((hf_i + 1) >> 1);
    int32_t hf_prev = hf_i;

    for (uint32_t id = 1; id < count; id++) {
        uint32_t k = id * 2;
        hf_i = in[(k + 1) * in_stride] - ((in[k * in_stride] + in[(k + 2) * in_stride]) >> 1);
        out_hf[id * hf_stride] = hf_i;
        out_lf[id * lf_stride] = in[k * in_stride] + ((hf_prev + hf_i + 2) >> 2);
        hf_prev = hf_i;
    }

    if (!(len & 1)) {
        uint32_t last = len / 2 - 1;
        hf_i = in[(len - 1) * in_stride] - in[(len - 2) * in_stride];
        out_hf[last * hf_stride] = hf_i;
        out_lf[last * lf_stride] = in[(len - 2) * in_stride] + ((hf_prev + hf_i + 2) >> 2);
    }
    else {
        out_lf[(len / 2) * lf_stride] = in[(len - 1) * in_stride] + ((hf_prev + 1) >> 1);
    }
}

/* One thread per row: splits `width` samples of each row into
 * width1=width-width/2 LF samples and width2=width/2 HF samples. */
__global__ void k_horizontal_lift(const int32_t* src, uint32_t pitch, uint32_t sx, uint32_t sy, uint32_t width, uint32_t height,
                                  int32_t* dst_lf, uint32_t lf_pitch, uint32_t lfx, uint32_t lfy, int32_t* dst_hf,
                                  uint32_t hf_pitch, uint32_t hfx, uint32_t hfy) {
    uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= height)
        return;
    const int32_t* in_row = src + (size_t)(sy + row) * pitch + sx;
    int32_t* lf_row = dst_lf + (size_t)(lfy + row) * lf_pitch + lfx;
    int32_t* hf_row = dst_hf + (size_t)(hfy + row) * hf_pitch + hfx;
    dwt_lift53(in_row, 1, lf_row, 1, hf_row, 1, width);
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
    int32_t val = src[(size_t)y * pitch + x];
    uint16_t out_v;
    if (val >= 0) {
        out_v = (uint16_t)((val + offset) >> shift);
    }
    else {
        int32_t v = (-val + offset) >> shift;
        out_v = (uint16_t)v;
        if (v) {
            out_v |= CUDA_DWT_SIGN_MASK;
        }
    }
    dst[(size_t)y * dst_pitch + x] = out_v;
}

int svt_cuda_dwt_component(const void* in_plane, uint32_t plane_stride, uint32_t comp_width, uint32_t comp_height,
                           uint32_t decom_h, uint32_t decom_v, uint8_t input_bit_depth, uint8_t hdr_Bw, uint8_t hdr_Fq,
                           uint16_t* out_pyramid_16bit) {
    if (comp_width < 2 || comp_height < 2 || decom_h < decom_v) {
        return 1;
    }

    const size_t elems = (size_t)comp_width * comp_height;
    const size_t bytes32 = elems * sizeof(int32_t);
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
        if (input_bit_depth <= 8) {
            k_nlt_scale_8bit<<<grid2d, block2d>>>(d_in_raw, plane_stride, d_cur, comp_width, comp_width, comp_height, shift, offset);
        }
        else {
            k_nlt_scale_16bit<<<grid2d, block2d>>>(
                (const uint16_t*)d_in_raw, plane_stride, d_cur, comp_width, comp_width, comp_height, shift, offset, input_bit_depth);
        }

        uint32_t active_w = comp_width, active_h = comp_height;

        /* Vertical decomposition levels: each level performs one full 2D
         * split (vertical lift, then horizontal lift on each vertical half)
         * producing 4 quadrants {LL,HL,LH,HH}; only LL continues recursion.
         * Band positions match Pi.c's pi_compute() geometry exactly. */
        for (uint32_t v = 0; v < decom_v; v++) {
            uint32_t h2 = active_h / 2, h1 = active_h - h2;
            uint32_t w2 = active_w / 2, w1 = active_w - w2;

            uint32_t gcol = (active_w + THREADS - 1) / THREADS;
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

            uint32_t grow_up = (h1 + THREADS - 1) / THREADS;
            k_horizontal_lift<<<grow_up, THREADS>>>(d_vert,
                                                    comp_width,
                                                    0,
                                                    0,
                                                    active_w,
                                                    h1, // source: V-LF rows
                                                    d_other,
                                                    comp_width,
                                                    0,
                                                    0, // LL -> next active region
                                                    d_pyramid,
                                                    comp_width,
                                                    w1,
                                                    0); // HL -> finalized

            uint32_t grow_down = (h2 + THREADS - 1) / THREADS;
            k_horizontal_lift<<<grow_down, THREADS>>>(d_vert,
                                                      comp_width,
                                                      0,
                                                      h1,
                                                      active_w,
                                                      h2, // source: V-HF rows
                                                      d_pyramid,
                                                      comp_width,
                                                      0,
                                                      h1, // LH -> finalized
                                                      d_pyramid,
                                                      comp_width,
                                                      w1,
                                                      h1); // HH -> finalized

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
            uint32_t grow = (active_h + THREADS - 1) / THREADS;
            k_horizontal_lift<<<grow, THREADS>>>(d_cur,
                                                 comp_width,
                                                 0,
                                                 0,
                                                 active_w,
                                                 active_h,
                                                 d_other,
                                                 comp_width,
                                                 0,
                                                 0, // LF -> next active region
                                                 d_pyramid,
                                                 comp_width,
                                                 w1,
                                                 0); // HF -> finalized
            active_w = w1;
            int32_t* tmp = d_cur;
            d_cur = d_other;
            d_other = tmp;
        }

        /* Final remaining LL band (band 0) sits in d_cur at (0,0,active_w,active_h). */
        if ((err = cudaMemcpy2D(d_pyramid,
                                comp_width * sizeof(int32_t),
                                d_cur,
                                comp_width * sizeof(int32_t),
                                active_w * sizeof(int32_t),
                                active_h,
                                cudaMemcpyDeviceToDevice)) != cudaSuccess)
            break;

        const int32_t shift_out = hdr_Fq;
        const int32_t offset_out = 1 << (hdr_Fq - 1);
        k_image_shift<<<grid2d, block2d>>>(d_pyramid, comp_width, comp_width, comp_height, d_out, comp_width, shift_out, offset_out);

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

int svt_cuda_dwt_component_ctx(const void* in_plane, uint32_t plane_stride, uint32_t comp_width, uint32_t comp_height,
                               uint32_t decom_h, uint32_t decom_v, uint8_t input_bit_depth, uint8_t hdr_Bw, uint8_t hdr_Fq,
                               uint8_t* d_in_raw, int32_t* d_cur, int32_t* d_other, int32_t* d_vert, int32_t* d_pyramid,
                               uint16_t* d_out_pyramid16, cudaStream_t stream) {
    if (comp_width < 2 || comp_height < 2 || decom_h < decom_v) {
        return 1;
    }

    const uint32_t THREADS = 256;
    cudaError_t err = cudaSuccess;

    size_t in_elem_size = (input_bit_depth <= 8) ? sizeof(uint8_t) : sizeof(uint16_t);
    if ((err = cudaMemcpy2DAsync(d_in_raw,
                                 plane_stride * in_elem_size,
                                 in_plane,
                                 plane_stride * in_elem_size,
                                 comp_width * in_elem_size,
                                 comp_height,
                                 cudaMemcpyHostToDevice,
                                 stream)) != cudaSuccess) {
        return (int)err;
    }

    const uint8_t shift = hdr_Bw - input_bit_depth;
    const int32_t offset = 1 << (hdr_Bw - 1);
    dim3 block2d(32, 8);
    dim3 grid2d((comp_width + block2d.x - 1) / block2d.x, (comp_height + block2d.y - 1) / block2d.y);
    if (input_bit_depth <= 8) {
        k_nlt_scale_8bit<<<grid2d, block2d, 0, stream>>>(
            d_in_raw, plane_stride, d_cur, comp_width, comp_width, comp_height, shift, offset);
    }
    else {
        k_nlt_scale_16bit<<<grid2d, block2d, 0, stream>>>(
            (const uint16_t*)d_in_raw, plane_stride, d_cur, comp_width, comp_width, comp_height, shift, offset, input_bit_depth);
    }

    uint32_t active_w = comp_width, active_h = comp_height;

    for (uint32_t v = 0; v < decom_v; v++) {
        uint32_t h2 = active_h / 2, h1 = active_h - h2;
        uint32_t w2 = active_w / 2, w1 = active_w - w2;

        uint32_t gcol = (active_w + THREADS - 1) / THREADS;
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

        uint32_t grow_up = (h1 + THREADS - 1) / THREADS;
        k_horizontal_lift<<<grow_up, THREADS, 0, stream>>>(d_vert,
                                                            comp_width,
                                                            0,
                                                            0,
                                                            active_w,
                                                            h1,
                                                            d_other,
                                                            comp_width,
                                                            0,
                                                            0,
                                                            d_pyramid,
                                                            comp_width,
                                                            w1,
                                                            0);

        uint32_t grow_down = (h2 + THREADS - 1) / THREADS;
        k_horizontal_lift<<<grow_down, THREADS, 0, stream>>>(d_vert,
                                                              comp_width,
                                                              0,
                                                              h1,
                                                              active_w,
                                                              h2,
                                                              d_pyramid,
                                                              comp_width,
                                                              0,
                                                              h1,
                                                              d_pyramid,
                                                              comp_width,
                                                              w1,
                                                              h1);

        active_w = w1;
        active_h = h1;
        int32_t* tmp = d_cur;
        d_cur = d_other;
        d_other = tmp;
    }

    for (uint32_t hh = decom_v; hh < decom_h; hh++) {
        uint32_t w2 = active_w / 2, w1 = active_w - w2;
        uint32_t grow = (active_h + THREADS - 1) / THREADS;
        k_horizontal_lift<<<grow, THREADS, 0, stream>>>(d_cur,
                                                         comp_width,
                                                         0,
                                                         0,
                                                         active_w,
                                                         active_h,
                                                         d_other,
                                                         comp_width,
                                                         0,
                                                         0,
                                                         d_pyramid,
                                                         comp_width,
                                                         w1,
                                                         0);
        active_w = w1;
        int32_t* tmp = d_cur;
        d_cur = d_other;
        d_other = tmp;
    }

    if ((err = cudaMemcpy2DAsync(d_pyramid,
                                 comp_width * sizeof(int32_t),
                                 d_cur,
                                 comp_width * sizeof(int32_t),
                                 active_w * sizeof(int32_t),
                                 active_h,
                                 cudaMemcpyDeviceToDevice,
                                 stream)) != cudaSuccess) {
        return (int)err;
    }

    const int32_t shift_out = hdr_Fq;
    const int32_t offset_out = 1 << (hdr_Fq - 1);
    k_image_shift<<<grid2d, block2d, 0, stream>>>(
        d_pyramid, comp_width, comp_width, comp_height, d_out_pyramid16, comp_width, shift_out, offset_out);

    return 0;
}
