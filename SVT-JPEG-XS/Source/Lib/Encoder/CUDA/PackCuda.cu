/*
* Copyright(c) 2024 Intel Corporation
* SPDX - License - Identifier: BSD - 2 - Clause - Patent
*/
#include <cuda_runtime.h>
#include "PackCuda.cuh"

#define PC_GROUP_SIZE 4
#define PC_SIGNIFICANCE_GROUP_SIZE 8
#define PC_SIGN_MASK ((uint16_t)1 << 15)      /* BITSTREAM_MASK_SIGN */
#define PC_SIGN_BIT_POS 15                    /* BITSTREAM_BIT_POSITION_SIGN */
#define PC_PRECINCT_HEADER_SIZE_BYTES 5
#define PC_CODING_MODE_FLAG_VERTICAL_PRED 1u   /* bit0 (always 0 in this Phase 3 scope) */
#define PC_CODING_MODE_FLAG_SIGNIFICANCE 2u    /* bit1 */

/* Minimal device-side port of bitstream_writer_t (BitstreamWriter.c),
 * matching its bit-exact semantics (MSB-first within a byte). */
struct PcWriter {
    uint8_t* mem;
    uint32_t offset;
    uint32_t bits_used;
};

__device__ __forceinline__ void pc_write_1_bit(PcWriter* bw, uint8_t input) {
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

/* Matches write_N_bits (BitstreamWriter.c), bits in [1,32], MSB-first. */
__device__ __forceinline__ void pc_write_n_bits(PcWriter* bw, uint32_t input, uint8_t bits) {
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

__device__ __forceinline__ void pc_write_4_bits_align4(PcWriter* bw, uint8_t input) {
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

__device__ __forceinline__ uint32_t pc_used_bits(const PcWriter* bw) {
    return bw->offset * 8 + bw->bits_used;
}

__device__ __forceinline__ void pc_align_byte(PcWriter* bw) {
    if (bw->bits_used) {
        bw->offset++;
        bw->bits_used = 0;
    }
}

__device__ __forceinline__ void pc_add_padding_bytes(PcWriter* bw, uint32_t nbytes) {
    for (uint32_t i = 0; i < nbytes; i++) {
        bw->mem[bw->offset + i] = 0;
    }
    bw->offset += nbytes;
}

/* Matches write_packet_header (PackHeaders.c), raw_coding always 0 (Phase 3
 * scope excludes RAW mode). Returns nothing (Phase 3 scope has no sign
 * sub-packet, so the sign_size offset write_packet_header would return is
 * never used). */
__device__ __forceinline__ void pc_write_packet_header(PcWriter* bw, uint32_t long_hdr, uint64_t data_size_bytes,
                                                        uint64_t bitplane_count_size_bytes) {
    uint8_t* mem = bw->mem + bw->offset;
    const uint64_t sign_size_bytes = 0;
    mem[0] = 0; /* raw_coding = 0 */
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

/* Matches vlc_encode_pack_bits / vlc_encode_simple (PackPrecinct.h). */
__device__ __forceinline__ void pc_vlc_encode_simple(PcWriter* bw, int32_t nbits) {
    if (nbits > 1) {
        uint32_t vlc_bits = ((1u << nbits) - 1) << 1;
        pc_write_n_bits(bw, vlc_bits, (uint8_t)(nbits + 1));
    }
    else if (nbits == 1) {
        pc_write_n_bits(bw, 2, 2); /* "10" */
    }
    else {
        pc_write_1_bit(bw, 0);
    }
}

/* Matches pack_significance (PackPrecinct.c). */
__device__ void pc_pack_significance(PcWriter* bw, uint8_t gtli, const uint8_t* sig_max, uint32_t width) {
    for (uint32_t i = 0; i < width; i++) {
        pc_write_1_bit(bw, sig_max[i] <= gtli ? 1 : 0);
    }
}

/* Matches pack_bitplane_count_no_significance. */
__device__ void pc_pack_bitplane_count_no_significance(PcWriter* bw, const uint8_t* bitplane, uint32_t width, int8_t gtli) {
    for (uint32_t i = 0; i < width; i++) {
        pc_vlc_encode_simple(bw, bitplane[i] > gtli ? (int32_t)bitplane[i] - gtli : 0);
    }
}

/* Matches pack_bitplane_count_significance. */
__device__ void pc_pack_bitplane_count_significance(PcWriter* bw, const uint8_t* bitplane, uint32_t width, int8_t gtli,
                                                     const uint8_t* sig_max, uint32_t group_size) {
    uint32_t groups = width / group_size;
    uint32_t leftover = width % group_size;
    uint32_t g = 0;
    for (; g < groups; g++) {
        if (sig_max[g] > gtli) {
            for (uint32_t i = 0; i < group_size; i++) {
                pc_vlc_encode_simple(bw, bitplane[i] > gtli ? (int32_t)bitplane[i] - gtli : 0);
            }
        }
        bitplane += group_size;
    }
    if (leftover) {
        if (sig_max[g] > gtli) {
            for (uint32_t i = 0; i < leftover; i++) {
                pc_vlc_encode_simple(bw, bitplane[i] > gtli ? (int32_t)bitplane[i] - gtli : 0);
            }
        }
    }
}

/* Matches pack_data_single_group_c. */
__device__ void pc_pack_data_single_group(PcWriter* bw, const uint16_t* buf, uint8_t gcli, uint8_t gtli) {
    uint16_t tmp[4];
    for (int i = 0; i < 4; i++) {
        tmp[i] = (uint16_t)((unsigned)buf[i] << ((PC_SIGN_BIT_POS + 1) - gcli));
    }
    for (int32_t bits = (int32_t)gcli - gtli - 1; bits >= 0; bits--) {
        uint16_t val = (uint16_t)(tmp[0] & PC_SIGN_MASK);
        tmp[0] = (uint16_t)(tmp[0] << 1);
        val = (uint16_t)(val | ((tmp[1] & PC_SIGN_MASK) >> 1));
        tmp[1] = (uint16_t)(tmp[1] << 1);
        val = (uint16_t)(val | ((tmp[2] & PC_SIGN_MASK) >> 2));
        tmp[2] = (uint16_t)(tmp[2] << 1);
        val = (uint16_t)(val | ((tmp[3] & PC_SIGN_MASK) >> 3));
        tmp[3] = (uint16_t)(tmp[3] << 1);
        val = (uint16_t)(val >> (PC_SIGN_BIT_POS - 3));
        pc_write_4_bits_align4(bw, (uint8_t)val);
    }
}

/* Matches pack_data_c with sign_flag==0 (Signs=OFF, this Phase 3 scope). */
__device__ void pc_pack_data(PcWriter* bw, const uint16_t* buf, uint32_t width, const uint8_t* gclis, uint8_t gtli) {
    uint32_t groups = width / PC_GROUP_SIZE;
    uint32_t leftover = width % PC_GROUP_SIZE;
    uint32_t group = 0;
    for (; group < groups; group++) {
        if (gclis[group] > gtli) {
            uint8_t signs = (uint8_t)((buf[0] & PC_SIGN_MASK) >> (PC_SIGN_BIT_POS - 3));
            signs |= (uint8_t)((buf[1] & PC_SIGN_MASK) >> (PC_SIGN_BIT_POS - 2));
            signs |= (uint8_t)((buf[2] & PC_SIGN_MASK) >> (PC_SIGN_BIT_POS - 1));
            signs |= (uint8_t)((buf[3] & PC_SIGN_MASK) >> (PC_SIGN_BIT_POS - 0));
            pc_write_4_bits_align4(bw, signs);
            pc_pack_data_single_group(bw, buf, gclis[group], gtli);
        }
        buf += PC_GROUP_SIZE;
    }
    if (leftover) {
        if (gclis[group] > gtli) {
            for (uint32_t i = 0; i < PC_GROUP_SIZE; i++) {
                if (i < leftover) {
                    pc_write_1_bit(bw, (uint8_t)(buf[i] >> PC_SIGN_BIT_POS));
                }
                else {
                    pc_write_1_bit(bw, 0);
                }
            }
            for (int32_t bits = (int32_t)gclis[group] - 1; bits >= gtli; bits--) {
                for (uint32_t i = 0; i < PC_GROUP_SIZE; i++) {
                    if (i < leftover) {
                        pc_write_1_bit(bw, (uint8_t)(buf[i] >> bits));
                    }
                    else {
                        pc_write_1_bit(bw, 0);
                    }
                }
            }
        }
    }
}

/* Single-thread precinct packer, matches pack_precinct() (PackPrecinct.c)
 * for the VPRED-disabled / Signs=OFF / hdr_Rl=0 path. */
__global__ void k_pack_precinct(const svt_cuda_pack_band_info_t* band_info, uint32_t bands_num_all, uint32_t bands_num_exists,
                                const uint8_t* gcli_data, const uint8_t* significance_data, const uint16_t* coeff_data,
                                uint32_t packets_num, const svt_cuda_pack_packet_t* packets, uint8_t use_short_header,
                                uint8_t pack_quantization, uint8_t pack_refinement, uint32_t pack_total_bytes,
                                uint32_t pack_padding_bytes, const uint32_t* packet_size_data_bytes,
                                const uint32_t* packet_size_gcli_bytes, const uint32_t* packet_size_significance_bytes,
                                uint8_t* out_buffer, uint32_t* out_used_bytes, int* out_error) {
    PcWriter bw;
    bw.mem = out_buffer;
    bw.offset = 0;
    bw.bits_used = 0;
    *out_error = 0;

    /* --- Precinct header --- */
    uint32_t header_len_bytes = (PC_PRECINCT_HEADER_SIZE_BYTES * 8 + 2 * bands_num_exists + 7) / 8;
    uint32_t packet_bytes_size = pack_total_bytes - header_len_bytes;
    pc_write_n_bits(&bw, packet_bytes_size, 24);
    pc_write_n_bits(&bw, pack_quantization, 8);
    pc_write_n_bits(&bw, pack_refinement, 8);

    for (uint32_t band = 0; band < bands_num_all; band++) {
        if (!band_info[band].exists) {
            continue;
        }
        uint8_t type_Dpb = 0;
        if (band_info[band].height_lines > 0) {
            type_Dpb = band_info[band].pack_method ? PC_CODING_MODE_FLAG_SIGNIFICANCE : 0;
        }
        pc_write_n_bits(&bw, type_Dpb, 2);
    }
    pc_align_byte(&bw);

    /* --- Packets --- */
    for (uint32_t p = 0; p < packets_num; p++) {
        int has_band = 0;
        for (uint32_t bidx = packets[p].band_start; bidx < packets[p].band_stop; bidx++) {
            if (packets[p].line_idx < band_info[bidx].height_lines) {
                has_band = 1;
                break;
            }
        }
        if (!has_band) {
            continue;
        }

        pc_write_packet_header(&bw, !use_short_header, packet_size_data_bytes[p], packet_size_gcli_bytes[p]);
        pc_align_byte(&bw);

        /* Significance sub-packet */
        uint32_t bits_last = pc_used_bits(&bw);
        for (uint32_t bidx = packets[p].band_start; bidx < packets[p].band_stop; bidx++) {
            uint32_t line = packets[p].line_idx;
            if (line >= band_info[bidx].height_lines)
                continue;
            const svt_cuda_pack_band_info_t& bi = band_info[bidx];
            if (bi.pack_method == 1) {
                const uint8_t* sig = significance_data + bi.significance_offset + (size_t)line * bi.significance_width;
                pc_pack_significance(&bw, bi.gtli, sig, bi.significance_width);
            }
        }
        pc_align_byte(&bw);
        if (pc_used_bits(&bw) - bits_last != packet_size_significance_bytes[p] * 8) {
            *out_error = 1;
            return;
        }

        /* GCLI (bitplane count) sub-packet */
        bits_last = pc_used_bits(&bw);
        for (uint32_t bidx = packets[p].band_start; bidx < packets[p].band_stop; bidx++) {
            uint32_t line = packets[p].line_idx;
            if (line >= band_info[bidx].height_lines)
                continue;
            const svt_cuda_pack_band_info_t& bi = band_info[bidx];
            const uint8_t* gcli = gcli_data + bi.gcli_offset + (size_t)line * bi.gcli_width;
            if (bi.pack_method == 1) {
                const uint8_t* sig = significance_data + bi.significance_offset + (size_t)line * bi.significance_width;
                pc_pack_bitplane_count_significance(&bw, gcli, bi.gcli_width, bi.gtli, sig, PC_SIGNIFICANCE_GROUP_SIZE);
            }
            else {
                pc_pack_bitplane_count_no_significance(&bw, gcli, bi.gcli_width, bi.gtli);
            }
        }
        pc_align_byte(&bw);
        if (pc_used_bits(&bw) - bits_last != packet_size_gcli_bytes[p] * 8) {
            *out_error = 2;
            return;
        }

        /* Data sub-packet */
        bits_last = pc_used_bits(&bw);
        for (uint32_t bidx = packets[p].band_start; bidx < packets[p].band_stop; bidx++) {
            uint32_t line = packets[p].line_idx;
            if (line >= band_info[bidx].height_lines)
                continue;
            const svt_cuda_pack_band_info_t& bi = band_info[bidx];
            const uint8_t* gcli = gcli_data + bi.gcli_offset + (size_t)line * bi.gcli_width;
            const uint16_t* coeff = coeff_data + bi.coeff_offset + (size_t)line * bi.width;
            pc_pack_data(&bw, coeff, bi.width, gcli, bi.gtli);
        }
        pc_align_byte(&bw);
        if (pc_used_bits(&bw) - bits_last != packet_size_data_bytes[p] * 8) {
            *out_error = 3;
            return;
        }
        /* Sign sub-packet: never present (Signs=OFF, this Phase 3 scope). */
    }

    if (pack_padding_bytes) {
        pc_add_padding_bytes(&bw, pack_padding_bytes);
    }
    *out_used_bytes = bw.offset;
}

int svt_cuda_pack_precinct(uint32_t bands_num_all, const svt_cuda_pack_band_info_t* band_info, uint32_t bands_num_exists,
                           const uint8_t* gcli_data, const uint8_t* significance_data, const uint16_t* coeff_data,
                           uint32_t packets_num, const svt_cuda_pack_packet_t* packets, uint8_t use_short_header,
                           uint8_t pack_quantization, uint8_t pack_refinement, uint32_t pack_total_bytes,
                           uint32_t pack_padding_bytes, const uint32_t* packet_size_data_bytes,
                           const uint32_t* packet_size_gcli_bytes, const uint32_t* packet_size_significance_bytes,
                           uint8_t* out_buffer, uint32_t* out_used_bytes) {
    uint32_t gcli_total = 0, sig_total = 0, coeff_total = 0;
    for (uint32_t b = 0; b < bands_num_all; b++) {
        if (!band_info[b].exists)
            continue;
        gcli_total = gcli_total > band_info[b].gcli_offset + band_info[b].height_lines * band_info[b].gcli_width
            ? gcli_total
            : band_info[b].gcli_offset + band_info[b].height_lines * band_info[b].gcli_width;
        sig_total = sig_total > band_info[b].significance_offset + band_info[b].height_lines * band_info[b].significance_width
            ? sig_total
            : band_info[b].significance_offset + band_info[b].height_lines * band_info[b].significance_width;
        coeff_total = coeff_total > band_info[b].coeff_offset + band_info[b].height_lines * band_info[b].width
            ? coeff_total
            : band_info[b].coeff_offset + band_info[b].height_lines * band_info[b].width;
    }

    svt_cuda_pack_band_info_t* d_band_info = NULL;
    uint8_t *d_gcli = NULL, *d_sig = NULL, *d_out = NULL;
    uint16_t* d_coeff = NULL;
    svt_cuda_pack_packet_t* d_packets = NULL;
    uint32_t *d_psd = NULL, *d_psg = NULL, *d_pss = NULL;
    uint32_t* d_used_bytes = NULL;
    int* d_error = NULL;
    cudaError_t err = cudaSuccess;

    do {
        if ((err = cudaMalloc(&d_band_info, bands_num_all * sizeof(svt_cuda_pack_band_info_t))) != cudaSuccess)
            break;
        if ((err = cudaMalloc(&d_gcli, gcli_total ? gcli_total : 1)) != cudaSuccess)
            break;
        if ((err = cudaMalloc(&d_sig, sig_total ? sig_total : 1)) != cudaSuccess)
            break;
        if ((err = cudaMalloc(&d_coeff, (coeff_total ? coeff_total : 1) * sizeof(uint16_t))) != cudaSuccess)
            break;
        if ((err = cudaMalloc(&d_packets, packets_num * sizeof(svt_cuda_pack_packet_t))) != cudaSuccess)
            break;
        if ((err = cudaMalloc(&d_psd, packets_num * sizeof(uint32_t))) != cudaSuccess)
            break;
        if ((err = cudaMalloc(&d_psg, packets_num * sizeof(uint32_t))) != cudaSuccess)
            break;
        if ((err = cudaMalloc(&d_pss, packets_num * sizeof(uint32_t))) != cudaSuccess)
            break;
        if ((err = cudaMalloc(&d_out, pack_total_bytes)) != cudaSuccess)
            break;
        if ((err = cudaMalloc(&d_used_bytes, sizeof(uint32_t))) != cudaSuccess)
            break;
        if ((err = cudaMalloc(&d_error, sizeof(int))) != cudaSuccess)
            break;

        if ((err = cudaMemset(d_out, 0, pack_total_bytes)) != cudaSuccess)
            break;
        if ((err = cudaMemcpy(d_band_info, band_info, bands_num_all * sizeof(svt_cuda_pack_band_info_t), cudaMemcpyHostToDevice)) !=
            cudaSuccess)
            break;
        if (gcli_total && (err = cudaMemcpy(d_gcli, gcli_data, gcli_total, cudaMemcpyHostToDevice)) != cudaSuccess)
            break;
        if (sig_total && (err = cudaMemcpy(d_sig, significance_data, sig_total, cudaMemcpyHostToDevice)) != cudaSuccess)
            break;
        if (coeff_total &&
            (err = cudaMemcpy(d_coeff, coeff_data, coeff_total * sizeof(uint16_t), cudaMemcpyHostToDevice)) != cudaSuccess)
            break;
        if ((err = cudaMemcpy(d_packets, packets, packets_num * sizeof(svt_cuda_pack_packet_t), cudaMemcpyHostToDevice)) !=
            cudaSuccess)
            break;
        if ((err = cudaMemcpy(d_psd, packet_size_data_bytes, packets_num * sizeof(uint32_t), cudaMemcpyHostToDevice)) !=
            cudaSuccess)
            break;
        if ((err = cudaMemcpy(d_psg, packet_size_gcli_bytes, packets_num * sizeof(uint32_t), cudaMemcpyHostToDevice)) !=
            cudaSuccess)
            break;
        if ((err = cudaMemcpy(d_pss, packet_size_significance_bytes, packets_num * sizeof(uint32_t), cudaMemcpyHostToDevice)) !=
            cudaSuccess)
            break;

        k_pack_precinct<<<1, 1>>>(d_band_info, bands_num_all, bands_num_exists, d_gcli, d_sig, d_coeff, packets_num, d_packets,
                                  use_short_header, pack_quantization, pack_refinement, pack_total_bytes, pack_padding_bytes,
                                  d_psd, d_psg, d_pss, d_out, d_used_bytes, d_error);

        int h_error = 0;
        if ((err = cudaMemcpy(&h_error, d_error, sizeof(int), cudaMemcpyDeviceToHost)) != cudaSuccess)
            break;
        if (h_error != 0) {
            err = cudaSuccess;
            cudaFree(d_band_info);
            cudaFree(d_gcli);
            cudaFree(d_sig);
            cudaFree(d_coeff);
            cudaFree(d_packets);
            cudaFree(d_psd);
            cudaFree(d_psg);
            cudaFree(d_pss);
            cudaFree(d_out);
            cudaFree(d_used_bytes);
            cudaFree(d_error);
            return 1;
        }
        if ((err = cudaMemcpy(out_buffer, d_out, pack_total_bytes, cudaMemcpyDeviceToHost)) != cudaSuccess)
            break;
        if ((err = cudaMemcpy(out_used_bytes, d_used_bytes, sizeof(uint32_t), cudaMemcpyDeviceToHost)) != cudaSuccess)
            break;
        err = cudaGetLastError();
    } while (0);

    cudaFree(d_band_info);
    cudaFree(d_gcli);
    cudaFree(d_sig);
    cudaFree(d_coeff);
    cudaFree(d_packets);
    cudaFree(d_psd);
    cudaFree(d_psg);
    cudaFree(d_pss);
    cudaFree(d_out);
    cudaFree(d_used_bytes);
    cudaFree(d_error);

    return err == cudaSuccess ? 0 : -(int)err;
}
