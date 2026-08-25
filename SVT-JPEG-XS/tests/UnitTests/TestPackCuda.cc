/*
* Copyright(c) 2024 Intel Corporation
* SPDX - License - Identifier: BSD - 2 - Clause - Patent
*/
#include "gtest/gtest.h"
#include "random.h"
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
#include "encoder_dsp_rtcd.h"
#include "common_dsp_rtcd.h"
#include "PackCuda.cuh"
#include <vector>
#include <cstdint>

/* Phase 3 bit-exact validation: pack_precinct() (real CPU reference) vs
 * svt_cuda_pack_precinct(). Reuses the same real pi_t/precinct_enc_t
 * construction pattern as TestRcQuantCuda.cc, then runs the real
 * rate_control_init_precinct()/rate_control_precinct()/precinct_quantization()
 * pipeline first (so gtli/pack_method/packet_size_*_bytes are genuine RC
 * output, not synthetic), and finally compares the packed bitstream byte
 * buffers with memcmp (matching TestPack.cc's "compare full output buffer"
 * convention). */

static void run_pack_cuda_vs_cpu(uint32_t width, uint32_t height, uint32_t decom_h, uint32_t decom_v, float bpp) {
    setup_common_rtcd_internal(CPU_FLAGS_ALL);
    setup_encoder_rtcd_internal(CPU_FLAGS_ALL);

    pi_t pi;
    pi_enc_t pi_enc;
    memset(&pi, 0, sizeof(pi));
    memset(&pi_enc, 0, sizeof(pi_enc));
    uint32_t sx[MAX_COMPONENTS_NUM] = {1, 2, 2, 0};
    uint32_t sy[MAX_COMPONENTS_NUM] = {1, 1, 1, 0};
    ASSERT_EQ(pi_compute(&pi, 1, 3, GROUP_SIZE, SIGNIFICANCE_GROUP_SIZE, width, height, decom_h, decom_v, 0, sx, sy, 0, height),
              SvtJxsErrorNone);
    ASSERT_EQ(weight_table_calculate(&pi, 0, COLOUR_FORMAT_PLANAR_YUV422), 0);
    ASSERT_EQ(pi_compute_encoder(&pi, &pi_enc, 1, 0, 0), 0);

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

    svt_jxs_test_tool::SVTRandom rand4(4, false);
    svt_jxs_test_tool::SVTRandom rand16(16, false);

    for (uint32_t flat = 0; flat < bands_num; flat++) {
        uint32_t c = pi.global_band_info[flat].comp_id;
        uint32_t b = pi.global_band_info[flat].band_id;
        if (b == BAND_NOT_EXIST) {
            continue;
        }
        const precinct_band_info_t& info = precinct.p_info->b_info[c][b];
        uint32_t hl = info.height;
        gcli[flat].resize((size_t)hl * info.gcli_width);
        sig[flat].resize((size_t)hl * info.significance_width, 0);
        coeff[flat].resize((size_t)hl * info.width);
        for (uint32_t line = 0; line < hl; line++) {
            uint8_t* gcli_line = &gcli[flat][(size_t)line * info.gcli_width];
            uint16_t* coeff_line = &coeff[flat][(size_t)line * info.width];
            for (uint32_t g = 0; g < info.gcli_width; g++) {
                uint8_t msb = (uint8_t)rand4.random();
                gcli_line[g] = msb;
                uint32_t base = g * GROUP_SIZE;
                for (uint32_t i = 0; i < GROUP_SIZE && base + i < info.width; i++) {
                    uint16_t mag = msb ? (uint16_t)((1u << (msb - 1)) | (rand16.random() & ((1u << (msb - 1)) - 1))) : 0;
                    uint16_t sign = (rand16.random() & 1) ? BITSTREAM_MASK_SIGN : 0;
                    coeff_line[base + i] = mag ? (mag | sign) : 0;
                }
            }
            uint8_t* sig_line = &sig[flat][(size_t)line * info.significance_width];
            for (uint32_t s = 0; s < info.significance_width; s++) {
                uint32_t end = std::min((s + 1) * SIGNIFICANCE_GROUP_SIZE, info.gcli_width);
                uint8_t m = 0;
                for (uint32_t g = s * SIGNIFICANCE_GROUP_SIZE; g < end; g++) {
                    if (gcli_line[g] > m)
                        m = gcli_line[g];
                }
                sig_line[s] = m;
            }
        }

        auto* band = &precinct.bands[c][b];
        for (uint32_t line = 0; line < hl; line++) {
            band->lines_common[line].gcli_data_ptr = &gcli[flat][(size_t)line * info.gcli_width];
            band->lines_common[line].significance_data_max_ptr = &sig[flat][(size_t)line * info.significance_width];
            band->lines_common[line].coeff_data_ptr_16bit = &coeff[flat][(size_t)line * info.width];
        }
        band->cache_buffers[0].gtli_rc_last_calculated = UINT8_MAX;
        band->cache_buffers[1].gtli_rc_last_calculated = UINT8_MAX;
        band->cache_actual = &band->cache_buffers[0];
    }
    precinct.p_info->packets_exist_num = pi.packets_num;

    uint32_t budget_bytes = (uint32_t)((double)width * height * bpp / 8.0);
    if (budget_bytes < 200)
        budget_bytes = 200;

    rate_control_init_precinct(&pcs, &precinct, SIGN_HANDLING_STRATEGY_OFF);
    ASSERT_EQ(rate_control_precinct(&pcs, &precinct, budget_bytes, METHOD_PRED_DISABLE, SIGN_HANDLING_STRATEGY_OFF),
              SvtJxsErrorNone);
    precinct_quantization(&pcs, &enc_common.pi, &precinct);

    /* --- Reference: real pack_precinct() --- */
    std::vector<uint8_t> ref_buf(precinct.pack_total_bytes + 16, 0xAB);
    bitstream_writer_t bw;
    bitstream_writer_init(&bw, ref_buf.data(), ref_buf.size());
    ASSERT_EQ(pack_precinct(&bw, &pi, &precinct, SIGN_HANDLING_STRATEGY_OFF), SvtJxsErrorNone);
    uint32_t ref_used_bytes = bitstream_writer_get_used_bytes(&bw);
    ASSERT_EQ(ref_used_bytes, precinct.pack_total_bytes);

    /* --- Flatten inputs for svt_cuda_pack_precinct() --- */
    std::vector<svt_cuda_pack_band_info_t> band_info(bands_num);
    std::vector<uint8_t> gcli_flat, sig_flat;
    std::vector<uint16_t> coeff_flat;
    for (uint32_t flat = 0; flat < bands_num; flat++) {
        uint32_t c = pi.global_band_info[flat].comp_id;
        uint32_t b = pi.global_band_info[flat].band_id;
        svt_cuda_pack_band_info_t& bi = band_info[flat];
        memset(&bi, 0, sizeof(bi));
        bi.gcli_offset = (uint32_t)gcli_flat.size();
        bi.significance_offset = (uint32_t)sig_flat.size();
        bi.coeff_offset = (uint32_t)coeff_flat.size();
        if (b == BAND_NOT_EXIST) {
            bi.exists = 0;
            continue;
        }
        bi.exists = 1;
        const precinct_band_info_t& info = precinct.p_info->b_info[c][b];
        bi.width = info.width;
        bi.gcli_width = info.gcli_width;
        bi.significance_width = info.significance_width;
        bi.height_lines = info.height;
        bi.pack_method = (precinct.bands[c][b].cache_actual->pack_method == METHOD_ZERO_SIGNIFICANCE_ENABLE) ? 1 : 0;
        bi.gtli = precinct.bands[c][b].gtli;
        gcli_flat.insert(gcli_flat.end(), gcli[flat].begin(), gcli[flat].end());
        sig_flat.insert(sig_flat.end(), sig[flat].begin(), sig[flat].end());
        coeff_flat.insert(coeff_flat.end(), coeff[flat].begin(), coeff[flat].end());
    }

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
    int err = svt_cuda_pack_precinct(bands_num, band_info.data(), pi.bands_num_exists, gcli_flat.data(), sig_flat.data(),
                                     coeff_flat.data(), pi.packets_num, packets.data(), (uint8_t)pi.use_short_header,
                                     precinct.pack_quantization, precinct.pack_refinement, precinct.pack_total_bytes,
                                     precinct.pack_padding_bytes, psd.data(), psg.data(), pss.data(), cuda_buf.data(),
                                     &cuda_used_bytes);
    ASSERT_EQ(err, 0);
    EXPECT_EQ(cuda_used_bytes, precinct.pack_total_bytes);
    EXPECT_EQ(memcmp(cuda_buf.data(), ref_buf.data(), precinct.pack_total_bytes), 0);
}

TEST(PackCuda, small_yuv422_matches_cpu_reference) {
    run_pack_cuda_vs_cpu(320, 64, 5, 2, 3.0f);
}

TEST(PackCuda, tight_budget_matches_cpu_reference) {
    run_pack_cuda_vs_cpu(320, 64, 5, 2, 0.8f);
}

TEST(PackCuda, loose_budget_matches_cpu_reference) {
    run_pack_cuda_vs_cpu(320, 64, 5, 2, 8.0f);
}

TEST(PackCuda, decom_v1_matches_cpu_reference) {
    run_pack_cuda_vs_cpu(256, 32, 3, 1, 3.0f);
}

TEST(PackCuda, real_size_4k_precinct_matches_cpu_reference) {
    run_pack_cuda_vs_cpu(3840, 4, 5, 2, 3.0f);
}
