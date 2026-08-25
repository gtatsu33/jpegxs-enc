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
#include "SvtType.h"
#include "Codestream.h"
#include "encoder_dsp_rtcd.h"
#include "common_dsp_rtcd.h"
#include "RcQuantCuda.cuh"
#include <vector>
#include <cstdint>

/* Phase 2 bit-exact validation: rate_control_precinct() + precinct_quantization()
 * (real CPU reference functions, VPRED disabled / Signs=OFF / significance
 * enabled / deadzone quant -- the encoder's default configuration) vs
 * svt_cuda_rc_quant_precinct(). A minimal but real pi_t/precinct_enc_t is
 * built via pi_compute()/weight_table_calculate()/pi_compute_encoder() (the
 * same functions the real encoder uses), with synthetic random gcli/coeff
 * data assigned directly into precinct_enc_t's band pointers (bypassing the
 * threaded pipeline's buffer allocation, which this test does not need). */

struct RcTestFixture {
    pi_t pi;
    pi_enc_t pi_enc;
    svt_jpeg_xs_encoder_common_t enc_common;
    PictureControlSet pcs;
    precinct_enc_t precinct_cpu;
    precinct_enc_t precinct_cuda_shadow; /* same synthetic input, quantized by CUDA path */

    std::vector<std::vector<uint16_t>> coeff_cpu, coeff_cuda;
    std::vector<std::vector<uint8_t>> gcli, significance; /* shared input (read-only by RC) */

    uint32_t bands_num = 0;
};

static void build_fixture(RcTestFixture& fx, uint32_t width, uint32_t height, uint32_t decom_h, uint32_t decom_v) {
    memset(&fx.pi, 0, sizeof(fx.pi));
    memset(&fx.pi_enc, 0, sizeof(fx.pi_enc));
    memset(&fx.enc_common, 0, sizeof(fx.enc_common));
    memset(&fx.precinct_cpu, 0, sizeof(fx.precinct_cpu));
    memset(&fx.precinct_cuda_shadow, 0, sizeof(fx.precinct_cuda_shadow));

    uint32_t sx[MAX_COMPONENTS_NUM] = {1, 2, 2, 0};
    uint32_t sy[MAX_COMPONENTS_NUM] = {1, 1, 1, 0};
    ASSERT_EQ(pi_compute(&fx.pi, 1, 3, GROUP_SIZE, SIGNIFICANCE_GROUP_SIZE, width, height, decom_h, decom_v, 0, sx, sy, 0, height),
              SvtJxsErrorNone);
    ASSERT_EQ(weight_table_calculate(&fx.pi, 0, COLOUR_FORMAT_PLANAR_YUV422), 0);
    ASSERT_EQ(pi_compute_encoder(&fx.pi, &fx.pi_enc, /*significance_flag*/ 1, /*vpred_flag*/ 0, /*verbose*/ 0), 0);

    fx.enc_common.pi = fx.pi;
    fx.enc_common.pi_enc = fx.pi_enc;
    fx.enc_common.coding_significance = 1;
    fx.enc_common.coding_vertical_prediction_mode = METHOD_PRED_DISABLE;
    fx.enc_common.coding_signs_handling = SIGN_HANDLING_STRATEGY_OFF;
    fx.enc_common.picture_header_dynamic.hdr_Qpih = QUANT_TYPE_DEADZONE;

    fx.pcs.enc_common = &fx.enc_common;

    fx.precinct_cpu.p_info = &fx.enc_common.pi.p_info[PRECINCT_NORMAL];
    fx.precinct_cuda_shadow.p_info = &fx.enc_common.pi.p_info[PRECINCT_NORMAL];

    /* IMPORTANT: pi->packets[]::band_start/band_stop index into
     * pi->global_band_info[], which interleaves components at each
     * decomposition level (comp0-band0, comp1-band0, comp2-band0,
     * comp0-band1, ...) -- NOT a component-major ordering. The flat band
     * array below (and the one built for the CUDA call) must use this same
     * global_band_info-derived ordering, including placeholder slots for
     * BAND_NOT_EXIST entries, or packet aggregation silently sums the wrong
     * bands. */
    fx.bands_num = fx.enc_common.pi.bands_num_all;

    svt_jxs_test_tool::SVTRandom rand4(4, false);  // gcli in [0,15]
    svt_jxs_test_tool::SVTRandom rand32(16, false); // coefficient noise

    fx.coeff_cpu.resize(fx.bands_num);
    fx.coeff_cuda.resize(fx.bands_num);
    fx.gcli.resize(fx.bands_num);
    fx.significance.resize(fx.bands_num);

    for (uint32_t flat = 0; flat < fx.bands_num; flat++) {
        uint32_t c = fx.enc_common.pi.global_band_info[flat].comp_id;
        uint32_t b = fx.enc_common.pi.global_band_info[flat].band_id;
        if (b == BAND_NOT_EXIST) {
            continue; // leave zeroed placeholder, never referenced by any packet
        }
        {
            const precinct_band_info_t& info = fx.precinct_cpu.p_info->b_info[c][b];
            uint32_t height_lines = info.height;

            fx.gcli[flat].resize((size_t)height_lines * info.gcli_width);
            fx.significance[flat].resize((size_t)height_lines * info.significance_width, 0);
            fx.coeff_cpu[flat].resize((size_t)height_lines * info.width);

            for (uint32_t line = 0; line < height_lines; line++) {
                uint8_t* gcli_line = &fx.gcli[flat][(size_t)line * info.gcli_width];
                uint16_t* coeff_line = &fx.coeff_cpu[flat][(size_t)line * info.width];
                for (uint32_t g = 0; g < info.gcli_width; g++) {
                    uint8_t msb = (uint8_t)rand4.random(); // 0..15
                    gcli_line[g] = msb;
                    uint32_t base = g * GROUP_SIZE;
                    for (uint32_t i = 0; i < GROUP_SIZE && base + i < info.width; i++) {
                        uint16_t mag = msb ? (uint16_t)((1u << (msb - 1)) | (rand32.random() & ((1u << (msb - 1)) - 1))) : 0;
                        uint16_t sign = (rand32.random() & 1) ? BITSTREAM_MASK_SIGN : 0;
                        coeff_line[base + i] = mag ? (mag | sign) : 0;
                    }
                }
                uint8_t* sig_line = &fx.significance[flat][(size_t)line * info.significance_width];
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
            fx.coeff_cuda[flat] = fx.coeff_cpu[flat]; // identical input, quantized independently below

            auto* band_cpu = &fx.precinct_cpu.bands[c][b];
            auto* band_shadow = &fx.precinct_cuda_shadow.bands[c][b];
            for (uint32_t line = 0; line < height_lines; line++) {
                band_cpu->lines_common[line].gcli_data_ptr = &fx.gcli[flat][(size_t)line * info.gcli_width];
                band_cpu->lines_common[line].significance_data_max_ptr = &fx.significance[flat][(size_t)line * info.significance_width];
                band_cpu->lines_common[line].coeff_data_ptr_16bit = &fx.coeff_cpu[flat][(size_t)line * info.width];
            }
            band_cpu->cache_buffers[0].gtli_rc_last_calculated = UINT8_MAX;
            band_cpu->cache_buffers[1].gtli_rc_last_calculated = UINT8_MAX;
            band_cpu->cache_index = 0;
            band_cpu->cache_actual = &band_cpu->cache_buffers[0];
            *band_shadow = *band_cpu; // same pointers for gcli/significance; coeff reassigned below
            for (uint32_t line = 0; line < height_lines; line++) {
                band_shadow->lines_common[line].coeff_data_ptr_16bit = &fx.coeff_cuda[flat][(size_t)line * info.width];
            }
        }
    }
    fx.precinct_cpu.p_info->packets_exist_num = fx.enc_common.pi.packets_num;
    fx.precinct_cuda_shadow.p_info->packets_exist_num = fx.enc_common.pi.packets_num;
}

static void run_rc_quant_cuda_vs_cpu(uint32_t width, uint32_t height, uint32_t decom_h, uint32_t decom_v, float bpp) {
    setup_common_rtcd_internal(CPU_FLAGS_ALL);
    setup_encoder_rtcd_internal(CPU_FLAGS_ALL);

    RcTestFixture fx;
    build_fixture(fx, width, height, decom_h, decom_v);

    uint32_t budget_bytes = (uint32_t)((double)width * height * bpp / 8.0);
    if (budget_bytes < 200)
        budget_bytes = 200;

    rate_control_init_precinct(&fx.pcs, &fx.precinct_cpu, SIGN_HANDLING_STRATEGY_OFF);
    SvtJxsErrorType_t ret = rate_control_precinct(&fx.pcs, &fx.precinct_cpu, budget_bytes, METHOD_PRED_DISABLE, SIGN_HANDLING_STRATEGY_OFF);
    ASSERT_EQ(ret, SvtJxsErrorNone);
    precinct_quantization(&fx.pcs, &fx.enc_common.pi, &fx.precinct_cpu);

    /* --- Flatten inputs for the CUDA API --- */
    std::vector<svt_cuda_rc_band_info_t> band_info(fx.bands_num);
    std::vector<uint8_t> gcli_flat, sig_flat;
    std::vector<uint16_t> coeff_flat;
    for (uint32_t flat = 0; flat < fx.bands_num; flat++) {
        uint32_t c = fx.enc_common.pi.global_band_info[flat].comp_id;
        uint32_t b = fx.enc_common.pi.global_band_info[flat].band_id;
        svt_cuda_rc_band_info_t& bi = band_info[flat];
        bi.gcli_offset = (uint32_t)gcli_flat.size();
        bi.significance_offset = (uint32_t)sig_flat.size();
        bi.coeff_offset = (uint32_t)coeff_flat.size();
        if (b == BAND_NOT_EXIST) {
            continue; // height_lines stays 0 (value-initialized), kernels skip it
        }
        const precinct_band_info_t& info = fx.precinct_cpu.p_info->b_info[c][b];
        bi.width = info.width;
        bi.gcli_width = info.gcli_width;
        bi.significance_width = info.significance_width;
        bi.height_lines = info.height;
        bi.gain = fx.enc_common.pi.components[c].bands[b].gain;
        bi.priority = fx.enc_common.pi.components[c].bands[b].priority;
        gcli_flat.insert(gcli_flat.end(), fx.gcli[flat].begin(), fx.gcli[flat].end());
        sig_flat.insert(sig_flat.end(), fx.significance[flat].begin(), fx.significance[flat].end());
        coeff_flat.insert(coeff_flat.end(), fx.coeff_cuda[flat].begin(), fx.coeff_cuda[flat].end());
    }

    std::vector<svt_cuda_rc_packet_t> packets(fx.enc_common.pi.packets_num);
    for (uint32_t p = 0; p < fx.enc_common.pi.packets_num; p++) {
        packets[p].band_start = fx.enc_common.pi.packets[p].band_start;
        packets[p].band_stop = fx.enc_common.pi.packets[p].band_stop;
        packets[p].line_idx = fx.enc_common.pi.packets[p].line_idx;
    }

    std::vector<uint8_t> out_gtli(fx.bands_num);
    uint8_t out_quantization = 0, out_refinement = 0;
    uint32_t out_data_bytes = 0, out_padding_bytes = 0, out_total_bytes = 0;

    int err = svt_cuda_rc_quant_precinct(fx.bands_num,
                                         band_info.data(),
                                         gcli_flat.data(),
                                         sig_flat.data(),
                                         coeff_flat.data(),
                                         fx.enc_common.pi.packets_num,
                                         packets.data(),
                                         fx.enc_common.pi.bands_num_exists,
                                         fx.precinct_cpu.p_info->packets_exist_num,
                                         (uint8_t)fx.enc_common.pi.use_short_header,
                                         (uint8_t)fx.enc_common.coding_significance,
                                         (uint8_t)fx.enc_common.picture_header_dynamic.hdr_Qpih,
                                         fx.enc_common.pi_enc.max_quantization,
                                         fx.enc_common.pi_enc.max_refinement,
                                         budget_bytes,
                                         out_gtli.data(),
                                         &out_quantization,
                                         &out_refinement,
                                         &out_data_bytes,
                                         &out_padding_bytes,
                                         &out_total_bytes,
                                         NULL,
                                         NULL,
                                         NULL,
                                         NULL);
    ASSERT_EQ(err, 0);

    /* Scatter the quantized flat coeff buffer back into fx.coeff_cuda[] for
     * the per-band comparison below (svt_cuda_rc_quant_precinct quantizes
     * coeff_flat in place; it does not know about fx's per-band vectors). */
    for (uint32_t flat = 0; flat < fx.bands_num; flat++) {
        uint32_t c = fx.enc_common.pi.global_band_info[flat].comp_id;
        uint32_t b = fx.enc_common.pi.global_band_info[flat].band_id;
        if (b == BAND_NOT_EXIST) {
            continue;
        }
        memcpy(fx.coeff_cuda[flat].data(), coeff_flat.data() + band_info[flat].coeff_offset,
               fx.coeff_cuda[flat].size() * sizeof(uint16_t));
    }

    EXPECT_EQ(out_quantization, fx.precinct_cpu.pack_quantization);
    EXPECT_EQ(out_refinement, fx.precinct_cpu.pack_refinement);
    EXPECT_EQ(out_padding_bytes, fx.precinct_cpu.pack_padding_bytes);
    EXPECT_EQ(out_total_bytes, fx.precinct_cpu.pack_total_bytes);

    for (uint32_t flat = 0; flat < fx.bands_num; flat++) {
        uint32_t c = fx.enc_common.pi.global_band_info[flat].comp_id;
        uint32_t b = fx.enc_common.pi.global_band_info[flat].band_id;
        if (b == BAND_NOT_EXIST) {
            continue;
        }
        EXPECT_EQ(out_gtli[flat], fx.precinct_cpu.bands[c][b].gtli) << "band " << flat;
        EXPECT_EQ(memcmp(fx.coeff_cuda[flat].data(), fx.coeff_cpu[flat].data(), fx.coeff_cuda[flat].size() * sizeof(uint16_t)), 0)
            << "band " << flat << " quantized coefficients differ";
    }
}

TEST(RcQuantCuda, small_yuv422_matches_cpu_reference) {
    run_rc_quant_cuda_vs_cpu(320, 64, 5, 2, 3.0f);
}

TEST(RcQuantCuda, tight_budget_matches_cpu_reference) {
    run_rc_quant_cuda_vs_cpu(320, 64, 5, 2, 0.8f);
}

TEST(RcQuantCuda, loose_budget_matches_cpu_reference) {
    run_rc_quant_cuda_vs_cpu(320, 64, 5, 2, 8.0f);
}

TEST(RcQuantCuda, decom_v1_matches_cpu_reference) {
    run_rc_quant_cuda_vs_cpu(256, 32, 3, 1, 3.0f);
}

TEST(RcQuantCuda, real_size_4k_precinct_matches_cpu_reference) {
    run_rc_quant_cuda_vs_cpu(3840, 4 /*one precinct's worth of rows, decom_v=2*/, 5, 2, 3.0f);
}
