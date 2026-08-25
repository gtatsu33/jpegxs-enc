/*
* Copyright(c) 2024 Intel Corporation
* SPDX - License - Identifier: BSD - 2 - Clause - Patent
*/
#ifndef GcCuda_cuh
#define GcCuda_cuh

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Matches gc_precinct_stage_scalar_c (GcStageProcess.c): for each group of
 * `group_size` (GROUP_SIZE=4) consecutive 16-bit sign+magnitude coefficients,
 * computes gcli = bit-length of the OR of their magnitudes (MSB position),
 * 0 if all-zero. One CUDA thread per group -- groups are fully independent
 * (no dependency across groups/lines/bands/components), so this is a
 * straightforward 1:1 port, unlike the DWT/NLT kernels.
 *
 * coeff_data_ptr_16bit: host input, `width` samples.
 * gcli_data_ptr: host output, ceil(width/group_size) bytes.
 * group_size: must be GROUP_SIZE (4), kept as a parameter to mirror the C signature.
 * Returns 0 on success, non-zero CUDA error code otherwise.
 */
int svt_cuda_gc_precinct_stage_scalar(uint8_t* gcli_data_ptr, const uint16_t* coeff_data_ptr_16bit, uint32_t group_size,
                                      uint32_t width);

/* Matches gc_precinct_sigflags_max_c (GcStageProcess.c): for each group of
 * `group_sign_size` (SIGNIFICANCE_GROUP_SIZE=8) consecutive gcli values,
 * computes their max. One CUDA thread per significance group.
 *
 * gcli_data_ptr: host input, `gcli_width` bytes.
 * significance_data_max_ptr: host output, ceil(gcli_width/group_sign_size) bytes.
 * Returns 0 on success, non-zero CUDA error code otherwise.
 */
int svt_cuda_gc_precinct_sigflags_max(uint8_t* significance_data_max_ptr, const uint8_t* gcli_data_ptr, uint32_t group_sign_size,
                                      uint32_t gcli_width);

#ifdef __cplusplus
}
#endif

#endif // GcCuda_cuh
