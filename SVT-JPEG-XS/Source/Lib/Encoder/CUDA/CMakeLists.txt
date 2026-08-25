#
# Copyright(c) 2024 Intel Corporation
# SPDX - License - Identifier: BSD - 2 - Clause - Patent
#

# Encoder/CUDA Directory CMakeLists.txt

include_directories(
    ${PROJECT_SOURCE_DIR}/Source/API/
    ${PROJECT_SOURCE_DIR}/Source/Lib/Common/Codec/
    ${PROJECT_SOURCE_DIR}/Source/Lib/Encoder/Codec/
    ${PROJECT_SOURCE_DIR}/Source/Lib/Encoder/CUDA/
    )

file(GLOB all_files
    "*.h"
    "*.cuh"
    "*.cu")

add_library(ENCODER_CUDA OBJECT ${all_files})
set_target_properties(ENCODER_CUDA PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    CUDA_STANDARD 17
    POSITION_INDEPENDENT_CODE ON)
target_link_libraries(ENCODER_CUDA PRIVATE CUDA::cudart)
