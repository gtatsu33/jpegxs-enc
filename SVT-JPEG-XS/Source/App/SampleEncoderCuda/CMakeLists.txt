#
# Copyright(c) 2024 Intel Corporation
# SPDX - License - Identifier: BSD - 2 - Clause - Patent
#

# SampleEncoderCuda Directory CMakeLists.txt
#
# Only added when ENABLE_CUDA is ON (see top-level CMakeLists.txt) -- this is
# the "CUDA-enabled build" sample referenced in PortingStrategy.txt section
# 12. Unlike SampleEncoder (public API only), this app reaches into the
# Encoder library's internal headers (EncHandle.h/Pi.h/PiEnc.h/PackHeaders.h/
# BitstreamWriter.h/Codestream.h) the same way tests/UnitTests/TestEncodeFrameCuda.cc
# does, because svt_cuda_encode_frame() is intentionally not wired into the
# public API (see PortingStrategy.txt section 7). Object files (not
# SvtJpegxsLib) are linked directly, same as SampleEncoder/CMakeLists.txt,
# both to reach these un-exported internal symbols and to reach ENCODER_CUDA's
# CUDA-only entry points the same way tests/UnitTests/CMakeLists.txt does.

include_directories(
    ${PROJECT_SOURCE_DIR}/Source/API/
    ${PROJECT_SOURCE_DIR}/Source/App/Common/
    ${PROJECT_SOURCE_DIR}/third_party/stb_image/
    ${PROJECT_SOURCE_DIR}/Source/Lib/Common/Codec/
    ${PROJECT_SOURCE_DIR}/Source/Lib/Encoder/Codec/
    ${PROJECT_SOURCE_DIR}/Source/Lib/Encoder/CUDA/
    )

file(GLOB all_files
    "../../API/*.h"
    "../Common/PixelIo.h"
    "../Common/PixelIo.c"
    "../Common/UtilityApp.h"
    "main.c"
)

add_executable(SvtJpegxsSampleEncoderCuda
    ${all_files})

set(lib_list_encoder_cuda
    $<TARGET_OBJECTS:COMMON_CODEC>
    $<TARGET_OBJECTS:COMMON_ASM_SSE2>
    $<TARGET_OBJECTS:COMMON_ASM_AVX2>
    $<TARGET_OBJECTS:ENCODER_CODEC>
    $<TARGET_OBJECTS:ENCODER_ASM_SSE2>
    $<TARGET_OBJECTS:ENCODER_ASM_SSE4_1>
    $<TARGET_OBJECTS:ENCODER_ASM_AVX2>
    $<TARGET_OBJECTS:ENCODER_ASM_AVX512>
    $<TARGET_OBJECTS:ENCODER_CUDA>
    cpuinfo_public
    )

target_link_libraries(SvtJpegxsSampleEncoderCuda
    ${lib_list_encoder_cuda}
    CUDA::cudart)

if(UNIX)
    target_link_libraries(SvtJpegxsSampleEncoderCuda
        pthread
        m)
endif()

add_dependencies(SvtJpegxsSampleEncoderCuda
    COMMON_CODEC
    COMMON_ASM_SSE2
    COMMON_ASM_AVX2
    ENCODER_CODEC
    ENCODER_ASM_SSE2
    ENCODER_ASM_SSE4_1
    ENCODER_ASM_AVX2
    ENCODER_ASM_AVX512
    ENCODER_CUDA
    cpuinfo_public)

install(TARGETS SvtJpegxsSampleEncoderCuda RUNTIME DESTINATION ${CMAKE_INSTALL_FULL_BINDIR})
