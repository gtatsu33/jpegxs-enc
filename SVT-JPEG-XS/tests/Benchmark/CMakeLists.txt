#
# Copyright(c) 2024 Intel Corporation
# SPDX - License - Identifier: BSD - 2 - Clause - Patent
#

cmake_minimum_required(VERSION 3.10)

include_directories(
    ${PROJECT_SOURCE_DIR}/tests/Benchmark
    ${PROJECT_SOURCE_DIR}/Source/API
    ${PROJECT_SOURCE_DIR}/Source/Lib/Common/Codec
    ${PROJECT_SOURCE_DIR}/Source/Lib/Encoder/Codec
    )

set(lib_list_bench
    $<TARGET_OBJECTS:COMMON_CODEC>
    $<TARGET_OBJECTS:COMMON_ASM_SSE2>
    $<TARGET_OBJECTS:COMMON_ASM_AVX2>
    $<TARGET_OBJECTS:ENCODER_CODEC>
    $<TARGET_OBJECTS:ENCODER_ASM_SSE2>
    $<TARGET_OBJECTS:ENCODER_ASM_SSE4_1>
    $<TARGET_OBJECTS:ENCODER_ASM_AVX2>
    $<TARGET_OBJECTS:ENCODER_ASM_AVX512>
    )

add_executable(SvtJpegxsBenchmark
    BenchTimer.h
    BenchReport.h
    BenchReport.cpp
    main.cpp
    )
target_link_libraries(SvtJpegxsBenchmark PRIVATE ${lib_list_bench} cpuinfo_public)

if(ENABLE_CUDA)
    include_directories(${PROJECT_SOURCE_DIR}/Source/Lib/Encoder/CUDA)
    target_compile_definitions(SvtJpegxsBenchmark PRIVATE SVT_ENABLE_CUDA)
    target_sources(SvtJpegxsBenchmark PRIVATE $<TARGET_OBJECTS:ENCODER_CUDA>)
    target_link_libraries(SvtJpegxsBenchmark PRIVATE CUDA::cudart)
endif()

add_dependencies(SvtJpegxsBenchmark SvtJpegxsLib)
