# Install script for directory: C:/Users/gtats/python codes/jpegxs-enc/SVT-JPEG-XS/Source/Lib

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "C:/Program Files/svt-jpegxs")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  if(CMAKE_INSTALL_CONFIG_NAME MATCHES "^([Dd][Ee][Bb][Uu][Gg])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY OPTIONAL FILES "C:/Users/gtats/python codes/jpegxs-enc/SVT-JPEG-XS/Bin/Debug/SvtJpegxs.lib")
  elseif(CMAKE_INSTALL_CONFIG_NAME MATCHES "^([Rr][Ee][Ll][Ee][Aa][Ss][Ee])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY OPTIONAL FILES "C:/Users/gtats/python codes/jpegxs-enc/SVT-JPEG-XS/Bin/Release/SvtJpegxs.lib")
  elseif(CMAKE_INSTALL_CONFIG_NAME MATCHES "^([Mm][Ii][Nn][Ss][Ii][Zz][Ee][Rr][Ee][Ll])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY OPTIONAL FILES "C:/Users/gtats/python codes/jpegxs-enc/SVT-JPEG-XS/Bin/MinSizeRel/SvtJpegxs.lib")
  elseif(CMAKE_INSTALL_CONFIG_NAME MATCHES "^([Rr][Ee][Ll][Ww][Ii][Tt][Hh][Dd][Ee][Bb][Ii][Nn][Ff][Oo])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY OPTIONAL FILES "C:/Users/gtats/python codes/jpegxs-enc/SVT-JPEG-XS/Bin/RelWithDebInfo/SvtJpegxs.lib")
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  if(CMAKE_INSTALL_CONFIG_NAME MATCHES "^([Dd][Ee][Bb][Uu][Gg])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin" TYPE SHARED_LIBRARY FILES "C:/Users/gtats/python codes/jpegxs-enc/SVT-JPEG-XS/Bin/Debug/SvtJpegxs.dll")
  elseif(CMAKE_INSTALL_CONFIG_NAME MATCHES "^([Rr][Ee][Ll][Ee][Aa][Ss][Ee])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin" TYPE SHARED_LIBRARY FILES "C:/Users/gtats/python codes/jpegxs-enc/SVT-JPEG-XS/Bin/Release/SvtJpegxs.dll")
  elseif(CMAKE_INSTALL_CONFIG_NAME MATCHES "^([Mm][Ii][Nn][Ss][Ii][Zz][Ee][Rr][Ee][Ll])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin" TYPE SHARED_LIBRARY FILES "C:/Users/gtats/python codes/jpegxs-enc/SVT-JPEG-XS/Bin/MinSizeRel/SvtJpegxs.dll")
  elseif(CMAKE_INSTALL_CONFIG_NAME MATCHES "^([Rr][Ee][Ll][Ww][Ii][Tt][Hh][Dd][Ee][Bb][Ii][Nn][Ff][Oo])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin" TYPE SHARED_LIBRARY FILES "C:/Users/gtats/python codes/jpegxs-enc/SVT-JPEG-XS/Bin/RelWithDebInfo/SvtJpegxs.dll")
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/pkgconfig" TYPE FILE FILES "C:/Users/gtats/python codes/jpegxs-enc/SVT-JPEG-XS/Build/windows_cuda/SvtJpegxs.pc")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for each subdirectory.
  include("C:/Users/gtats/python codes/jpegxs-enc/SVT-JPEG-XS/Build/windows_cuda/Source/Lib/Common/Codec/cmake_install.cmake")
  include("C:/Users/gtats/python codes/jpegxs-enc/SVT-JPEG-XS/Build/windows_cuda/Source/Lib/Common/ASM_SSE2/cmake_install.cmake")
  include("C:/Users/gtats/python codes/jpegxs-enc/SVT-JPEG-XS/Build/windows_cuda/Source/Lib/Common/ASM_AVX2/cmake_install.cmake")
  include("C:/Users/gtats/python codes/jpegxs-enc/SVT-JPEG-XS/Build/windows_cuda/Source/Lib/Encoder/Codec/cmake_install.cmake")
  include("C:/Users/gtats/python codes/jpegxs-enc/SVT-JPEG-XS/Build/windows_cuda/Source/Lib/Encoder/ASM_SSE2/cmake_install.cmake")
  include("C:/Users/gtats/python codes/jpegxs-enc/SVT-JPEG-XS/Build/windows_cuda/Source/Lib/Encoder/ASM_SSE4_1/cmake_install.cmake")
  include("C:/Users/gtats/python codes/jpegxs-enc/SVT-JPEG-XS/Build/windows_cuda/Source/Lib/Encoder/ASM_AVX2/cmake_install.cmake")
  include("C:/Users/gtats/python codes/jpegxs-enc/SVT-JPEG-XS/Build/windows_cuda/Source/Lib/Encoder/ASM_AVX512/cmake_install.cmake")
  include("C:/Users/gtats/python codes/jpegxs-enc/SVT-JPEG-XS/Build/windows_cuda/Source/Lib/Decoder/Codec/cmake_install.cmake")
  include("C:/Users/gtats/python codes/jpegxs-enc/SVT-JPEG-XS/Build/windows_cuda/Source/Lib/Decoder/ASM_SSE4_1/cmake_install.cmake")
  include("C:/Users/gtats/python codes/jpegxs-enc/SVT-JPEG-XS/Build/windows_cuda/Source/Lib/Decoder/ASM_AVX2/cmake_install.cmake")
  include("C:/Users/gtats/python codes/jpegxs-enc/SVT-JPEG-XS/Build/windows_cuda/Source/Lib/Decoder/ASM_AVX512/cmake_install.cmake")
  include("C:/Users/gtats/python codes/jpegxs-enc/SVT-JPEG-XS/Build/windows_cuda/Source/Lib/Encoder/CUDA/cmake_install.cmake")

endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
if(CMAKE_INSTALL_LOCAL_ONLY)
  file(WRITE "C:/Users/gtats/python codes/jpegxs-enc/SVT-JPEG-XS/Build/windows_cuda/Source/Lib/install_local_manifest.txt"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
endif()
