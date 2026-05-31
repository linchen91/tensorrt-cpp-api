# FindTensorRT.cmake -- locate a TensorRT install (NVIDIA apt repo OR tarball) and expose
# the imported target TensorRT::TensorRT (nvinfer + nvonnxparser + headers). Supports
# TensorRT 10.0 through 11.x and errors clearly otherwise. Relocatable: it bakes no
# build-tree paths, so it can be installed alongside the package config (Phase E14/H).
#
# Hints: set -DTensorRT_DIR=<tarball root> (or the env var) to point at a tarball; on a
# host with libnvinfer-dev from the NVIDIA apt repo no hint is needed.

set(_trt_hints)
if(TensorRT_DIR)
    list(APPEND _trt_hints "${TensorRT_DIR}")
endif()
if(DEFINED ENV{TensorRT_DIR})
    list(APPEND _trt_hints "$ENV{TensorRT_DIR}")
endif()

# On Windows, scan common TensorRT install locations
if(WIN32)
    file(GLOB _trt_win_roots "C:/Program Files/NVIDIA GPU Computing Toolkit/TensorRT/*")
    list(APPEND _trt_hints ${_trt_win_roots})
endif()

find_path(TensorRT_INCLUDE_DIR
    NAMES NvInfer.h
    HINTS ${_trt_hints}
    PATH_SUFFIXES include
    PATHS /usr/include/x86_64-linux-gnu /usr/include /usr/local/include /usr/local/tensorrt/include)

find_library(TensorRT_nvinfer_LIBRARY
    NAMES nvinfer nvinfer_10 nvinfer_11 nvinfer.lib nvinfer_10.lib nvinfer_11.lib
    HINTS ${_trt_hints}
    PATH_SUFFIXES lib lib64 lib/x64 targets/x86_64-linux/lib
    PATHS /usr/lib/x86_64-linux-gnu /usr/lib /usr/local/lib)

find_library(TensorRT_nvonnxparser_LIBRARY
    NAMES nvonnxparser nvonnxparser_10 nvonnxparser_11 nvonnxparser.lib nvonnxparser_10.lib nvonnxparser_11.lib
    HINTS ${_trt_hints}
    PATH_SUFFIXES lib lib64 lib/x64 targets/x86_64-linux/lib
    PATHS /usr/lib/x86_64-linux-gnu /usr/lib /usr/local/lib)

# Extract TensorRT version from NvInferVersion.h. Two styles exist:
#   1.  #define NV_TENSORRT_MAJOR 10           (OSS / apt builds)
#   2.  #define NV_TENSORRT_MAJOR TRT_MAJOR_ENTERPRISE  (Enterprise / package builds)
#        #define TRT_MAJOR_ENTERPRISE 10
# Try both.
if(TensorRT_INCLUDE_DIR AND EXISTS "${TensorRT_INCLUDE_DIR}/NvInferVersion.h")
    file(READ "${TensorRT_INCLUDE_DIR}/NvInferVersion.h" _trt_ver_content)
    # Style 1: direct numeric NV_TENSORRT_MAJOR
    string(REGEX MATCH "#define NV_TENSORRT_MAJOR[ \t]+([0-9]+)" _trt_direct "${_trt_ver_content}")
    if(_trt_direct)
        set(TensorRT_VERSION_MAJOR "${CMAKE_MATCH_1}")
        string(REGEX MATCH "#define NV_TENSORRT_MINOR[ \t]+([0-9]+)" _trt_direct "${_trt_ver_content}")
        set(TensorRT_VERSION_MINOR "${CMAKE_MATCH_1}")
        string(REGEX MATCH "#define NV_TENSORRT_PATCH[ \t]+([0-9]+)" _trt_direct "${_trt_ver_content}")
        set(TensorRT_VERSION_PATCH "${CMAKE_MATCH_1}")
    else()
        # Style 2: enterprise macros
        string(REGEX MATCH "#define TRT_MAJOR_ENTERPRISE[ \t]+([0-9]+)" _trt_ent "${_trt_ver_content}")
        if(_trt_ent)
            set(TensorRT_VERSION_MAJOR "${CMAKE_MATCH_1}")
            string(REGEX MATCH "#define TRT_MINOR_ENTERPRISE[ \t]+([0-9]+)" _trt_ent "${_trt_ver_content}")
            set(TensorRT_VERSION_MINOR "${CMAKE_MATCH_1}")
            string(REGEX MATCH "#define TRT_PATCH_ENTERPRISE[ \t]+([0-9]+)" _trt_ent "${_trt_ver_content}")
            set(TensorRT_VERSION_PATCH "${CMAKE_MATCH_1}")
        endif()
    endif()
    if(TensorRT_VERSION_MAJOR AND TensorRT_VERSION_MINOR AND TensorRT_VERSION_PATCH)
        set(TensorRT_VERSION "${TensorRT_VERSION_MAJOR}.${TensorRT_VERSION_MINOR}.${TensorRT_VERSION_PATCH}")
    endif()
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(TensorRT
    REQUIRED_VARS TensorRT_nvinfer_LIBRARY TensorRT_nvonnxparser_LIBRARY TensorRT_INCLUDE_DIR
    VERSION_VAR TensorRT_VERSION)

if(TensorRT_FOUND)
    if(TensorRT_VERSION VERSION_LESS "10.0" OR NOT TensorRT_VERSION VERSION_LESS "12.0")
        message(FATAL_ERROR
            "tensorrt_cpp_api requires TensorRT 10.0 - 11.x, but found ${TensorRT_VERSION} at "
            "${TensorRT_INCLUDE_DIR}.\n"
            "  Point -DTensorRT_DIR=<root> at a supported tarball, or install libnvinfer-dev from "
            "the NVIDIA apt repo (scripts/install_deps.sh).")
    endif()

    if(NOT TARGET TensorRT::nvonnxparser)
        add_library(TensorRT::nvonnxparser UNKNOWN IMPORTED)
        set_target_properties(TensorRT::nvonnxparser PROPERTIES IMPORTED_LOCATION "${TensorRT_nvonnxparser_LIBRARY}")
    endif()
    if(NOT TARGET TensorRT::TensorRT)
        add_library(TensorRT::TensorRT UNKNOWN IMPORTED)
        set_target_properties(TensorRT::TensorRT PROPERTIES
            IMPORTED_LOCATION "${TensorRT_nvinfer_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${TensorRT_INCLUDE_DIR}"
            INTERFACE_LINK_LIBRARIES TensorRT::nvonnxparser)
    endif()
    set(TensorRT_LIBRARIES ${TensorRT_nvinfer_LIBRARY} ${TensorRT_nvonnxparser_LIBRARY})
    set(TensorRT_INCLUDE_DIRS ${TensorRT_INCLUDE_DIR})
endif()

mark_as_advanced(TensorRT_INCLUDE_DIR TensorRT_nvinfer_LIBRARY TensorRT_nvonnxparser_LIBRARY)
