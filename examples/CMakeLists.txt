cmake_minimum_required(VERSION 3.22)

# These examples are meant to be built against the INSTALLED library:
#   cmake -S examples -B build/examples -DCMAKE_PREFIX_PATH=<install-prefix>
# When this directory is instead pulled into the parent build (-DTRT_CPP_API_BUILD_EXAMPLES=ON),
# the library targets already exist, so we declare no project and skip find_package.
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    project(tensorrt_cpp_api_examples LANGUAGES CXX)
    find_package(tensorrt_cpp_api REQUIRED)
endif()

# Image decode/encode is handled by the vendored stb single-header libs (common/), so the
# examples have NO third-party image dependency -- they build anywhere the library does.
if(MSVC)
    set(TRTCPP_EXAMPLE_COMMON_FLAGS /utf-8)
    set(TRTCPP_EXAMPLE_COMMON_DEFS _CRT_SECURE_NO_WARNINGS)
endif()

function(add_example name)
    add_executable(${name} ${name}/main.cpp ${CMAKE_CURRENT_SOURCE_DIR}/common/stb_impl.cpp)
    target_link_libraries(${name} PRIVATE tensorrt_cpp_api::tensorrt_cpp_api tensorrt_cpp_api::preproc)
    target_compile_features(${name} PRIVATE cxx_std_20)
    if(MSVC)
        target_compile_options(${name} PRIVATE ${TRTCPP_EXAMPLE_COMMON_FLAGS})
        target_compile_definitions(${name} PRIVATE ${TRTCPP_EXAMPLE_COMMON_DEFS})
        set_target_properties(${name} PROPERTIES DEBUG_POSTFIX "d")
    endif()
endfunction()

add_example(classification)
add_example(detection)
add_example(segmentation)

# Inference-throughput benchmark (no image I/O, so no stb): the C++ baseline for the Python
# zero-copy perf-parity check.
add_executable(benchmark benchmark/main.cpp)
target_link_libraries(benchmark PRIVATE tensorrt_cpp_api::tensorrt_cpp_api)
target_compile_features(benchmark PRIVATE cxx_std_20)
if(MSVC)
    target_compile_options(benchmark PRIVATE ${TRTCPP_EXAMPLE_COMMON_FLAGS})
    target_compile_definitions(benchmark PRIVATE ${TRTCPP_EXAMPLE_COMMON_DEFS})
    set_target_properties(benchmark PROPERTIES DEBUG_POSTFIX "d")
endif()

# On Windows with a shared (DLL) core library, copy tensorrt_cpp_api.dll next to each
# example executable at build time so they can run from the build directory.
if(MSVC AND BUILD_SHARED_LIBS)
    foreach(_tgt classification detection segmentation benchmark)
        if(TARGET ${_tgt})
            add_custom_command(TARGET ${_tgt} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "$<TARGET_FILE:tensorrt_cpp_api>" "$<TARGET_FILE_DIR:${_tgt}>/"
                COMMAND_EXPAND_LISTS
            )
        endif()
    endforeach()
endif()
