cmake_minimum_required(VERSION 3.22)
project(FuzzFormer LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CUDA_ARCHITECTURES 86 CACHE STRING "" FORCE)

find_package(PkgConfig QUIET)

pkg_check_modules(LIBUV REQUIRED libuv)
if(LIBUV_FOUND)
  message(STATUS "libuv found: ${LIBUV_VERSION}")
endif()

option(FUZZFORMER_USE_TORCH "Enable libtorch integration" ON)

set(FUZZFORMER_HAS_TORCH OFF)
if(FUZZFORMER_USE_TORCH)
  find_package(Torch QUIET)
  if(Torch_FOUND)
    set(FUZZFORMER_HAS_TORCH ON)
    if(DEFINED Torch_VERSION)
      message(STATUS "Torch found: ${Torch_VERSION}")
    else()
      message(STATUS "Torch found")
    endif()
  else()
    message(WARNING "libtorch not found; building with stub interfaces.")
    message(STATUS "To enable full functionality:")
    message(STATUS "  1. Download libtorch from https://pytorch.org/get-started/locally/")
    message(STATUS "  2. Extract and set: export LIBTORCH_HOME=/path/to/libtorch")
    message(STATUS "  3. Reconfigure with: cmake -DCMAKE_PREFIX_PATH=${LIBTORCH_HOME} ..")
  endif()
endif()

add_library(fuzzformer_core STATIC
  src/core/fuzzyAttention.cpp
  src/core/fuzzyAttention.cu
  src/core/transformerBlock.cpp
  src/core/model.cpp
  src/runtime/eventLoop.cpp
  src/runtime/asyncScheduler.cpp
  src/runtime/metricsCollector.cpp
  src/visual/renderer.cpp
  src/visual/terminalHeatmap.cpp
  src/utils/tensorUtils.cpp
  src/utils/timer.cpp
  src/utils/logger.cpp
)

target_include_directories(fuzzformer_core
  PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)

target_include_directories(fuzzformer_core PUBLIC ${LIBUV_INCLUDE_DIRS})
target_link_libraries(fuzzformer_core PUBLIC ${LIBUV_LIBRARIES})

if(FUZZFORMER_HAS_TORCH)
  target_include_directories(fuzzformer_core PUBLIC ${TORCH_INCLUDE_DIRS})
  target_link_libraries(fuzzformer_core PUBLIC ${TORCH_LIBRARIES})
  target_compile_definitions(fuzzformer_core
    PUBLIC
      -DTORCH_API_INCLUDE_EXTENSION_H
      -D_GLIBCXX_USE_CXX11_ABI=1
      -DFUZZFORMER_HAS_TORCH
  )
else()
endif()

set_target_properties(fuzzformer_core PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  POSITION_INDEPENDENT_CODE ON
)

add_executable(fuzzformer
  src/main.cpp
)

target_link_libraries(fuzzformer
  PRIVATE
    fuzzformer_core
)

option(FUZZFORMER_BUILD_TESTS "Build the test suite" ON)

if(FUZZFORMER_BUILD_TESTS)
  include(CTest)
  include(FetchContent)

  FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
  )
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)

  add_executable(fuzzformer_tests
    tests/smoke.cpp
    tests/utils_test.cpp
    tests/test_transformer_block.cpp
    tests/test_model_inference.cpp
    tests/test_fuzzy_attention.cpp
    tests/test_async_runtime.cpp
    tests/test_visualization.cpp
    tests/test_metrics.cpp
    tests/benchmark_kernels.cpp
  )

  target_link_libraries(fuzzformer_tests
    PRIVATE
      fuzzformer_core
      GTest::gtest_main
  )

  include(GoogleTest)
  gtest_discover_tests(fuzzformer_tests)
endif()

