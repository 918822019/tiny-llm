cmake_minimum_required(VERSION 3.16)
project(tinyqwen CXX)

# ---------------------------------------------------------------------------
# tinyqwen: single-model tiny runtime for Qwen2.5-like decoder-only models.
#
# Scope is intentionally narrow:
#   - CPU-only reference path (FP32), batch = 1, token-by-token decode.
#   - No graph executor, no operator registry, no scheduler.
#   - Android is supported via the NDK toolchain file (see docs/build_android.md).
# ---------------------------------------------------------------------------

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_compile_options(-Wall -Wextra)

add_subdirectory(kernels)
add_subdirectory(runtime)

option(TINYQWEN_BUILD_TESTS "Build tinyqwen unit tests" ON)
if(TINYQWEN_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()
