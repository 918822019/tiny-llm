cmake_minimum_required(VERSION 3.16)
project(tinyqwen CXX)

# ---------------------------------------------------------------------------
# tinyqwen：面向单模型（Qwen2.5-like decoder-only）的实验 runtime。
#
# 范围刻意收窄：
#   - 只做 CPU-only 的 reference 路径（FP32），batch = 1，token-by-token decode。
#   - 不做 graph executor、不做 operator registry、不做 scheduler。
#   - Android 通过 NDK toolchain 文件支持（见 docs/build_android.md）。
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
