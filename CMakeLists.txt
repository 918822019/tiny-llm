cmake_minimum_required(VERSION 3.16)
project(tinyqwen CXX)

# ---------------------------------------------------------------------------
# tinyqwen：面向单模型（Qwen2.5-like decoder-only）的实验 runtime。
#
# 范围刻意收窄：
#   - CPU-only 的 reference 路径（FP32）是主线，batch = 1，token-by-token decode；
#     matvec 的 CUDA 变体是可选的第二条腿（见下方 CUDA 检测块），检测不到
#     nvcc 时完全不参与构建，不影响这条主线。
#   - 不做 graph executor、不做 operator registry、不做 scheduler。
#   - Android 通过 NDK toolchain 文件支持（见 docs/android.md）。
# ---------------------------------------------------------------------------

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif ()

add_compile_options(-Wall -Wextra)

# ---------------------------------------------------------------------------
# 可选 CUDA 支持：必须在这里（第一个 add_subdirectory / 第一个目标创建之前）
# 检测 + 设置 CMAKE_CUDA_ARCHITECTURES——它是"目标创建那一刻"从这个变量取
# 默认值的，晚了会留空触发 CMP0104 警告，且 nvcc 实际不知道编译到哪个架构。
# 其他平台（没装 CUDA 工具链）check_language 失败，TINYQWEN_HAS_CUDA=OFF，
# kernels/CMakeLists.txt 里整段 cuda 相关配置会被跳过，等同于没有这一块。
# ---------------------------------------------------------------------------
include(CheckLanguage)
check_language(CUDA)
if (CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    if (NOT CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES 86) # A10 = Ampere SM86；换卡按需覆盖
    endif ()
    set(TINYQWEN_HAS_CUDA ON)
else ()
    message(STATUS "tinyqwen: 未检测到 CUDA 编译器，跳过 matvec 的 cuda 变体")
    set(TINYQWEN_HAS_CUDA OFF)
endif ()

add_subdirectory(kernels)
add_subdirectory(runtime)

option(TINYQWEN_BUILD_TESTS "Build tinyqwen unit tests" ON)
if (TINYQWEN_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif ()
