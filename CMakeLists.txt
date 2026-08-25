# =============================================================================
# CMakeLists.txt —— tinyqwen 项目顶层构建配置
# =============================================================================
# 文件级别说明：
#   本文件是 tinyqwen 项目的 CMake 顶层构建脚本。它定义了项目的
#   基本属性（C++ 标准、编译选项）、可选的 CUDA 支持检测、以及
#   子目录的构建顺序。
#
# 项目定位：
#   tinyqwen：面向单模型（Qwen2.5-like decoder-only）的实验 runtime。
#
# 范围刻意收窄：
#   - CPU-only 的 reference 路径（FP32）是主线，batch = 1，token-by-token decode；
#     matvec 的 CUDA 变体是可选的第二条腿（见下方 CUDA 检测块），检测不到
#     nvcc 时完全不参与构建，不影响这条主线。
#   - 不做 graph executor、不做 operator registry、不做 scheduler。
#   - Android 通过 NDK toolchain 文件支持（见 docs/android.md）。
#
# 构建结构：
#   - kernels/   : 计算内核（matvec、非 matvec 算子、CUDA kernel）
#   - runtime/   : 运行时核心（模型加载、推理引擎、配置）
#   - tests/     : 单元测试（可选，通过 TINYQWEN_BUILD_TESTS 控制）
# =============================================================================

cmake_minimum_required(VERSION 3.16)   # 最低 CMake 版本要求
project(tinyqwen CXX)                  # 项目名称和语言（C++）

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

# C++ 编译标准设置
set(CMAKE_CXX_STANDARD 17)           # 使用 C++17 标准
set(CMAKE_CXX_STANDARD_REQUIRED ON)   # 强制要求 C++17（编译器不支持则报错）
set(CMAKE_CXX_EXTENSIONS OFF)         # 禁用编译器扩展（保证可移植性）

# 默认构建类型为 Release（优化编译）
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif ()

# 编译警告选项：开启所有警告 + 额外警告
add_compile_options(-Wall -Wextra)

# ---------------------------------------------------------------------------
# 可选 CUDA 支持：必须在这里（第一个 add_subdirectory / 第一个目标创建之前）
# 检测 + 设置 CMAKE_CUDA_ARCHITECTURES——它是"目标创建那一刻"从这个变量取
# 默认值的，晚了会留空触发 CMP0104 警告，且 nvcc 实际不知道编译到哪个架构。
# 其他平台（没装 CUDA 工具链）check_language 失败，TINYQWEN_HAS_CUDA=OFF，
# kernels/CMakeLists.txt 里整段 cuda 相关配置会被跳过，等同于没有这一块。
# ---------------------------------------------------------------------------
include(CheckLanguage)                # 引入 CheckLanguage 模块（检测编译器支持）
check_language(CUDA)                  # 检测 CUDA 编译器是否可用
if (CMAKE_CUDA_COMPILER)
    # CUDA 编译器可用：启用 CUDA 语言支持
    enable_language(CUDA)
    if (NOT CMAKE_CUDA_ARCHITECTURES)
        # 默认 CUDA 架构：SM86（NVIDIA A10 = Ampere 架构）
        # 换卡时可按需覆盖此值
        set(CMAKE_CUDA_ARCHITECTURES 86)
    endif ()
    # 设置全局宏：指示项目启用了 CUDA 支持
    set(TINYQWEN_HAS_CUDA ON)
else ()
    # CUDA 编译器不可用：跳过 CUDA 相关构建
    message(STATUS "tinyqwen: 未检测到 CUDA 编译器，跳过 matvec 的 cuda 变体")
    set(TINYQWEN_HAS_CUDA OFF)
endif ()

# 构建子目录：计算内核（matvec、非 matvec 算子、CUDA kernel）
add_subdirectory(kernels)
# 构建子目录：运行时核心（模型加载、推理引擎、配置）
add_subdirectory(runtime)

# 单元测试（可选）
option(TINYQWEN_BUILD_TESTS "Build tinyqwen unit tests" ON)  # 默认开启
if (TINYQWEN_BUILD_TESTS)
    enable_testing()                    # 启用 CTest 测试框架
    add_subdirectory(tests)             # 构建测试子目录
endif ()

# 算子级微基准（可选）：枚举各注册表全部 matvec 实现，校验 + 计时对比。
# 新增算子变体自注册后即自动纳入评测，见 benchmarks/bench_kernels.cpp。
option(TINYQWEN_BUILD_BENCHMARKS "Build tinyqwen kernel benchmarks" ON)
if (TINYQWEN_BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif ()
