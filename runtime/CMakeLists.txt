# ============================================================================
# runtime/CMakeLists.txt — Runtime 库 + CLI 可执行文件构建配置
# ============================================================================
# 本文件定义两个构建目标：
#   1. tinyqwen_runtime — OBJECT 库，包含模型加载、推理引擎、KV Cache 等核心组件
#   2. tinyqwen         — CLI 可执行文件，链接 runtime + kernels 提供命令行推理入口
#
# 为什么用 OBJECT 库而不是 STATIC 库？
#   kernel 变体（NEON/CUDA/ref）靠静态注册器（static Registrar）接入 dispatch。
#   静态库归档是"按需提取"成员——没人引用的 .o（比如注册器所在的变体文件）
#   不会被链入，注册就丢了。OBJECT 库配合下方可执行的"显式直链"
#   （tinyqwen / tinyqwen_tests 各自直链 runtime + kernels），把所有 .o
#   直接放上链接行，注册器必然生效。
# ============================================================================

# ---------------------------------------------------------------------------
# OBJECT 库：tinyqwen_runtime
# ---------------------------------------------------------------------------
# add_library(... OBJECT) 只编译不链接，产出 .o 文件供后续目标使用
add_library(tinyqwen_runtime OBJECT
        tensor.cpp              # Tensor 视图 / 张量操作
        model_loader.cpp        # .tqwen 模型文件加载器
        kv_cache.cpp            # KV Cache 内存管理（arena 分配 + 追加）
        gdn_state.cpp           # GDN 状态管理（Gated Delta Network 的递归状态）
        profiler.cpp            # 性能剖析器（计时 + JSON 输出）
        config.cpp              # 运行时配置解析
        backend_cpu.cpp         # CPU 后端实现
        qwen_model.cpp          # Qwen 模型结构定义 + 参数绑定
        qwen_forward_token.cpp  # Decode 阶段单 token forward
        qwen_forward_prefill.cpp # Prefill 阶段 prompt forward（Qwen2.x 批量 + 分发）
        qwen_forward_prefill_qwen35.cpp # Qwen3.5 批量 prefill（GEMM 路径，权重只读一遍）
        ../quantization/quant.cpp # 量化子系统实现（放在 quantization/ 目录）
)

# ---------------------------------------------------------------------------
# CUDA 后端（仅 CUDA 构建时编译）
# ---------------------------------------------------------------------------
# TINYQWEN_HAS_CUDA 由顶层 CMakeLists.txt 在检测到 CUDA 编译器时设置
if (TINYQWEN_HAS_CUDA)
    # 向已有 OBJECT 库追加 CUDA 后端源文件
    target_sources(tinyqwen_runtime PRIVATE backend_cuda.cpp)
endif ()

# 设置头文件搜索路径：PUBLIC 表示使用此库的目标也自动获得该 include 路径
target_include_directories(tinyqwen_runtime PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# 链接 kernels OBJECT 库（runtime 调用 kernel dispatch 函数）
target_link_libraries(tinyqwen_runtime PUBLIC tinyqwen_kernels)

# Qwen3.5 批量 prefill 的 GEMM 后端：Apple Silicon 用 Accelerate（cblas_sgemm，
# 自动走 AMX/多核）。框架经 OBJECT 库的 PUBLIC 接口传播给所有链接方
# （tinyqwen / tinyqwen_tests）。其他平台（如 Android）无此后端，
# forward_prefill_qwen35_batch 运行期返回 -2 回退逐 token 路径。
if (APPLE)
    target_link_libraries(tinyqwen_runtime PUBLIC "-framework Accelerate")
endif ()

# 批量 prefill 文件里：融合 W4A8 matmul 需 +dotprod（SDOT），
# fp16→fp32 组头转换走硬件需 +fp16（守卫 __ARM_FEATURE_FP16_SCALAR_ARITHMETIC，
# 同 sdot4 的教训）。
if (CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
    set_source_files_properties(qwen_forward_prefill_qwen35.cpp PROPERTIES
            COMPILE_OPTIONS "-march=armv8.2-a+dotprod+fp16")
endif ()

# ---------------------------------------------------------------------------
# CLI 可执行文件：tinyqwen
# ---------------------------------------------------------------------------
# main.cpp 提供命令行参数解析 + 推理循环入口
add_executable(tinyqwen main.cpp)

# 显式链两个 OBJECT 库：
# CMake 3.21 之前，OBJECT 库的 .o 不会沿依赖传递到最终目标，
# 只有"直接链给可执行"才把 .o 真正放上链接行。
# 这保证了 kernel 变体的静态注册器不会被链接器丢弃。
target_link_libraries(tinyqwen PRIVATE tinyqwen_runtime tinyqwen_kernels)
