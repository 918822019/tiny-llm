# ============================================================================
# tests/CMakeLists.txt — 单元测试构建配置
# ============================================================================
# 所有 test_*.cpp 编进一个可执行文件 tinyqwen_tests，共用 test_framework.h。
# test_main.cpp 提供 main() 入口，调用 tinytest::run_all() 运行全部用例。
# ============================================================================

# 定义测试可执行文件，列出所有测试源文件
# add_executable 将多个 .cpp 编译链接为一个二进制
add_executable(tinyqwen_tests
        test_main.cpp           # main() 入口 + run_all() 调用
        test_format.cpp         # 模型文件格式（.tqwen）解析测试
        test_loader.cpp         # 模型加载器测试
        test_rmsnorm.cpp        # RMSNorm 算子测试
        test_rope.cpp           # 旋转位置编码测试
        test_matvec.cpp         # 矩阵向量乘法全变体测试（f32/f16/i4）
        test_matvec_i4.cpp      # INT4 matvec/matmul 专项测试
        test_matvec_vq2.cpp     # VQ2（2-bit 向量量化）matvec/pair/qkv/matmul 测试
        test_biip_rotate.cpp    # BiIP 激活旋转（分块快速 Hadamard）正确性测试
        test_softmax_argmax.cpp # Softmax / Argmax 测试
        test_attention.cpp      # Attention decode 测试
        test_kv_cache.cpp       # KV Cache 内存管理测试
        test_config.cpp         # 配置解析测试
        test_ops_neon.cpp       # NEON 通用算子 vs ref 对齐测试
        test_gdn_ops.cpp        # GDN 算子（ref）正确性测试
        test_gdn_ops_neon.cpp   # GDN 算子 NEON vs ref 对齐测试
        test_backend.cpp        # Backend 抽象层测试
        test_quantization.cpp   # 量化子系统测试
)

# ---------------------------------------------------------------------------
# CUDA 后端测试（仅 CUDA 构建时编译）
# ---------------------------------------------------------------------------
# TINYQWEN_HAS_CUDA 由顶层 CMakeLists.txt 在检测到 CUDA 编译器时设置
if (TINYQWEN_HAS_CUDA)
    # target_sources PRIVATE 向已有目标追加源文件
    target_sources(tinyqwen_tests PRIVATE test_backend_cuda.cpp)
endif ()

# ---------------------------------------------------------------------------
# GPU decode engine 算子对齐测试（仅 CUDA 构建时编译）
# ---------------------------------------------------------------------------
# 引用了 device 指针启动器（gpu_kernels.cuh），需要链接 kernels 的 CUDA OBJECT。
# 无 CUDA 平台的构建不受影响。
if (TINYQWEN_HAS_CUDA)
    target_sources(tinyqwen_tests PRIVATE test_gpu_ops.cpp)
endif ()

# ---------------------------------------------------------------------------
# 链接依赖库
# ---------------------------------------------------------------------------
# 显式链两个 OBJECT 库：同 runtime/CMakeLists.txt 注释——OBJECT 库的 .o
# 要直接链给可执行才上链接行。kernel 变体的静态注册器（static Registrar）
# 依赖这一点才能被链接器保留（否则被当作"未引用符号"丢弃）。
target_link_libraries(tinyqwen_tests PRIVATE tinyqwen_runtime tinyqwen_kernels)

# 注册为 CTest 测试用例（cmake --build && ctest 可自动发现并运行）
add_test(NAME tinyqwen_tests COMMAND tinyqwen_tests)
