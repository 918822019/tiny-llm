# 单元测试：所有 test_*.cpp 编进一个可执行，共用 test_framework.h
add_executable(tinyqwen_tests
        test_main.cpp
        test_format.cpp
        test_loader.cpp
        test_rmsnorm.cpp
        test_rope.cpp
        test_matvec.cpp
        test_softmax_argmax.cpp
        test_attention.cpp
        test_kv_cache.cpp
        test_config.cpp
        test_ops_neon.cpp
        test_gdn_ops.cpp
        test_gdn_ops_neon.cpp
)
# GPU decode engine 的算子对齐测试只在 CUDA 构建编译（引用了 device 指针
# 启动器，链接 kernels 的 CUDA OBJECT）；无 CUDA 平台不受影响。
if (TINYQWEN_HAS_CUDA)
    target_sources(tinyqwen_tests PRIVATE test_gpu_ops.cpp)
endif ()
# 显式链两个 OBJECT 库：同 runtime/CMakeLists.txt 注释——OBJECT 库的 .o
# 要直接链给可执行才上链接行（变体的静态注册器依赖这一点）。
target_link_libraries(tinyqwen_tests PRIVATE tinyqwen_runtime tinyqwen_kernels)
add_test(NAME tinyqwen_tests COMMAND tinyqwen_tests)
