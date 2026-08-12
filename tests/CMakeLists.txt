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
)
target_link_libraries(tinyqwen_tests PRIVATE tinyqwen_runtime)
add_test(NAME tinyqwen_tests COMMAND tinyqwen_tests)
