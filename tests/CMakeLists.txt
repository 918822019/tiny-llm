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
)
target_link_libraries(tinyqwen_tests PRIVATE tinyqwen_runtime)
add_test(NAME tinyqwen_tests COMMAND tinyqwen_tests)
