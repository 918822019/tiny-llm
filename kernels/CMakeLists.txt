# 朴素标量 reference kernels：正确性优先，v1 不用 SIMD。
add_library(tinyqwen_kernels STATIC
  rmsnorm_ref.cpp
  rope_ref.cpp
  matvec_f32_ref.cpp
  softmax_ref.cpp
  attention_decode_ref.cpp
  silu_ref.cpp
  argmax_ref.cpp
)
target_include_directories(tinyqwen_kernels PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
