# 朴素标量 reference kernels：正确性优先，v1 不用 SIMD。
# *_ref.cpp 是参考实现（base，保留）；将来的优化版以
# <op>_<dtype>_<variant>.cpp 命名新增；dispatch.cpp 负责在它们之间选择。
add_library(tinyqwen_kernels STATIC
        rmsnorm_ref.cpp
        rope_ref.cpp
        matvec_f32_ref.cpp
        softmax_ref.cpp
        attention_decode_ref.cpp
        silu_ref.cpp
        argmax_ref.cpp
        dispatch.cpp
)
target_include_directories(tinyqwen_kernels PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
