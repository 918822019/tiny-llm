# 朴素标量 reference kernels：正确性优先，v1 不用 SIMD。
# 每个算子一个文件夹：<op>/<op>_<dtype>_<variant>.cpp。
# *_ref.cpp 是参考实现（base，保留）；将来的优化版以
# <op>_<dtype>_<variant>.cpp 命名，放进对应算子文件夹；
# dispatch.cpp 负责在实现之间选择。
add_library(tinyqwen_kernels STATIC
        rmsnorm/rmsnorm_ref.cpp
        rope/rope_ref.cpp
        matvec/matvec_f32_ref.cpp
        softmax/softmax_ref.cpp
        attention/attention_decode_ref.cpp
        silu/silu_ref.cpp
        argmax/argmax_ref.cpp
        dispatch.cpp
)
target_include_directories(tinyqwen_kernels PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
