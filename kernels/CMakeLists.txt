# 朴素标量 reference kernels：正确性优先，v1 不用 SIMD。
# 每个算子一个文件夹：<op>/<op>_<dtype>_<variant>.cpp。
# *_ref.cpp 是参考实现（base，保留）；将来的优化版以
# <op>_<dtype>_<variant>.cpp 命名，放进对应算子文件夹，
# 文件末尾用 TINYQWEN_MATVEC_VARIANT 宏自注册进 dispatch。
#
# 注意：这里是 OBJECT 库而不是 STATIC——变体靠文件里的静态注册器接入
# dispatch，而静态库归档按"被引用才提取"丢弃没人引用的 .o，注册器会丢。
# OBJECT 库把 .o 直接交给最终链接；配套的 runtime 也是 OBJECT 库
# （见 runtime/CMakeLists.txt 注释），两者共同保证注册器必然生效。
add_library(tinyqwen_kernels OBJECT
        rmsnorm/rmsnorm_ref.cpp
        rope/rope_ref.cpp
        matvec/matvec_f32_ref.cpp
        matvec/matvec_f32_double_2_float.cpp
        softmax/softmax_ref.cpp
        attention/attention_decode_ref.cpp
        silu/silu_ref.cpp
        argmax/argmax_ref.cpp
        dispatch.cpp
)
target_include_directories(tinyqwen_kernels PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
