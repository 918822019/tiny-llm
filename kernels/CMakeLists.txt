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
        # 归因阶梯变体（每层只加一个技术，便于 A/B 归因）：
        # acc4 = 标量 4 链累加（隔离"累加结构"，所有平台）；
        # neon_nofma = NEON 向量化但乘加分开（隔离"SIMD 宽度"，仅 aarch64）；
        # neon = 完整版（再加 FMA 融合，仅 aarch64）。后两者其他平台编译为空。
        matvec/matvec_f32_acc4.cpp
        matvec/matvec_f32_neon_nofma.cpp
        matvec/matvec_f32_neon.cpp
        # neon_mt = neon + 多线程行切分（常驻线程池，仅 aarch64；其他平台编译为空）。
        matvec/matvec_f32_neon_mt.cpp
        # neon_mt_bal = neon_mt + 自校准加权分块（大小核异构下按实测吞吐分行；
        # 实测带宽墙下收益≈0，归档保留；仅 aarch64）。
        matvec/matvec_f32_neon_mt_bal.cpp
        # neon_mt_kv = neon_mt + k/v 成对融合（共享输入的两个小 matvec 合成一次
        # fork-join；pair 通用入口见 dispatch.h；仅 aarch64）。
        matvec/matvec_f32_neon_mt_kv.cpp
        # neon_mt_kv_nt = neon_mt_kv + 流式加载（权重 LDNP 内联汇编 + 对齐兜底；
        # 实验性：带宽墙下的加载侧微调，预期≈0；仅 aarch64）。
        matvec/matvec_f32_neon_mt_kv_nt.cpp
        softmax/softmax_ref.cpp
        attention/attention_decode_ref.cpp
        silu/silu_ref.cpp
        argmax/argmax_ref.cpp
        dispatch.cpp
)
target_include_directories(tinyqwen_kernels PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# neon_mt 用 std::thread 建常驻线程池。OBJECT 库的 PUBLIC 链接需求会传播给
# 所有链 tinyqwen_kernels 的目标（runtime/tests），无需各处重复写。
find_package(Threads REQUIRED)
target_link_libraries(tinyqwen_kernels PUBLIC Threads::Threads)
