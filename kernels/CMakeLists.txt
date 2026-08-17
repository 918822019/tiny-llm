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
        # rmsnorm NEON 变体（float 累加平方和 + 向量化缩放；仅 aarch64，其他平台编译为空）。
        rmsnorm/rmsnorm_neon.cpp
        rope/rope_ref.cpp
        # rope NEON 变体（cos/sin 表与 ref 逐字相同，旋转循环向量化；仅 aarch64）。
        rope/rope_neon.cpp
        matvec/matvec_f32_ref.cpp
        # f16 权重族：参考实现（所有平台，f16 模型的正确性基准 + 兜底）。
        matvec/matvec_f16_ref.cpp
        # f16 满栈变体：NEON(fp32 累加) + 线程池 + k/v 融合 + LDNP，与 f32
        # 阶梯顶层同名注册，按模型 dtype 解析（仅 aarch64；其他平台编译为空）。
        matvec/matvec_f16_neon_mt_kv_nt.cpp
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
        # attention NEON 变体（online softmax 结构不变，点积 + oh 更新向量化；仅 aarch64）。
        attention/attention_decode_neon.cpp
        silu/silu_ref.cpp
        # swiglu 融合：ref（全平台，正确性锚点 + 兜底）+ NEON（向量化 exp；仅 aarch64）。
        silu/swiglu_ref.cpp
        silu/swiglu_neon.cpp
        argmax/argmax_ref.cpp
        # argmax NEON 变体（两遍法：NEON 求 max + NEON 找首个相等下标，与 ref 逐位一致；仅 aarch64）。
        argmax/argmax_neon.cpp
        dispatch.cpp
)
target_include_directories(tinyqwen_kernels PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# neon_mt 用 std::thread 建常驻线程池。OBJECT 库的 PUBLIC 链接需求会传播给
# 所有链 tinyqwen_kernels 的目标（runtime/tests），无需各处重复写。
find_package(Threads REQUIRED)
target_link_libraries(tinyqwen_kernels PUBLIC Threads::Threads)

# ---------------------------------------------------------------------------
# matvec 的 CUDA 参考变体（可选，只在检测到 CUDA 编译器时才编译进去）。
# CUDA 语言探测 + CMAKE_CUDA_ARCHITECTURES 已在顶层 CMakeLists.txt 里做过
# （必须在任何目标创建之前设置，见那里的注释），这里只消费
# TINYQWEN_HAS_CUDA 这个开关。
#
# 和 NEON 变体的"其他平台编译为空"不是同一种机制——NEON 靠预处理器
# `#if defined(__aarch64__)` 在源码层面自行留空；.cu 文件离不开 CUDA
# 语言支持，没有 nvcc 时必须在 CMake 层就不把它加进编译，否则配置阶段
# 直接报错。TINYQWEN_HAS_CUDA 为 OFF 时这一整块跳过，其他平台
# （mac / Android / 没装 CUDA 的 CI）不受任何影响。
# ---------------------------------------------------------------------------
if (TINYQWEN_HAS_CUDA)
    find_package(CUDAToolkit REQUIRED)
    target_sources(tinyqwen_kernels PRIVATE
            # cuda = 参考版（每调用重传权重，量传输税的锚点）；
            # cuda_resident = 权重常驻显存（按 host 指针缓存 device 副本，
            # kernel 与参考版逐字相同，隔离"常驻"这一个变量）；
            # cuda_resident_ws = cuda_resident + x/y 常驻 workspace（grow-only，
            # 删掉每调用的 cudaMalloc/cudaFree，kernel 仍逐字不变）。
            matvec/matvec_f32_cuda.cu
            matvec/matvec_f32_cuda_resident.cu
            matvec/matvec_f32_cuda_resident_ws.cu
            # cuda_resident_coal = cuda_resident + 合并访存 kernel（block-per-row
            # + warp shuffle 归约，warp 内连续读权重，吃满 HBM 带宽）。
            matvec/matvec_f32_cuda_resident_coal.cu
            # f16 权重版 cuda_resident_coal（权重流量减半）；与 f32 同名注册，
            # 按模型 dtype 解析（f16 模型 + --matvec-impl cuda_resident_coal 用本文件）。
            matvec/matvec_f16_cuda_resident_coal.cu
            # cuda_resident_coal_ws = coal kernel + x/y 常驻 workspace + 删冗余 sync
            # （砍每调用固定开销；kernel 与 coal 逐字相同）。
            matvec/matvec_f32_cuda_resident_coal_ws.cu
            # f16 版权重常驻+合并访存+砍开销（f16 满栈）；与 f32 同名注册按 dtype 解析。
            matvec/matvec_f16_cuda_resident_coal_ws.cu
            # f16 融合：qkv 三合一 + gate/up(pair) 二合一，减调用次数砍启动/拷贝开销；
            # 注册进 pair/qkv f16 注册表，同名 cuda_resident_coal_ws。
            matvec/matvec_f16_cuda_fused.cu
            # GPU decode engine 的算子 kernel（device 指针形态，供 engine 编排 +
            # test_gpu_ops 对齐 *_ref）。engine 本体 gpu_engine.cu 后续加入。
            cuda/gpu_kernels.cu)
    set_target_properties(tinyqwen_kernels PROPERTIES CUDA_STANDARD 17)
    target_link_libraries(tinyqwen_kernels PUBLIC CUDA::cudart)
endif ()
