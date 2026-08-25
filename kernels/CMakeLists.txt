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
        # INT4 weight-only：非对称 uint4 per-group 量化（正确性锚 ref + NEON + 多线程）。
        matvec/matvec_i4_ref.cpp
        matvec/matvec_i4_neon.cpp
        matvec/matvec_i4_neon_mt.cpp
        # VQ2（2-bit 向量量化）：每权重 uint8 码本索引 + per-tensor 码本，
        # 反量化纯查表（2.0bit 部署洁净点）。ref=double 累加锚；neon=fp32+4路ILP。
        matvec/matvec_vq2_ref.cpp
        matvec/matvec_vq2_neon.cpp
        # neon_mr = neon + 4 行并行（独立 FMA 链隐藏延迟 + 共享 x4 加载），
        # 归因阶梯下一层（对齐 i4 族 sdot2 的 multi-row 路径；仅 aarch64）。
        matvec/matvec_vq2_neon_mr.cpp
        # neon_mr_mt = neon_mr + 常驻线程池行切分（对齐 i4 族 *_mt 路径，
        # 粒度阈值 262144 元素；仅 aarch64）。
        matvec/matvec_vq2_neon_mr_mt.cpp
        # BiIP 激活旋转（推理时对激活施加旋转配对逆变换，配合旋转量化权重）。
        biip/biip_rotate.cpp
        # W4A8 SDOT：权重解包 int8 + 激活 int8 对称量化 + SDOT 整数点积（armv8.2
        # dotprod；把 unpack 指令/字节从 ~3.5 打到 ~0.9，目标逼近带宽墙）。
        matvec/matvec_i4_sdot.cpp
        # SDOT v2：预计算 scale/zero 为 f32 数组（消 per-group memcpy+half_to_float）
        # + 2-row 并行内循环（2 条独立 SDOT 链 + 共享激活加载，提高 ILP）。
        matvec/matvec_i4_sdot2.cpp
        # SDOT v3：动态行调度（work-stealing，异构核无拖尾）+ 内联组头
        # 硬件 FCVT（砍掉预计算缓存的 +25% 流量）+ 128 位解包（-37% 指令）。
        matvec/matvec_i4_sdot3.cpp
        # SDOT v4：修正组头转换的特性守卫——clang +fp16 不定义
        # __ARM_FEATURE_FP16（sdot3 守卫失效，热循环静默跑软件转换），
        # 改用 __ARM_FEATURE_FP16_SCALAR_ARITHMETIC + _Float16（2 条指令）。
        matvec/matvec_i4_sdot4.cpp
        # SDOT v5：对称量化版（zero=8），无 zero 修正项与前缀和，省 per-group 标量。
        matvec/matvec_i4_sdot5.cpp
        matvec/matmul_ref.cpp
        matvec/matmul_i4_ref.cpp
        # Qwen3.5 Gated DeltaNet（GDN）算子：l2norm / causal conv1d 单步 /
        # gated delta rule 递归步 / 门控 RMSNorm / partial RoPE（v2 新增）。
        gdn/gdn_ops_ref.cpp
        # GDN NEON 变体（仅 aarch64，其他平台编译为空）。
        gdn/l2norm_inplace_neon.cpp
        gdn/rmsnorm_gated_neon.cpp
        gdn/causal_conv1d_update_neon.cpp
        gdn/gdn_step_neon.cpp
        dispatch.cpp
)
target_include_directories(tinyqwen_kernels PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# SDOT kernel 需要 armv8.2 dotprod。只给这一个源文件开 +dotprod，不动全局 arch。
# Apple Silicon 与 Android arm64 都支持；源码里另有 __ARM_FEATURE_DOTPROD 守卫，
# 平台不开此 flag 时该文件自动编译为空。
if (CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
    set_source_files_properties(matvec/matvec_i4_sdot.cpp PROPERTIES
            COMPILE_OPTIONS "-march=armv8.2-a+dotprod")
    set_source_files_properties(matvec/matvec_i4_sdot2.cpp PROPERTIES
            COMPILE_OPTIONS "-march=armv8.2-a+dotprod")
    # sdot3 另开 +fp16：组头的 fp16→fp32 走硬件 FCVT（__ARM_FEATURE_FP16）。
    # Apple Silicon 与 Android arm64（armv8.2+dotprod 的芯片）均支持；
    # 源码里另有 __ARM_FEATURE_FP16 守卫，缺特性时退回软件转换（仍逐位一致）。
    set_source_files_properties(matvec/matvec_i4_sdot3.cpp PROPERTIES
            COMPILE_OPTIONS "-march=armv8.2-a+dotprod+fp16")
    # sdot4 同开 +fp16（守卫修正为 __ARM_FEATURE_FP16_SCALAR_ARITHMETIC，
    # 这才是 clang 实际定义的宏；见文件头注释的翻车记录）。
    set_source_files_properties(matvec/matvec_i4_sdot4.cpp PROPERTIES
            COMPILE_OPTIONS "-march=armv8.2-a+dotprod+fp16")
    # sdot5 同开 +dotprod+fp16（对称量化版，与 sdot4 同 SDK 需求）。
    set_source_files_properties(matvec/matvec_i4_sdot5.cpp PROPERTIES
            COMPILE_OPTIONS "-march=armv8.2-a+dotprod+fp16")
endif ()

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
    # gpu_engine.cu 需要读 runtime 的 ModelFile/TensorView（仅头文件依赖，
    # 不产生链接环：model_loader.h 不反向 include kernels 头）。
    target_include_directories(tinyqwen_kernels PRIVATE ${CMAKE_SOURCE_DIR}/runtime)
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
            # test_gpu_ops 对齐 *_ref）。
            cuda/gpu_kernels.cu
            # GPU decode engine 本体：权重/激活/KV 常驻显存，单 stream 跑整段
            # forward，自注册进 decode-engine registry（名字 "cuda"）。
            cuda/gpu_engine.cu)
    set_target_properties(tinyqwen_kernels PROPERTIES CUDA_STANDARD 17)
    target_link_libraries(tinyqwen_kernels PUBLIC CUDA::cudart)
endif ()
