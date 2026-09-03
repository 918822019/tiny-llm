#pragma once

// ============================================================================
// 文件: metal_prefill.h
// 作用: Metal GPU prefill 引擎的 C++ 接口 —— 让 main.cpp 无需接触 ObjC 头文件
//
// 为什么要有这个文件?
//   实现（metal_prefill.mm）必须用 ObjC++ 才能调 Metal / MetalPerformanceShaders，
//   而 main.cpp 是纯 C++。本头文件只暴露纯 C++17 的 API：不透明句柄 +
//   free function，因此 main.cpp 可以正常 #include 它。
//
// 与现有两条 GPU 路径的关系（见 runtime/README.md「两条 GPU 路径的区别」）:
//   --backend cuda ：IBackend 逐算子 CUDA，供 A/B，非性能路径
//   --engine  cuda ：整段 **decode** forward 常驻显存（gpu_engine.cu）
//   --engine  metal：整段 **prefill** forward 跑在 Apple GPU 上（本文件）
//
//   注意 cuda 那条是 decode-only（main.cpp 注释：「GPU-resident engine 没有
//   批量 prefill 入口」）。本引擎正好补上 prefill 这一半：整批 prompt 一次
//   GEMM 前向算完，产出 [seq, vocab] logits，并把 post-RoPE 的 K/V 写进
//   runtime 的 KvCache，于是后续 decode 仍可走 CPU 路径（位置从 n 接续）。
//
// 适用范围（metal_prefill_create 会 fail fast 校验）:
//   - 全部为 full attention 层（Qwen3.5 的 GDN 混合架构不支持）
//   - 权重 dtype 为 f16 或 f32（i4 / vq2 不支持）
//   - 全 RoPE（partial_rotary_factor == 1.0）
//   - 仅 Apple 平台（非 Apple 构建 metal_prefill_available() 返回 false）
// ============================================================================

#include <string>

#include "kv_cache.h"
#include "model_loader.h"

namespace tinyqwen {
    // -------------------------------------------------------------------------
    // MetalPrefillEngine: 不透明句柄
    //
    // 说明:
    //   真实定义在 metal_prefill.mm 里（持有 Metal device / command queue /
    //   权重 GPU buffer）。这里只做前向声明，让 C++ 侧拿到的是纯指针。
    // -------------------------------------------------------------------------
    struct MetalPrefillEngine;

    // ---------------------------------------------------------------------
    // metal_prefill_available: 当前构建是否带 Metal prefill 引擎
    //
    // 返回值:
    //   Apple 构建返回 true；其他平台（Linux / Android / 无 CUDA 的通用构建）
    //   返回 false —— 此时 main.cpp 应当拒绝 --engine metal 而不是崩溃。
    // ---------------------------------------------------------------------
    bool metal_prefill_available();

    // ---------------------------------------------------------------------
    // metal_prefill_create: 建引擎并把全部权重搬上 GPU
    //
    // 参数:
    //   file:        已加载并校验过的 ModelFile（引擎只借它的内存视图，
    //                file 必须比引擎活得久）
    //   max_seq_len: 激活 buffer 的容量上限（与 --max-seq-len 一致）
    //   err:         输出参数；失败时写入原因
    //   out:         输出参数；成功时写入新引擎指针
    //
    // 返回值:
    //   成功 true / 失败 false
    //
    // 校验项（任一不满足即 fail fast，避免算出错数还查不出来）:
    //   - Metal device 是否可用
    //   - 模型是否全 full attention（full_attention_interval <= 1）
    //   - 权重 dtype 是否 f16/f32
    //   - partial_rotary_factor 是否 1.0
    //   - 必需的 tensor 是否都在文件里（缺 tensor 直接报错而不是静默算错）
    // ---------------------------------------------------------------------
    bool metal_prefill_create(const ModelFile *file, int max_seq_len, std::string *err,
                              MetalPrefillEngine **out);

    // ---------------------------------------------------------------------
    // metal_prefill_destroy: 释放引擎（device / queue / GPU buffer）
    //
    // 参数:
    //   engine: 待释放的引擎；传 nullptr 是安全的（no-op）
    // ---------------------------------------------------------------------
    void metal_prefill_destroy(MetalPrefillEngine *engine);

    // ---------------------------------------------------------------------
    // metal_prefill_run: 对一整批 prompt token 做 GPU prefill
    //
    // 参数:
    //   engine:     已创建的引擎
    //   tokens:     prompt token ids
    //   n:          token 个数（必须 >= 1 且 <= create 时的 max_seq_len）
    //   logits_out: 输出 buffer，容量必须 >= n * vocab_size 个 float；
    //               写入 [n, vocab] 行主序 fp32 logits（每个位置一行）。
    //               传 nullptr 表示只要 argmax、不要 logits。
    //   kv:         可选。非 nullptr 时把 post-RoPE 的 K/V 写进该 cache 的
    //               位置 [0, n)，并 advance(n)，使后续 CPU decode 能从位置 n
    //               正确接续。
    //   err:        输出参数；失败时写入原因
    //
    // 返回值:
    //   最后一个位置（第 n-1 行）logits 的 argmax，也就是下一个 token id；
    //   失败返回 -1
    //
    // 语义对齐:
    //   与 QwenModel::forward_prefill 一致 —— 都是"整批前向 + 写 KV +
    //   advance"。区别只是算子跑在 GPU 上（GEMM 走 MPS，其余走 Metal compute）。
    //
    // KV 接续（投机解码的前提）:
    //   引擎自己持有一份 GPU 侧 KV cache，每次调用把 n 个 token 追加在已有上下文
    //   之后（落在 [kv_len, kv_len+n)）。所以可以反复调用：首次 prefill 填 prompt，
    //   之后每次传 K 个草稿 token 做 verify pass —— attention 会 attend 到全部
    //   已有上下文（Q_len = n，KV_len = kv_len + n，非对称）。
    //   要重新开始一段序列，先调 metal_prefill_reset_kv。
    //   若同时传了 CPU 的 KvCache，它必须与引擎长度一致，否则报错。
    // ---------------------------------------------------------------------
    int metal_prefill_run(MetalPrefillEngine *engine, const int *tokens, int n,
                          float *logits_out, KvCache *kv, std::string *err);

    // ---------------------------------------------------------------------
    // metal_prefill_reset_kv: 清空引擎的 GPU KV cache，回到位置 0
    //
    // 参数:
    //   engine: 引擎
    //
    // 说明:
    //   开始一段新序列前必须调用。不清空的话下一次 run 会把新 token 追加到旧
    //   上下文后面，attention 会读到上一段序列的历史。
    //   注意：本函数只清引擎的 GPU cache，调用方若还用了 CPU 的 KvCache，
    //   需要自己另行 reset。
    // ---------------------------------------------------------------------
    void metal_prefill_reset_kv(MetalPrefillEngine *engine);

    // ---------------------------------------------------------------------
    // metal_prefill_kv_len: 引擎当前已缓存的位置数
    // ---------------------------------------------------------------------
    int metal_prefill_kv_len(const MetalPrefillEngine *engine);

    // ---------------------------------------------------------------------
    // metal_prefill_device_name: 返回 Metal 设备名（日志用）
    //
    // 参数:
    //   engine: 引擎
    //
    // 返回值:
    //   设备名字符串；引擎为 nullptr 时返回空串
    // ---------------------------------------------------------------------
    std::string metal_prefill_device_name(const MetalPrefillEngine *engine);
} // namespace tinyqwen
