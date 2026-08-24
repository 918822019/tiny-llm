#pragma once

// ============================================================================
// 文件: ref_ops.h
// 作用: 所有"reference kernel"（参考实现算子）的函数签名集合
//
// 什么是 reference kernel?
//   就是每个数学运算的"最简单、最直白"的实现: 不用 SIMD、不优化、只求正确。
//   它有两个用途:
//     1. 作为数值基准 —— 后续写高性能 kernel（INT4/NEON 等）时，结果必须和它对齐；
//     2. v1 阶段直接拿它跑通整条推理链路。
//
// 本目录所有 kernel 的共同约定:
//   - 正确性和可读性优先，不用 SIMD、不开多线程；
//   - 输入/输出指针由调用方提供，kernel 不为其分配内存（避免隐藏的 new/delete）；
//   - shape 全部以显式参数传入，kernel 无隐藏状态；
//   - 每个 kernel 在 tests/ 都有小 shape 单元测试。
// ============================================================================

#include <cmath>
#include <cstdint>
#include <cstring>

namespace tinyqwen {
    // =========================================================================
    // log_softmax 在 target 处的取值（数值稳定：先减最大值）
    // = logits[target] - logsumexp(logits)。PPL 评测（forward_ppl）用它对
    // 每个位置的 logits 按目标 token 计负对数似然。
    // =========================================================================
    inline double log_softmax_at(const float *logits, int vocab, int target) {
        float mx = logits[0];
        for (int v = 1; v < vocab; ++v) if (logits[v] > mx) mx = logits[v];
        double sum = 0.0;
        for (int v = 0; v < vocab; ++v) sum += std::exp(static_cast<double>(logits[v]) - mx);
        return (static_cast<double>(logits[target]) - mx) - std::log(sum);
    }

    // =========================================================================
    // IEEE binary16（half）<-> float32 的可移植位操作转换
    //
    // 不依赖编译器扩展（__fp16 不是所有平台都有），runtime 与 kernels 共用:
    //   - f16_ref kernel 用它把权重转回 float 再累加；
    //   - runtime 的 embed 查表 / 测试的数据构造也用它。
    //
    // float_to_half 采用 round-to-nearest-even（与 numpy astype("float16")
    // 一致），保证导出端和 C++ 端量化结果逐位相同。
    // =========================================================================

    // 将 IEEE 754 binary16 位模式转换为 float32
    // 处理三种情况: 规格化数、非规格化数、特殊值（inf/NaN）
    inline float half_to_float(uint16_t h) {
        const uint32_t sign = static_cast<uint32_t>(h & 0x8000) << 16; // 符号位移到 float32 位置
        uint32_t exp = (h >> 10) & 0x1F;   // 5 位指数
        uint32_t mant = h & 0x3FF;          // 10 位尾数
        uint32_t bits;
        if (exp == 0) {
            if (mant == 0) {
                bits = sign; // ±0: 直接返回带符号的零
            } else {
                // 非规格化数: 规格化成 float 的 1.m 形式
                exp = 127 - 15 + 1; // +1 是因为隐含位从 0 变 1
                while ((mant & 0x400) == 0) { mant <<= 1; --exp; } // 逐位左移直到找到第一个 1
                mant &= 0x3FF; // 去掉隐含的 1
                bits = sign | (exp << 23) | (mant << 13);
            }
        } else if (exp == 31) {
            bits = sign | 0x7F800000u | (mant << 13); // inf / NaN: 保留尾数位
        } else {
            bits = sign | ((exp - 15 + 127) << 23) | (mant << 13); // 规格化数: 调整指数偏移
        }
        float f;
        std::memcpy(&f, &bits, sizeof(f)); // 位模式直接拷贝，避免 UB
        return f;
    }

    // 将 float32 转换为 IEEE 754 binary16 位模式
    // 采用 RNE 舍入: 处理溢出、规格化、非规格化、下溢四种情况
    inline uint16_t float_to_half(float f) {
        uint32_t x;
        std::memcpy(&x, &f, sizeof(x)); // 位模式读取
        const uint16_t sign = static_cast<uint16_t>((x >> 16) & 0x8000); // 提取符号位
        x &= 0x7FFFFFFFu; // 幅值部分单独处理，符号最后拼回
        if (x >= 0x47800000u) {            // |f| >= 65536: 溢出或 inf/nan
            if (x > 0x7F800000u) return static_cast<uint16_t>(sign | 0x7E00u); // NaN
            return static_cast<uint16_t>(sign | 0x7C00u);                       // ±inf / 溢出
        }
        if (x >= 0x38800000u) {            // f16 规格化范围: |f| >= 2^-14
            const uint32_t exp16 = ((x >> 23) & 0xFF) - 127 + 15; // 计算 half 指数
            const uint32_t r = (exp16 << 23) | (x & 0x7FFFFFu);   // 构造含尾数的值
            // RNE: 丢弃低 13 位，加"半格 + 保留位 LSB"进位
            const uint32_t rounded = r + 0x0FFFu + ((r >> 13) & 1u);
            return static_cast<uint16_t>(sign | static_cast<uint16_t>(rounded >> 13));
        }
        if (x < 0x33800000u) return sign;  // |f| < 2^-24: 下溢成 ±0
        // f16 非规格化范围: 2^-24 <= |f| < 2^-14
        const int exp = static_cast<int>((x >> 23) & 0xFF) - 127; // fp32 无偏指数
        const uint32_t mant = (x & 0x7FFFFFu) | 0x800000u;        // 含隐含 1（24 位）
        const int shift = -1 - exp; // 需要右移的位数，∈ [14, 23]
        const uint32_t full = mant >> shift;        // 移位后的整数部分
        const uint32_t rem = mant & ((1u << shift) - 1u); // 余数部分
        const uint32_t half = 1u << (shift - 1);          // 半格值
        uint32_t bits10 = full;
        if (rem > half || (rem == half && (full & 1u))) ++bits10; // RNE: tie to even
        return static_cast<uint16_t>(sign | bits10);
    }

    // RMSNorm: y = x / sqrt(mean(x^2) + eps) * weight
    // x: 输入 [n], weight: 缩放权重 [n], y: 输出 [n], eps: 防除零常数
    void rmsnorm_ref(const float *x, const float *weight, float *y, int n, float eps);

    // matvec_f32_ref: y = W @ x
    // W 为行主序 [out_dim, in_dim]（HF 布局，不做转置）
    void matvec_f32_ref(const float *w, const float *x, float *y, int out_dim, int in_dim);

    // matvec_f16_ref: f16 权重版 matvec
    // 权重是 IEEE binary16（按 uint16_t 搬运），x/y 仍是 f32
    // weight-only 半精度: 只省权重的存储/搬运，计算精度不降
    // 实现 = 逐元素 half_to_float 后累加 —— f16 族的正确性基准
    void matvec_f16_ref(const uint16_t *w, const float *x, float *y, int out_dim, int in_dim);

    // 数值稳定 softmax: y[i] = exp(x[i] - max(x)) / sum_j exp(x[j] - max(x))
    // 减去 max(x) 防止 exp 溢出，数学上等价于直接 softmax
    void softmax_ref(const float *x, float *y, int n);

    // SiLU: y = x * sigmoid(x) = x / (1 + exp(-x))
    void silu_ref(const float *x, float *y, int n);

    // SwiGLU 融合: gate[i] = silu(gate[i]) * up[i]（就地修改 gate）
    // 等价于先 silu_ref(gate, gate, n) 再逐元素 gate[i] *= up[i]——逐位一致
    // 融合成一个 kernel 减少内存访问次数（FFN 热点优化）
    void swiglu_ref(float *gate, const float *up, int n);

    // 返回第一个最大值的下标（平局取靠前者）
    int argmax_ref(const float *logits, int n);

    // 旋转位置编码 RoPE（in-place）
    // q: [n_heads * head_dim], k: [n_kv_heads * head_dim]
    // 与 HF Qwen2 的 rotate-half 约定一致:
    //   out[i]        = x[i] * cos - x[i + half] * sin
    //   out[i + half] = x[i + half] * cos + x[i] * sin
    // inv_freq[i] = theta ^ (-2i / head_dim), angle = pos * inv_freq[i]
    void rope_ref(float *q, float *k, int n_heads, int n_kv_heads, int head_dim, int pos,
                  float theta);

    // decode 阶段 attention（单个 query 位置）对 KV cache 做 attention
    // q: [n_heads * head_dim], k_cache/v_cache: [n_kv_heads][max_seq_len][head_dim]
    // out: [n_heads * head_dim], 有效位置 [0, seq_len)
    // score = dot(q_h, k_{kv(h), t}) * scale; 采用 online softmax，无暂存内存
    // GQA 映射: query head h 对应的 kv head 为 h / (n_heads / n_kv_heads)
    void attention_decode_ref(const float *q, const float *k_cache, const float *v_cache,
                              int seq_len, int max_seq_len, int n_heads, int n_kv_heads,
                              int head_dim, float scale, float *out);

    // fp16-KV 融合 attention：k_cache/v_cache 为 fp16（uint16_t*），
    // 在寄存器内转 fp32 后按与 attention_decode_ref 完全相同的 online softmax
    // 计算。消灭"独立反量化遍历"，是 fp16 KV 的正确加速方式。
    // 仅 aarch64 提供 NEON 实现；其余平台无实现（调用方需保证不走到）。
    void attention_decode_f16kv_neon(const float *q, const uint16_t *k_cache,
                                     const uint16_t *v_cache, int seq_len, int max_seq_len,
                                     int n_heads, int n_kv_heads, int head_dim, float scale,
                                     float *out);
} // namespace tinyqwen