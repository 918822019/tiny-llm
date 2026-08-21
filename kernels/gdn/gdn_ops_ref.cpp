// ============================================================================
// gdn_ops_ref.cpp — GDN（Gated DeltaNet）算子的标量参考实现
// ============================================================================
//
// 本文件提供 GDN 线性注意力层 4 个核心算子 + partial RoPE 的纯标量、fp32、
// 正确性优先的参考实现。这些函数是数值对齐的"金标准"——所有 NEON/优化变体
// 的输出都必须与这里的结果在容差范围内一致。
//
// 包含的算子：
//   1. l2norm_inplace_ref      — L2 归一化（就地），用于 q/k 向量归一化
//   2. causal_conv1d_update_ref — 因果深度卷积状态更新，捕获局部时序依赖
//   3. gdn_step_ref            — Gated Delta Rule 单步递归，线性注意力的核心
//   4. rmsnorm_gated_ref       — 门控 RMSNorm，GDN 输出的归一化+门控
//   5. partial_rope_ref        — Partial RoPE 旋转位置编码
//
// 设计原则：
//   - 使用 double 累加确保数值精度（NEON 变体用 float 累加，会有微小差异）
//   - 不做任何平台特定优化，代码可读性优先
//   - 永不修改此文件的逻辑——它是正确性的锚点
//
// 数学口径与 HuggingFace torch 兜底实现对齐（见 kernels/gdn_ops.h）。
// ============================================================================

#include "gdn_ops.h"   // GDN 算子声明头文件

#include <cmath>       // std::exp, std::sqrt, std::pow, std::cos, std::sin
#include <cstdio>      // std::fprintf（错误报告）
#include <cstdlib>     // std::abort（致命错误时终止）

namespace tinyqwen {
    namespace {  // 匿名命名空间：内部辅助函数不导出符号表
        // ================================================================
        // silu_scalar — 标量 SiLU（Sigmoid Linear Unit）激活函数
        // ================================================================
        // 公式：silu(x) = x / (1 + exp(-x)) = x * sigmoid(x)
        // 用于 conv1d 输出激活和门控 RMSNorm 中的 gate 激活。
        inline float silu_scalar(float x) { return x / (1.0f + std::exp(-x)); }
    } // namespace

    // ====================================================================
    // l2norm_inplace_ref — L2 归一化（就地）
    // ====================================================================
    // 功能：将向量 x 除以其 L2 范数，使结果向量的 L2 范数为 1。
    // 公式：x[i] = x[i] / sqrt(Σ x[j]² + eps)
    // 参数：
    //   x   — 输入/输出向量 [n]，就地修改
    //   n   — 向量维度（Qwen3.5 中 = 128，即 qk_head_dim）
    //   eps — 防止除零的小常数（通常 1e-6）
    // 说明：
    //   使用 double 累加平方和以获得更高精度。NEON 变体用 float 累加，
    //   128 维下相对误差 ~1e-7，在容差范围内。
    void l2norm_inplace_ref(float *x, int n, float eps) {
        double sumsq = 0.0;  // double 累加器，避免 float 大数吞小数
        for (int i = 0; i < n; ++i) {
            const double xi = static_cast<double>(x[i]);  // 提升到 double 精度
            sumsq += xi * xi;                              // 累加平方
        }
        // 计算逆范数：1/sqrt(sumsq + eps)
        const float inv_norm = 1.0f / std::sqrt(static_cast<float>(sumsq) + eps);
        // 逐元素乘以逆范数完成归一化
        for (int i = 0; i < n; ++i) x[i] *= inv_norm;
    }

    // ====================================================================
    // causal_conv1d_update_ref — 因果深度卷积状态更新（标量参考）
    // ====================================================================
    // 功能：对每个通道独立执行一步因果卷积，并更新历史状态。
    // 算法（每通道 c）：
    //   acc = Σ_{i=0}^{state_len-1} weight[c][i] * state[c][i]
    //         + weight[c][state_len] * x[c]
    //   out[c] = SiLU(acc)
    //   state[c] 左移一格，推入原始输入 x[c]
    // 参数：
    //   x           — 当前时间步输入 [dim]
    //   conv_state  — 卷积状态 [dim][kernel_size-1]，就地更新
    //   weight      — 卷积权重 [dim][kernel_size]
    //   out         — 输出 [dim]（可与 x 同址）
    //   dim         — 通道数
    //   kernel_size — 卷积核大小（≥2）
    void causal_conv1d_update_ref(const float *x, float *conv_state, const float *weight,
                                  float *out, int dim, int kernel_size) {
        // 参数合法性检查：kernel_size 必须 ≥ 2（至少 1 个历史 + 1 个当前）
        if (kernel_size < 2) {
            std::fprintf(stderr, "causal_conv1d_update_ref: kernel_size %d < 2\n", kernel_size);
            std::abort();
        }
        const int state_len = kernel_size - 1;  // 历史状态长度 = kernel_size - 1
        // 逐通道处理
        for (int c = 0; c < dim; ++c) {
            // 该通道的 state 起始地址（通道主序布局）
            float *st = conv_state + static_cast<size_t>(c) * state_len;
            // 该通道的权重起始地址
            const float *w = weight + static_cast<size_t>(c) * kernel_size;
            // 注意 x 与 out 可能同址（就地）：先把当前输入存下来，避免下面
            // out[c]=silu(...) 覆盖 x[c] 后，再往 state 里推进的是被覆盖的值。
            const float xc = x[c];
            // 卷积：state 里是最近 state_len 个输入（旧->新），拼上当前 xc。
            float acc = 0.0f;
            for (int i = 0; i < state_len; ++i) acc += w[i] * st[i];  // 历史部分
            acc += w[state_len] * xc;                                   // 当前输入部分
            out[c] = silu_scalar(acc);  // SiLU 激活
            // 状态左移一格，把原始输入 xc 推进最新槽位（存 silu 前的值，与 HF 一致）。
            for (int i = 0; i < state_len - 1; ++i) st[i] = st[i + 1];  // 左移
            st[state_len - 1] = xc;  // 最新槽位 ← 原始输入
        }
    }

    // ====================================================================
    // gdn_step_ref — Gated Delta Rule 单步递归（标量参考）
    // ====================================================================
    // 功能：执行一次 GDN 状态更新，这是线性注意力的核心递归步骤。
    // 算法（单 head、单 token）：
    //   Step 1: S *= exp(g)                — 衰减门控（遗忘旧信息）
    //   Step 2: kv_mem = S^T @ k           — 用衰减后的状态读出旧 key-value 记忆
    //   Step 3: delta = beta * (v - kv_mem) — delta rule 纠错量
    //   Step 4: S += outer(k, delta)        — 将纠错量写入状态矩阵
    //   Step 5: o = S^T @ q                 — 从更新后的状态读出输出
    // 参数：
    //   S       — 状态矩阵 [qk_dim × v_dim]，就地更新（每层的持久记忆）
    //   q       — query 向量 [qk_dim]
    //   k       — key 向量 [qk_dim]
    //   v       — value 向量 [v_dim]
    //   g       — 门控标量（log-space），decay = exp(g)
    //   beta    — delta rule 的学习率/缩放因子
    //   o       — 输出向量 [v_dim]
    //   qk_dim  — query/key 维度（Qwen3.5 = 128）
    //   v_dim   — value 维度（Qwen3.5 = 128）
    // 内存布局：
    //   S 按行主序存储：S[j][d] = S[j * v_dim + d]
    //   总大小 = 128 × 128 × 4B = 64KB（恰好可放入 Apple M 系列的 128KB L1D）
    // 注意：
    //   Step 2 的 kv_mem 必须用衰减后的 S 计算（HF 先乘 decay 再读）。
    //   Step 5 使用 double 累加以确保精度（NEON 变体用 float，误差 ~1e-4）。
    void gdn_step_ref(float *S, const float *q, const float *k, const float *v, float g,
                      float beta, float *o, int qk_dim, int v_dim) {
        // v_dim 上限检查：栈上临时数组的大小限制
        constexpr int kMaxV = 512;  // 最大支持的 v_dim
        if (v_dim <= 0 || v_dim > kMaxV) {
            std::fprintf(stderr, "gdn_step_ref: unsupported v_dim %d\n", v_dim);
            std::abort();
        }

        const float decay = std::exp(g);  // 将 log-space 门控转为乘法衰减因子

        // ---- 第 1 遍：S *= decay，同时求 kv_mem = S(衰减后)^T @ k ----
        // kv_mem[d] = Σ_j S[j][d] * k[j]，其中 S 已经是衰减后的值
        // 注意：kv_mem 必须在所有 j 行都衰减完之后才能用于 delta 计算
        float kv_mem[kMaxV];  // 栈上分配，避免每 token 堆分配
        for (int d = 0; d < v_dim; ++d) kv_mem[d] = 0.0f;  // 初始化 kv_mem 为零
        for (int j = 0; j < qk_dim; ++j) {
            float *Srow = S + static_cast<size_t>(j) * v_dim;  // S 的第 j 行起始地址
            const float kj = k[j];                              // 当前行的 key 值
            for (int d = 0; d < v_dim; ++d) {
                Srow[d] *= decay;           // Step 1: 衰减 S[j][d]
                kv_mem[d] += Srow[d] * kj;  // Step 2: 累加 S[j][d] * k[j]
            }
        }

        // ---- 第 2 遍：delta = beta * (v - kv_mem)；S += outer(k, delta) ----
        // delta[d] 表示 value 空间中需要纠正的量
        float delta[kMaxV];  // 栈上分配
        for (int d = 0; d < v_dim; ++d) delta[d] = beta * (v[d] - kv_mem[d]);  // delta rule
        // outer product 更新：S[j][d] += k[j] * delta[d]
        for (int j = 0; j < qk_dim; ++j) {
            float *Srow = S + static_cast<size_t>(j) * v_dim;  // S 的第 j 行
            const float kj = k[j];                              // 当前行的 key 值
            for (int d = 0; d < v_dim; ++d) Srow[d] += kj * delta[d];  // 外积累加
        }

        // ---- 第 3 遍：o = S^T @ q（double 累加）----
        // o[d] = Σ_j S[j][d] * q[j]
        // 使用 double 累加减少浮点舍入误差（128 维点积的累积误差可达 ~1e-4）
        for (int d = 0; d < v_dim; ++d) {
            double acc = 0.0;  // double 累加器
            for (int j = 0; j < qk_dim; ++j) {
                acc += static_cast<double>(S[static_cast<size_t>(j) * v_dim + d]) *
                       static_cast<double>(q[j]);
            }
            o[d] = static_cast<float>(acc);  // 截断回 float
        }
    }

    // ====================================================================
    // rmsnorm_gated_ref — 门控 RMSNorm（标量参考）
    // ====================================================================
    // 功能：对输入向量做 RMSNorm 归一化后乘以可学习权重，再乘以 SiLU(gate)。
    // 公式：y[i] = (x[i] / sqrt(mean(x²) + eps) * weight[i]) * silu(gate[i])
    // 参数：
    //   x      — 输入向量 [n]
    //   gate   — 门控向量 [n]（经 SiLU 激活后作为逐元素乘数）
    //   weight — 可学习缩放权重 [n]
    //   y      — 输出向量 [n]
    //   n      — 向量维度（Qwen3.5 = 128，即 v_head_dim）
    //   eps    — 防止除零的小常数
    // 说明：
    //   门控机制让模型可以动态调节每个维度的输出强度。
    //   当 gate[i] → -∞ 时 silu(gate[i]) → 0，相当于关闭该维度。
    void rmsnorm_gated_ref(const float *x, const float *gate, const float *weight, float *y,
                           int n, float eps) {
        // Pass 1：计算平方和（double 精度累加）
        double sumsq = 0.0;
        for (int i = 0; i < n; ++i) {
            const double xi = static_cast<double>(x[i]);
            sumsq += xi * xi;
        }
        // scale = 1 / sqrt(mean(x²) + eps) = 1 / sqrt(sumsq/n + eps)
        const float scale = 1.0f / std::sqrt(static_cast<float>(sumsq / n) + eps);
        // Pass 2：归一化 × 权重 × 门控
        for (int i = 0; i < n; ++i) {
            const float normed = x[i] * scale * weight[i];  // RMSNorm + 可学习权重
            y[i] = normed * silu_scalar(gate[i]);            // × SiLU(gate)
        }
    }

    // ====================================================================
    // partial_rope_ref — Partial RoPE 旋转位置编码（标量参考）
    // ====================================================================
    // 功能：对 q 和 k 向量的前 rotary_dim 个分量施加旋转位置编码。
    // 与标准 RoPE 的区别：频率按 rotary_dim（而非 head_dim）归一化，
    // 且只旋转前 rotary_dim 维，[rotary_dim, head_dim) 区间保持不变。
    // 这允许 Qwen3.5 在不同 head 上使用不同的旋转维度。
    // 参数：
    //   q          — query 向量 [n_heads × head_dim]，就地修改
    //   k          — key 向量 [n_kv_heads × head_dim]，就地修改
    //   n_heads    — query 头数量
    //   n_kv_heads — key/value 头数量（GQA 中可小于 n_heads）
    //   head_dim   — 每个头的维度
    //   rotary_dim — 参与旋转的维度数（必须为偶数，≤ head_dim）
    //   pos        — 当前 token 的位置索引
    //   theta      — RoPE 基频（通常 10000.0）
    void partial_rope_ref(float *q, float *k, int n_heads, int n_kv_heads, int head_dim,
                          int rotary_dim, int pos, float theta) {
        // 参数合法性检查
        if (rotary_dim <= 0 || rotary_dim > head_dim || rotary_dim % 2 != 0) {
            std::fprintf(stderr, "partial_rope_ref: bad rotary_dim %d (head_dim %d)\n",
                         rotary_dim, head_dim);
            std::abort();
        }
        const int half = rotary_dim / 2;  // rotate-half 配对：前半与后半对应旋转
        constexpr int kMaxHalf = 256;      // 栈上 cos/sin 表的最大大小
        if (half > kMaxHalf) {
            std::fprintf(stderr, "partial_rope_ref: rotary_dim %d too large\n", rotary_dim);
            std::abort();
        }

        // 预计算 cos/sin 表：频率按 rotary_dim 归一化（partial RoPE 的关键）
        // inv_freq[i] = theta^(-2i/rotary_dim)，angle = pos * inv_freq
        float cs[kMaxHalf];  // cos 表
        float sn[kMaxHalf];  // sin 表
        for (int i = 0; i < half; ++i) {
            const float exponent = -static_cast<float>(2 * i) / static_cast<float>(rotary_dim);
            const float inv_freq = std::pow(theta, exponent);      // θ^(-2i/D)
            const float angle = static_cast<float>(pos) * inv_freq; // pos × 频率
            cs[i] = std::cos(angle);  // cos(pos × θ^(-2i/D))
            sn[i] = std::sin(angle);  // sin(pos × θ^(-2i/D))
        }

        // apply lambda：对单个头的向量施加旋转
        // rotate-half 配对：x[i] 与 x[i+half] 组成一对进行二维旋转
        // 只旋转前 rotary_dim 个分量，[rotary_dim, head_dim) 原样保留
        const auto apply = [&](float *x) {
            for (int i = 0; i < half; ++i) {
                const float x0 = x[i];          // 前半部分
                const float x1 = x[i + half];   // 后半部分
                const float c = cs[i];           // cos 系数
                const float s = sn[i];           // sin 系数
                x[i] = x0 * c - x1 * s;         // 旋转后的前半
                x[i + half] = x1 * c + x0 * s;  // 旋转后的后半
            }
        };
        // 对所有 query 头施加 RoPE
        for (int h = 0; h < n_heads; ++h) {
            apply(q + static_cast<size_t>(h) * head_dim);
        }
        // 对所有 key 头施加 RoPE（GQA 中 n_kv_heads ≤ n_heads）
        for (int h = 0; h < n_kv_heads; ++h) {
            apply(k + static_cast<size_t>(h) * head_dim);
        }
    }
} // namespace tinyqwen
