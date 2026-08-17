#pragma once

// Qwen3.5 Gated DeltaNet（GDN）线性注意力用到的 reference 算子集合。
//
// 这些算子是 v2 新增的（Qwen2.x 不需要），目前只有标量参考实现；
// 将来若做性能优化，再按 dispatch.h 的自注册模式接入变体。
//
// 数学口径的权威来源是 HF transformers 的 torch 兜底实现
// （torch_recurrent_gated_delta_rule / torch_causal_conv1d_update /
// Qwen3_5RMSNormGated），本头文件各函数的注释给出对应公式。

namespace tinyqwen {
    // L2 归一化（就地）：x = x / sqrt(sum(x^2) + eps)。
    // GDN 的 q/k 在进入递归前做 l2norm（use_qk_l2norm_in_kernel=True），
    // eps = 1e-6，与 FLA 库的实现对齐。
    void l2norm_inplace_ref(float *x, int n, float eps);

    // causal depthwise conv1d 的单步（decode）更新：
    //   - conv_state: [dim][kernel_size - 1]，保存每个通道最近 kernel-1 个
    //     输入（时间顺序，旧 -> 新）；
    //   - weight: [dim][kernel_size]（HF conv1d.weight.squeeze(1) 后的形状）；
    //   - 输出 out[c] = silu( sum_i w[c][i] * [state_c..., x[c]][i] )，
    //     同时把 x[c] 推入 state（最旧的一个被挤出去）。
    // 与 HF torch_causal_conv1d_update 逐位等价（conv 后接 silu）。
    void causal_conv1d_update_ref(const float *x, float *conv_state, const float *weight,
                                  float *out, int dim, int kernel_size);

    // Gated delta rule 的单个 head、单个 token 的递归步（就地更新状态 S）。
    // HF torch_recurrent_gated_delta_rule 的每步：
    //   S = exp(g) * S                        （按 head 的衰减门控）
    //   kv_mem = S^T @ k                      （用旧状态读 key）
    //   delta  = beta * (v - kv_mem)          （delta rule 的"纠错量"）
    //   S = S + outer(k, delta)
    //   o = S^T @ q
    // 参数：
    //   S    : [qk_dim, v_dim]，该 head 的状态矩阵（行主序），就地更新；
    //   q, k : 已经 l2norm 过、q 已乘 scale = 1/sqrt(qk_dim)（调用方负责）；
    //   g    : 衰减对数（负数），kernel 内部取 exp；
    //   beta : 写入门（调用方已 sigmoid）。
    void gdn_step_ref(float *S, const float *q, const float *k, const float *v, float g,
                      float beta, float *o, int qk_dim, int v_dim);

    // 带门控的 RMSNorm（Qwen3_5RMSNormGated）：
    //   y = (x / sqrt(mean(x^2) + eps) * weight) * silu(gate)
    // GDN 输出在 out_proj 之前按 v head 维度（v_head_dim）做这个归一化。
    void rmsnorm_gated_ref(const float *x, const float *gate, const float *weight, float *y,
                           int n, float eps);

    // 部分旋转位置编码（partial RoPE，rotate-half 约定）：
    // 只旋转每个 head 的前 rotary_dim 个分量，其余原样保留。
    // inv_freq[i] = theta ^ (-2i / rotary_dim)，与 HF
    // Qwen3_5TextRotaryEmbedding 一致（dim = head_dim * partial_rotary_factor）。
    // 纯文本输入下 mRoPE 的三轴位置相同，交织退化为普通 RoPE。
    void partial_rope_ref(float *q, float *k, int n_heads, int n_kv_heads, int head_dim,
                          int rotary_dim, int pos, float theta);
} // namespace tinyqwen
