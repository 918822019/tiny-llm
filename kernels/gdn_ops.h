#pragma once

// ============================================================================
// 文件: gdn_ops.h
// 作用: Qwen3.5 Gated DeltaNet（GDN）线性注意力用到的 reference 算子集合
//
// 这些算子是 v2 新增的（Qwen2.x 不需要），目前只有标量参考实现；
// 将来若做性能优化，再按 dispatch.h 的自注册模式接入变体。
//
// 数学口径的权威来源:
//   HF transformers 的 torch 兜底实现:
//     - torch_recurrent_gated_delta_rule
//     - torch_causal_conv1d_update
//     - Qwen3_5RMSNormGated
//   本头文件各函数的注释给出对应公式。
//
// 与 Qwen2.x 算子的区别:
//   - Qwen2.x 用 full attention（KV cache 随序列长度线性增长）
//   - Qwen3.5 GDN 用线性注意力（递归状态 O(1) 大小，与序列长度无关）
// ===========================================================================

namespace tinyqwen {
    // -------------------------------------------------------------------------
    // l2norm_inplace_ref: L2 归一化（就地）
    //
    // 计算: x = x / sqrt(sum(x^2) + eps)
    //
    // 参数:
    //   x:   输入向量 [n]，就地修改
    //   n:   向量长度
    //   eps: 防止除零的小常数（1e-6）
    //
    // 说明:
    //   GDN 的 q/k 在进入递归前做 l2norm（use_qk_l2norm_in_kernel=True），
    //   eps = 1e-6，与 FLA 库的实现对齐。
    // -------------------------------------------------------------------------
    void l2norm_inplace_ref(float *x, int n, float eps);

    // -------------------------------------------------------------------------
    // causal_conv1d_update_ref: causal depthwise conv1d 的单步（decode）更新
    //
    // 参数:
    //   x:           当前输入 [dim]
    //   conv_state:  卷积状态 [dim][kernel_size - 1]，保存每个通道最近 kernel-1
    //                个输入（时间顺序，旧 -> 新），就地更新
    //   weight:      卷积权重 [dim][kernel_size]（HF conv1d.weight.squeeze(1) 后）
    //   out:         输出 [dim]
    //   dim:         通道数
    //   kernel_size: 卷积核大小
    //
    // 计算:
    //   out[c] = silu( sum_i w[c][i] * [state_c..., x[c]][i] )
    //   同时把 x[c] 推入 state（最旧的一个被挤出，实现滑动窗口）
    //
    // 说明:
    //   与 HF torch_causal_conv1d_update 逐位等价（conv 后接 silu）。
    //   kernel_size 典型值为 4（Qwen3.5），所以 state 保存 3 个历史值。
    // -------------------------------------------------------------------------
    void causal_conv1d_update_ref(const float *x, float *conv_state, const float *weight,
                                  float *out, int dim, int kernel_size);

    // -------------------------------------------------------------------------
    // gdn_step_ref: Gated delta rule 的单个 head、单个 token 的递归步
    //
    // 就地更新状态矩阵 S，并计算输出 o。
    //
    // 参数:
    //   S:      状态矩阵 [qk_dim, v_dim]，行主序，就地更新
    //   q:      query 向量 [qk_dim]（已 l2norm，已乘 scale = 1/sqrt(qk_dim)）
    //   k:      key 向量 [qk_dim]（已 l2norm）
    //   v:      value 向量 [v_dim]
    //   g:      衰减对数（负数），kernel 内部取 exp(g)
    //   beta:   写入门（调用方已 sigmoid 处理）
    //   o:      输出向量 [qk_dim]
    //   qk_dim: Q/K 头的维度
    //   v_dim:  V 头的维度
    //
    // 计算（HF torch_recurrent_gated_delta_rule 的每步）:
    //   S = exp(g) * S                        // 按 head 的衰减门控，遗忘旧信息
    //   kv_mem = S^T @ k                      // 用旧状态读 key，提取记忆
    //   delta  = beta * (v - kv_mem)          // delta rule 的"纠错量"
    //   S = S + outer(k, delta)               // 用新信息更新状态
    //   o = S^T @ q                           // 用更新后的状态读 query，得到输出
    //
    // 说明:
    //   这是 GDN 的核心计算，将 O(n) 的 attention 压缩为 O(1) 的递归状态更新。
    //   每个 head 独立维护一个 [qk_dim, v_dim] 的状态矩阵。
    // -------------------------------------------------------------------------
    void gdn_step_ref(float *S, const float *q, const float *k, const float *v, float g,
                      float beta, float *o, int qk_dim, int v_dim);

    // -------------------------------------------------------------------------
    // rmsnorm_gated_ref: 带门控的 RMSNorm（Qwen3_5RMSNormGated）
    //
    // 计算: y = (x / sqrt(mean(x^2) + eps) * weight) * silu(gate)
    //
    // 参数:
    //   x:      输入向量 [n]
    //   gate:   门控向量 [n]（用于 silu 激活）
    //   weight: 可学习的缩放权重 [n]（gamma 参数）
    //   y:      输出向量 [n]
    //   n:      向量长度
    //   eps:    防止除零的小常数
    //
    // 说明:
    //   GDN 输出在 out_proj 之前按 v head 维度（v_head_dim）做这个归一化。
    //   与普通 RMSNorm 的区别: 多了一个 silu(gate) 的逐元素乘法。
    // -------------------------------------------------------------------------
    void rmsnorm_gated_ref(const float *x, const float *gate, const float *weight, float *y,
                           int n, float eps);

    // -------------------------------------------------------------------------
    // partial_rope_ref: 部分旋转位置编码（partial RoPE，rotate-half 约定）
    //
    // 参数:
    //   q:           query 向量 [n_heads * head_dim]，就地修改
    //   k:           key 向量 [n_kv_heads * head_dim]，就地修改
    //   n_heads:     query 注意力头数
    //   n_kv_heads:  key/value 注意力头数
    //   head_dim:    每个头的维度
    //   rotary_dim:  需要旋转的维度（= head_dim * partial_rotary_factor）
    //   pos:         当前 token 在序列中的位置
    //   theta:       RoPE 的底数
    //
    // 说明:
    //   只旋转每个 head 的前 rotary_dim 个分量，其余原样保留。
    //   inv_freq[i] = theta ^ (-2i / rotary_dim)（注意分母是 rotary_dim 而非 head_dim）。
    //   与 HF Qwen3_5TextRotaryEmbedding 一致。
    //   纯文本输入下 mRoPE 的三轴位置相同，交织退化为普通 RoPE。
    // -------------------------------------------------------------------------
    void partial_rope_ref(float *q, float *k, int n_heads, int n_kv_heads, int head_dim,
                          int rotary_dim, int pos, float theta);
} // namespace tinyqwen