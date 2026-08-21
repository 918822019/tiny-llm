# Qwen forward 数学定义与 shape 约定

对应实现：`runtime/qwen_forward_token.cpp`（decode，token-by-token）与
`runtime/qwen_forward_prefill.cpp`（批量 prefill）；激活全程 fp32，权重可为
f32/f16/i4（经 IBackend 分发，见 `architecture.md`）。
符号：`H`=hidden_size(896)，`I`=intermediate_size(4864)，`V`=vocab_size(151936)，
`nh`=n_heads(14)，`nkv`=n_kv_heads(2)，`hd`=head_dim(64)，`qd=nh*hd`，`kvd=nkv*hd`。

## 单 token 前向（position = p）

```text
x = embed_tokens[token_id]                       # [H]

for i in 0..n_layers-1:
    n1  = RMSNorm(x, input_layernorm[i], eps)    # [H]
    q   = q_proj[i] @ n1 + q_bias[i]             # [qd]   W:[qd,H]
    k   = k_proj[i] @ n1 + k_bias[i]             # [kvd]  W:[kvd,H]
    v   = v_proj[i] @ n1 + v_bias[i]             # [kvd]
    q,k = RoPE(q, k, pos=p, theta=1e6)           # rotate-half, 见下；bias 在 RoPE 前
    KV[i][p] = (k, v)                            # append 到 cache
    a   = Attention(q, K[i][0..p], V[i][0..p])   # [qd], GQA, scale=1/sqrt(hd)
    x   = x + o_proj[i] @ a                      # W:[H,qd]
    n2  = RMSNorm(x, post_attention_layernorm[i], eps)
    g   = gate_proj[i] @ n2                      # [I]
    u   = up_proj[i] @ n2                        # [I]
    f   = down_proj[i] @ (SiLU(g) * u)           # [H]
    x   = x + f

x     = RMSNorm(x, model.norm, eps)
logits = lm_head @ x                             # [V]（tied 时 = embed_tokens @ x）
next  = argmax(logits)                           # greedy
```

## RMSNorm

```text
RMSNorm(x, w) = x / sqrt(mean(x^2) + eps) * w,   eps = 1e-6
```

只有 weight，没有 bias。

## RoPE（与 HF Qwen2 rotate-half 一致）

```text
inv_freq[i] = theta ^ (-2i / hd),   i = 0..hd/2-1
对每对 (x0=x[i], x1=x[i+hd/2])，angle = p * inv_freq[i]：
    out[i]        = x0 * cos(angle) - x1 * sin(angle)
    out[i + hd/2] = x1 * cos(angle) + x0 * sin(angle)
```

作用于 q 的全部 `nh` 个 head 和 k 的全部 `nkv` 个 head；v 不做 RoPE。

## GQA attention（decode，单 query 位置）

```text
kv(h) = h / (nh / nkv)                    # 14 个 q head 共享 2 个 kv head
score(h, t) = dot(q_h, K[kv(h)][t]) / sqrt(hd),   t = 0..p
prob(h, :) = softmax(score(h, :))
out_h = sum_t prob(h, t) * V_cache[kv(h)][t]
```

decode 阶段 query 只有当前 token，天然不需要额外 causal mask
（cache 里只有 ≤ p 的位置）。

## SwiGLU FFN

```text
down_proj( SiLU(gate_proj(x)) * up_proj(x) )
```

注意先 gate 后 up、SiLU 只作用在 gate 分支、逐元素相乘后再 down。

## KV cache 约定

```text
K/V: [n_layers][n_kv_heads][max_seq_len][head_dim], fp32
```

attention 读取 `seq_len = p + 1`（含当前 token，先 append 再 attend）。

## 与 HF 实现的差异点（v1）

| 点         | HF         | tinyqwen v1                          |
|-----------|------------|--------------------------------------|
| prefill   | 一次并行       | token-by-token 串行（数值等价，causal 下结果一致） |
| attention | SDPA/eager | 手写 decode kernel（online softmax）     |
| 累加精度      | fp32       | matvec/rmsnorm 用 double 累加，输出 fp32   |
| tokenizer | 内置         | 外部 Python 提供 token ids               |

## Qwen3.5 混合架构（v2）

Qwen3.5 用 **Gated DeltaNet（linear attention）+ full attention** 按
`full_attention_interval`（3:1 → 4）交替。FFN/residual/norm 结构与上面一致，
差异只在 token mixer。对应实现仍在 `runtime/qwen_forward_token.cpp`。

### full attention 层（无 bias，有 QK-norm / partial RoPE / 输出门）

```text
n1   = RMSNorm(x, input_layernorm)                 # zero-centered，权重已折 +1
qg   = q_proj @ n1                                  # [2*qd]：每头前 hd 是 q、后 hd 是 gate
q, gate = deinterleave(qg)                          # q:[qd], gate:[qd]
k    = k_proj @ n1                                  # [kvd]
v    = v_proj @ n1                                  # [kvd]
q_h  = RMSNorm(q_h, q_norm)   (per head)            # zero-centered，已折 +1
k_h  = RMSNorm(k_h, k_norm)   (per head)
q,k  = PartialRoPE(q, k, pos, theta, rotary_dim)    # 只旋转每头前 rotary_dim 维
KV.append(k, v)
a    = Attention(q, K, V)                           # GQA, scale=1/sqrt(hd)
a    = a * sigmoid(gate)                            # 输出门
x    = x + o_proj @ a
```

partial RoPE：`rotary_dim = head_dim * partial_rotary_factor`（0.8B = 256·0.25 = 64），
只旋转每头前 `rotary_dim` 个分量（rotate-half），其余原样保留；频率按
`rotary_dim`（而非 head_dim）归一化。纯文本下 mRoPE 三轴位置相同，交织退化为普通 RoPE。

### linear attention 层（Gated DeltaNet，O(1) 状态）

```text
n1    = RMSNorm(x, input_layernorm)
mixed = in_proj_qkv @ n1                            # [conv_dim=2*key_dim+value_dim]
z     = in_proj_z @ n1                              # [value_dim]
b     = in_proj_b @ n1                              # [n_v_heads]
a     = in_proj_a @ n1                              # [n_v_heads]
mixed = SiLU(CausalConv1d_step(mixed, conv_state))  # depthwise, kernel=4；state 推进
q,k,v = split(mixed)                                # key_dim/key_dim/value_dim
q_h   = l2norm(q_h) / sqrt(qk_head_dim)  (per qk head)
k_h   = l2norm(k_h)                      (per qk head)
g     = -exp(A_log) * softplus(a + dt_bias)         # per v head（衰减对数）
beta  = sigmoid(b)                                  # per v head
# delta rule 递归步（per v head，状态 S:[qk_dim, v_dim]）：
S     = exp(g) * S
kv_mem = S^T @ k
S     = S + outer(k, beta * (v - kv_mem))
o     = S^T @ q
o     = RMSNormGated(o, z, norm)  = RMSNorm(o)*norm*SiLU(z)   # per v head
x     = x + out_proj @ o
```

GDN 无 KV cache：`conv_state`（最近 kernel-1 个输入）与递归矩阵 `S` 大小固定，
与序列长度无关（`runtime/gdn_state.h`）。full attention 层才用 KV cache（紧凑下标）。

### 数值要点

- `l2norm(x)=x/sqrt(sum(x^2)+1e-6)`；q 归一化后再乘 `1/sqrt(qk_head_dim)`。
- `softplus(t)=log(1+exp(t))`（t>20 时取 t，避免溢出）；g、beta 均为 fp32。
- causal conv1d 的 state 存 **silu 前**的原始投影值；就地更新时须先保存当前输入，
  防止被输出覆盖（有专门单测 `conv1d_inplace_state_not_corrupted`）。

---

相关文档：实现见 `runtime/qwen_forward_token.cpp`（op 顺序与本文一一对应）；
数值验收见 `pytorch_alignment.md`；概念背景见 `infra_primer.md`。
