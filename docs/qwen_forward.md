# Qwen forward 数学定义与 shape 约定

对应实现：`runtime/qwen_model.cpp`（token-by-token，batch=1，fp32）。
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

| 点 | HF | tinyqwen v1 |
|---|---|---|
| prefill | 一次并行 | token-by-token 串行（数值等价，causal 下结果一致） |
| attention | SDPA/eager | 手写 decode kernel（online softmax） |
| 累加精度 | fp32 | matvec/rmsnorm 用 double 累加，输出 fp32 |
| tokenizer | 内置 | 外部 Python 提供 token ids |
