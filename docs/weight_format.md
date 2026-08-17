# tiny binary format（.tqwen）v1 / v2

`.tqwen` 是 tinyqwen 的权重文件格式。设计目标：**无依赖、可 mmap、64B 对齐、
Python 导出 / C++ 加载零歧义**。格式的唯一契约是 `runtime/tiny_format.h`，本文档
与它保持同步；修改格式必须同时改 exporter 并 bump version。

## 1. 总体布局

```text
offset 0        TinyHeader (192 B)
offset 192      TensorEntry[tensor_count] (每个 120 B)
                padding 到 64B 边界
data_offset     tensor 数据区（每个 tensor 的 offset 均为 64B 对齐）
```

- 所有整数 **little-endian**；所有 offset 均为**文件内绝对偏移**。
- C++ 侧结构体自然对齐即等于磁盘布局（有 `static_assert` 保护）。
- Python 侧用 `struct` 标准大小（`<` 前缀），不依赖平台 padding：
    - header：`<8s 12I ff 4Q 96s` → 192 B
    - entry：`<64s II 4Q QQ` → 120 B

## 2. TinyHeader（192 B）

| 字段                  | 类型       | 说明                                                      |
|---------------------|----------|---------------------------------------------------------|
| magic               | char[8]  | `"TINYQWEN"`                                            |
| version             | u32      | 格式版本，v1 = 1                                             |
| dtype               | u32      | 全部 tensor 的默认 dtype：0=f32（v1 仅支持 0），1=f16，2=i8，3=i4（保留） |
| n_layers            | u32      | Qwen2.5-0.5B: 24                                        |
| hidden_size         | u32      | 896                                                     |
| intermediate_size   | u32      | 4864                                                    |
| n_heads             | u32      | 14                                                      |
| n_kv_heads          | u32      | 2                                                       |
| head_dim            | u32      | 64                                                      |
| vocab_size          | u32      | 151936                                                  |
| max_seq_len         | u32      | 32768（HF `max_position_embeddings`）                     |
| tied_embeddings     | u32      | 1 = lm_head 与 embed_tokens 共享权重（0.5B 为 1）               |
| reserved_u32        | u32      | 必须为 0                                                   |
| rms_norm_eps        | f32      | 1e-6                                                    |
| rope_theta          | f32      | 1e6                                                     |
| tensor_count        | u64      | tensor 数量                                               |
| tensor_table_offset | u64      | v1 固定 = 192                                             |
| data_offset         | u64      | 第一个 tensor 数据偏移，64B 对齐                                  |
| total_bytes         | u64      | 文件总字节数                                                  |
| reserved            | char[96] | v1 必须全 0；v2 前 64 B 按 `TinyHeaderV2Ext` 解读（见 §7），其余为 0 |

## 3. TensorEntry（120 B）

| 字段     | 类型       | 说明                  |
|--------|----------|---------------------|
| name   | char[64] | NUL 补齐；满 64 字符时无终止符 |
| dtype  | u32      | 同 header dtype      |
| ndim   | u32      | 1..4                |
| shape  | u64[4]   | shape[ndim..3] = 0  |
| offset | u64      | 数据绝对偏移，% 64 == 0    |
| nbytes | u64      | numel × dtype_size  |

## 4. tensor 命名与 shape（Qwen2.5-0.5B）

命名与 HuggingFace 保持一致，便于对齐排查：

| name                                               | shape                                          |
|----------------------------------------------------|------------------------------------------------|
| `model.embed_tokens.weight`                        | [151936, 896]                                  |
| `model.layers.{i}.input_layernorm.weight`          | [896]                                          |
| `model.layers.{i}.self_attn.q_proj.weight`         | [896, 896]（n_heads*head_dim × hidden）          |
| `model.layers.{i}.self_attn.k_proj.weight`         | [128, 896]（n_kv_heads*head_dim × hidden）       |
| `model.layers.{i}.self_attn.v_proj.weight`         | [128, 896]                                     |
| `model.layers.{i}.self_attn.q_proj.bias`           | [896]                                          |
| `model.layers.{i}.self_attn.k_proj.bias`           | [128]                                          |
| `model.layers.{i}.self_attn.v_proj.bias`           | [128]                                          |
| `model.layers.{i}.self_attn.o_proj.weight`         | [896, 896]                                     |
| `model.layers.{i}.post_attention_layernorm.weight` | [896]                                          |
| `model.layers.{i}.mlp.gate_proj.weight`            | [4864, 896]                                    |
| `model.layers.{i}.mlp.up_proj.weight`              | [4864, 896]                                    |
| `model.layers.{i}.mlp.down_proj.weight`            | [896, 4864]                                    |
| `model.norm.weight`                                | [896]                                          |
| `lm_head.weight`                                   | [151936, 896]（tied 时不存在，loader 用 embed_tokens） |

layout 约定：**linear 权重保持 HF 的 `[out_dim, in_dim]` 行主序，v1 不转置**；
matvec reference 直接按行点积。后续 INT4/KronQ kernel 如需 packing，
另定义专用 dtype 与 layout，不复用 f32 路径。

bias 约定：Qwen2/2.5 的 attention **q/k/v 有 bias**（HF `attention_bias=True`），
o_proj 与 MLP 无 bias。bias 必须在 RoPE 之前加到 q/k/v 上。

## 5. 校验规则（loader 必须 fail fast）

1. 文件 ≥ 192 B，magic/version 匹配；
2. `total_bytes` == 实际文件大小；
3. `tensor_table_offset` == 192，entry 区不越界；
4. 每个 entry：ndim ∈ [1,4]，dtype = 0（v1），`nbytes == numel * 4`，
   `offset % 64 == 0`，`offset ≥ data_offset`，`offset + nbytes ≤ total_bytes`；
5. name 不重复；
6. header 中 config 字段非零且 `n_heads % n_kv_heads == 0`。

## 6. 体积估算（Qwen2.5-0.5B, f32）

约 0.5B 参数（tied，embedding 计一次）→ 约 2 GB。这是 v1 的已知代价，
INT8/INT4 阶段会显著缩小；runtime 侧 KV cache 与权重内存独立核算。

## 7. v2：Qwen3.5 混合架构扩展

v2 不改 header 大小与整体布局，仅把 `reserved[96]` 的前 64 字节定义为
`TinyHeaderV2Ext`（`runtime/tiny_format.h`），用于描述 Gated DeltaNet + full
attention 的混合层结构。loader 接受 version ∈ {1, 2}；v1 文件不受影响。

### 7.1 TinyHeaderV2Ext（64 B，位于 reserved 前 64 B）

字段顺序（`<7I f 2I 24s`）：

| 字段                       | 类型  | 说明                                                |
|--------------------------|-----|---------------------------------------------------|
| model_type               | u32 | 0 = Qwen2.x，1 = Qwen3.5 混合                          |
| linear_num_qk_heads      | u32 | GDN 的 Q/K 头数（0.8B: 16）                             |
| linear_num_v_heads       | u32 | GDN 的 V 头数（0.8B: 16）                               |
| linear_qk_head_dim       | u32 | GDN 每个 Q/K 头维度（0.8B: 128）                          |
| linear_v_head_dim        | u32 | GDN 每个 V 头维度（0.8B: 128）                            |
| linear_conv_kernel_dim   | u32 | GDN causal conv1d kernel（0.8B: 4）                  |
| full_attention_interval  | u32 | 每 N 层一个 full attention（3:1 → 4）                    |
| partial_rotary_factor    | f32 | full attention RoPE 只旋转 head_dim 的该比例（0.8B: 0.25） |
| eos_token_id             | u32 | stop token（Qwen3.5: 248044）                        |
| pad                      | u32 | 必须为 0                                              |
| ext_reserved             | char[24] | 必须全 0                                          |

层类型：`layer i` 为 full attention 当且仅当 `(i+1) % full_attention_interval == 0`，
其余为 linear（GDN）。KV cache 只为 full 层分配；GDN 层用固定大小的递归状态
（`runtime/gdn_state.h`），与序列长度无关。

### 7.2 Qwen3.5 tensor 命名与 shape（0.8B）

HF 权重带 `model.language_model.` 前缀，导出时统一去掉以复用 Qwen2.x 命名。
**full attention 层**（`self_attn`）：

| name                                        | shape                        | 备注                              |
|---------------------------------------------|------------------------------|---------------------------------|
| `model.layers.{i}.self_attn.q_proj.weight`  | [2·qd, hidden]               | 每头前 head_dim 是 query、后半是输出门 gate |
| `model.layers.{i}.self_attn.k_proj.weight`  | [kvd, hidden]                |                                 |
| `model.layers.{i}.self_attn.v_proj.weight`  | [kvd, hidden]                |                                 |
| `model.layers.{i}.self_attn.o_proj.weight`  | [hidden, qd]                 |                                 |
| `model.layers.{i}.self_attn.q_norm.weight`  | [head_dim]                   | zero-centered，导出时已 **+1**        |
| `model.layers.{i}.self_attn.k_norm.weight`  | [head_dim]                   | zero-centered，导出时已 **+1**        |

**linear attention 层**（`linear_attn`，Gated DeltaNet）：

| name                                          | shape                 | 备注                       |
|-----------------------------------------------|-----------------------|--------------------------|
| `model.layers.{i}.linear_attn.in_proj_qkv.weight` | [conv_dim, hidden] | conv_dim=2·key_dim+value_dim |
| `model.layers.{i}.linear_attn.in_proj_z.weight` | [value_dim, hidden] | 输出门 z                    |
| `model.layers.{i}.linear_attn.in_proj_b.weight` | [n_v_heads, hidden] | beta（写入门）投影             |
| `model.layers.{i}.linear_attn.in_proj_a.weight` | [n_v_heads, hidden] | a（dt）投影                  |
| `model.layers.{i}.linear_attn.out_proj.weight` | [hidden, value_dim]  |                          |
| `model.layers.{i}.linear_attn.conv1d.weight`  | [conv_dim, kernel]    | HF [d,1,k] 导出时 squeeze 成 [d,k] |
| `model.layers.{i}.linear_attn.A_log`          | [n_v_heads]           | 衰减 A 的 log               |
| `model.layers.{i}.linear_attn.dt_bias`        | [n_v_heads]           | dt 偏置                    |
| `model.layers.{i}.linear_attn.norm.weight`    | [v_head_dim]          | RMSNormGated，标准（不 +1）    |

**共享**：`input_layernorm.weight` / `post_attention_layernorm.weight` /
`model.norm.weight` 为 zero-centered RMSNorm，导出时统一 **+1**；`mlp.*` 与
Qwen2.x 一致（SwiGLU，无 bias）。Qwen3.5 attention 无 q/k/v bias。

### 7.3 zero-centered RMSNorm 折叠约定

Qwen3.5 的 `Qwen3_5RMSNorm` 是 `(1+w)·norm(x)`（权重初始 0）。为让 C++ 复用同一个
`rmsnorm` kernel，导出端把 `+1` 折进权重（存 `1+w`）。适用：input/post/model norm、
q_norm、k_norm。**例外**：`linear_attn.norm` 是 `Qwen3_5RMSNormGated`
（`w·norm(x)·silu(z)`），为标准形式，不折叠，由 `rmsnorm_gated` kernel 处理。

---

相关文档：格式的 C/Python 共同契约在 `runtime/tiny_format.h`（修改格式必须
同时 bump 版本并改 exporter）；对齐/字节序概念见 `infra_primer.md` §4–5；
导出端 `tools/export_qwen_to_tiny.py`、加载端 `runtime/model_loader.cpp`。
