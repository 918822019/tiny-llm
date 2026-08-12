# 与 PyTorch reference 对齐流程

目标：C++ runtime 的每一步都能和 HuggingFace/PyTorch 的 fp32 eager 前向对上，
出问题时能把误差定位到具体 op。

## 0. 无真模型的快速自检（改完 forward 先跑这个）

```bash
python tools/align_fake_model.py
```

随机权重小模型（hidden=16，2 层），C++ `--dump-logits` 全量 logits 与 HF
Qwen2（fp32/eager）逐位置比对，覆盖 RMSNorm / q-k-v bias / RoPE / GQA /
SwiGLU / tied lm_head。当前基线：worst max_abs_err ≈ 4e-7。

## 流程

1. 固定输入：

   ```bash
   python tools/tokenize_prompt.py --model $MODEL --prompt "你好" --chat \
     --out prompt_tokens.json
   ```

2. dump 参考值（fp32 / cpu / eager / eval）：

   ```bash
   python tools/dump_qwen_reference.py --model $MODEL \
     --tokens-json prompt_tokens.json --out ref.npz --topk 10
   ```

3. C++ 侧运行同样的 token 序列，导出对应中间量
   （v1 可先只比 logits/top-k；逐 op 比对可加 debug dump 开关，见下）。

4. 按下面的优先级比对。

## 对齐顺序（从粗到细）

1. **top-k token id**：最后位置 top-10 的 index 集合必须一致
   （value 允许小误差）。这是最终验收线。
2. **logits**：最后位置全量 logits：`max_abs_err` 与相对误差。
3. **final_norm** 输出。
4. **layer 0 细粒度**：`attn_norm → q/k/v → attn_out → post_attn_residual
   → ffn_norm → gate/up → ffn_out → layer_output`，
   与 `ref.npz` 中同名 tensor 逐位置比对。
5. 若 layer 0 对上而整体不对，二分中间层。

## 建议容差（fp32, CPU）

| 量                                     | max_abs_err 经验容差    |
|---------------------------------------|---------------------|
| rmsnorm / rope / 单 matvec（小 shape 单测） | 1e-5                |
| layer 输出（896 维，24 层之前）                | 1e-4 ~ 1e-3         |
| logits                                | 1e-2 以内且 top-k 集合一致 |

注意：误差随层数累积是正常现象；关键是**序关系和结构**不能变
（例如某个 op 误差突然比相邻 op 大一个量级 → 该 op 实现有问题）。

## 关键一致性检查点（最容易错的地方）

- RoPE：rotate-half 的拼接方向、`inv_freq` 的指数、position 从 0 开始；
- GQA：q head → kv head 的映射是整除 `h / (nh/nkv)`；
- attention scale：`1/sqrt(head_dim)`，不是 `1/sqrt(hidden)`；
- SwiGLU：SiLU 只作用在 gate 分支；
- residual 顺序：先 attn residual，再 ffn residual；
- tied lm_head：0.5B 用 embed_tokens 当 lm_head；
- `position_ids`：reference dump 用显式 `arange`，C++ 侧 pos = KV seq_len，
  两者必须一致；
- dtype：reference 固定 fp32 + eager attention，避免 SDPA 内核差异混入。

## 排查建议

- 每发现一个对不上的 op，先在 tests/ 里加一个小 shape 复现用例；
- C++ 逐 op dump 可以用一个 debug 环境变量开关（后续补），
  dump 命名必须与 `ref.npz` 的 key 一致；
- 不要让分析脚本替代逐 tensor 比对：先看 max_abs_err，再看分布。
