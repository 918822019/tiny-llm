# docs/archive/ — 已归档的历史模型配置

这里的 5 个 yaml 是 **Qwen2.5-0.5B** 早期优化阶梯的模型注册表。它们指向的
`.tqwen` 权重文件已不在仓库（当时是 0.5B 小模型，后来项目主线切到
Qwen3.5-0.8B / 4B），因此从根目录移到这里归档，`tools/model_registry.py`
不再加载它们。

**为什么保留**：这些配置对应优化日志（`../optimization_log.md`）里 fp32→f16→i4
的完整归因阶梯（fp32-baseline → neon → neon_mt → f16 满栈 → i4 HQQ →
lm_head i4 → W4A8 SDOT）。数字和结论都在日志里，这里只是配套的元数据快照。

| 文件 | 对应模型 | 阶段 |
|---|---|---|
| `model.yaml` | Qwen2.5-0.5B f32 | fp32 基线 |
| `model_f16.yaml` | Qwen2.5-0.5B f16 | f16 满栈 |
| `model_i4.yaml` | Qwen2.5-0.5B i4 RTN | i4 初步 |
| `model_i4_hqq.yaml` | Qwen2.5-0.5B i4 HQQ | HQQ 量化 |
| `model_i4_hqq_lmh.yaml` | Qwen2.5-0.5B i4 HQQ + lm_head i4 | W4A8 SDOT 前的最终形态 |

当前在用的模型配置在仓库根目录（`model_qwen35_*.yaml`）。
