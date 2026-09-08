# profiling JSON schema

由 `runtime/profiler.cpp` 输出（`--profile-out profile.json`）。手写 JSON，无第三方库。

```json
{
  "model": "qwen2.5-0.5b",
  "backend": "cpu_ref",
  "precision": "fp32",
  "prompt_tokens": 9,
  "generated_tokens": 16,
  "total_ms": 0.0,
  "first_token_ms": 0.0,
  "decode_avg_ms": 0.0,
  "tokens": [
    {
      "index": 0,
      "pos": 0,
      "is_prefill": true,
      "latency_ms": 0.0,
      "ops": {
        "layer_0.input_layernorm": 0.0,
        "layer_0.q_proj": 0.0
      }
    }
  ],
  "op_totals": {
    "layer_0.q_proj": {
      "calls": 25,
      "total_ms": 0.0
    }
  }
}
```

字段语义：

| 字段                 | 说明                                                                                                      |
|--------------------|---------------------------------------------------------------------------------------------------------|
| `prompt_tokens`    | prompt 长度（runtime 用 `set_counts` 显式声明；批量 prefill 下只有一条记录，≠ 记录数）                                         |
| `generated_tokens` | 实际生成的 token 数（= decode 步数 + 1：最后一个 token 不再 forward）                                                    |
| `total_ms`         | 所有 token latency 之和（不含加载模型；批量 prefill 计入整批那条记录）                                                         |
| `first_token_ms`   | TTFT：prefill 记录耗时之和。批量 prefill（默认 CPU 路径）= 整批记录的耗时；逐 token 路径（verbose / GPU engine）= 各 prefill token 之和 |
| `decode_avg_ms`    | decode 阶段平均每 token 耗时                                                                                   |
| `tokens[].ops`     | 该 token 内按执行顺序记录的 op 耗时。**同一 token 内同名 op 先按名合并求和再写入**（如 MoE 每层多次调用的 `expert_load`/`expert_ffn`），避免 JSON 重名键被 `json.load` 静默丢弃（8d7ad38 修复，见 AGENTS.md 坑 #34） |
| `op_totals`        | 跨 token 聚合（profiler 内部逐记录累加，不受重名影响），用于看 op 占比                                                      |

注意事项：

- token-by-token prefill 时 `first_token_ms` 随 prompt 长度线性增长，
  与 batched prefill 的 TTFT 不直接可比；
- `enter/leave` 必须配对（ScopedTimer 保证），不配对会打 warning；
- profiler 关闭时所有调用接近零开销；
- Android 上使用 `steady_clock`，不受墙钟跳变影响。

---

相关文档：测量方法与 top op 归因见 `optimization.md` §6；真机拉回 profile
见 `android.md` §4。
