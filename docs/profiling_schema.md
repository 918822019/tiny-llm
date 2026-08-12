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
      "ops": {"layer_0.input_layernorm": 0.0, "layer_0.q_proj": 0.0}
    }
  ],
  "op_totals": {
    "layer_0.q_proj": {"calls": 25, "total_ms": 0.0}
  }
}
```

字段语义：

| 字段 | 说明 |
|---|---|
| `prompt_tokens` / `generated_tokens` | `is_prefill` 为 true / false 的 token 数 |
| `total_ms` | 所有 token latency 之和（不含加载模型） |
| `first_token_ms` | prefill 阶段总耗时（token-by-token prefill 下 = 各 prefill token 之和，近似 TTFT） |
| `decode_avg_ms` | decode 阶段平均每 token 耗时 |
| `tokens[].ops` | 该 token 内按执行顺序记录的 op 耗时；op 命名如 `layer_3.q_proj`、`lm_head` |
| `op_totals` | 跨 token 聚合，用于看 op 占比 |

注意事项：

- token-by-token prefill 时 `first_token_ms` 随 prompt 长度线性增长，
  与 batched prefill 的 TTFT 不直接可比；
- `enter/leave` 必须配对（ScopedTimer 保证），不配对会打 warning；
- profiler 关闭时所有调用接近零开销；
- Android 上使用 `steady_clock`，不受墙钟跳变影响。
