# v1 已知限制

刻意且公开的限制，避免误用。

> **当前性能基线**：Qwen2.5-0.5B，Apple Silicon 单线程 fp32，
> decode ≈ **220 ms/token**（已落地变体 `double_2_float`；
> 精确数字与优化进展以 `optimization_log.md` 为准）。

## 运行时

- 仅 FP32 reference 路径，无 SIMD、无多线程、无量化 kernel
  （但可插拔的 dispatch 接缝 + 配置开关已就位，见 `optimization.md`）；
- prefill 也是 token-by-token（O(n²) attention 总量与 batched 相同，
  但没有矩阵并行，长 prompt 会明显慢）；
- 单 batch，无 continuous batching；
- greedy decode，无采样（temperature / top-p 未实现）；
- C++ 侧无 tokenizer：token ids 由 Python 工具提供；
- loader 一次性 fread 整个文件进内存（未用 mmap）；
- KV cache 一次性分配 fp32，`--max-seq-len` 过大会有内存压力；
- 超出 max_seq_len / KV 溢出直接 abort（fail loud）。

## 格式

- dtype 支持 f32 / f16 / i4（i4 为混合 dtype：matmul 权重 uint4-packed，
  embed/norm/bias 保留 fp32）；i8 已在 header 中预留但 loader 拒绝；
- f16 的数值验收只做过 canonical prompt 的 greedy 对照（16 token 与 fp32
  逐位一致），长文本/敏感任务未覆盖；
- tensor name 上限 64 字符；
- 小端假设（目标设备 ARM64 均为小端）。

## profiling

- token-by-token prefill 下 `first_token_ms` 不是 batched TTFT；
- profiler 开启时每 op 一次 steady_clock 调用，有小开销
  （对 ms 级 op 可忽略，对 μs 级 op 有相对放大）；
- 不记录内存带宽、cache miss 等硬件计数器（后续 perf 阶段补）。

## 未实现（路线图，不在 v1）

- INT4 路线进行中：HQQ 量化导出 + ref/neon kernel 已落地（见优化日志
  i4-hqq-android），但 **i4 kernel 尚不如 f16 满栈快**——缺多线程变体、
  nibble unpack 慢，且 tied lm_head 仍是 fp32（÷4 流量红利未拿满）；
- INT8 weight-only reference quantization；
- KronQ packing；
- Speculative decoding（draft/verify/rollback）；
- 多 LoRA adapter 调度；
- KV cache 的 speculative rollback / truncate；
- mmap 加载、Android App。

---

相关文档：优化进展与当前数字见 `optimization_log.md`；
优化方法与纪律见 `optimization.md`。
