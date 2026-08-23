# v1 已知限制

刻意且公开的限制，避免误用。

> **当前性能基线**：因 dtype / 硬件而异，权威数字见 `optimization_log.md`
> 汇总表（Qwen2.5-0.5B：macOS f16 满栈 ~6.35 ms/tok、i4 sdot2_mt ~3.67 ms/tok；
> A10 GPU engine ~4.89 ms/tok；Android f16 17~21 ms/tok、i4 SDOT ~22.4 ms/tok）。

## 运行时

- greedy decode，无采样（temperature / top-p 未实现）；
- 单 batch，无 continuous batching（`--batch-tokens-jsonl` 是一次进程顺序跑
  多条 prompt，逐条独立 prefill+decode，不是并行批处理）;
- prefill 已是批量 GEMM（`forward_prefill`），但无 chunked prefill；
  GPU engine 路径无批量 prefill 入口（逐 token 喂）；
- C++ 侧无 tokenizer：token ids 由 Python 工具提供；
- loader 一次性 fread 整个文件进内存（未用 mmap）；
- KV cache 一次性分配 fp32，`--max-seq-len` 过大会有内存压力；
- 超出 max_seq_len / KV 溢出直接 abort（fail loud）；
- speculative rollback / KV truncate 未实现。

## 后端与硬件

- CPUBackend 是主线：f32/f16/i4 全 dtype + GDN 全路径覆盖；
- CUDABackend（`--backend cuda`）逐算子调用、每次带 H2D/D2H：INT4 matvec、
  partial_rope、top_k_logits、GDN 四算子**未实现（触发即 abort）**；仅用于
  A/B 与单算子验证，不是性能路径；
- GPU-resident engine（`--engine cuda`）是 GPU 性能路径，但只支持
  Qwen2.x + greedy：不支持 `--topk` / `--dump-logits` / 批量模式；
- Metal / Vulkan 后端仅 IBackend 接口预留，无实现。

## 格式

- dtype 支持 f32 / f16 / i4（i4 为混合 dtype：大矩阵 uint4-packed，
  embed/norm/bias 保留 fp32）；i8 已在 header 中预留但 loader 拒绝；
- f16/i4 的数值验收只做过 canonical prompt 的 greedy 对照与随机权重假模型
  logits 对齐，长文本/敏感任务未覆盖；
- tensor name 上限 64 字符；
- 小端假设（目标设备 ARM64 均为小端）。

## profiling

- 批量 prefill 在 profiler 里记为一条 prefill 记录（整批计时），不做 op 级拆分；
- profiler 开启时每 op 一次 steady_clock 调用，有小开销
  （对 ms 级 op 可忽略，对 μs 级 op 有相对放大）；
- 不记录内存带宽、cache miss 等硬件计数器（端侧用采样工具补，见 android.md §6）。

## 未实现（路线图，不在 v1）

- INT4 继续：sdot3 已消预计算缓存与异构核拖尾（见优化日志），下一刀候选
  = 4-row 内循环（单核效率）/ i4 gate_up·qkv 融合（dispatch 入口已有、
  无实现）/ 激活量化 SIMD；Android 真机验证 sdot3 未做；
- INT8 weight-only reference quantization；
- KronQ packing；
- GPTQ / AWQ 等量化算法（接入流程见 `quantization_guide.md`）；
- Speculative decoding（draft/verify/rollback）；
- 多 LoRA adapter 调度；
- mmap 加载、Android App。

---

相关文档：优化进展与当前数字见 `optimization_log.md`；
优化方法与纪律见 `optimization.md`；分层架构见 `architecture.md`。
