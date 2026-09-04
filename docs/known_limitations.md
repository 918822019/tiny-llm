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
  Qwen3.5 混合架构亦有批量 prefill（`forward_prefill_qwen35_batch`，GEMM 路径，
  ≥`kBatchPrefillMinQwen35` token 才启用，否则回退逐 token）；
  GPU engine 路径无批量 prefill 入口（逐 token 喂）；
- C++ 侧无 tokenizer：token ids 由 Python 工具提供；
- loader 一次性 fread 整个文件进内存（未用 mmap）；
- KV cache 一次性分配，`--max-seq-len` 过大会有内存压力；默认 fp32，
  `--kv-f16` 可切 fp16（内存减半，长上下文用；解码慢 ~8%，因寄存器内
  fp16→fp32 转换抵消读带宽减半——是内存特性不是提速，见优化日志
  `fp16_kv_fused`）；fp16-KV 仅 CPU（CUDA 后端调用即 abort）；
- 超出 max_seq_len / KV 溢出直接 abort（fail loud）；
- speculative rollback / KV truncate 未实现。

## 后端与硬件

- CPUBackend 是主线：f32/f16/i4 全 dtype + GDN 全路径覆盖；
- CUDABackend（`--backend cuda`）逐算子调用、每次带 H2D/D2H：INT4 matvec、
  partial_rope、top_k_logits、GDN 四算子**未实现（触发即 abort）**；仅用于
  A/B 与单算子验证，不是性能路径；
- GPU-resident engine（`--engine cuda`）是 GPU 性能路径，但只支持
  Qwen2.x + greedy：不支持 `--topk` / `--dump-logits` / 批量模式；
- **Metal prefill（`--engine metal`）仅覆盖 prefill 阶段**，且限制较多：
  - 仅 Apple 平台（其他平台编译 `metal_prefill_stub.cpp`，运行期报错而非链接失败）；
  - 权重 dtype 只支持 f16 / f32（i4 / vq2 的亚字节布局需要专门反量化 kernel）；
  - `head_dim % 4 == 0` 且 ≤ 256（float4 对齐 + 片上 qs 数组容量）；
    rotary_dim 必须为偶数且 ≤ head_dim（支持 partial RoPE，如 Qwen3.5 的 64/256）；
  - **支持 Qwen3.5 的 GDN + full attention 混合架构**：GDN 四算子有 Metal 实现。
    跑混合架构时 `metal_prefill_run` 必须传 `GdnState*`，否则 prefill 首 token
    正确但 decode 发散；
  - **支持 KV 接续**（投机解码 verify pass 的前提）：引擎自持一份 GPU 侧 KV cache，
    每次调用把 n 个 token 追加在已有上下文之后，attention 读 `[0, pos0+n)`。
    开始新序列前必须调 `metal_prefill_reset_kv()`，否则会读到上一段的历史。
    若同时传 CPU KvCache，两者长度必须一致，否则报错；
  - `max_seq_len <= 1024`：attention 的片上分数数组 `sc[1024]` 是固定容量，
    更长上下文需要改成分块（flash-attention）才能突破；
  - 额外显存：GPU KV cache 约 235 MB（0.6B @ max_seq_len=1024），与 CPU 的
    KvCache 是两份独立内存；
  - 不支持 `--topk`（用 `--dump-logits` 代替）、不支持批量模式；
  - 权重常驻 GPU：f16 模型以 fp16 存（0.6B ≈ 1.20 GB，RSS 2.46 GB），f32 模型以
    fp32 存（≈ 2.40 GB）。fp16 **只省内存不提速** —— 长 prompt prefill 的 GEMM
    是算力受限而非带宽受限；
  - 短 prompt 不划算：seq=16 仍比 CPU 慢（0.84×），交叉点约在 seq≈32；
  - **投机解码区间是 MPS 逐调用开销受限**：verify pass 权重流量 ~1.19 GB
    （带宽下限 ~10 ms），但实测 42–87 ms —— 112 次 MPS encode 的固定开销占大头。
    所以 K（草稿长度）越大越划算：L=128 时 K=4 是 10.46 ms/tok、K=16 是 3.31 ms/tok；
  - 数值口径与 CPU 同为 fp32，末位 logits `max_abs_err ≈ 3.2e-05`（CPU ref 的
    RMSNorm 用 fp64 累加，GPU 用 fp32，故略大于纯 CPU 路径间的差异）；
- Vulkan 后端仅 IBackend 接口预留，无实现。

## 格式

- dtype 支持 f32 / f16 / i4（i4 为混合 dtype：大矩阵 uint4-packed，
  embed/norm/bias 保留 fp32）；i8 已在 header 中预留但 loader 拒绝；
- f16/i4 的数值验收只做过 canonical prompt 的 greedy 对照与随机权重假模型
  logits 对齐，长文本/敏感任务未覆盖；
- **i4 批量 prefill 与逐 token decode 数值不一致（已知且刻意）**：批量路径
  反量化权重后激活走 fp32（weight-only，更贴近 HF）；decode 逐 token 走
  W4A8（激活量化 int8，提速）。末位 prefill logits 可差 ~1 量级，临界
  argmax 偶发翻转——同一 prompt 贪心输出可能因长度跨过批量阈值而不同；
- tensor name 上限 64 字符；
- 小端假设（目标设备 ARM64 均为小端）。

## profiling

- 批量 prefill 在 profiler 里记为一条 prefill 记录（整批计时），不做 op 级拆分；
- profiler 开启时每 op 一次 steady_clock 调用，有小开销
  （对 ms 级 op 可忽略，对 μs 级 op 有相对放大）；
- 不记录内存带宽、cache miss 等硬件计数器（端侧用采样工具补，见 android.md §6）。

## 未实现（路线图，不在 v1）

- INT4 继续：sdot4 已兑现硬件组头转换（4B 46.9→36.5 ms/tok）；TTFT 侧已跳过
  非末位 prefill logits（1.14×）+ Qwen3.5 批量 prefill GEMM 化（4B-61tok 1.72×、
  0.8B-33tok 1.38×）+ 反量化并行化（1.08×）。批量 prefill 到此基本触顶：
  **已调研确认两条路走不通**——① Accelerate 无 fp16/int8 GEMM（cblas 仅
  s/d/c/z），"反量化到 fp16 减半流量"不可行；② 融合 W4A8 批量 matmul 实测
  更慢（0.70×，手写 NEON SDOT 干不过 AMX，证伪归档，代码保留 `TINYQWEN_FUSED_MM`）。
  剩余大头 = AMX-sgemm（~77%）+ 单线程 GDN 递归扫描（~20%，跨 token 顺序
  依赖难并行）。4-row 内循环 / i4 gate_up·qkv 融合 / 激活量化 SIMD 仍是
  decode 侧候选；Android 真机验证 sdot3/sdot4 与批量 prefill 未做
  （无 BLAS 后端时自动回退逐 token）；
- INT8 weight-only reference quantization；
- KronQ packing；
- GPTQ / AWQ 等量化算法（接入流程见 `quantization_guide.md`）；
- Speculative decoding（draft/verify/rollback）；
- 多 LoRA adapter 调度；
- mmap 加载、Android App。

---

相关文档：优化进展与当前数字见 `optimization_log.md`；
优化方法与纪律见 `optimization.md`；分层架构见 `architecture.md`。
