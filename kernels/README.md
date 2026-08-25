# kernels/ 导读

这里放所有算子（kernel）。结构围绕一条原则：**参考实现（base）永远保留，
优化版只增不删，model 通过分发层选用**。

## 文件约定

**每个算子一个文件夹**，参考实现和所有优化版都放在里面；签名契约和分发层
全算子共用，留在根目录。加新 kernel 不需要做任何结构决策——放进对应
算子文件夹即可。

```text
ref_ops.h                     # 所有 kernel 的签名契约（共用）
dispatch.h / dispatch.cpp     # 分发层 + 注册表：model 只调通用入口（共用）；
                              # 变体在文件末尾用 TINYQWEN_MATVEC_VARIANT 宏自注册
<op>/                         # 每个算子一个文件夹
├── <op>_<dtype>_ref.cpp      # 参考实现（base）：正确性基准 + 兜底，永不删
└── <op>_<dtype>_<variant>.cpp  # 【改进位】优化版，只增不删
```

命名示例：

| 文件                                     | 含义                              |
|----------------------------------------|---------------------------------|
| `matvec/matvec_f32_ref.cpp`            | fp32 参考实现（base）                 |
| `matvec/matvec_f32_neon.cpp`           | fp32 + NEON SIMD（归因阶梯 L4）       |
| `matvec/matvec_f32_neon_mt.cpp`        | fp32 + NEON + 多线程               |
| `matvec/matvec_f16_neon_mt_kv_nt.cpp`  | fp16 权重满栈移植（流量减半）               |
| `matvec/matvec_i4_sdot2.cpp`           | INT4 W4A8 SDOT + 预计算（首个反超 f16）  |
| `matvec/matvec_i4_sdot4.cpp`           | **当前 decode 最佳**：W4A8 + work-stealing + 硬件 FCVT |

## 当前状态

- 参考实现齐全：rmsnorm / rope / matvec（f32/f16/i4）/ matmul（f32/i4，prefill 用）/
  softmax / attention / silu / argmax + GDN 四算子（Qwen3.5）。
- matvec 走**四套独立注册表**（f32/f16/i4/vq2，按模型 dtype 解析，配错 fail fast）：
  - f32：`ref`（默认）/ `double_2_float` / `acc4` / `neon_nofma` / `neon` /
    `neon_mt` / `neon_mt_bal` / `neon_mt_kv` / `neon_mt_kv_nt`（CPU 阶梯顶层）
    / `cuda*` 系列（CUDA 构建）；
  - f16：`ref` / `neon_mt_kv_nt`（f16 满栈）/ `cuda_resident_coal(_ws)`；
  - i4：`ref` / `neon` / `neon_mt` / `sdot(_mt)` / `sdot2(_mt)` / `sdot3(_mt)` /
    **`sdot4(_mt)`（当前 decode 最佳）** / `sdot5(_mt)`（对称量化实验，配
    `--symmetric` 模型）。
  - vq2（2-bit 块向量量化）：`ref` / `neon` / `neon_mr` / `neon_mr_mt` /
    **`neon_mr_mt_wl`（当前最佳，推荐）**。
- **vq2 kernel 阶梯**（归因用，只增不删）：
  `ref`（double 累加锚，纯标量查表）→ `neon`（4 宽 SIMD：查表喂 1 条
  float32x4 FMA 链）→ `neon_mr`（+4 行并行：4 条独立 FMA 链隐藏延迟 +
  共享 x4 加载，vs neon 2.9–3.5×）→ `neon_mr_mt`（+常驻线程池行切分，
  粒度阈值 262144 元素，vs neon_mr 再 3.2–3.8×）→ `neon_mr_mt_wl`
  （+索引 32 位字加载：4 块展开，每块 load 数 9→6；**尺寸门**——实测
  收益只在 DRAM 流式大形状，索引区 ≥8MB 才走字加载体，否则退回字节体。
  lm_head 1.35 ms（分进程实测 1.48×），中小形状零回归）。
  lm_head vs ref 合计 **~76×**。
  评测用 `./scripts/bench_kernels.sh --family vq2`（见 docs/optimization.md §6；
  mt 变体对比须分进程，避免常驻池互扰）。
- **i4 kernel 阶梯**（归因用，只增不删）：
  `sdot`（W4A8 SDOT 首版）→ `sdot2`（+预计算 scale/zero + 2-row 并行，首次反超 f16）
  → `sdot3`（+work-stealing 调度 + 内联组头硬件 FCVT + 128 位解包）→
  `sdot4`（修正 FCVT 特性守卫，4B 46.9→36.5 ms/tok）→ `sdot5`（对称量化，
  无 zero 修正，实验性 1.045×）。**推荐 `sdot4_mt`**（非对称模型）/
  `sdot5_mt`（对称模型）。
- **被证伪的尝试**（代码保留供对照，默认不启用）：融合 W4A8 批量 matmul
  （`qwen_forward_prefill_qwen35.cpp` 内，`TINYQWEN_FUSED_MM` 开关）——
  手写 NEON SDOT 干不过 AMX sgemm，0.70×。
- 非 matvec 算子（rmsnorm/rope/attention/swiglu/argmax + GDN 四算子）共用 ops
  注册表：`ref` 兜底 + `neon` 变体，`--ops-impl` 开关选择；partial_rope 仅 ref。
- **批量 prefill（Qwen3.5）**：`runtime/qwen_forward_prefill_qwen35.cpp`——
  prompt ≥32 token 时，线性投影反量化到 fp32 走 Accelerate/AMX sgemm
  （权重每层只读一遍），GDN 递归与因果 attention 保留逐 token 顺序扫描。
  4B 61-token TTFT 2072→1206ms（1.72×）。
- **fp16-KV 融合 attention**：`attention/attention_decode_neon.cpp` 内的
  `attention_decode_f16kv_neon`——`--kv-f16` 时 KV 存 fp16，attention 读 fp16、
  寄存器内转 fp32 计算（消灭独立反量化遍）。省一半 KV 内存（长上下文用），
  解码慢 ~8%（内存特性非提速）。`QwenModel::attention_kv` 按 KV 精度分发。
- CUDA 两条腿：matvec 单算子变体（`matvec/*.cu`，走 dispatch）+ GPU-resident
  decode engine（`cuda/gpu_engine.cu`，`--engine cuda` 整段 forward，与逐算子
  分发正交）。runtime 侧另有逐算子 CUDABackend（`--backend cuda`，A/B 用）。
- 数字与归因见 `../docs/optimization_log.md`；调用方（CPUBackend）如何选择
  dtype 入口见 `../docs/architecture.md`。
- 优化版往哪加、怎么接、怎么测提速：见 `../docs/optimization.md`（优化手册）。

## 为什么 `_ref` 很重要

任何优化版在宣称"快"之前，必须先和 `_ref` 对齐（误差在容差内）才算"算对了"。
`_ref` 也是出问题时的兜底——切回它永远有正确结果。
