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
| `biip/biip_rotate_neon.cpp`            | BiIP 激活旋转 NEON 版（旋转 VQ2 模型必经，与 ref 逐位一致） |

## 当前状态

- 参考实现齐全：rmsnorm / rope / matvec（f32/f16/i4）/ matmul（f32/i4，prefill 用）/
  softmax / attention / silu / argmax + GDN 四算子（Qwen3.5）+ BiIP 激活旋转。
- matvec 走**四套独立注册表**（f32/f16/i4/vq2，按模型 dtype 解析，配错 fail fast）：
  - f32：`ref`（默认）/ `double_2_float` / `acc4` / `neon_nofma` / `neon` /
    `neon_mt` / `neon_mt_bal` / `neon_mt_kv` / `neon_mt_kv_nt`（CPU 阶梯顶层）
    / `cuda*` 系列（CUDA 构建）；
  - f16：`ref` / `neon_mt_kv_nt`（f16 满栈）/ `cuda_resident_coal(_ws)`；
  - i4：`ref` / `neon` / `neon_mt` / `sdot(_mt)` / `sdot2(_mt)` / `sdot3(_mt)` /
    **`sdot4(_mt)`（当前 decode 最佳）** / `sdot5(_mt)`（对称量化实验，配
    `--symmetric` 模型）。
  - vq2（2-bit 块向量量化）：`ref` / `neon` / `neon_mr` / `neon_mr_mt` /
    `neon_mr_mt_wl` / **`neon_mr_mt_wl_nt`（当前最佳，推荐）**。
- **vq2 kernel 阶梯**（归因用，只增不删）：
  `ref`（double 累加锚，纯标量查表）→ `neon`（4 宽 SIMD：查表喂 1 条
  float32x4 FMA 链）→ `neon_mr`（+4 行并行：4 条独立 FMA 链隐藏延迟 +
  共享 x4 加载，vs neon 2.9–3.5×）→ `neon_mr_mt`（+常驻线程池行切分，
  粒度阈值 262144 元素，vs neon_mr 再 3.2–3.8×）→ `neon_mr_mt_wl`
  （+索引 32 位字加载：4 块展开，每块 load 数 9→6；**尺寸门**——实测
  收益只在 DRAM 流式大形状，索引区 ≥8MB 才走字加载体，否则退回字节体。
  lm_head 1.35 ms（分进程实测 1.48×），中小形状零回归）
  → **`neon_mr_mt_wl_nt`**（+索引流一条 `ldnp x,x` 取 **16** 个索引。
  **归因（做过隔离实验）**：收益来自**载入变宽**——每层 7 投影之和 0.313→0.205 ms
  （**1.53×**，用同宽度的普通 `ldp` 测得）；而 `ldnp` 的**非临时语义拿不出证据**，
  换成 `ldp` 后 lm_head 读数与之完全重叠（效应在噪声之下）。原假设"码本 4KB
  必须常驻 L1、索引流会挤它"的**前提也未验证**（4KB 落在 128KB 8 路 L1、每周期
  都访问，可能压根没被挤过）。名字里的 `nt` 只表示用了哪条指令。
  用 GPR 对而非 q 寄存器对，因为索引要参与地址计算。
  端到端 decode 1.077×、Qwen3.5-0.8B VQ2 1.28×，逐位一致。
  行起点须 8B 对齐（`n_blocks % 8 == 0`），否则退回上一级）。
  lm_head vs ref 合计 **~89×**。
  评测用 `./scripts/bench_kernels.sh --family vq2`（见 docs/optimization.md §6；
  mt 变体对比须分进程，避免常驻池互扰）。
- **GPTQ kernel 阶梯**（AutoGPTQ 列主序，与 i4 的 HQQ interleaved 布局不兼容，
  是**独立注册表**）：
  - matvec：`ref`（double 累加锚）→ `neon`（c8-outer/o-inner + o 分块 64 +
    **反量化因式分解** `Σ_k((nib_k-z)·s·x_k) = s·[Σ_k(nib_k·x_k) - z·Σ_k(x_k)]`）
    → `neon_mt`（+常驻线程池动态领取 o_block）。decode 9157 → 308 ms/tok。
  - matmul（批量 GEMM，MoE 批量 prefill 用）：`ref` → `neon`。结构与
    `matvec_gptq_neon` 同构，只是外面多一层列循环。
    **注意：X 是列主序 [K,N]，元素 (r,c) 在 `c*K+r`——固定 r 变化 c 步长是 K，
    不连续，不能对列向量化。正确方向是 o**（初版搞错，实测结果全错）。
- **MoE 批量 prefill 的收益上限由 I/O 原本占比决定**：批量路径把 I/O 降了 96%
  （n=128：230 GB → 8.4 GB），但 TTFT 只快 1.44×——因为逐 token prefill 里 I/O
  只占 28.4%，省掉 96% 的理论上限就是 `1/(1-0.284×0.96)` = 1.39×。
  **且批量路径的 matmul 必须有 NEON 版**：初版只有标量 ref 时 prefill 反而慢
  3.5×（用 5× 计算变慢换 96% I/O 降幅，净亏）。
- **i4 kernel 阶梯**（归因用，只增不删）：
  `sdot`（W4A8 SDOT 首版）→ `sdot2`（+预计算 scale/zero + 2-row 并行，首次反超 f16）
  → `sdot3`（+work-stealing 调度 + 内联组头硬件 FCVT + 128 位解包）→
  `sdot4`（修正 FCVT 特性守卫，4B 46.9→36.5 ms/tok）→ `sdot5`（对称量化，
  无 zero 修正，实验性 1.045×）。**推荐 `sdot4_mt`**（非对称模型）/
  `sdot5_mt`（对称模型）。
- **GPTQ kernel 阶梯**（归因用，只增不删；AutoGPTQ 列主序，与 i4 的 HQQ
  interleaved 布局不兼容，是**独立注册表**）：
  `ref`（double 累加锚，纯标量，每 MAC 约 5 op）→ `neon`（c8-outer/o-inner
  遍历 + o 分块 64 + **反量化因式分解**：同一 u32 字的 8 个 nibble 共享 s/z，
  故 `Σ_k ((nib_k - z)·s·x_k) = s·[Σ_k(nib_k·x_k) - z·Σ_k(x_k)]`，`Σ_k(x_k)`
  只依赖 c8、与 o 无关可预算。共享内层逻辑在 `matvec_gptq_neon_common.h`，
  neon 与 neon_mt 共用避免数值行为分叉）→ `neon_mt`（+常驻线程池动态领取
  o_block）。Qwen3-30B-A3B-GPTQ 实测 **decode 9157 → 616 ms/tok（14.9×）**，
  TTFT 67.1 → 5.3 s。act-order（g_idx 非均匀）自动降级到 slow 路径——
  算错不报错是本仓库最忌讳的失效模式，单测有护栏。
  **`neon_mt` 对 MoE 零收益甚至更慢**：专家矩阵 `[768,2048]` 只有 12 个
  o_block 给 10 线程，且每 token 有 3 matvec × 8 专家 × 48 层 = **1152 次
  fork-join**，同步开销压过收益（提高粒度阈值到 4M 反而更差）。MoE 要并行
  得在**专家层**并行（top-8 彼此独立，48 次 fork-join/token），是 runtime 级
  改动。保留 `neon_mt` 供非 MoE 的大 GPTQ 矩阵用。**加 MT 变体前先算
  fork-join 次数。**
- **被证伪的尝试**（代码保留供对照，默认不启用）：融合 W4A8 批量 matmul
  （`qwen_forward_prefill_qwen35.cpp` 内，`TINYQWEN_FUSED_MM` 开关）——
  手写 NEON SDOT 干不过 AMX sgemm，0.70×。
- 非 matvec 算子（rmsnorm/rope/attention/swiglu/argmax + GDN 四算子 + BiIP
  激活旋转）共用 ops 注册表：`ref` 兜底 + `neon` 变体，`--ops-impl` 开关选择；
  partial_rope 仅 ref。
- **BiIP 激活旋转**：`biip/biip_rotate.cpp`（ref）+ `biip/biip_rotate_neon.cpp`。
  旋转量化模型每个子层有各自的 sign/scale，旋转不能复用也不能融合——
  Qwen3-0.6B 每 token 调 196 次（7 子层 × 28 层），是 decode 的必经开销。
  NEON 版内核 2.30×（1.047→0.456 ms/token），**与 ref 逐位一致**：除法用
  `vdivq_f32` 不用近似倒数、butterfly 不重结合、末尾 `1/sqrt(bs)` 单独一遍。
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
