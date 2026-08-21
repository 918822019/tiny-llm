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

## 当前状态

- 参考实现齐全：rmsnorm / rope / matvec（f32/f16/i4）/ matmul（f32/i4，prefill 用）/
  softmax / attention / silu / argmax + GDN 四算子（Qwen3.5）。
- matvec 走**三套独立注册表**（f32/f16/i4，按模型 dtype 解析，配错 fail fast）：
  - f32：`ref`（默认）/ `double_2_float` / `acc4` / `neon_nofma` / `neon` /
    `neon_mt` / `neon_mt_bal` / `neon_mt_kv` / `neon_mt_kv_nt`（CPU 阶梯顶层）
    / `cuda*` 系列（CUDA 构建）；
  - f16：`ref` / `neon_mt_kv_nt`（f16 满栈）/ `cuda_resident_coal(_ws)`；
  - i4：`ref` / `neon` / `neon_mt` / `sdot(_mt)` / `sdot2(_mt)`（W4A8 SDOT，
    预计算 + 2-row 并行，macOS 首次反超 f16）。
- 非 matvec 算子（rmsnorm/rope/attention/swiglu/argmax + GDN 四算子）共用 ops
  注册表：`ref` 兜底 + `neon` 变体，`--ops-impl` 开关选择；partial_rope 仅 ref。
- CUDA 两条腿：matvec 单算子变体（`matvec/*.cu`，走 dispatch）+ GPU-resident
  decode engine（`cuda/gpu_engine.cu`，`--engine cuda` 整段 forward，与逐算子
  分发正交）。runtime 侧另有逐算子 CUDABackend（`--backend cuda`，A/B 用）。
- 数字与归因见 `../docs/optimization_log.md`；调用方（CPUBackend）如何选择
  dtype 入口见 `../docs/architecture.md`。
- 优化版往哪加、怎么接、怎么测提速：见 `../docs/optimization.md`（优化手册）。

## 为什么 `_ref` 很重要

任何优化版在宣称"快"之前，必须先和 `_ref` 对齐（误差在容差内）才算"算对了"。
`_ref` 也是出问题时的兜底——切回它永远有正确结果。
