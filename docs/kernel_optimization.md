# Kernel 优化指南（保留 base，往哪加改进）

> 这篇回答两个问题：**参考实现放哪、改进版往哪加**。核心是"base 永远保留、
> 优化版只增不删、model 不感知具体实现"。

---

## 1. 目录约定

```text
kernels/
├── ref_ops.h                     # 所有 kernel 的签名契约（共用）
├── dispatch.h / dispatch.cpp     # 分发层：model 只调通用入口，由它选实现（共用）
└── <op>/                         # 每个算子一个文件夹
    ├── <op>_<dtype>_ref.cpp      # 参考实现（base）：永不删、永不覆盖
    └── <op>_<dtype>_<variant>.cpp  # 优化版（改进位）：只增不删
```

三条铁律：

- **每个算子一个文件夹**，参考实现和所有优化版都放里面——加新 kernel
  不做结构决策，放进对应文件夹即可。
- **`_ref` 是标准答案**。任何优化版都要先和它对齐（误差在容差内）才算"算对了"，
  然后才谈"快不快"。它也是出问题时的一键兜底。
- **优化版永远是新文件**，命名 `<op>_<dtype>_<variant>`，绝不覆盖 `_ref`。

## 2. 命名约定

`<算子>_<数据类型>_<优化手段>`，可叠加：

| 文件名                  | 含义                      |
|----------------------|-------------------------|
| `matvec_f32_ref`     | fp32 参考实现（base）         |
| `matvec_f32_neon`    | fp32 + NEON SIMD        |
| `matvec_f32_neon_mt` | fp32 + NEON + 多线程       |
| `matvec_i8_neon`     | INT8 weight-only + NEON |
| `rmsnorm_neon`       | RMSNorm 的 NEON 版        |

## 3. 分发层：model 不感知具体实现

model（`qwen_model.cpp`）只调**通用入口** `matvec_f32(...)`，不直接调
`matvec_f32_ref`。由 `dispatch.cpp` 根据当前选择决定跑哪个实现：

```text
qwen_model.cpp ──> matvec_f32()  ──dispatch──> matvec_f32_ref()   (默认)
                                             └──> matvec_f32_neon()  (启用后)
```

好处：

- **加优化不用动 model**——只加文件 + 枚举值 + dispatch 的 case；
- **可 A/B**：同一负载，切换实现各跑一遍直接对比；
- **可兜底**：切回 ref 永远有正确结果。

选择方式有两种，优先级 **CLI 开关 > 配置文件 > 默认**：

- CLI：`--matvec-impl ref`（benchmark 时随手切换）；
- 配置文件：`tinyqwen.conf` 里写 `matvec_impl = ref`（用 `--config` 指定）。

micro-benchmark 时也能在同一个程序里同时调 ref 和优化版做对比。

## 4. 加一个新优化算子：标准流程

以"给 matvec 加 NEON 版"为例：

1. **写实现**：新建 `kernels/matvec/matvec_f32_neon.cpp`，声明
   `void matvec_f32_neon(const float*, const float*, float*, int, int);`
   （签名和 `_ref` 完全一致）。
2. **接进分发**：
    - `dispatch.h`：给 `MatvecImpl` 加 `kNeon`；
    - `dispatch.cpp`：`switch` 里加 `case kNeon: matvec_f32_neon(...);`，
      并在 `matvec_impl_name()` 加名字。
3. **CMake**：把新 `.cpp` 加进 `kernels/CMakeLists.txt`。
4. **加开关值**：`main.cpp` 的 `--matvec-impl` 接受 `neon`。
5. **正确性门禁**：写单测/脚本，确认 `neon` 输出与 `ref` 误差在容差内
   （**先证明算对了**）。
6. **测速**：micro-benchmark 测 kernel 提速，`bench.sh` 测整机提速，
   两个数都填进 `optimization_log.md`（见 `benchmarking.md` 第 7–9 节）。

## 5. 正确性门禁（不可跳过）

优化版在宣称"快"之前，必须先过对齐：

```text
对同一组输入：max | optimized(x) - ref(x) | < 容差
```

- fp32 之间的优化（如 NEON）：容差可紧些（~1e-4），因为只是计算顺序不同；
- 量化（INT8/INT4）：容差放宽，但要在 `dump_qwen_reference` / 真模型上
  确认最终 logits top-k 不变。

对齐工具：`tests/`（小 shape 单测）+ `tools/align_fake_model.py`（整模型）。

## 6. 当前状态

- 已接入 dispatch 的算子：**matvec**（唯一热点，优化主攻方向）。
- 其余算子（rmsnorm/rope/attention/...）目前直接调 `_ref`；将来要优化哪个，
  照第 4 节给它也加一个通用入口即可。
