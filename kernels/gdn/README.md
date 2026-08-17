# GDN（Gated DeltaNet）算子内核

Qwen3.5 的线性注意力层（Gated DeltaNet）专属算子集合。每层调用一次 conv1d +
32 次 l2norm + 16 次 gdn_step + 16 次 rmsnorm_gated。

## 架构概览

```text
runtime/qwen_model.cpp
    │
    │ 通用入口（dispatch.h）
    ├──> causal_conv1d_update()  ──dispatch──> _ref / _neon
    ├──> l2norm_inplace()        ──dispatch──> _ref / _neon
    ├──> gdn_step()              ──dispatch──> _ref / _neon
    └──> rmsnorm_gated()         ──dispatch──> _ref / _neon
```

开关：`--ops-impl neon`（与 rmsnorm/rope/attention/swiglu/argmax 共用同一命名空间）。
未注册时自动兜底到 `_ref`，不影响正确性。

## 算子规格（Qwen3.5-0.8B）

| 算子 | 函数签名 | 维度 | 每层调用次数 | 热度 |
|------|---------|------|------------|------|
| causal_conv1d_update | `(x, state, weight, out, dim=6144, ks=4)` | 6144 通道 | 1 | 中 |
| l2norm_inplace | `(x, n=128, eps)` | 128 (qk_head_dim) | 32 (16q+16k) | 低 |
| gdn_step | `(S, q, k, v, g, beta, o, qk=128, v=128)` | S=[128×128]=64KB | 16 (n_heads) | **高** |
| rmsnorm_gated | `(x, gate, weight, y, n=128, eps)` | 128 (v_head_dim) | 16 | 低 |

> gdn_step 占 GDN 非投影算子 91% 的耗时，是优化主目标。

## 文件结构

```text
kernels/gdn/
├── gdn_ops_ref.cpp               # 4 个参考实现（标量、正确性优先、永不修改）
├── gdn_step_neon.cpp             # delta rule 递归 NEON 版
├── causal_conv1d_update_neon.cpp # depthwise conv1d NEON 版
├── l2norm_inplace_neon.cpp       # L2 归一化 NEON 版
└── rmsnorm_gated_neon.cpp        # 门控 RMSNorm NEON 版
```

声明在 `kernels/gdn_ops.h`（ref 签名）和 `kernels/dispatch.h`（通用入口 + 注册宏）。

## NEON 优化策略

### gdn_step_neon（2.62×）

核心算法三步：衰减 → delta 更新 → 输出读取。

```
Pass 1: S *= decay;  kv_mem = S^T @ k       （必须独立：delta 依赖完整 kv_mem）
Pass 2+3（融合）: S += outer(k, delta);  o = S^T @ q  （同一遍完成更新与输出）
```

关键技术：
- **遍融合**：原三遍各遍历 64KB → 合并为两遍，节省 33% 内存搬运
- **分块 chunk=64**：16 个 float32x4_t 累加器，留寄存器给 vkj/vqj/临时值
- **4× 循环展开**：内层每次处理 16 float，减少循环开销
- **软件预取**：`__builtin_prefetch(S + (j+2)*v_dim, 1, 3)` 隐藏 L2 延迟
- 数值精度：o 误差 ≤1e-4（float 累加 128 维），S 误差 ≤1e-5

### causal_conv1d_update_neon（3.81×）

```
跨通道向量化：4 个通道一组（vld3q state + vld4q weight → dot → SiLU → vst3q shift）
双路 8 通道展开：两组独立 exp 管线重叠执行，隐藏 7-FMA 延迟链
```

关键技术：
- **vld3q_f32 / vst3q_f32**：硬件 deinterleave，完美匹配 state_len=3 布局
- **vld4q_f32**：匹配 kernel_size=4 的权重布局
- **向量化 exp**：6 阶泰勒展开（整数分离指数 + 多项式逼近小数），相对误差 <2e-7

### l2norm_inplace_neon（2.06×）

n=128 时计算量极小，标量 sqrt+div 开销占 37%。

关键技术：
- **vrsqrteq_f32 + 2 步 Newton**：替代 `1.0f/sqrt()`，避免标量/向量域切换
- **4 路累加器**：隐藏 FMA 延迟链
- **16 元素展开**：Pass 2 归一化从 32 次循环降到 8 次

### rmsnorm_gated_neon（3.27×）

关键技术：
- **vrsqrte 计算 scale**（同 l2norm）
- **向量化 SiLU**：vexpq_f32 多项式 + vdivq_f32
- **双路 8 元素展开**：两组 SiLU 管线 OoO 重叠

### Apple Silicon 特殊考量

| 特性 | Apple M 系列 | Cortex-A78 | 影响 |
|------|------------|-----------|------|
| vdivq_f32 延迟 | ~7 cycles | 12-15 cycles | M 系列不需要 vrecpe 替代 |
| L1D 大小 | 128KB (M2+) | 32-64KB | 64KB S 矩阵可能全部命中 L1 |
| vld3q/vld4q | 2-3 μops | 3-5 μops | conv1d 的跨通道策略两者都高效 |

## A/B 测试方法

```bash
# 构建
cmake --build build -j

# 单测门禁（78 tests, 0 failed）
./build/tests/tinyqwen_tests

# A 组（ref 基线）
./build/runtime/tinyqwen --model /tmp/fake35_bench.tqwen \
    --tokens 3,7,11 --max-new-tokens 32 --eos -1 \
    --ops-impl ref --profile-out /tmp/ref.json

# B 组（neon 优化）
./build/runtime/tinyqwen --model /tmp/fake35_bench.tqwen \
    --tokens 3,7,11 --max-new-tokens 32 --eos -1 \
    --ops-impl neon --profile-out /tmp/neon.json

# 对比（Python 提取 op_totals）
python3 -c "import json; ..."
```

测试模型：4 层 fake Qwen3.5（3 GDN + 1 Attention），真实 GDN 维度
（hidden=896, conv_dim=6144, qk_dim=128, v_dim=128, 16 heads）。

## 数值对齐

| 算子 | 容差 | 精度差异来源 |
|------|------|------------|
| l2norm | 1e-5 | float vs double 累加 |
| conv1d output | 1e-5 | exp 多项式逼近 |
| conv1d state | 1e-6 | 精确一致（纯移位） |
| gdn_step output | 1e-4 | float 累加 128 维点积（ref 用 double） |
| gdn_step state | 1e-5 | 乘加顺序差异 |
| rmsnorm_gated | 1e-4 | exp 多项式 + float 平方和 |

## 添加新变体

遵循项目约定（`docs/optimization.md` 第 4 节）：

1. 新建 `kernels/gdn/<op>_<variant>.cpp`
2. `#include "dispatch.h"`，末尾加 `TINYQWEN_GDN_STEP_VARIANT(fn, "variant_name");`
3. `kernels/CMakeLists.txt` 加一行
4. `tests/test_gdn_ops_neon.cpp` 加对应正确性门禁
5. 跑 `./build/tests/tinyqwen_tests` + A/B benchmark
