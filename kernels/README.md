# kernels/ 导读

这里放所有算子（kernel）。结构围绕一条原则：**参考实现（base）永远保留，
优化版只增不删，model 通过分发层选用**。

## 文件约定

```text
ref_ops.h                     # 所有 kernel 的签名契约
<op>_<dtype>_ref.cpp          # 参考实现（base）：正确性基准 + 兜底，永不删
<op>_<dtype>_<variant>.cpp    # 【改进位】优化版，只增不删
dispatch.h / dispatch.cpp     # 分发层：model 只调通用入口，由它选实现
```

命名示例：

| 文件 | 含义 |
|---|---|
| `matvec_f32_ref.cpp` | fp32 参考实现（base） |
| `matvec_f32_neon.cpp` | fp32 + NEON SIMD（以后加） |
| `matvec_f32_neon_mt.cpp` | fp32 + NEON + 多线程（以后加） |
| `matvec_i8_neon.cpp` | INT8 量化 + NEON（以后加） |

## 当前状态

- 已有参考实现：rmsnorm / rope / matvec / softmax / attention / silu / argmax。
- 已接入分发层的算子：**matvec**（热点，优化主攻方向）。其余仍直接调 `_ref`。
- 优化版往哪加、怎么接：见 `../docs/kernel_optimization.md`。
- 怎么测提速：见 `../docs/benchmarking.md`。

## 为什么 `_ref` 很重要

任何优化版在宣称"快"之前，必须先和 `_ref` 对齐（误差在容差内）才算"算对了"。
`_ref` 也是出问题时的兜底——切回它永远有正确结果。
