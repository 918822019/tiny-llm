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

| 文件                                 | 含义                     |
|------------------------------------|------------------------|
| `matvec/matvec_f32_ref.cpp`        | fp32 参考实现（base）        |
| `matvec/matvec_f32_neon.cpp`       | fp32 + NEON SIMD（以后加）  |
| `matvec/matvec_f32_neon_mt.cpp`    | fp32 + NEON + 多线程（以后加） |
| `matvec/matvec_i8_neon.cpp`        | INT8 量化 + NEON（以后加）    |

## 当前状态

- 已有参考实现：rmsnorm / rope / matvec / softmax / attention / silu / argmax。
- 已接入分发层的算子：**matvec**（热点，优化主攻方向），已注册实现：
  `ref`（默认）、`double_2_float`。其余算子仍直接调 `_ref`。
- 优化版往哪加、怎么接、怎么测提速：见 `../docs/optimization.md`（优化手册）。

## 为什么 `_ref` 很重要

任何优化版在宣称"快"之前，必须先和 `_ref` 对齐（误差在容差内）才算"算对了"。
`_ref` 也是出问题时的兜底——切回它永远有正确结果。
