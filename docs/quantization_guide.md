# 量化算法接入指南

## 当前支持的量化算法

| 算法 | 类型 | 精度损失 | 压缩比 | 导出工具 |
|------|------|---------|--------|---------|
| FP32 | `kF32` | 无 | 1× | `tools/export_qwen_to_tiny.py` |
| FP16 | `kF16` | 极小 | 2× | `tools/export_qwen_to_tiny.py --dtype f16` |
| INT4 (RTN) | `kI4` | 中等 | 4× | `tools/export_qwen_to_tiny_i4.py` |
| INT4 (HQQ) | `kI4` | 较小 | 4× | `tools/export_qwen_to_tiny_i4.py --hqq` |

## 接入新的量化算法

以 GPTQ 为例，说明接入流程。

### 1. 定义量化类型

在 `quantization/quant.h` 的 `Type` 枚举中添加：

```cpp
enum class Type {
    kF32 = 0,
    kF16 = 1,
    kI4 = 2,
    kGPTQ = 3,  // 新增
};
```

### 2. 实现量化器（Python）

创建 `tools/export_gptq.py`：

```python
import torch
from auto_gptq import AutoGPTQ

def quantize_gptq(model_name, output_path, group_size=128):
    """使用 GPTQ 量化模型"""
    model = AutoGPTQ.from_pretrained(model_name)
    model.quantize(group_size=group_size)
    # 导出为 .tqwen 格式
    # ...
```

### 3. 实现后端 kernel（C++）

创建 `kernels/matvec/matvec_gptq_neon.cpp`：

```cpp
#include "dispatch.h"

namespace tinyqwen {
namespace {

// GPTQ 解包 + matvec
void matvec_gptq_neon(const uint8_t* w, const float* x, float* y,
                     int out_dim, int in_dim, int group_size) {
    // 1. 解包 qweight（int4）
    // 2. 反量化：w_fp16 = (qweight - qzeros) * scales
    // 3. matvec：y = w_fp16 @ x
    // ...
}

} // namespace

TINYQWEN_MATVEC_VARIANT(matvec_gptq_neon, "gptq_neon");
} // namespace tinyqwen
```

### 4. 注册量化类型

在 `runtime/backend_cpu.cpp` 的 `matvec()` 里添加分支：

```cpp
void CPUBackend::matvec(const WeightTensor& w, ...) {
    if (w.quant_type == QuantType::kGPTQ) {
        // 解析 GPTQ 参数
        // 调用 matvec_gptq_*()
    } else if (w.quant_type == QuantType::kI4) {
        matvec_i4(...);
    }
    // ...
}
```

### 5. 更新导出工具

修改 `tools/export_qwen_to_tiny.py`，支持 GPTQ：

```python
parser.add_argument("--quant", choices=["none", "f16", "i4", "gptq"],
                    default="none", help="quantization algorithm")

if args.quant == "gptq":
    from export_gptq import quantize_gptq
    quantize_gptq(args.model, args.out, group_size=args.group_size)
```

### 6. 添加测试

创建 `tests/test_gptq.cpp`：

```cpp
#include "test_framework.h"
#include "backend.h"
#include "backend_cpu.h"

TEST (gptq_matches_fp32) {
    // 1. 加载 FP32 权重
    // 2. 用 GPTQ 量化
    // 3. 对比输出与 FP32 的差异
    // 4. 验证 MSE < threshold
}
```

## 量化参数存储格式

### INT4 (当前实现)

每组（group_size 个元素）一个单元，**组内交错存储**（与
`runtime/tiny_format.h` 的 `kI4Group*` 常量一致）：

```
[scale_fp16 2B][zero_fp16 2B][packed_uint4 group_size/2 B]
```

- 低 nibble 在前：`byte & 0x0F` = 偶数下标元素，`byte >> 4` = 奇数下标元素；
- 反量化：`float_val = (uint4_val - zero) * scale`；
- group_size 典型 64（HQQ@64，已验证 recipe）；格式默认常量 128。

### GPTQ（未来）

```
[qweight]   量化后的权重（int4 packed）
[qzeros]    量化零点（int4 packed）
[scales]    量化 scale（fp16）
[g_idx]     group index（int32）
```

**布局**：分开存储，便于 GPU 并行解包。

## 性能考虑

### 量化算法选择

| 算法 | 精度 | 速度 | 适用场景 |
|------|------|------|---------|
| FP32 | 最高 | 最慢 | 基线 |
| FP16 | 极高 | 快 | GPU/NPU |
| INT4 (RTN) | 中等 | 最快 | 内存受限 |
| INT4 (HQQ) | 较高 | 最快 | 内存受限 + 精度敏感 |
| GPTQ | 高 | 中等 | 需要校准数据 |
| AWQ | 高 | 中等 | 需要激活统计 |

### Group Size 选择

- **group_size=32**：精度最高，但开销大（scale/zero 占 1/16）
- **group_size=64**：平衡点（推荐）
- **group_size=128**：开销最小，但精度下降

## 调试技巧

### 对齐 PyTorch

```bash
# 导出 FP32 参考
python tools/export_qwen_to_tiny.py --model Qwen/Qwen2.5-0.5B --out model_f32.tqwen

# 导出量化版本
python tools/export_qwen_to_tiny_i4.py --model Qwen/Qwen2.5-0.5B --out model_i4.tqwen

# 精度审计：i4 反量化权重 vs fp32 逐层 MSE / max-abs / cosine
python tools/verify_i4_accuracy.py --model-i4 model_i4.tqwen --model-fp32 model_f32.tqwen
```

端到端数值对齐（C++ vs HF logits）走随机权重假模型链路：
`tools/make_fake_model.py` + `tools/align_fake_model.py`。

### 性能分析

```bash
# A/B 测试
./scripts/record.sh gptq_test --extra-args "--matvec-impl gptq_neon"

# 查看 profiler
python tools/visualize.py all profile.json -o viz.html
```

## 参考资料

- [GPTQ 论文](https://arxiv.org/abs/2210.17323)
- [AWQ 论文](https://arxiv.org/abs/2306.00978)
- [HQQ 论文](https://mobiusml.github.io/hqq_blog/)
- [llama.cpp 量化方案](https://github.com/ggerganov/llama.cpp#quantization)
