# 量化算法接入指南

## 当前支持的量化算法

| 算法 | 类型 | 说明 | 导出工具 |
|------|------|------|---------|
| FP32 | `kF32` | 无量化，基线 | `tools/export_qwen_to_tiny.py` |
| FP16 | `kF16` | 半精度浮点 | `tools/export_qwen_to_tiny.py --dtype f16` |
| INT4 (RTN) | `kI4` | Round-to-nearest 量化 | `tools/export_qwen_to_tiny_i4.py` |
| INT4 (HQQ) | `kI4` | Half-Quadratic Quantization | `tools/export_qwen_to_tiny_i4.py --hqq` |

## 接入新的量化算法

### 1. 定义量化类型

在 `quantization/quant.h` 的 `Type` 枚举中添加新类型：

```cpp
enum class Type {
    kF32 = 0,
    kF16 = 1,
    kI4 = 2,
    kGPTQ = 3,  // 新增
};
```

### 2. 实现量化器（Python）

在 `tools/` 下新增导出脚本，例如 `export_gptq.py`：

```python
def quantize_gptq(weight, group_size):
    # 实现 GPTQ 量化算法
    # 返回量化后的数据（bytes）
    pass
```

### 3. 实现后端 kernel（C++）

在 `kernels/matvec/` 下新增 kernel，例如 `matvec_gptq_neon.cpp`：

```cpp
void matvec_gptq_neon(const uint8_t* w, const float* x, float* y,
                     int out_dim, int in_dim, int group_size) {
    // 实现 GPTQ 解包 + matvec
}
```

### 4. 注册 kernel

在 kernel 文件末尾注册：

```cpp
TINYQWEN_MATVEC_VARIANT(matvec_gptq_neon, "gptq_neon");
```

### 5. 更新后端分发

在 `runtime/backend_cpu.cpp` 的 `matvec()` 里添加分支：

```cpp
void CPUBackend::matvec(const WeightTensor& w, ...) {
    if (w.quant_type == QuantType::kGPTQ) {
        matvec_gptq(...);
    } else if (...) {
        // ...
    }
}
```

### 6. 测试

在 `tests/` 下新增测试，验证数值对齐：

```cpp
TEST(gptq_matches_ref) {
    // 对比 GPTQ 量化后的输出 vs FP32 参考
}
```

## 量化参数存储格式

### INT4 (当前实现)

```
[data]      每行：ceil(cols/2) bytes，2 个 4-bit 值 packed
[scale]     每行每组：1 个 fp16
[zero]      每行每组：1 个 fp16
```

### GPTQ（未来）

```
[qweight]   量化后的权重（int4 packed）
[qzeros]    量化零点（int4 packed）
[scales]    量化 scale（fp16）
[g_idx]     group index（int32）
```

## 参考资料

- RTN：Round-to-nearest，最简单的量化
- HQQ：Half-Quadratic Quantization，无需校准数据
- GPTQ：Group-wise Post-Training Quantization，需要校准数据
- AWQ：Activation-aware Weight Quantization，需要激活统计
