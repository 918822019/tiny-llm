# 量化算法接入指南

## 当前支持的量化算法

| 算法 | 类型 | 精度损失 | 压缩比 | 导出工具 |
|------|------|---------|--------|---------|
| FP32 | `kF32` | 无 | 1× | `tools/export_qwen_to_tiny.py` |
| FP16 | `kF16` | 极小 | 2× | `tools/export_qwen_to_tiny.py --dtype f16` |
| INT4 (RTN) | `kI4` | 中等 | 4× | `tools/export_qwen_to_tiny_i4.py` |
| INT4 (HQQ) | `kI4` | 较小 | 4× | `tools/export_qwen_to_tiny_i4.py --hqq` |
| VQ2 朴素 (k-means) | `kVQ2` | **大（无旋转，退化）** | 8×（线性层） | `tools/export_qwen_to_tiny_vq2.py` |
| VQ2 完整配方 (旋转+TwoPass) | `kVQ2` | 小 | 8×（线性层） | kronq 仓 `scripts/export_tiny_vq2.py`（GPU） |

## VQ2（2-bit 块向量量化）与 BiIP 旋转

**格式**：块大小 d=4、码本 K=256，每 4 个连续权重共用 1 字节索引，码率 2 bit/权重。
张量布局 = `[码本 [256,4] fp16 = 2048B][索引 uint8]`，详见 `weight_format.md`。
反量化纯查表、字节对齐、免 bit-pack（2.0bit"部署洁净点"）。

**为什么有两条路径**：2-bit 单码本罩不住非平稳权重分布，必须靠 **BiIP 旋转**
（消融：无旋转 −46.6% / 有旋转 −9.4%）。旋转在量化前对权重做变换、推理时对激活做
配对逆变换，自抵消。因此：
- **朴素导出器**（`tools/export_qwen_to_tiny_vq2.py`）：仅 k-means、无旋转。
  免校准、快，但 2-bit 精度差（生成退化）。用于链路/格式验证与基线。
- **完整配方桥接**（kronq 仓 `scripts/export_tiny_vq2.py`，GPU 上跑）：
  BiIP 旋转 + GPTQ 误差补偿 + TwoPass 码本精化 + K-FAC。产出精度可用的 2-bit，
  同时把旋转参数（`*.rot_sign`/`*.rot_scale`）一并写进 `.tqwen`。

**运行时**：`QwenModel` 检测 `*.rot_sign` 判定 `rotated_`；`mv_rot`/`mm_rot` 在
量化 matvec 前对激活施加 `x → blockHadamard((x/scale)⊙sign)`（内核
`kernels/biip/biip_rotate.cpp`）。旋转使 q/k/v、gate/up 融合失效，旋转模型自动
去融合逐子层投影；非旋转模型路径不受影响。

**验证手段**（都在 `tools/`）：
- `add_biip_rotation.py`：给已有模型注入旋转，做**自抵消测试**（旋转前后输出须一致）。
- `make_rotated_vq2_fake.py`：合成"旋转+VQ2"假模型，端到端验证格式可加载运行。
- 单元：`tests/test_biip_rotate.cpp`（Hadamard）、`tests/test_matvec_vq2.cpp`（VQ2）。

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

## i4 导出器的性能设计（2026-08 重构）

大模型（4B+）的 i4 导出曾耗时 ~40 分钟，重构后降到 ~10 分钟量级。关键设计：

1. **打包向量化**：nibble 打包用 numpy 三维视图一次组装（`pack_i4_groups`），
   替代三重纯 Python 循环。lm_head 量级张量从 ~10 分钟降到秒级。
2. **RTN 向量化**：min/max/scale/zero 按组轴批量计算，与旧逐组循环**逐位一致**
   （取整链 round→int32→clip→uint4 完全相同，含 fp16 下溢/退化组边界）。
3. **流式两遍写盘**：第一遍取 shape 定布局，第二遍边加载边量化边写——
   内存峰值只与在途任务数相关，不再随模型大小线性膨胀（旧版一次性持有
   全模型 fp32，4B ≈ 15GB，16GB 机器靠交换硬撑）。
4. **并行量化**（`--workers`，默认自动 = 核数/3）：进程池量化 + "鲸鱼优先"
   提交（lm_head 最先开跑，与其余张量重叠，消除串行长尾）。
5. **行拆分（鲸鱼张量）**：超大张量按行拆成 ~64M 元素块分给多 worker。
   必须拆的两个理由：a) 消除 lm_head 串行长尾；b) 内存——HQQ 整块量化
   lm_head（635M 元素）单 worker 峰值 ~10GB（fp32 输入 + torch 副本 +
   unpack 临时区），16GB 机器实测 OOM 被杀。RTN 拆分逐位可证；HQQ 拆分
   有 fp16 ULP 级漂移（torch 归约顺序依赖行数）：全文件仅几十字节差异，
   scale/q 一致、zero 偶有 ±1 ULP，质量等价（ULP ≪ 量化步长）。

正确性回归手段（改导出器必跑）：

```bash
# 1. 新旧实现逐字节对照自测（随机数据 + 边界 + 拆分一致性）
python tools/selftest_export_i4.py

# 2. 真实模型字节级回归：重导出与既有文件 cmp（同时验证 HQQ 确定性）
#    注意必须 --workers 1：并行 worker 内 torch 线程数不同，归约累加顺序
#    会变，产生 fp16 ULP 级差异（全文件仅几十字节，质量等价但字节不同）。
#    workers>1 的导出自身可复现（同 workers 配置逐位一致），只是与串行
#    参考存在 ULP 差。
python tools/export_qwen_to_tiny_i4.py --model models/Qwen3.5-0.8B \
    --out /tmp/re.tqwen --method hqq --group-size 64 --workers 1
cmp /tmp/re.tqwen model_qwen35_i4.tqwen
```

已知坑：
- **HQQ 鲸鱼张量的 ULP 漂移**：并行拆分路径下 lm_head 输出与串行整块参考
  存在几十字节的 ULP 级差异（质量等价）。需要字节级复现时统一用
  `--workers 1`（串行路径不拆分、torch 默认线程，逐位可复现）。
- **worker 内存峰值**：HQQ 量化的单 worker 峰值 ≈ 张量 fp32 × 4~5 倍
  （输入 + torch Linear 副本 + unpack float32 临时区 + HQQ 内部量），
  这是鲸鱼必须行拆分的硬约束，调大 SHARD_TARGET 前先算内存账。

实测（M4，Qwen3.5-0.8B HQQ g64）：重构前 ~5 分钟 → 串行 86s → 并行 77s。
4B 量级收益更大（打包循环从 ~15 分钟降到 ~1 分钟，层间量化再吃并行收益）。

## 参考资料

- [GPTQ 论文](https://arxiv.org/abs/2210.17323)
- [AWQ 论文](https://arxiv.org/abs/2306.00978)
- [HQQ 论文](https://mobiusml.github.io/hqq_blog/)
- [llama.cpp 量化方案](https://github.com/ggerganov/llama.cpp#quantization)
