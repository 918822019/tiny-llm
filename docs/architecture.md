# tinyqwen 架构

## 分层架构

```
┌─────────────────────────────────────┐
│  Model Layer (QwenModel)            │  # 只关心 forward 逻辑
│  - forward_token / forward_prefill  │
│  - attention / FFN / GDN            │
│  - 不关心 dtype、量化、硬件           │
└──────────────┬──────────────────────┘
               ↓ 调用抽象接口
┌─────────────────────────────────────┐
│  Backend Layer (IBackend)           │  # 抽象后端接口
│  - matvec / matmul                  │
│  - rmsnorm / rope / attention       │
│  - 不关心的 dtype 和量化              │
└──────────────┬──────────────────────┘
               ↓ 分发到具体后端
┌─────────────────────────────────────┐
│  Concrete Backends                  │
│  ├── CPUBackend (NEON/SDOT，主线)     │
│  ├── CUDABackend (已实现，逐算子 A/B)   │
│  ├── MetalBackend (未来)             │
│  └── VulkanBackend (未来)            │
└──────────────┬──────────────────────┘
               ↓ 每个后端支持多种量化
┌─────────────────────────────────────┐
│  Quantization Layer                 │
│  ├── FP32                           │
│  ├── FP16                           │
│  ├── INT4 (RTN / HQQ)               │
│  └── GPTQ / AWQ (未来)               │
└─────────────────────────────────────┘
```

## 目录结构

```
tinyqwen/
├── runtime/
│   ├── backend.h              # IBackend 抽象接口 + WeightTensor
│   ├── backend_cpu.h/cpp      # CPUBackend 实现（包装 dispatch，主线）
│   ├── backend_cuda.h/cpp     # CUDABackend 实现（逐算子，需 CUDA 构建）
│   ├── qwen_model.h           # QwenModel 接口
│   ├── qwen_model.cpp         # create() + 公共逻辑
│   ├── qwen_forward_token.cpp # forward_token（decode 路径）
│   ├── qwen_forward_prefill.cpp # forward_prefill（prefill 路径）
│   ├── model_loader.cpp       # 权重文件加载
│   ├── kv_cache.cpp           # KV cache 管理
│   ├── gdn_state.cpp          # GDN 状态管理（Qwen3.5）
│   └── main.cpp               # CLI 入口（后端/engine 选择）
│
├── kernels/
│   ├── dispatch.h/cpp         # 通用入口 + 自注册
│   ├── ref_ops.h              # 标量参考实现
│   ├── matvec/                # matvec 变体（f32/f16/i4/cuda）
│   ├── rmsnorm/               # RMSNorm 变体
│   ├── rope/                  # RoPE 变体
│   ├── attention/             # attention decode 变体
│   ├── silu/                  # SwiGLU 变体
│   ├── gdn/                   # GDN 算子（Qwen3.5）
│   └── cuda/                  # CUDA 变体
│
├── quantization/
│   ├── quant.h/cpp            # 量化类型元数据
│   └── README.md              # 量化算法接入指南
│
├── tools/                     # Python 工具
│   ├── export_qwen_to_tiny.py       # FP32/FP16 导出
│   ├── export_qwen_to_tiny_i4.py    # INT4 导出（RTN/HQQ）
│   ├── tokenize_prompt.py           # tokenizer
│   ├── bench.py                     # 基准测试
│   └── align_*.py                   # 与 PyTorch 对齐
│
├── tests/                     # 单元测试
│   ├── test_backend.cpp       # 后端接口测试
│   ├── test_quantization.cpp  # 量化层测试
│   └── test_*.cpp             # 其他测试
│
└── docs/                      # 文档
    ├── architecture.md        # 本文档
    ├── quantization_guide.md  # 量化算法接入指南
    └── ...
```

## 关键设计

### 1. 后端抽象（IBackend）

QwenModel 只调用 `IBackend` 接口，不关心具体的硬件实现：

```cpp
class IBackend {
public:
    virtual void matvec(const WeightTensor& w, const float* x, float* y,
                       int out_dim, int in_dim) = 0;
    virtual void rmsnorm(const float* x, const float* weight, float* y,
                        int n, float eps) = 0;
    // ...
};
```

**好处**：
- 新增后端（CUDA/Metal/Vulkan）只需实现 IBackend
- QwenModel 代码不需要修改
- 可以在运行时切换后端

### 2. 量化抽象（WeightTensor）

权重的 dtype 和量化参数封装在 `WeightTensor` 里：

```cpp
struct WeightTensor {
    const void* data;          // 原始数据（fp32/fp16/int4/...）
    QuantType quant_type;      // 量化类型
    int rows, cols;
    int group_size;            // 量化 group size
};
```

**好处**：
- QwenModel 不需要知道权重的具体格式
- 后端自己解析量化参数
- 新增量化算法只需新增 QuantType 和对应 kernel

### 3. Kernel 自注册

每个 kernel 变体在自己的 .cpp 末尾注册：

```cpp
TINYQWEN_MATVEC_VARIANT(matvec_f32_neon, "neon");
```

**好处**：
- 新增 kernel 变体不需要改 dispatch 代码
- 可以按名字选择实现（`--matvec-impl neon`）
- 未注册的变体自动兜底到 ref

### 4. Backend ≠ Engine（两条 GPU 路径，易混淆）

| | `--backend cuda`（CUDABackend） | `--engine cuda`（gpu_engine） |
|---|---|---|
| 形态 | 实现 IBackend，**逐算子**调 CUDA kernel | **整段 forward** 常驻显存、单 stream |
| 数据流 | 每算子 H2D/D2H + cudaMalloc | 权重/激活/KV 常驻，每步仅 4B argmax 过 PCIe |
| 定位 | A/B 测试、单算子调试 | GPU 性能路径（A10 实测 4.89 ms/tok） |
| 覆盖 | INT4/partial RoPE/top_k/GDN 未实现（触发即 abort） | 仅 Qwen2.x + greedy，无 topk/dump-logits |
| 位置 | `runtime/backend_cuda.*` | `kernels/cuda/gpu_engine.cu`（不走 IBackend） |

CPU 主线没有这个区分：CPUBackend 就是唯一路径。

## 数据流

### 导出（Python）

```
HuggingFace 模型
    ↓
tools/export_qwen_to_tiny*.py（量化）
    ↓
.tqwen 文件（量化后的权重）
```

### 运行（C++）

```
.tqwen 文件
    ↓
model_loader.cpp 加载
    ↓
main.cpp 按 --backend 创建后端（默认 CPU），传入 QwenModel::create()
    ↓
forward_token() / forward_prefill()
    ↓
IBackend::matvec() / rmsnorm() / ...（WeightTensor 携带 quant_type）
    ↓
CPUBackend::matvec() → matvec_f32() / matvec_f16() / matvec_i4() / matvec_vq2()
    ↓
具体 kernel（dispatch 按名字分发，未命中兜底 _ref）
```

> **VQ2 + BiIP 旋转**：旋转量化模型（存在 `*.rot_sign`）在 `forward` 里每个量化
> matvec 前先对激活做配对旋转（`mv_rot`/`mm_rot` → `biip_rotate_activation`），
> 再走 `matvec_vq2`/`matmul_vq2` 查表。旋转使 qkv/gate-up 融合失效，旋转模型自动
> 去融合。格式契约见 `weight_format.md`，两条导出路径见 `quantization_guide.md`。

## 扩展指南

### 新增后端

1. 实现 `IBackend` 接口（参考 `backend_cpu.cpp`）
2. 在 `main.cpp` 里添加后端选择逻辑
3. 添加测试

### 新增量化算法

1. 在 `quantization/quant.h` 的 `Type` 枚举中添加新类型
2. 实现 Python 导出工具
3. 实现 C++ kernel
4. 在后端里添加分发逻辑
5. 添加测试

详见 `quantization/README.md`。

## 性能考虑

### 为什么不用 graph executor？

刻意不做。原因：
- 单模型、batch=1，graph 优化收益小
- 手写 forward 更容易调试（与 PyTorch 对齐时）
- 避免过度设计

### 为什么 Kernel 自注册？

- 新增变体不需要改 dispatch 代码
- 可以在运行时按名字选择实现
- 便于 A/B 测试

### 为什么拆分 forward_token 和 forward_prefill？

- forward_token 是 decode 路径（batch=1，matvec）
- forward_prefill 是 prefill 路径（batch>1，GEMM）
- 两者的优化策略不同，拆开更清晰
