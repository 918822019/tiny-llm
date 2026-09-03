# 目录职责说明

```text
tinyqwen/
├── CMakeLists.txt          # 顶层：C++17、Release 默认、开关测试
├── tinyqwen.conf           # 运行时配置（key=value；CLI 开关优先于它）
├── model*.yaml             # 模型注册表（架构/量化/配方/实测数字；
│                           #   tools/model_registry.py 读取；Qwen2.5 旧档在 docs/archive/）
│
├── runtime/                # 运行时（OBJECT 库 tinyqwen_runtime + 可执行 tinyqwen）
│   ├── tiny_format.h       # .tqwen 二进制格式的 C/Python 共同契约（头文件即规范）
│   ├── tensor.h/.cpp       # TensorView：name/shape/dtype/data 指针，不拥有内存
│   ├── model_loader.h/.cpp # 读取并校验 .tqwen，name -> TensorView 映射（fail fast）
│   ├── kv_cache.h/.cpp     # [n_layers][n_kv_heads][max_seq_len][head_dim] K/V 缓存
│   ├── gdn_state.h/.cpp    # Qwen3.5 GDN 状态：递归矩阵 + conv 窗口（O(1)，与 seq 无关）
│   ├── backend.h           # IBackend 抽象接口 + WeightTensor（模型层不感知 dtype/量化/硬件）
│   ├── backend_cpu.h/.cpp  # CPUBackend：按 quant_type 包装 kernels/dispatch 通用入口（主线）
│   ├── backend_cuda.h/.cpp # CUDABackend：逐算子 CUDA（需 CUDA 构建；A/B 用，非性能路径）
│   ├── config.h/.cpp       # key=value 配置解析（零依赖，不引 YAML）
│   ├── qwen_model.h/.cpp   # create()：权重绑定/校验 + KV/workspace 分配 + 公共逻辑
│   ├── qwen_forward_token.cpp   # forward_token（decode 路径，matvec）
│   ├── qwen_forward_prefill.cpp # forward_prefill（Qwen2.x 批量 GEMM + Qwen3.5 分发）
│   ├── qwen_forward_prefill_qwen35.cpp # Qwen3.5 批量 prefill：prompt≥32 时线性投影
│   │                       #   反量化 fp32 走 Accelerate/AMX sgemm（权重每层读一遍），
│   │                       #   GDN 递归/因果 attention 保留逐 token 顺序扫描
│   ├── profiler.h/.cpp     # ScopedTimer + per-token/per-op 记录 + JSON 输出
│   ├── metal_prefill.h     # Apple GPU prefill 引擎接口（纯 C++，main.cpp 不必碰 ObjC）
│   ├── metal_prefill.mm    # 实现（ObjC++）：GEMM 走 MPS、其余算子走自写 Metal compute
│   │                       #   kernel（shader 源码内嵌）；自持 GPU KV cache，支持接续
│   │                       #   → 投机解码 verify pass。仅 APPLE 构建时编译
│   ├── metal_prefill_stub.cpp # 非 Apple 平台占位（运行期报错而非链接失败）
│   └── main.cpp            # CLI 入口（后端/engine 选择 + prefill + decode 循环）
│
├── kernels/                # kernel 库（OBJECT 库 tinyqwen_kernels）
│   ├── ref_ops.h           # 所有 reference kernel 的签名约定（共用）
│   ├── dispatch.h/.cpp     # 分发层 + 注册表：model 调通用入口 matvec_f32()；
│   │                       # 变体用 TINYQWEN_MATVEC_VARIANT 宏自注册（共用）
│   ├── <op>/               # 每个算子一个文件夹：rmsnorm/rope/matvec/softmax/attention/silu/argmax
│   │   ├── <op>_<dtype>_ref.cpp          # 参考实现（base）
│   │   └── <op>_<dtype>_<variant>.cpp    # 【改进位】优化版 kernel，只增不删（见 optimization.md）
│   ├── gdn/               # Qwen3.5 GDN（Gated DeltaNet）专属算子
│   │   ├── gdn_ops_ref.cpp               # 4 个 ref 实现（l2norm/conv1d/gdn_step/rmsnorm_gated）
│   │   ├── gdn_step_neon.cpp             # delta rule 递归步 NEON（融合遍 + 4× 展开）
│   │   ├── causal_conv1d_update_neon.cpp  # depthwise conv1d NEON（8 通道解交错）
│   │   ├── l2norm_inplace_neon.cpp        # L2 归一化 NEON（vrsqrte + 16 元素展开）
│   │   └── rmsnorm_gated_neon.cpp         # 门控 RMSNorm NEON（双路 exp 展开）
│   └── cuda/              # GPU-resident decode engine（--engine cuda；整段 forward
│       ├── gpu_engine.cu  #   常驻显存、单 stream，与逐算子 dispatch 正交）
│       └── gpu_kernels.cu/.cuh  # engine 内部 kernel（不走 dispatch 注册表）
│
├── quantization/           # 量化抽象层（元数据，不含 kernel）
│   ├── quant.h/.cpp        # 量化类型 Type/group_size/算法名 + 字节数计算
│   └── README.md           # 接入速览（完整版见 docs/quantization_guide.md）
│
├── tools/                  # Python 工具（不进 CMake）
│   ├── export_qwen_to_tiny.py     # HF safetensors/bf16 -> .tqwen（f32/f16，v1/v2 格式）
│   ├── export_qwen_to_tiny_i4.py  # INT4 量化导出（RTN/HQQ，per-group 打包；
│   │                       #   --symmetric 对称量化配 sdot5，--no-lm-head-i4 对照）
│   ├── tokenize_prompt.py         # prompt -> token ids JSON
│   ├── tokenize_batch.py          # 数据集 -> 批量模式 JSONL
│   ├── dump_qwen_reference.py     # PyTorch 参考值 dump（npz，对齐用）
│   ├── make_fake_model.py         # 随机权重小 .tqwen（冒烟测试，不下载真模型）
│   ├── make_fake_qwen35_model.py  # 随机权重 Qwen3.5 混合架构小模型
│   ├── align_fake_model.py        # C++ vs HF 逐位置 logits 对齐（改 forward 后先跑）
│   ├── align_fake_qwen35_model.py # Qwen3.5 版对齐（覆盖 GDN 全路径）
│   ├── verify_i4_accuracy.py      # INT4 精度审计（反量化 vs fp32 逐层 MSE）
 │   ├── bench.py                   # 可复现基准：固定负载 + 稳态统计 + 环境记录
 │   ├── bench_android.py           # Android 端基准（热节流门控 + 同场 A/B）
 │   ├── bench_dataset.py           # 数据集负载 TTFT/TOPT 测试
 │   ├── metal_prefill_qwen3.py     # PyTorch MPS 基线探针：Metal vs CPU prefill 对照
 │   │                       #   （C++ Metal 引擎的可行性验证起点，保留作对照基线）
 │   ├── metal_prefill_scaling.py   # MPS prefill 随 seq 的扩展性探针：判断算力受限
 │   │                       #   还是开销/带宽受限（roofline 归因用）
│   ├── model_registry.py          # model*.yaml 模型注册表读取
│   ├── record_optimization.py     # record.sh 的测速 + 写日志主体
│   ├── visualize.py               # profile/日志 -> 零依赖单文件 HTML
│   ├── web_console.py             # Android 端测 Web 控制台（纯 stdlib）
│   └── webui/                     # web_console/visualize 共享的前端资源
│
├── tests/                  # 单元测试，自带最小测试框架（test_framework.h）
├── scripts/                # 优化 pipeline（verify/bench/record/set_baseline/commit_opt）
│                           #   + Android 全家桶（build/run/pull/doctor/bench/record）
│                           #   + bench_metal_prefill.sh（Metal prefill vs CPU 同场 A/B，带离散度列）
├── benchmarks/             # 测速数据：baseline*.json + history*.jsonl + jobs/ + series/
├── experiments/            # 预留：run_decode.cpp / run_layer_bench.cpp
└── docs/                   # 规范与流程文档（见下表）
```

## 文档索引

| 文档                     | 内容                                      |
|------------------------|-----------------------------------------|
| `infra_primer.md`      | infra 新手导读：内存/对齐/字节序/KV cache/RAII      |
| `architecture.md`      | 分层架构：IBackend 后端抽象 / 量化抽象 / 数据流         |
| `optimization.md`      | **优化手册**：kernel 怎么加（自注册）+ 性能怎么测（A/B/纪律） |
| `optimization_log.md`  | 优化日志：每次改了什么/提升多少/为什么                    |
| `weight_format.md`     | .tqwen 二进制格式规范                          |
| `qwen_forward.md`      | Qwen forward 数学定义与 shape 约定             |
| `profiling_schema.md`  | profiler JSON 输出 schema                 |
| `pytorch_alignment.md` | C++ 与 PyTorch 对齐流程                      |
| `quantization_guide.md`| 量化算法接入指南（新量化类型怎么加 / INT4 布局）            |
| `android.md`           | Android 端侧：NDK 编译 / adb 运行 / 常见坑        |
| `known_limitations.md` | v1 已知限制                                 |
| `project_structure.md` | 本文件                                     |

## 依赖方向

```text
main.cpp ──> QwenModel ──> IBackend ──> CPUBackend ──> dispatch 通用入口 ──> 自注册变体 / _ref 兜底
     │           │             │
     │           │             └─> CUDABackend（逐算子，--backend cuda）
     │           └─> tiny_format.h <── tools/*.py（二进制契约）
     └─> --engine cuda ──> gpu_decode engine（kernels/cuda/gpu_engine.cu，整段 forward）
```

## 当前主线模型与配方（2026-08-24）

| 模型 | 文件 | 量化 | decode 配方 | decode 实测（M4） |
|---|---|---|---|---|
| Qwen3.5-4B | `model_qwen35_4b_i4.tqwen` | i4 HQQ@64 | `sdot4_mt` | 36.5 ms/tok |
| Qwen3.5-0.8B | `model_qwen35_i4.tqwen` | i4 HQQ@64 | `sdot4_mt` | 8.08 ms/tok |
| Qwen3.5-0.8B | `model_qwen35_i4_sym.tqwen` | i4 对称（实验） | `sdot5_mt` | 7.72 ms/tok |
| Qwen3.5-0.8B | `model_qwen35_f16.tqwen` | f16 | `neon_mt_kv_nt` | ~17.9 ms/tok |

- prompt ≥32 token 时自动走**批量 prefill GEMM 路径**（见上文 `qwen_forward_prefill_qwen35.cpp`）。
- **fp16 KV cache（`--kv-f16`，opt-in）**：KV 存 fp16，attention 经 `attention_kv`
  分发到融合 `attention_decode_f16kv_neon`（读 fp16、寄存器内转 fp32）。KV 内存减半、
  长上下文可用，解码慢 ~8%（内存特性非提速）。
- decode 是带宽瓶颈（batch=1 matvec，算术强度低），4B 当前用到持续带宽墙的 ~86%。
- 完整数字与归因见 `optimization_log.md`；机器极限见 `../benchmarks/machine_ceiling/`。

## 设计原则

- loader 只负责读，不负责计算；
- TensorView 只保存 pointer / shape / dtype，不做引用计数；
- kernel 输入输出指针由调用方提供，kernel 不分配输入输出内存；
- `_ref` 参考实现是正确性基准，只增不删；优化版走 dispatch 可插拔；
- QwenModel 只调 IBackend 抽象，不感知 dtype/量化/硬件；量化参数封装在
  WeightTensor 里，由后端自行解析（见 `architecture.md`）；
- 不引入 graph / Node / OperatorRegistry / MemoryPlanner 等抽象。
