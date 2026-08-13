# 目录职责说明

```text
tinyqwen/
├── CMakeLists.txt          # 顶层：C++17、Release 默认、开关测试
├── tinyqwen.conf           # 运行时配置（key=value；CLI 开关优先于它）
│
├── runtime/                # 运行时（静态库 tinyqwen_runtime + 可执行 tinyqwen）
│   ├── tiny_format.h       # .tqwen 二进制格式的 C/Python 共同契约（头文件即规范）
│   ├── tensor.h/.cpp       # TensorView：name/shape/dtype/data 指针，不拥有内存
│   ├── model_loader.h/.cpp # 读取并校验 .tqwen，name -> TensorView 映射（fail fast）
│   ├── kv_cache.h/.cpp     # [n_layers][n_kv_heads][max_seq_len][head_dim] K/V 缓存
│   ├── config.h/.cpp       # key=value 配置解析（零依赖，不引 YAML）
│   ├── qwen_model.h/.cpp   # 固定结构的 Qwen forward_one_token（不做图抽象）
│   ├── profiler.h/.cpp     # ScopedTimer + per-token/per-op 记录 + JSON 输出
│   └── main.cpp            # CLI 入口（prefill + decode 循环）
│
├── kernels/                # kernel 库（静态库 tinyqwen_kernels）
│   ├── ref_ops.h           # 所有 reference kernel 的签名约定（共用）
│   ├── dispatch.h/.cpp     # 分发层：model 调通用入口 matvec_f32()，由它选实现（共用）
│   └── <op>/               # 每个算子一个文件夹：rmsnorm/rope/matvec/softmax/attention/silu/argmax
│       ├── <op>_<dtype>_ref.cpp          # 参考实现（base）
│       └── <op>_<dtype>_<variant>.cpp    # 【改进位】优化版 kernel，只增不删（见 kernel_optimization.md）
│
├── tools/                  # Python 工具（不进 CMake）
│   ├── export_qwen_to_tiny.py   # HF safetensors/bf16 -> .tqwen（fp32）
│   ├── tokenize_prompt.py       # prompt -> token ids JSON
│   ├── dump_qwen_reference.py   # PyTorch 参考值 dump（npz，对齐用）
│   ├── make_fake_model.py       # 随机权重小 .tqwen（冒烟测试，不下载真模型）
│   ├── align_fake_model.py      # C++ vs HF 逐位置 logits 对齐（改 forward 后先跑）
│   └── bench.py                 # 可复现基准：固定负载 + 稳态统计 + 环境记录
│
├── tests/                  # 单元测试，自带最小测试框架（test_framework.h）
├── scripts/                # bench.sh + build/run/pull Android 脚本
├── experiments/            # 预留：run_decode.cpp / run_layer_bench.cpp
└── docs/                   # 规范与流程文档（见下表）
```

## 文档索引

| 文档                                        | 内容                                 |
|-------------------------------------------|------------------------------------|
| `infra_primer.md`                         | infra 新手导读：内存/对齐/字节序/KV cache/RAII |
| `benchmarking.md`                         | 测量方法论 + 多优化组合评测                    |
| `optimization_log.md`                     | 优化日志：每次改了什么/提升多少/为什么               |
| `kernel_optimization.md`                  | 算子优化指南：保留 base、改进版往哪加              |
| `weight_format.md`                        | .tqwen 二进制格式规范                     |
| `qwen_forward.md`                         | Qwen forward 数学定义与 shape 约定        |
| `profiling_schema.md`                     | profiler JSON 输出 schema            |
| `pytorch_alignment.md`                    | C++ 与 PyTorch 对齐流程                 |
| `build_android.md` / `android_runbook.md` | NDK 编译 / adb 运行                    |
| `known_limitations.md`                    | v1 已知限制                            |
| `project_structure.md`                    | 本文件                                |

## 依赖方向

```text
main.cpp ──> tinyqwen_runtime ──> tinyqwen_kernels
     │            │                     │
     │            └─> tiny_format.h <───┘（与 tools/*.py 共享的二进制契约）
     └─> dispatch.matvec_f32() ──> matvec_f32_ref / 将来的优化版
```

## 设计原则

- loader 只负责读，不负责计算；
- TensorView 只保存 pointer / shape / dtype，不做引用计数；
- kernel 输入输出指针由调用方提供，kernel 不分配输入输出内存；
- `_ref` 参考实现是正确性基准，只增不删；优化版走 dispatch 可插拔；
- 不引入 graph / Node / OperatorRegistry / MemoryPlanner 等抽象。
