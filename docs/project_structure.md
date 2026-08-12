# 目录职责说明

```text
tinyqwen/
├── CMakeLists.txt          # 顶层：C++17、Release 默认、开关测试
├── runtime/                # 运行时（静态库 tinyqwen_runtime + 可执行 tinyqwen）
│   ├── tiny_format.h       # tiny binary format 的 C/C++/Python 共同契约（头文件即规范）
│   ├── tensor.h/.cpp       # TensorView：name/shape/dtype/data 指针，不拥有内存
│   ├── model_loader.h/.cpp # 读取并校验 .tqwen，name -> TensorView 映射
│   ├── kv_cache.h/.cpp     # [n_layers][n_kv_heads][max_seq_len][head_dim] K/V 缓存
│   ├── qwen_model.h/.cpp   # 固定结构的 Qwen forward_one_token（不做图抽象）
│   ├── profiler.h/.cpp     # ScopedTimer + per-token/per-op 记录 + JSON 输出
│   └── main.cpp            # CLI 入口
├── kernels/                # 朴素标量 reference kernels（静态库 tinyqwen_kernels）
│   ├── ref_ops.h           # 所有 reference kernel 的签名约定
│   └── *_ref.cpp           # rmsnorm/rope/matvec/softmax/attention/silu/argmax
├── tools/                  # Python 工具（不进 CMake）
│   ├── export_qwen_to_tiny.py   # HF safetensors -> .tqwen flat binary（write_tqwen 为通用写入口）
│   ├── tokenize_prompt.py       # prompt -> token ids JSON
│   ├── dump_qwen_reference.py   # PyTorch 参考值 dump（npz，真机对齐用）
│   ├── make_fake_model.py       # 随机权重小 .tqwen（冒烟测试，不下载真模型）
│   └── align_fake_model.py      # C++ vs HF Qwen2 逐位置 logits 数值对齐（改 forward 后先跑）
├── tests/                  # 单元测试，自带最小测试框架（test_framework.h）
├── scripts/                # build_android.sh / run_android.sh / pull_profile.sh
├── experiments/            # 预留：run_decode.cpp / run_layer_bench.cpp
└── docs/                   # 规范与流程文档
```

依赖方向：

```text
main.cpp -> tinyqwen_runtime -> tinyqwen_kernels
                       \-> tiny_format.h（与 tools/*.py 共享的二进制契约）
```

原则：

- loader 只负责读，不负责计算；
- TensorView 只保存 pointer / shape / dtype，不做引用计数；
- kernels 输入输出指针均由调用方提供，kernel 不分配输入输出内存；
- 不引入 graph / Node / OperatorRegistry / MemoryPlanner 等抽象。
