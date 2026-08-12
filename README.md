# tinyqwen

面向单模型（Qwen2.5-0.5B-like decoder-only）的端侧实验 runtime。
第一版目标：**CPU-only、FP32 reference、batch=1、token-by-token decode**。
不追求通用推理框架，不做 graph executor。

## 目录结构

```text
tinyqwen/
├── CMakeLists.txt        # 顶层构建
├── runtime/              # loader / tensor view / kv cache / qwen forward / profiler / CLI
├── kernels/              # 朴素标量 reference kernels（正确性优先）
├── tools/                # Python 侧：exporter / tokenize / reference dump / profile 分析
├── tests/                # 单元测试（无第三方测试框架）
├── scripts/              # Android NDK 编译 / adb 运行脚本
├── experiments/          # 预留：run_decode / run_layer_bench 等实验入口
└── docs/                 # 格式规范、forward 推导、对齐流程、Android runbook
```

## 本地编译（Linux / macOS）

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

要求：CMake ≥ 3.16，C++17 编译器，Python 3（仅工具侧）。

Android NDK 交叉编译见 `docs/build_android.md`，真机运行流程见 `docs/android_runbook.md`。

## 端到端流程

1. **导出权重**（Python，一次）：

   ```bash
   python tools/export_qwen_to_tiny.py \
     --model Qwen/Qwen2.5-0.5B \
     --out model.tqwen
   ```

2. **生成 token ids**（C++ v1 不内置 tokenizer）：

   ```bash
   python tools/tokenize_prompt.py \
     --model Qwen/Qwen2.5-0.5B \
     --prompt "你好" --chat \
     --out prompt_tokens.json
   ```

3. **运行 decode**：

   ```bash
   ./build/runtime/tinyqwen \
     --model model.tqwen \
     --tokens-json prompt_tokens.json \
     --max-new-tokens 16 \
     --profile-out profile.json
   ```

4. **与 PyTorch 对齐**（数值验收）：

   ```bash
   python tools/dump_qwen_reference.py \
     --model Qwen/Qwen2.5-0.5B \
     --tokens-json prompt_tokens.json \
     --out ref.npz
   ```

   对齐口径和容差见 `docs/pytorch_alignment.md`。

## 不需要真模型的验证

用随机权重小模型打通并数值验证整条链路（loader → forward → KV cache →
profiler → PyTorch eager 对齐）：

```bash
python tools/make_fake_model.py --out /tmp/fake.tqwen
./build/runtime/tinyqwen --model /tmp/fake.tqwen --tokens 3,7,11,2 \
  --max-new-tokens 8 --topk 5 --profile-out /tmp/profile.json
python tools/align_fake_model.py        # C++ vs HF Qwen2 逐位置 logits，~1e-7
```

`--dump-logits PATH` 可导出每步全量 logits（fp32 binary），用于逐位置排查。

## 文档

| 文档 | 内容 |
|---|---|
| `docs/weight_format.md` | tiny binary format（header / tensor table / 对齐） |
| `docs/qwen_forward.md` | Qwen forward 数学定义与 shape 约定 |
| `docs/profiling_schema.md` | profiler JSON 输出 schema |
| `docs/pytorch_alignment.md` | C++ 与 PyTorch reference 对齐流程 |
| `docs/build_android.md` | NDK 交叉编译 |
| `docs/android_runbook.md` | adb push / run / pull 全流程 |
| `docs/project_structure.md` | 目录职责说明 |
| `docs/known_limitations.md` | v1 已知限制 |

## 边界声明

本项目刻意**不做**：通用 graph executor、C++ tokenizer、safetensors C++ parser、
Android App / JNI。研究核心（INT4/KronQ kernel、speculative decoding 策略、
多 LoRA 调度）不在 v1 范围内，v1 只提供正确的 FP32 reference 路径和 profiling 基础设施。
