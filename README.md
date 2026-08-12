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
├── tools/                # Python 侧：exporter / tokenize / reference dump / 数值对齐验证
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

## 性能基准与优化记录

优化前先测基线、优化后再测、把结果记进日志——这是本项目的纪律
（详见 `docs/benchmarking.md`）：

```bash
./scripts/bench.sh fp32-baseline      # 跑标准负载，输出 decode ms/token 等统计
```

每个优化单独一个 commit，测完把一行记录追加到 `docs/optimization_log.md`。
当前基线：**fp32 单线程 decode ≈ 230 ms/token**（commit `5f679ea`）。

优化会逐步叠加（NEON、多线程、量化……）。叠加时加速比**不能简单相乘**，
要同时记"vs 原始基线"和"vs 上一配置"两个数——组合评测方法见
`docs/benchmarking.md` 第 7–9 节。

## CLI 参考

```bash
tinyqwen --model <model.tqwen> [options]
```

| 参数 | 默认 | 说明 |
|---|---|---|
| `--model PATH` | 必填 | .tqwen 权重文件 |
| `--tokens CSV` | 二选一 | 逗号分隔的 token ids |
| `--tokens-json PATH` | 二选一 | `tokenize_prompt.py` 输出的 JSON |
| `--max-new-tokens N` | 16 | 最多生成 token 数 |
| `--max-seq-len N` | 1024 | KV cache 容量上限，不得超过 header max_seq_len |
| `--topk K` | 0 | 输出 top-k logits 行（0 = 关闭） |
| `--dump-logits PATH` | 无 | 每次 forward 后写全量 logits（fp32 binary，按位置顺序逐行） |
| `--profile-out PATH` | 无 | profiler JSON 输出 |
| `--eos ID` | 151645 | stop token，-1 禁用 |
| `--verbose` | 关 | 模型 summary + prefill 细节（stderr） |

stdout 输出语义（面向脚本化）：

```text
topk 13:1.253097 25:1.182579   # logits 分布；第一行在 prefill 结束后输出
gen 0 13                       # 生成 token g0；每个 topk 行描述下一个 gen 行的 token
...
generated_ids: 13 13 13 13     # 末尾汇总全部生成 ids
```

注意：第一个生成 token g0 = 最后一个 prompt 位置 logits 的 argmax，在 prefill
阶段产生；decode 每步消费上一个生成 token。进度/报错信息走 stderr。

## 文档

| 文档 | 内容 |
|---|---|
| `docs/infra_primer.md` | **infra 新手导读**：内存布局/对齐/字节序/KV cache/RAII 等概念 |
| `docs/benchmarking.md` | **测量方法论**：怎么科学测性能、怎么归因 |
| `docs/optimization_log.md` | **优化日志**：每次优化改了什么/提升多少/为什么 |
| `docs/kernel_optimization.md` | **算子优化指南**：保留 base、改进版往哪加 |
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
