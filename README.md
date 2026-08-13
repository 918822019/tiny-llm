# tinyqwen

面向单模型（Qwen2.5-0.5B-like decoder-only）的端侧实验 runtime。
第一版目标：**CPU-only、FP32 reference、batch=1、token-by-token decode**。
不追求通用推理框架，不做 graph executor。

> **当前状态**
> - ✅ 已在 macOS 跑通真实 Qwen2.5-0.5B，16 个生成 token 与 HuggingFace 逐位一致；
> - 性能：单线程 fp32 **decode ≈ 220 ms/token**（已落地变体 `double_2_float`，
>   同场 A/B ~1.09–1.12×；当前数字以 `docs/optimization_log.md` 为准）；
> - 已就位：可复现基准（内置同场 A/B + 漂移警告）、优化日志、kernel 分发层
>   （变体自注册）、key=value 配置；
> - 下一步：NEON matvec。
>
> 新手建议先读 [`docs/infra_primer.md`](docs/infra_primer.md)。

## 目录结构

```text
tinyqwen/
├── CMakeLists.txt        # 顶层构建
├── runtime/              # loader / tensor view / kv cache / qwen forward / profiler / CLI
├── kernels/              # reference kernels + 优化变体（每算子一个文件夹，变体自注册）
├── tools/                # Python 侧：exporter / tokenize / reference dump / bench
├── tests/                # 单元测试（无第三方测试框架）
├── scripts/              # 优化 pipeline（verify/bench/record/commit_opt）+ Android 编译运行
├── experiments/          # 预留：run_decode / run_layer_bench 等实验入口
└── docs/                 # 手册与规范（见下方「文档」）
```

详细目录职责、依赖方向见 `docs/project_structure.md`；
kernels/ 与 runtime/ 各有自己的 README 导读。

## 本地编译（Linux / macOS）

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

要求：CMake ≥ 3.16，C++17 编译器，Python 3（仅工具侧）。

Android NDK 交叉编译与真机运行流程见 `docs/android.md`。

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

纪律：优化前先测基线、优化后再测、结果记进日志。正式记录一条命令：

```bash
./scripts/record.sh <label>                        # 门禁 + 稳定测速 + 自动写日志
./scripts/record.sh <label> --extra-args "--matvec-impl <名>"   # 测变体：自动同场 A/B
```

方法论、测量纪律、工具细节全部在 `docs/optimization.md`；
所有历史数字与当前基线在 `docs/optimization_log.md`（唯一权威）。

## CLI 参考

```bash
tinyqwen --model <model.tqwen> [options]
```

| 参数                   | 默认     | 说明                                          |
|----------------------|--------|---------------------------------------------|
| `--model PATH`       | 必填     | .tqwen 权重文件                                 |
| `--tokens CSV`       | 二选一    | 逗号分隔的 token ids                             |
| `--tokens-json PATH` | 二选一    | `tokenize_prompt.py` 输出的 JSON               |
| `--max-new-tokens N` | 16     | 最多生成 token 数                                |
| `--max-seq-len N`    | 1024   | KV cache 容量上限，不得超过 header max_seq_len       |
| `--topk K`           | 0      | 输出 top-k logits 行（0 = 关闭）                   |
| `--dump-logits PATH` | 无      | 每次 forward 后写全量 logits（fp32 binary，按位置顺序逐行） |
| `--profile-out PATH` | 无      | profiler JSON 输出                            |
| `--eos ID`           | 151645 | stop token，-1 禁用                            |
| `--config PATH`      | 无      | key=value 配置文件（见下；CLI 开关优先于它）               |
| `--matvec-impl NAME` | ref    | matvec kernel 实现：任意已注册名（当前 `ref` / `double_2_float`），未知值报错并列出可用 |
| `--verbose`          | 关      | 模型 summary + prefill 细节（stderr）             |

### 配置文件

纯文本 `key = value`（`#` 注释），零依赖、不引 YAML。仓库里带了一份
`tinyqwen.conf`。**优先级：CLI 开关 > 配置文件 > 默认值**——benchmark 时可
随时用命令行覆盖。当前可配：

```text
matvec_impl = ref     # matvec kernel 实现（任意已注册名；变体自注册，见 kernels/dispatch.h）
```

```bash
./build/runtime/tinyqwen --config tinyqwen.conf --model model.tqwen ...
```

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

| 文档                            | 内容                                             |
|-------------------------------|------------------------------------------------|
| `docs/infra_primer.md`        | **infra 新手导读**：内存布局/对齐/字节序/KV cache/RAII 等概念   |
| `docs/optimization.md`        | **优化手册**：kernel 怎么加（自注册）+ 性能怎么测（A/B/纪律）        |
| `docs/optimization_log.md`    | **优化日志**：每次优化改了什么/提升多少/为什么                     |
| `docs/weight_format.md`       | tiny binary format（header / tensor table / 对齐） |
| `docs/qwen_forward.md`        | Qwen forward 数学定义与 shape 约定                    |
| `docs/profiling_schema.md`    | profiler JSON 输出 schema                        |
| `docs/pytorch_alignment.md`   | C++ 与 PyTorch reference 对齐流程                   |
| `docs/android.md`             | Android 端侧：NDK 编译 / adb 运行 / 常见坑               |
| `docs/project_structure.md`   | 目录职责说明                                         |
| `docs/known_limitations.md`   | v1 已知限制                                        |
| `kernels/README.md`           | kernels/ 导读：文件约定、已注册实现、_ref 的意义                |
| `runtime/README.md`           | runtime/ 角色地图、数据流、阅读顺序                         |

按角色的阅读路线：

- **入门**：`infra_primer` → `project_structure` → `runtime/README` → `qwen_forward`
- **做优化**：`optimization`（手册）→ `optimization_log`（历史与数字）→ `kernels/README`
- **上端侧**：`android` → `profiling_schema`
- **改格式/对齐**：`weight_format` → `pytorch_alignment`

## 边界声明

本项目刻意**不做**：通用 graph executor、C++ tokenizer、safetensors C++ parser、
Android App / JNI。研究核心（INT4/KronQ kernel、speculative decoding 策略、
多 LoRA 调度）不在 v1 范围内，v1 只提供正确的 FP32 reference 路径和 profiling 基础设施。
