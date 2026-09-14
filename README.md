# tinyqwen

面向单模型（Qwen2.5-0.5B / Qwen3.5-0.8B decoder-only）的端侧实验 runtime。
第一版目标：**CPU-only、FP32 reference、batch=1、token-by-token decode**。
不追求通用推理框架，不做 graph executor。

> **当前状态**
> - ✅ 已在 macOS 跑通真实 Qwen2.5-0.5B / Qwen3.5-0.8B / **Qwen3.5-4B**，生成 token
    > 与 HuggingFace 逐位一致；随机权重小模型与 HF eager 对齐 max_abs_err ≈ 1e-6
    > （`tools/align_fake_qwen35_model.py`）；
> - **当前最佳 decode 配方**（i4 HQQ@64，`--matvec-impl sdot4_mt --ops-impl neon`）：
    > Qwen3.5-4B ≈ 36.5 ms/tok、Qwen3.5-0.8B ≈ 8.1 ms/tok（M4）。i4 kernel 阶梯
    > `ref→neon→neon_mt→sdot→sdot2→sdot3→sdot4`（work-stealing + 内联组头硬件
    > FCVT + 128 位解包）；`sdot5` 为对称量化实验变体（配 `--symmetric` 导出）。
    > 数字与归因见 `docs/optimization_log.md`；
> - **VQ2 2-bit 块向量量化 + BiIP 旋转**：块大小 d=4、码本 K=256（2 bit/权重），
    > 查表反量化、字节对齐；旋转在量化前对权重做变换、推理时对激活做配对逆变换
    > （自抵消）。朴素导出器 `tools/export_qwen_to_tiny_vq2.py`（无旋转、精度受限）；
    > 精度可用版走 kronq 完整配方桥接（旋转+GPTQ+TwoPass+K-FAC）。
    > 格式/旋转契约见 `docs/weight_format.md`，路径说明见 `docs/quantization_guide.md`；
> - **批量 prefill（Qwen3.5）**：prompt ≥32 token 时线性投影反量化 fp32 走
    > Accelerate/AMX sgemm（权重每层只读一遍），4B 61-token TTFT 2072→1206ms（1.72×）；
> - **fp16 KV cache（opt-in，`--kv-f16`）**：KV 存 fp16 + 融合 attention（寄存器内
    > 转 fp32），KV 内存减半、长上下文可用；解码慢 ~8%（内存特性非提速）；
> - 已就位：可复现基准（内置同场 A/B + 漂移警告）、优化日志、两套 kernel 分发层
    > （matvec / 非 matvec ops，变体自注册 + 未注册兜底 ref）、key=value 配置、
    > 归因阶梯方法论、机器极限账表（`benchmarks/machine_ceiling/`）、端侧资源采样
    > （见 `docs/android.md` §6）；
> - **后端抽象已就位**：`IBackend` 接口 + CPU 后端（包装 kernel dispatch，主线）；
    > CUDABackend（`--backend cuda`，逐算子，供 A/B）与 GPU-resident engine
    > （`--engine cuda`，整段 **decode** forward 常驻显存，A10 实测 4.89 ms/tok）并存，
    > 见 `docs/architecture.md`。
> - **Apple GPU prefill（`--engine metal`）**：整批 prompt 跑在 Metal/MPS 上，
    > 与 `--engine cuda` 互补（cuda 管 decode，metal 管 prefill）。Qwen3-0.6B f16
    > 实测 vs CPU 满栈：**seq=512 提速 5.01×、seq=256 4.29×、seq=128 3.23×**；
    > 交叉点在 seq≈32（更短不值得走 GPU）。prefill 后把 K/V 写回 KvCache，
    > decode 仍走 CPU 且逐 token 一致。测速 `scripts/bench_metal_prefill.sh`，
    > 归因与证伪记录见 `docs/optimization_log.md`。
> - **Android Vulkan FP16（`--backend vulkan`）**：普通模型提供逐算子 FP16
    > matvec/matmul 正确性路径；DFlash 自动切换为 GPU 常驻的整块 drafter，把
    > 3 层 DFlare、共享 lm_head、Markov 修正和 argmax 合并成每块一次提交。
    > PLK110 / Adreno 840 已通过 208 项真机测试并与 CPU greedy 逐 token 一致；
    > 当前目标模型验证仍走 CPU，因此这是可用的实验后端，尚不能宣称端到端加速。
>
> 新手建议先读 [`docs/infra_primer.md`](docs/infra_primer.md)。

## 目录结构

```text
tinyqwen/
├── CMakeLists.txt        # 顶层构建
├── runtime/              # loader / tensor view / kv cache / 后端抽象 / qwen forward / profiler / CLI
├── kernels/              # reference kernels + 优化变体（每算子一个文件夹，变体自注册）
├── quantization/         # 量化类型元数据（与后端 WeightTensor 对应）
├── tools/                # Python 侧：exporter / 量化导出 / tokenize / reference dump / bench
├── tests/                # 单元测试（无第三方测试框架）
├── scripts/              # 优化 pipeline（verify/bench/record/commit_opt）+ Android 编译运行
├── benchmarks/           # 基线与历史测速数据（baseline*.json / history*.jsonl / jobs / series）
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
   # Qwen2.x（写 v1 格式）
   python tools/export_qwen_to_tiny.py \
     --model Qwen/Qwen2.5-0.5B \
     --out model.tqwen

   # Qwen3.5 混合架构（自动识别 model_type，写 v2 格式；
   # HF 权重需先下载到本地目录，见脚本内 find_local_dir 说明）
   python tools/export_qwen_to_tiny.py \
     --model <本地 Qwen3.5-0.8B 目录> \
     --out model_qwen35.tqwen
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

### 投机解码

目标模型保持最终裁决权，较小的草稿模型先提出 K 个 token；目标模型用一次
continuation batch 验证，接受首个不一致位置之前的前缀，并在拒绝点改用目标
token。当前实现是精确 greedy 语义，包含 KV truncate、Qwen3.5 GDN 快照恢复、
全接受 bonus token 和统计输出：

```bash
./build/runtime/tinyqwen \
  --model target.tqwen \
  --draft-model draft.tqwen \
  --tokens-json prompt_tokens.json \
  --speculative-tokens 4 \
  --max-new-tokens 32 \
  --speculative-stats-out spec.json
```

两个模型必须使用同一 tokenizer（运行时会校验 vocab 和 EOS 元数据）。在 Apple
上可加 `--engine metal`，目标验证走 Metal continuation；其他平台走 CPU。
正确性验收应把 `generated_ids` 与不带 `--draft-model` 的目标模型 greedy 输出逐位比较。
仓库内置的快速门禁会同时覆盖全接受与首拒绝回退：

```bash
./scripts/verify_speculative.sh
```

DFlash / DFlare + Markov checkpoint 使用专用的 `--dflash-model` 路径。它会捕获
Qwen3 目标模型的指定层残差流、并行构造 mask block、应用低秩 Markov 修正，再由
目标模型精确验证：

```bash
.venv/bin/python tools/export_dflash_to_tiny.py \
  --model <dflash-checkpoint-dir> --out draft.tqwen
./build/runtime/tinyqwen --model qwen3_target.tqwen \
  --dflash-model draft.tqwen --tokens-json prompt_tokens.json \
  --speculative-tokens 2 --max-new-tokens 32
```

当前支持 FP16、greedy、单链，以及 CPU 路径 / Android Vulkan GPU-resident
DFlash + target decode；GPU 模式仍用 CPU prefill 作为数值锚点，随后一次导入 KV。
算法、导出、Android 真机结果与性能限制见 `docs/dflash.md`。Android 用法是在同一
命令末尾加 `--backend vulkan`。

Qwen3-0.6B 也支持 SpecForge EAGLE3 target-feature drafter。它从目标模型第
`1,13,24` 层捕获残差流，使用训练时的 next-token shifted embedding 约定生成
greedy proposal chain，并在拒绝后用已确认的 target hidden 重建草稿状态：

```bash
.venv/bin/python tools/export_eagle3_to_tiny.py \
  --model models/SGLang-EAGLE3-Qwen3-0.6B-SpecForge \
  --out eagle3_qwen3_06b_specforge_f16.tqwen
./build/runtime/tinyqwen --model model_qwen3_06b_f16_ctx2048.tqwen \
  --eagle3-model eagle3_qwen3_06b_specforge_f16.tqwen \
  --tokens-json prompt_tokens.json --speculative-tokens 2 --max-new-tokens 64 \
  --matvec-impl neon_mt_kv_nt --ops-impl neon --kv-f16
```

EAGLE3 的 `--speculative-tokens` 是包含 pending root 的验证块宽度，所以宽度 2/4
分别提出最多 1/3 个 token。当前 CPU 路径完整可用；Android `--backend vulkan`
只把 target 放到通用逐算子 Vulkan 后端，drafter 仍在 CPU。转换格式、状态语义、
原始权重对齐和 PLK110 实测见 `docs/eagle3.md`。

## 不需要真模型的验证

用随机权重小模型打通并数值验证整条链路（loader → forward → KV cache →
profiler → PyTorch eager 对齐）：

```bash
# Qwen2.x（full attention）
python tools/make_fake_model.py --out /tmp/fake.tqwen
./build/runtime/tinyqwen --model /tmp/fake.tqwen --tokens 3,7,11,2 \
  --max-new-tokens 8 --max-seq-len 32 --topk 5
python tools/align_fake_model.py        # C++ vs HF Qwen2 逐位置 logits，~1e-7

# Qwen3.5 混合架构（Gated DeltaNet + full attention）
python tools/make_fake_qwen35_model.py --out /tmp/fake35.tqwen --hf-out /tmp/fake35.pt
./build/runtime/tinyqwen --model /tmp/fake35.tqwen --tokens 3,7,11,2 \
  --max-new-tokens 8 --max-seq-len 32 --topk 5
python tools/align_fake_qwen35_model.py # C++ vs HF Qwen3_5 逐位置 logits，~1e-6
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

看一眼数据长什么样（火焰图 / token 时序 / op 占比 / 优化历史趋势，
独立 HTML 零依赖）：

```bash
python tools/visualize.py all profile.json -o viz.html && open viz.html
```

## CLI 参考

```bash
tinyqwen --model <model.tqwen> [options]
```

| 参数                        | 默认     | 说明                                                                          |
|---------------------------|--------|-----------------------------------------------------------------------------|
| `--model PATH`            | 必填     | .tqwen 权重文件                                                                 |
| `--draft-model PATH`      | 无      | 启用精确 greedy 投机解码；草稿模型须与目标模型共用 tokenizer                              |
| `--dflash-model PATH`     | 无      | 启用 DFlare + Markov 块草稿；当前为 Qwen3 稠密目标、FP16、greedy；Android 可用 Vulkan drafter |
| `--eagle3-model PATH`     | 无      | 启用 EAGLE3 target-feature 单链草稿；当前为 Qwen3-0.6B、FP16、greedy                        |
| `--speculative-tokens K`  | 4      | AR 路径为 proposal 数；DFlash/EAGLE3 为包含 pending root 的 verify block 宽度                  |
| `--speculative-stats-out PATH` | 无 | 写接受率、拒绝/回退、目标调用数及 `draft_ms`/`target_verify_ms` 分项耗时 JSON                  |
| `--draft-matvec-impl NAME` | 无 | 单独选择草稿模型 dtype 的 matvec kernel；目标/草稿同 dtype 时两边共享该选择                      |
| `--tokens CSV`            | 三选一    | 逗号分隔的 token ids                                                             |
| `--tokens-json PATH`      | 三选一    | `tokenize_prompt.py` 输出的 JSON                                               |
| `--batch-tokens-jsonl PATH` | 三选一  | 批量模式（数据集测试）：JSONL 每行一条 `{"tokens": [...]}`，需配 `--batch-out`；与 topk/dump-logits/engine/profile-out 互斥 |
| `--batch-out PATH`        | 无      | 批量模式每条 prompt 的 TTFT/decode 时序 JSON                                         |
| `--max-new-tokens N`      | 16     | 最多生成 token 数                                                                |
| `--max-seq-len N`         | 1024   | KV cache 容量上限，不得超过 header max_seq_len                                       |
| `--topk K`                | 0      | 输出 top-k logits 行（0 = 关闭）                                                   |
| `--dump-logits PATH`      | 无      | 每次 forward 后写全量 logits（fp32 binary，按位置顺序逐行）                                 |
| `--profile-out PATH`      | 无      | profiler JSON 输出                                                            |
| `--eos ID`                | 随模型    | stop token：默认取 v2 文件头 eos_token_id（v1 文件用 151645），-1 禁用                    |
| `--config PATH`           | 无      | key=value 配置文件（见下；CLI 开关优先于它）                                               |
| `--matvec-impl NAME`      | ref    | matvec kernel 实现：任意已注册名；f32/f16/i4 三套独立注册表，按模型 dtype 解析，未知值报错并列出可用          |
| `--ops-impl NAME`         | ref    | 非 matvec 算子（rmsnorm/rope/attention/swiglu/argmax + GDN 四算子）：`ref` / `neon` |
| `--backend NAME`          | CPU    | 计算后端：空 = CPU；`cuda` = 逐算子 CUDA；`vulkan` = Android FP16（DFlash 时自动用 GPU 常驻整块 drafter） |
| `--engine NAME`           | 无      | engine：`cuda` = GPU-resident 整段 **decode**（需 CUDA 构建；仅 Qwen2.x + greedy）；`metal` = Apple GPU 整批 prefill，并可用于投机 continuation verify（仅 Apple；两者互斥） |
| `--no-fuse-gate-up`       | 关      | 禁用 FFN gate/up 成对融合（A/B 用）                                                  |
| `--no-fuse-qkv`           | 关      | 禁用 q/k/v 三路融合（A/B 用）                                                        |
| `--kv-f16`                | 关      | KV cache 存 fp16（内存减半，长上下文用；解码慢 ~8%，内存特性非提速）                                    |
| `--no-batch-prefill`      | 关      | 禁用 Qwen3.5 批量 prefill GEMM 路径（A/B 用；默认 prompt≥32 自动启用）                        |
| `--verbose`               | 关      | 模型 summary + prefill 细节（stderr）                                             |

### 配置文件

纯文本 `key = value`（`#` 注释），零依赖、不引 YAML。仓库里带了一份
`tinyqwen.conf`。**优先级：CLI 开关 > 配置文件 > 默认值**——benchmark 时可
随时用命令行覆盖。当前可配：

```text
matvec_impl = ref      # matvec kernel 实现（任意已注册名；变体自注册，见 kernels/dispatch.h）
ops_impl = ref         # 非 matvec 算子实现（ref / neon）
engine =               # decode engine（空 = CPU forward；cuda = GPU-resident）
fuse_gate_up = true    # FFN gate/up 成对融合
fuse_qkv = true        # q/k/v 三路融合
batch_prefill = true   # Qwen3.5 批量 prefill GEMM 路径（prompt≥32 自动启用）
```

注：`--kv-f16`（KV 存 fp16）仅 CLI 开关，不进配置文件。

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

| 文档                          | 内容                                             |
|-----------------------------|------------------------------------------------|
| `docs/infra_primer.md`      | **infra 新手导读**：内存布局/对齐/字节序/KV cache/RAII 等概念   |
| `docs/architecture.md`      | **分层架构**：IBackend 后端抽象 / WeightTensor 量化抽象 / 数据流 |
| `docs/optimization.md`      | **优化手册**：kernel 怎么加（自注册）+ 性能怎么测（A/B/纪律）        |
| `docs/optimization_log.md`  | **优化日志**：每次优化改了什么/提升多少/为什么                     |
| `docs/weight_format.md`     | tiny binary format（header / tensor table / 对齐；INT4 与 VQ2 packing、BiIP 旋转参数） |
| `docs/qwen_forward.md`      | Qwen forward 数学定义与 shape 约定                    |
| `docs/profiling_schema.md`  | profiler JSON 输出 schema                        |
| `docs/pytorch_alignment.md` | C++ 与 PyTorch reference 对齐流程                   |
| `docs/quantization_guide.md`| 量化算法接入指南（新量化类型怎么加；INT4 / VQ2 布局；BiIP 旋转两条路径） |
| `docs/android.md`           | Android 端侧：NDK 编译 / adb 运行 / 常见坑               |
| `docs/dflash.md`          | DFlash / DFlare + Markov：导出、算法、CLI 与 Android 实测  |
| `docs/eagle3.md`          | Qwen3-0.6B EAGLE3：转换、状态语义、正确性与 Android 实测     |
| `docs/project_structure.md` | 目录职责说明                                         |
| `docs/known_limitations.md` | v1 已知限制                                        |
| `kernels/README.md`         | kernels/ 导读：文件约定、已注册实现、_ref 的意义                |
| `runtime/README.md`         | runtime/ 角色地图、数据流、阅读顺序                         |

按角色的阅读路线：

- **入门**：`infra_primer` → `project_structure` → `runtime/README` → `qwen_forward`
- **做优化**：`optimization`（手册）→ `optimization_log`（历史与数字）→ `kernels/README`
- **上端侧**：`android` → `profiling_schema`
- **改格式/对齐**：`weight_format` → `pytorch_alignment`
- **加后端/量化**：`architecture` → `quantization_guide` → `quantization/README`

## 边界声明

本项目刻意**不做**：通用 graph executor、C++ tokenizer、safetensors C++ parser、
Android App / JNI、采样式 speculative decoding、多 LoRA 调度不在 v1 范围内。
v1 提供：正确的多精度推理路径（f32 reference + f16/i4 优化 kernel）、可插拔的
后端/量化抽象（IBackend / WeightTensor）和 profiling + 可复现基准设施。
