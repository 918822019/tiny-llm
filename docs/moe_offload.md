# MoE 动态路由 + SSD 专家卸载

## 命题

Qwen3.5-35B-A3B（256 路由专家、top-8、1 共享专家，激活 ~3B/总 35B）的 Int4
权重约 17.5GB，装不进端侧内存。但每个 token 只激活 k 个专家。把**路由专家
权重留在 .tqwen 文件（SSD），只有被路由门选中的专家才 pread 进一个固定大小
的 LRU 缓存参与计算**；未激活专家永远不进 RAM。attention / 路由门 / 共享专家 /
embed / norm 常驻内存（resident）。

## 架构

支持两种 MoE 拓扑（`ModelType`）：

| ModelType | attention | 共享专家 | 对应 HF model_type | 真模型 |
|---|---|---|---|---|
| `kQwen35MoE` (2) | GDN + full 3:1 混合 | **有** | `qwen3_5_moe` | Qwen3.5-35B-A3B |
| `kQwen3MoE` (3) | 全层 full attention | **无**（`n_shared_experts=0`） | `qwen3_moe` | Qwen3-30B-A3B |

两者都是"既有 attention 代码复用 + MoE FFN 替换 dense SwiGLU FFN"。每层 FFN：

```
有共享专家： post_attn_norm → router_gate → topk_softmax → 共享专家 + Σ 路由专家
无共享专家： post_attn_norm → router_gate → topk_softmax →              Σ 路由专家
                                                        （ffn_acc 从 0 起算）
```

`kQwen3MoE` 的 `full_attention_interval` 必须为 0 或 1（`is_linear_layer()` 的
`<= 1` 守卫据此判定"全层 full attention"）。注意 `full_layer_cache_index()` 原先
缺这个守卫会除零，已补（见 AGENTS.md 坑 #16）。

**prefill 路径的坑**：MoE 的逐 token 分支必须放在架构判断**之前**。原先它在
`if (is_qwen35)` 里面，而 `kQwen3MoE` 不属于 `is_qwen35`，会掉进 Qwen2 的批量
prefill 路径 —— 那条路径算的是 dense SwiGLU 而非 MoE，**不报错但结果全错**。

路由专家权重访问二选一（`--moe-ssd` 开关）：
- **resident 模式**（默认，正确性锚点）：ModelFile 整文件读入，专家权重直接绑自
  内存指针。与既有 dense 路径同源，是正确性 oracle。
- **SSD 模式**：ModelFile 走**稀疏加载**——`data_` 只含 header + tensor 表 +
  resident tensor，路由专家的字节**一整个不进 RAM**（其 `TensorView.data` 为
  `nullptr`，只记 `file_offset`/`nbytes`）。forward 经 `ExpertStore`（
  `runtime/expert_store.h`）pread + LRU 槽缓存按需取。两者须逐位一致（见验证）。

> 稀疏加载是"未激活专家永远不进 RAM"这一命题真正成立的前提。没有它，即便走
> ExpertStore 的 pread 路径，整文件缓冲也会让 17.5GB 专家权重全量常驻 ——
> pread 只是在演示机制，内存一点没省。

- **真 MoE GPTQ 导出器**（`tools/export_qwen_moe_to_tiny.py`）：从 HF/AutoGPTQ
  checkpoint 导出，repack 成 in-band GPTQ 块（qzeros int4 解包 + zp+1）。支持
  稠密 MoE（`qwen3_moe`，无共享专家）与混合架构 MoE（`qwen3_5_moe`）。
  Qwen3-30B-A3B-GPTQ-Int4 实测导出 18867 tensors / 17.16 GB，其中 18432 个专家
  张量走卸载。逐块校验与 `repack_gptq_from_hf` 输出逐位一致。

## 数据流（decode，单 token，MoE FFN 段）

```
mv(router, normed) → gate_logits[n_experts]
topk_softmax(gate_logits, k) → topk_idx[k], topk_w[k]      # backend_->topk_softmax
mv_pair(shared_gate, shared_up, normed) → swiglu → mv(shared_down) → shared_out[hidden]
ffn_acc = shared_out
for t in 0..k:
    eg,eu,ed = moe_ssd ? expert_store.get(layer, topk_idx[t])   # 命中 or pread
                       : w.moe_experts[topk_idx[t]]              # resident 指针
    mv_pair(eg, eu, normed) → swiglu → mv(ed) → expert_out[hidden]
    ffn_acc += topk_w[t] * expert_out
residual += ffn_acc
```

## 关键组件

- **稀疏加载**（`runtime/model_loader.cpp`，`load(path, err, offload_experts)`）：
  卸载判据是名字含 `.mlp.experts.`（路由门 `mlp.gate.weight`、共享专家
  `mlp.shared_experts.*` 都不含该子串，天然区分）。resident tensor 紧凑打包进
  `data_`，每个起点仍 64B 对齐（kernel 有按对齐选路的分支，丢对齐会静默走慢路）。
  专家 tensor 的 in-band GPTQ magic 仍校验，但只 pread 前 4 字节——保住 fail fast
  又不把整个专家读进内存。归因访问器：`resident_bytes()` / `offloaded_bytes()` /
  `offloaded_count()` / `file_bytes()`。
- **ExpertStore**（`runtime/expert_store.{h,cpp}`）：持有 .tqwen fd + (layer,expert)
  → 文件 offset 映射 + LRU 槽缓存。`get()` 命中返零拷贝指针；未命中淘汰 LRU +
  pread 三块（gate/up/down）。`pread` 而非 `mmap`——mmap 的 readahead 会把未激活
  专家也读进 page cache，违背"只读激活"命题。统计 hits/misses/evictions/bytes_read。
  offset 取自 `TensorView::file_offset`（稀疏紧凑重排后 `(data - base())` 不再等于
  文件内偏移，指针算式会算错）。
- **原生 GPTQ dequant**（`kernels/matvec/matvec_gptq_ref.cpp`）：AutoGPTQ 列主序
  int32 打包（qweight/qzeros/scales/g_idx），in-band 存（仿 VQ2 码本哲学，
  `WeightTensor` 不加字段）。dispatch 族 `matvec_gptq`（`--matvec-impl` 选，默认 ref）。
- **topk_softmax**（`kernels/topk_softmax/topk_softmax_ref.cpp`）：MoE 路由门选 k +
  softmax 归一。ops 分发（`--ops-impl`）。
- **fake MoE 模型**（`tools/make_fake_qwen35_moe_model.py`）：极小拓扑（hidden=16，
  8 专家，top-2，GPTQ group=8，GDN+full 3:1），全 GPTQ 投影 + fp32 embed/norm。

## dtype 路由（真模型才暴露的坑）

`QwenModel::mv()` 原先一律用 `quant_type_of(dtype_)`（文件级 master dtype）。
GPTQ MoE 模型里 **router（`mlp.gate.weight`）与非 tied 的 `lm_head` 是 fp32，
而 master 是 kGPTQ4** —— fp32 数据被当 GPTQ in-band 块解析，router logits 全错
→ 选错专家 → 输出完全乱码，**且不报任何错**。

修法：新增 `mv_typed(w, x, y, out_dim, in_dim, dtype, group_size)`，绑定时记录
张量自身 dtype（`moe_router_dtype_` / `lm_head_dtype_`），forward 用它路由。
注意 `lm_head_is_f32_/f16_/i4_` 那组标志**只在 tied 分支设置**（`lm_head_ = embed_`
时按 embed dtype 路由），非 tied 会落到 `mv()` —— 这是同一类 bug 的第二处。

**为什么 fake 模型测不出**：fake 生成器把 router 也做成了 GPTQ
（`add_quant(p + "mlp.gate.weight", ...)`），恰好掩盖。而
`align_fake_qwen35_moe_model.py` 只比 **resident vs SSD（两边都是 C++）**，从不
与独立参考比 —— 所以 MoE FFN 的数值正确性此前从未被验证过。

## 真模型验证（独立参考对齐）

fake 模型只能证明"两条 C++ 路径一致"，证不了"算对了"。真模型对齐必须比**独立
参考实现**。技巧：**单 token + pos=0** 让参考前向退化成极简形式——

- RoPE 是恒等变换（pos=0）
- attention 只有一个位置，softmax 权重必为 1，输出 = `v[0]`
- 于是只需 rmsnorm + 一次 GPTQ dequant matvec + topk + 专家 FFN + lm_head

```bash
# 导 1 层调试模型，跑 C++ 拿 logits
./build/runtime/tinyqwen --model /tmp/moe_1l.tqwen --tokens 9707 \
  --max-new-tokens 0 --max-seq-len 8 --moe-ssd --moe-expert-cache-slots 8 \
  --dump-logits /tmp/c1.bin
# numpy 参考前向逐位比（见 tools/validate_gptq_numeric.py 同款 dequant 逻辑）
```

结果：ref argmax 4182 == C++ argmax 4182，**CosSim = 1.00000012**，
max|diff| = 3.81e-06（fp32 舍入量级）。

48 层端到端生成验证：

```bash
./build/runtime/tinyqwen --model model_qwen3_30b_moe_i4.tqwen \
  --tokens 9707,11,358,1229 --max-new-tokens 16 --max-seq-len 64 \
  --moe-ssd --moe-expert-cache-slots 16
# generated_ids: 13 358 2776 4460 311 11625 419 3491 25 362 220 16 15 15 15 20972
# decode: 'Hello, Ining' → ". I'm trying to solve this problem: A 1000 kg"
```

> **读 GPTQ 张量的坑**：`safetensors.torch.load_file` 后 `.float()` 会把 int32
> 的 `qweight/qzeros/g_idx` 转成 float32，再 `.view(np.uint32)` 就是把浮点位模式
> 当整数读 → 全是垃圾（qzeros 本该恒为 7，算出 mean=13.88）。必须用
> `safe_open(framework='numpy')` 保留 dtype。

## 验证（正确性锚点：resident vs SSD 逐位一致）

```bash
.venv/bin/python tools/make_fake_qwen35_moe_model.py --out /tmp/fake_moe.tqwen
# resident
./build/runtime/tinyqwen --model /tmp/fake_moe.tqwen --tokens 3,7,11,2 \
  --max-new-tokens 8 --max-seq-len 32 --matvec-impl ref --ops-impl ref \
  --dump-logits /tmp/lr.bin
# SSD 全 miss
./build/runtime/tinyqwen --model /tmp/fake_moe.tqwen --tokens 3,7,11,2 \
  --max-new-tokens 8 --max-seq-len 32 --matvec-impl ref --ops-impl ref \
  --moe-ssd --moe-expert-cache-slots 0 --dump-logits /tmp/l0.bin
cmp /tmp/lr.bin /tmp/l0.bin && echo "resident == ssd0 ✓"
```
或一键：`.venv/bin/python tools/align_fake_qwen35_moe_model.py`。

两种 SSD 模式（slots=0 全 pread / slots=N 缓存）的 logits 与 resident **逐字节
一致**——证明 pread 加载的专家与常驻指针计算的专家结果相同。

### 内存真降的证明（稀疏加载）

logits 一致只证明**算对了**，不证明**内存省了**。后者看启动归因：

```bash
./build/runtime/tinyqwen --model /tmp/fake_moe.tqwen --tokens 3,7,11,2 \
  --max-new-tokens 2 --max-seq-len 32 --matvec-impl ref --ops-impl ref \
  --moe-ssd --verbose
# [init] weights: resident 0.04 MB, offloaded 96 tensors / 0.02 MB on disk (file 0.07 MB)
# print_summary 里被卸载的 tensor 行尾标 [SSD offloaded]
```

fake 模型：文件 70144 B → resident 39424 B，96 个专家 tensor / 25344 B 留盘
（差额 5376 B 是 header + tensor 表 + 对齐填充，账对得上）。真 35B 下 resident
只剩 attention/router/共享专家/embed，占比会降到个位数百分比。

单测护栏（`tests/test_loader.cpp`）：
- `loader_sparse_offloads_experts`：专家 `data == nullptr` 但 `file_offset`/`nbytes`
  正确、`resident_bytes() < file_bytes()`、常驻 tensor 仍 64B 对齐
- `loader_sparse_resident_bytes_identical`：两种加载方式同名 tensor 逐字节相同
- `loader_full_keeps_entire_file`：`offload_experts=false` 时整文件驻留、零卸载
  （非 MoE 路径与 resident 正确性锚点的回归护栏）

## CLI

| 参数 | 说明 |
|---|---|
| `--moe-ssd` | 路由专家走稀疏加载 + ExpertStore pread+LRU（专家字节不进 RAM）；默认 resident 指针 |
| `--moe-expert-cache-slots N` | LRU 槽数（0 = 每次现 pread；默认 4） |

运行结束 stderr 输出 `[moe] expert cache: hits=.. misses=.. evictions=.. bytes_read=..`；
启动时输出 `[init] weights: resident .. offloaded ..`。

## 性能现实（真模型实测，M4 / 16GB）

**原预测被证伪。** 本文档早期版本推测"decode 受 I/O 主导，最坏全 miss 数百
MB/token → NVMe 下数百 ms/token"。真模型实测推翻了它——见下表。教训同
AGENTS.md 坑 #11：**不能从理论倒推组件占比，必须隔离测量**。

模型：Qwen3-30B-A3B-GPTQ-Int4（48 层 / 128 专家 / top-8 / 17.16 GB 文件）。
负载：prompt 7 tok + decode 8 tok，`--profile-out` 归因，单遍。

### 内存边界（命题成立）

```
[init] weights: resident 2883.98 MB, offloaded 18432 tensors / 14688.14 MB on disk
                (file 17573.10 MB)
```

| | |
|---|---|
| 文件大小 | 17573 MB |
| **resident（真进 RAM）** | **2884 MB（16.4%）** |
| offloaded（永留 SSD） | 14688 MB（83.6%）/ 18432 tensors |
| 物理内存 | 16 GB → **17.16 GB 文件装不下，不开稀疏加载无法运行** |

命题成立：只有 attention / router / embed / norm / lm_head 常驻，18432 个
专家张量的字节一个都没进 RAM。

### I/O 不是瓶颈（关键反直觉结论）

同场 A/B，其余参数完全相同，只改 LRU 槽数：

| 配置 | total | expert_load（I/O） | bytes_read |
|---|---|---|---|
| `slots=0`（每次访问都 pread） | 132.2 s | **9.0 s（6.8%）** | 13.48 GB |
| `slots=4096`（首轮后全命中） | 140.2 s | **4.0 s（2.9%）** | 5.05 GB |

少读 **8.43 GB** 反而**更慢 8 s**。消除全部 I/O 换来的收益是零（甚至负）。

profiler 归因（slots=0）：

| op | 耗时 | 占比 | 类别 |
|---|---|---|---|
| `expert_ffn` | 77.4 s | **58.5%** | 计算 |
| `qkv_proj` | 23.6 s | 17.8% | 计算 |
| `o_proj` | 18.9 s | 14.3% | 计算 |
| `expert_load` | 9.0 s | **6.8%** | **I/O** |
| `lm_head` | 3.1 s | 2.3% | 计算 |
| `moe_router` | 0.2 s | 0.2% | 计算 |

**计算占 90.6%，I/O 占 6.8%。** 瓶颈是标量 GPTQ matvec，不是磁盘。

### Roofline 与瓶颈定位

每 token MAC 数（hidden=2048, moe_inter=768, 48 层, top-8, vocab=151936）：

| 段 | GMAC/token | 占比 |
|---|---|---|
| MoE FFN | 1.812 | 70.3% |
| attention | 0.453 | 17.6% |
| lm_head | 0.311 | 12.1% |
| **合计** | **2.576** | |

实测 decode = 9.40 s/tok → **274 MMAC/s = 0.548 GFLOPS**。

M4 10 核 NEON fp32 上限 ≈ 270 GFLOPS，即**当前只用 0.20%**。根因：
`kernels/matvec/` 下 GPTQ 变体族**只有一个实现** `matvec_gptq_ref`（标量参考）。
AGENTS.md 坑 #1 的 `sdot4_mt` 是给 HQQ/i4 interleaved 打包用的，**不适用于
GPTQ 列主序布局**——`--matvec-impl` 的可用列表里没有 GPTQ 优化变体。

标量 GPTQ 每个 MAC 要做：nibble 提取（移位+掩码）→ 减零点 → 乘 scale → FMA，
约 5 op/MAC 且无 SIMD 无多线程。

### 预估（用仓库已实测 kernel 锚定，非理论上限）

锚点：Qwen3.5-4B i4 实测 3.74 GMAC/token @ 36.5 ms/tok = **102.5 GMAC/s
（205 GFLOPS，NEON 上限的 76%）**。

| 假设 | 吞吐 | decode | 加速 |
|---|---|---|---|
| 乐观（同 4B 吞吐） | 102.5 GMAC/s | 25.1 ms/tok | 374× |
| 保守（打 3 折） | 30.7 GMAC/s | 83.8 ms/tok | 112× |
| 悲观（打 1 折） | 10.2 GMAC/s | 251.4 ms/tok | 37× |
| **现状** | **0.274 GMAC/s** | **9400 ms/tok** | 1× |

MoE 的专家矩阵小（`[768,2048]`=0.8 MB），比 4B 的大矩阵更难喂饱 10 核，且每
token 384 次独立 dispatch，真实值大概率落在**保守～悲观区间**。

**结论：GPTQ NEON+MT kernel 是唯一的数量级优化机会，覆盖 90.6% 的运行时。**
异步预取（Phase B）在当前瓶颈下收益 <4%，优先级应下调。

### TTFT

prefill 7 tok = 66.4 s（≈9.4 s/tok）。MoE 走**逐 token** prefill（批量 GEMM
的 gather/scatter 未实现，Phase C）。长 prompt 会线性放大。

## 已交付与后续

### ✅ Phase A：机制（fake 模型验证）
格式 v3 + GPTQ dequant + topk_softmax + ExpertStore（pread+LRU）+ MoE forward
+ fake 模型 + 对齐（resident vs SSD 逐位一致）+ **稀疏加载**（专家字节真不进
RAM，内存归因与单测护栏）。

### ✅ Phase A+：真模型端到端（本轮完成）
- `tools/export_qwen_moe_to_tiny.py`：真 MoE GPTQ 导出器，支持 dense-attention
  MoE（无共享专家）+ AutoGPTQ 张量 repack（qzeros int4 解包 + zp+1）。
  48 层全量导出 18867 tensors / 17.16 GB。
- `kQwen3MoE` 架构支持 + 共享专家可选 + prefill 分支顺序修正 +
  `full_layer_cache_index()` 除零守卫。
- **matvec 按张量自身 dtype 路由**（见下「dtype 路由」）。
- 正确性验收：单 token pos=0 logits **CosSim = 1.00000012**（max|diff| 3.81e-06），
  48 层生成连贯文本。

### ⏳ Phase B：I/O 优化 —— **优先级已下调**
异步预取 overlap、`bench_expert_store`、cache 抖动归因。

**下调理由**：实测 I/O 只占 6.8%，消除 8.43 GB 读取换来零收益（见「性能现实」）。
异步预取的收益上限 <4%。除非未来专家矩阵变大或改用更快的计算 kernel
（届时 I/O 占比会相对上升），否则不值得投入。

### ⏳ Phase C：**GPTQ 优化 kernel（当前唯一数量级机会）**
覆盖 90.6% 运行时，预估 37–374× 加速。具体：
1. **GPTQ NEON+MT matvec**（最高优先级）：GPTQ 变体族现在只有 `matvec_gptq_ref`
   标量实现。可参考 HQQ 侧的 `sdot4_mt` 阶梯（work-stealing + 内联组头硬件
   FCVT + 128 位解包），但须适配 AutoGPTQ 列主序布局。
2. **MoE 批量 prefill**：同批 token 路由到不同专家的 gather/scatter GEMM，
   解决 TTFT 66.4 s / 7 tok 的线性放大。
3. **多线程专家并行**：top-8 专家的 FFN 彼此独立，天然可并行。
