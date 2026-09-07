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
[init] weights: resident 1079.07 MB, offloaded 18433 tensors / 15281.64 MB on disk
                (file 16361.70 MB)
```

| | |
|---|---|
| 文件大小 | 16362 MB |
| **resident（真进 RAM）** | **1079 MB（6.6%）** |
| offloaded（永留 SSD） | 15282 MB（93.4%）/ 18433 tensors |
| 物理内存 | 16 GB → **16.36 GB 文件装不下，不开稀疏加载无法运行** |

命题成立：只有 attention / router / norm / lm_head 常驻，18433 个卸载张量
（专家 + embed_tokens）的字节一个都没进 RAM。

#### resident 构成与两个降内存杠杆

resident 曾高达 2884 MB，构成分析（估算与实测吻合 2881.8 vs 2884 MB）：

| 张量 | 大小 | 占比 | 降内存杠杆 |
|---|---|---|---|
| **lm_head** | 1187 MB (fp32) | **41.2%** | 保留源 fp16 → 594 MB |
| **embed_tokens** | 1187 MB (fp32) | **41.2%** | **卸载到 SSD**（查表只读 1 行） |
| q_proj ×48 | 204 MB | 7.1% | — |
| o_proj ×48 | 204 MB | 7.1% | — |
| router ×48 | 48 MB | 1.7% | — |
| k/v_proj ×48 | 51 MB | 1.8% | — |

**lm_head + embed 两个张量占 resident 的 82%。**

**杠杆 ①：embed_tokens 卸载（纯内存优化，零数值代价）**
embed 是**查表**：每 token 只读 1 行（`hidden*2` = 4 KB），却占 1187 MB。留盘按需
pread 单行，实测开销 **0.03%**。`is_offloadable()` 判据扩展 + **tied 模型守卫**
（tied 时 lm_head = embed，卸载会让指针悬空 → 段错误）。

**杠杆 ②：保留源 dtype，不升 fp32（exporter 的纯浪费 bug）**
源 checkpoint 的 lm_head / embed_tokens / layernorm **本来就是 fp16**，exporter
硬编码 `DTYPE_F32` 把它们升 fp32 —— **白白翻倍且零收益**。实测 fp16 相对 RMS
误差 **0.0000%**、argmax 一致率 **100%**、CosSim **1.0**（就是原值）。

两个杠杆合计：**resident 2884 → 1079 MB（-62.6%）**，decode 616 → 538 ms/tok
（lm_head 占 decode 从 20.6% 降到 0.9%，fp16 省一半读取）。
**logits 与 fp32 版逐位一致**（CosSim 1.00000000, max|Δ| 0.0000e+00），
generated_ids 完全一致。

#### i4 lm_head 量化为何不安全（实测，非假设）

`--quant-lm-head`（RTN → GPTQ in-band，复用专家同一格式与 NEON kernel）可把
lm_head 压到 158 MB，但实测代价：

| | fp32 | **fp16** | i4 RTN |
|---|---|---|---|
| lm_head 大小 | 1187 MB | **594 MB** | 158 MB |
| 权重相对 RMS 误差 | — | **0.0000%** | 9.989% |
| **top-1 argmax 一致率** | — | **100%** | **75%** |
| CosSim | — | **1.00000000** | 0.994987 |

**top-1 argmax 一致率只有 75%** —— greedy decode 每 4 个 token 就有 1 个分叉，
长序列累积偏离。10% 相对误差不是 bug，是 RTN i4 的理论值（step/√12 ÷ 权重 RMS
≈ 11.5%），但对**直接产生 logits 的 lm_head** 来说偏高。故默认走 fp16 保留，
量化仅作 opt-in。

> **教训**：量化收益必须用 argmax 一致率衡量，不能只看 CosSim。CosSim 0.995
> 看起来"很好"，但 top-1 一致率 75% 意味着生成会分叉。

#### 内存上限校验（swap 是写操作，伤 SSD）

启动时校验 `resident + 专家缓存预算` vs **当前可用内存**（不是物理总量）：

```
[init] mem budget: resident 1079 MB + expert cache 10 MB = 1089 MB
[init] mem available: 3524 MB (物理 16384 MB), 留 10% 余量后 上限 3172 MB
```

**为什么不用物理总量**：wired（内核与不可换出部分）+ 其他进程已占掉大半，实测
16 GB 机器 wired 就有 8 GB、可用只剩 1.5 GB。按物理总量校验会宽松 **7×**，
放行后照样把机器推进换页（实测 swap used 11.3 GB / 12 GB、pageouts 132 万）。

超限 **fail-fast** 而非静默换页；`--force-over-memory-budget` 可强制继续（会警告）。
`--moe-expert-cache-mb N` 按字节预算限界专家缓存（单专家装不下时也 fail-fast，
不静默降级成 slots=0——那会让用户误以为预算生效）。

### 瓶颈定位随优化而转移（两轮实测，结论相反）

**第一轮（标量 GPTQ kernel）**：I/O 不是瓶颈。

| 配置 | total | expert_load（I/O） | bytes_read |
|---|---|---|---|
| `slots=0`（每次访问都 pread） | 132.2 s | **9.0 s（6.8%）** | 13.48 GB |
| `slots=4096`（首轮后全命中） | 140.2 s | **4.0 s（2.9%）** | 5.05 GB |

少读 8.43 GB 反而更慢 8 s。profiler：`expert_ffn` 58.5% + `qkv_proj` 17.8%
+ `o_proj` 14.3% = **计算 90.6%**。→ 砍 GPTQ kernel。

**第二轮（GPTQ NEON kernel 落地后）**：I/O 变成瓶颈。

| op | 标量 ref | NEON | 变化 |
|---|---|---|---|
| `expert_load`（I/O） | 3.0% | **61.5%** | ← **新瓶颈** |
| `expert_ffn` | 59.7% | 20.1% | ↓ |
| `qkv_proj` | 17.2% | 8.1% | ↓ |
| `lm_head` | 6.0% | 2.5% | ↓ |

I/O 绝对耗时不变（它不是计算），占比却涨 20×。**→ 下一刀砍 I/O（异步预取），
不是继续砍计算。**

> **教训**：组件占比的结论有**时效性**。"I/O 不重要"只在 compute 占 90.6% 时
> 成立；compute 快 14.9× 后同一个结论就反了。这就是优化日志模板里
> 「瓶颈转移」字段必须每次重测的原因（AGENTS.md 坑 #11）。

### Roofline 与瓶颈定位（历史，标量时期）

每 token MAC 数（hidden=2048, moe_inter=768, 48 层, top-8, vocab=151936）：

| 段 | GMAC/token | 占比 |
|---|---|---|
| MoE FFN | 1.812 | 70.3% |
| attention | 0.453 | 17.6% |
| lm_head | 0.311 | 12.1% |
| **合计** | **2.576** | |

标量时期 decode = 9.40 s/tok → 274 MMAC/s = 0.548 GFLOPS（NEON 上限的 0.20%）。
根因：GPTQ 变体族当时只有 `matvec_gptq_ref`（标量）。已修，见下。

### 实测加速（原预估 vs 实际）

原预估用 Qwen3.5-4B i4 的 102.5 GMAC/s 锚定，给出 37–374× 区间，并判断
"真实值大概率落在保守～悲观区间（37–112×）"。

**实际结果：14.9×** —— 落在预估区间**之外**（比悲观值 37× 还低）。原因正是
当时列出的两条限制，且比预想更严重：

1. 专家矩阵小（`[768,2048]`=0.8 MB），只有 `768/64 = 12` 个 o_block 给 10 线程
2. 每 token **3 matvec × 8 专家 × 48 层 = 1152 次 fork-join**，同步开销压过收益
   —— 所以 `neon_mt` 对 MoE **零收益甚至更慢**（提高粒度阈值到 4M 反而 1.242
   s/tok）。MoE 要并行得在**专家层**并行（top-8 彼此独立），是 runtime 级改动
3. 加速后 I/O 立刻顶到 61.5%，把计算侧收益吃掉一大半

### cache slots 扫描（必做，否则测速结论无效）

**反直觉：小 cache 更快。** 同场 A/B，3 次取 min：

| slots | min decode | hits | bytes_read |
|---|---|---|---|
| **4（仓库默认）** | **616 ms/tok** | ~0 | 13.48 GB |
| 0 | 633 | 0 | 13.48 GB |
| 8 | 631 | ~0 | 13.48 GB |
| 16 | 655 | 0 | 13.48 GB |
| 64 | 689 | 低 | — |
| 128 | 755 | 低 | — |
| 4096 | **1126（慢 1.8×）** | 3361 | 5.05 GB |

`slots=16` 几乎全 miss（hits=0）依然比全命中的 4096 快。**原因不是命中率，
而是工作集能否常驻 CPU cache**：13.48 GB 顺序读进 ~40 MB 热缓冲，胜过读
5.05 GB 进 ~5 GB 缓冲反复 thrash。同一份数学的 `expert_ffn` 耗时随 slots 变
**2.5×**（2.16 s vs 5.35 s）就是证据。

**仓库默认值 4 本来就最优。** 且 `slots=4096` 会把机器推进重度换页
（实测 `vm.swapusage used = 11.3 GB / 12 GB`、pageouts 132 万）——cache 应按
**字节预算**而非槽数限界。

### TTFT

prefill 7 tok：标量 66.4 s → **NEON 5.3 s（12.7×）**。MoE 走**逐 token**
prefill（批量 GEMM 的 gather/scatter 未实现）。长 prompt 仍线性放大。

## 已交付与后续

### ✅ Phase A：机制（fake 模型验证）
格式 v3 + GPTQ dequant + topk_softmax + ExpertStore（pread+LRU）+ MoE forward
+ fake 模型 + 对齐（resident vs SSD 逐位一致）+ **稀疏加载**（专家字节真不进
RAM，内存归因与单测护栏）。

### ✅ Phase A+：真模型端到端
- `tools/export_qwen_moe_to_tiny.py`：真 MoE GPTQ 导出器，支持 dense-attention
  MoE（无共享专家）+ AutoGPTQ 张量 repack（qzeros int4 解包 + zp+1）。
  48 层全量导出 18867 tensors / 17.16 GB。
- `kQwen3MoE` 架构支持 + 共享专家可选 + prefill 分支顺序修正 +
  `full_layer_cache_index()` 除零守卫。
- **matvec 按张量自身 dtype 路由**（见「dtype 路由」）。
- 正确性验收：单 token pos=0 logits **CosSim = 1.00000012**（max|diff| 3.81e-06），
  48 层生成连贯文本。

### ✅ Phase C-1：GPTQ NEON kernel（decode 14.9×）
`kernels/matvec/matvec_gptq_neon{,_mt}.cpp` + 共享内层 `matvec_gptq_neon_common.h`。
三处改动：① c8-outer/o-inner 遍历（连续内存）② o 分块 64 ③ **反量化因式分解**
`Σ_k ((nib_k - z)·s·x_k) = s·[Σ_k(nib_k·x_k) - z·Σ_k(x_k)]`，`Σ_k(x_k)` 只依赖
c8、与 o 无关，可预算。act-order（g_idx 非均匀）自动降级到 slow 路径——算错
不报错是最危险的失效模式，必须有护栏（单测覆盖）。

`neon_mt` 已实现但对 MoE 无收益（原因见上）。保留供非 MoE 的大 GPTQ 矩阵用。

### ⏳ Phase B：I/O 优化 —— **最高优先级（已从"下调"改回）**
**上调理由**：NEON kernel 让 compute 快 14.9× 后，**I/O 从 3.0% 涨到 61.5%**，
成为最大单项。原"收益上限 <4%"的结论前提是 compute 占 90.6%，该前提已失效。

#### 先测清受限类型，再动手（关键前置）

**I/O 是带宽受限，不是延迟受限。**

| | |
|---|---|
| 每 token 读 | 1.685 GB |
| expert_load 耗时 | 378.8 ms |
| **实测有效带宽** | **4.45 GB/s** |
| M4 NVMe 顺序读峰值 | ~5–7 GB/s |
| **已达峰值** | **74%** |

这个判断决定收益上限，两者差 2.1 倍：

| 若 I/O 是… | overlap 能藏住什么 | 收益上限 |
|---|---|---|
| 延迟受限 | 全部 I/O | **2.60×** |
| **带宽受限（实测）** | **只能藏计算，藏不住带宽本身** | **1.25×** |

计算：理想 = `max(I/O, 计算)` = max(379, 124) = 379 ms，省下 124 ms，
decode 616 → **492 ms/tok（1.25×）**。

> **教训**：动手前先测受限类型。按"延迟受限"假设设计会把收益高估 2.1 倍，
> 做完发现只有 1.25× 会误判为"实现有问题"。同 AGENTS.md 坑 #11。

#### B-1：单次 pread 合并三块 ✅ 已完成（1.47×）

每个专家原做 **3 次独立 pread**（gate/up/down）。已验证三块在文件内**连续**：
- exporter 按 `gate_proj → up_proj → down_proj` 顺序写入同一专家
- 稀疏加载保留**原始文件偏移**（`model_loader.cpp:420` `view.file_offset = e.offset`），
  卸载张量不参与紧凑重排
- 块间仅 64B 对齐填充，≤63 字节

合并成 1 次 pread 读 `down_off + down_nbytes - gate_off`（~2.507 MB）。实测：

| | B-1 前 | **B-1 后** | 加速 |
|---|---|---|---|
| decode (slots=4) | 538 ms/tok | **365 ms/tok** | **1.47×** |
| decode (slots=0) | 633 ms/tok | **344 ms/tok** | **1.84×** |
| expert_load (I/O) | 5.11s (60.5%) | **2.71s (45.1%)** | **-47%** |
| 有效带宽 | ~2.6 GB/s | **~5.0 GB/s** | ≈ NVMe 峰值 83% |

**实际 1.47× 高于预估 1.19×**：预估只算了"把带宽推向峰值"，漏了 syscall 从 3
降到 1 本身的收益（每 token 384 次专家加载 × 省 2 次 syscall = 768 次）。
**估算 I/O 优化收益时要同时算带宽与 syscall 两项。**

累计 vs ref 基线：**9157 → 365 ms/tok = 25.1×**。generated_ids 与优化前完全一致。

#### B-2：层内异步预取 —— **收益空间已收窄，需重估**

B-1 把 I/O 从 60.5% 压到 45.1%，带宽已达 NVMe 峰值 83%。剩余 I/O 接近硬件下限，
异步预取能藏住的只剩计算部分：

| | B-1 后 |
|---|---|
| decode | 365 ms/tok |
| I/O | 45.1% ≈ 165 ms |
| 计算 | 54.9% ≈ 200 ms |

理想 overlap = `max(I/O, 计算)` = 200 ms → 收益上限 **1.83×**（365 → 200 ms/tok）。
比 B-1 前的 1.25× 上限**反而变大了**——因为计算现在比 I/O 慢，藏住 I/O 更划算。

数据流（`qwen_forward_token.cpp:433-491`）：

```
moe_router → topk_softmax → (shared_ffn) → for t in 0..k: [expert_load + expert_ffn]
```

**可 overlap 的窗口**：`topk_softmax` 之后，本层全部 k=8 个专家 id 已知，但当前
逐个 load+compute 串行。改成：

```
topk_softmax 完成 → 后台线程一次性预取 expert[0..7]
主线程：wait(expert[0]) → compute(expert[0]) → wait(expert[1]) → compute(expert[1]) …
```

**不可 overlap 的部分（硬约束）**：层 i+1 的 router 依赖层 i 的 FFN 输出，
所以**跨层无法预取**——不知道下一层选哪些专家。这决定了收益上限只能是
`max(I/O, 计算)`，不可能藏住全部 I/O。

**风险（必须先解决）**：
1. `ExpertStore::get()` 会修改 `slots_` / `slot_index_` / `stats_`，**当前非线程安全**。
   预取线程与主线程并发访问需要加锁或改无锁设计。
2. 返回的 `ExpertWeights` 指针指向槽内部缓冲，契约是"仅在下次淘汰前有效"。
   预取 8 个专家时若槽数 < 8，先预取的会被后预取的淘汰 → **指针失效**。
   必须保证槽数 ≥ k，或改返回 owning handle。
   （B-1 后这一点更微妙：连续路径下三个指针指向同一 `span` 缓冲，淘汰即全部失效。）
3. profiler 的 `ScopedTimer` 归因会失真：I/O 与计算重叠后，两者耗时之和不再
   等于墙钟时间。**归因方法要改**（按墙钟 + 各自的 CPU 时间分开记）。

#### B-3：投机跨 token 预取（上限更高，但是启发式）

专家选择在相邻 token 间常有重叠。可在层 i-1 的 FFN 期间，按**上一个 token 的
选择**投机预取层 i 的专家；猜错则回退到正常 pread（正确性不受影响）。

上限高于 B-2（可藏住部分 I/O），但收益取决于命中率，需先测相邻 token 的专家
重叠率再决定是否投入。**不要在没有命中率数据前实现。**

#### 执行顺序建议

1. ✅ **B-1**（合并 pread）：已完成，1.47×
2. ✅ 重测归因：I/O 60.5% → 45.1%，带宽达峰值 83%
3. **B-2**（层内预取）：收益上限升到 1.83×（计算现在比 I/O 慢）。先解决线程
   安全与指针生命周期。
4. 测相邻 token 专家重叠率，数据支持才做 **B-3**

### ⏳ Phase C-2/3：剩余计算优化
1. **专家层多线程并行**：top-8 专家 FFN 彼此独立。当前 kernel 级 MT 无效
   （1152 fork-join/token），改到 runtime 级只有 48 次/token。
2. **MoE 批量 prefill**：同批 token 路由到不同专家的 gather/scatter GEMM，
   解决长 prompt 的线性放大。
3. **Metal prefill**：`metal_prefill_create` 目前 fail-fast 拒绝 GPTQ
   （亚字节布局需专门反量化 kernel）。且 Metal 只覆盖 prefill，对 decode
   无帮助——**不是当前杠杆**。
