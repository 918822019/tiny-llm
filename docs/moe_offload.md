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

## 性能现实（本机 4.8GB / 4 核）

35B-Int4 ~17.5GB 装不进内存 → SSD 流式是唯一可行方式。预期 decode 受 I/O 主导
（每 token 各 MoE 层读 k 个专家，最坏全 miss 数百 MB/token → NVMe 下数百 ms/token
量级）。这是"研究 SSD 卸载机制本身"，非交互式速度。fake 模型极小，I/O 不可测；
真实 I/O 归因需在真模型 + `bench_expert_store`（Phase B，见下）。

## 已交付（Phase A）与后续（Phase B/C）

- ✅ Phase A：格式 v3 + GPTQ dequant + topk_softmax + ExpertStore（pread+LRU）
  + MoE forward + fake 模型 + 对齐（resident vs SSD 逐位一致）+ **稀疏加载**
  （专家字节真不进 RAM，内存归因与单测护栏）。
- ⏳ Phase B：异步预取 overlap（router 选完 top-k → 入队预取，CPU 同时算 shared/
  下一层 attention）；`benchmarks/bench_expert_store`（cache hit/miss + cold-load
  延迟 + pread 吞吐，带离散度列）；cache 抖动归因。
- ⏳ Phase C（明确推迟）：真 35B GPTQ 导出器；GPTQ 优化 kernel（marlin/sdot 风格）；
  MoE 批量 prefill（同批 token 路由到不同专家的 gather/scatter GEMM）；多线程专家并行。
