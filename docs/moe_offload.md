# MoE 动态路由 + SSD 专家卸载

## 命题

Qwen3.5-35B-A3B（256 路由专家、top-8、1 共享专家，激活 ~3B/总 35B）的 Int4
权重约 17.5GB，装不进端侧内存。但每个 token 只激活 k 个专家。把**路由专家
权重留在 .tqwen 文件（SSD），只有被路由门选中的专家才 pread 进一个固定大小
的 LRU 缓存参与计算**；未激活专家永远不进 RAM。attention / 路由门 / 共享专家 /
embed / norm 常驻内存（resident）。

## 架构

`ModelType::kQwen35MoE` = Qwen3.5 attention（GDN + full attention 3:1 混合，**全部
既有代码复用**）+ MoE FFN 替换 dense SwiGLU FFN。每层 FFN 都是 MoE：

```
post_attn_norm → router_gate(resident) → topk_softmax → 共享专家(resident) + Σ 路由专家
```

路由专家权重访问二选一（`--moe-ssd` 开关）：
- **resident 模式**（默认，正确性锚点）：专家权重直接绑自 ModelFile 内存（fake
  模型小，可全驻留）。与既有 dense 路径同源，是正确性 oracle。
- **SSD 模式**：专家权重经 `ExpertStore`（`runtime/expert_store.h`）——pread +
  LRU 槽缓存按需加载。两者须逐位一致（见验证）。

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

- **ExpertStore**（`runtime/expert_store.{h,cpp}`）：持有 .tqwen fd + (layer,expert)
  → 文件 offset 映射 + LRU 槽缓存。`get()` 命中返零拷贝指针；未命中淘汰 LRU +
  pread 三块（gate/up/down）。`pread` 而非 `mmap`——mmap 的 readahead 会把未激活
  专家也读进 page cache，违背"只读激活"命题。统计 hits/misses/evictions/bytes_read。
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

## CLI

| 参数 | 说明 |
|---|---|
| `--moe-ssd` | 路由专家走 ExpertStore pread+LRU（默认 resident 指针） |
| `--moe-expert-cache-slots N` | LRU 槽数（0 = 每次现 pread；默认 4） |

运行结束 stderr 输出 `[moe] expert cache: hits=.. misses=.. evictions=.. bytes_read=..`。

## 性能现实（本机 4.8GB / 4 核）

35B-Int4 ~17.5GB 装不进内存 → SSD 流式是唯一可行方式。预期 decode 受 I/O 主导
（每 token 各 MoE 层读 k 个专家，最坏全 miss 数百 MB/token → NVMe 下数百 ms/token
量级）。这是"研究 SSD 卸载机制本身"，非交互式速度。fake 模型极小，I/O 不可测；
真实 I/O 归因需在真模型 + `bench_expert_store`（Phase B，见下）。

## 已交付（Phase A）与后续（Phase B/C）

- ✅ Phase A：格式 v3 + GPTQ dequant + topk_softmax + ExpertStore（pread+LRU）
  + MoE forward + fake 模型 + 对齐（resident vs SSD 逐位一致）。
- ⏳ Phase B：异步预取 overlap（router 选完 top-k → 入队预取，CPU 同时算 shared/
  下一层 attention）；`benchmarks/bench_expert_store`（cache hit/miss + cold-load
  延迟 + pread 吞吐，带离散度列）；cache 抖动归因。
- ⏳ Phase C（明确推迟）：真 35B GPTQ 导出器；GPTQ 优化 kernel（marlin/sdot 风格）；
  MoE 批量 prefill（同批 token 路由到不同专家的 gather/scatter GEMM）；多线程专家并行。
