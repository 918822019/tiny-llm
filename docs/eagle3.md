# Qwen3-0.6B EAGLE3 投机解码

本项目可以把 SpecForge 发布的 Qwen3-0.6B EAGLE3 checkpoint 转成统一 FP16 的
`.tqwen` 文件，并用目标模型的三层残差流执行精确 greedy 投机解码。当前实现是
`topk=1` 的单链：草稿只负责提案，Qwen3-0.6B 目标模型仍逐块验证并拥有最终裁决权，
因此输出应与不启用草稿的 greedy 路径逐 token 一致。

当前范围：Qwen3-0.6B target、batch=1、FP16 权重、greedy、EAGLE3 单链。CPU 是主线；
Android 可让 target 使用通用逐算子 Vulkan 后端，但 EAGLE3 drafter 仍在 CPU，尚不是
target + drafter 整图 GPU 常驻实现。

## 1. Checkpoint 与下载

已验证的草稿模型是
[`GavinLucky/SGLang-EAGLE3-Qwen3-0.6B-SpecForge`](https://huggingface.co/GavinLucky/SGLang-EAGLE3-Qwen3-0.6B-SpecForge)。
模型卡说明它由 SpecForge 在 ShareGPT/Vicuna 68K 样本上训练 10 个 epoch，未用目标
模型重新生成训练响应；官方 SGLang 示例使用 `num_steps=3`、`topk=1` 和
`num_draft_tokens=4`。

通过 Hugging Face 镜像下载：

```bash
HF_ENDPOINT=https://hf-mirror.com .venv/bin/hf download \
  GavinLucky/SGLang-EAGLE3-Qwen3-0.6B-SpecForge \
  --local-dir models/SGLang-EAGLE3-Qwen3-0.6B-SpecForge
```

本次下载的原始权重校验值：

```text
model.safetensors
sha256 cd1b033ea6e147b00c5089ece598a43f8dca186252a270c59dfa19778c57c60c
size   112090960 bytes
```

## 2. 转换为统一 FP16

```bash
.venv/bin/python tools/export_eagle3_to_tiny.py \
  --model models/SGLang-EAGLE3-Qwen3-0.6B-SpecForge \
  --out eagle3_qwen3_06b_specforge_f16.tqwen
```

转换器会严格校验架构、全部 tensor 名称和 shape、`d2t`/`t2d` 的一致性，然后把源
BF16 权重转为 FP16。输出中包括：

- `fc.weight`：把目标模型第 `1,13,24` 层后的三个残差流从 `3H` 投影到 `H`；
- 一层 EAGLE3 attention + SwiGLU recurrent layer；
- 32K draft vocabulary 的 FP16 `lm_head`；
- draft token 到 Qwen3 151936 词表的精确映射；
- 目标残差层号 `1,13,24`（均为零基、表示执行该层后的 residual stream）。

目标 token id 最大超过 FP16 能精确表示的连续整数范围，所以词表映射不能直接写成
单个 FP16。导出器将每个 id 拆成两个 base-256 FP16 limb：`high=id//256`、
`low=id%256`；两部分都能精确表示，loader 再还原并检查范围与唯一性。这样既保持
全文件 tensor dtype 统一为 FP16，也不会损坏词表映射。

本次确定性转换结果：

```text
eagle3_qwen3_06b_specforge_f16.tqwen
sha256 0ba5f5fb4cc45a019e57a65458f04c8e03cdc4e0039d0ef76757166e84608c58
size   111811648 bytes (106.63 MiB)
```

配套 FP16 target 为 `model_qwen3_06b_f16_ctx2048.tqwen`，大小 1192137280 bytes。
因此草稿文件约为 target 文件的 9.38%；但每个草稿步仍需运行一层网络及 32K
`lm_head`，文件小不等于端侧计算开销为零。

## 3. 推理语义

EAGLE3 的关键不是把普通小语言模型直接接到 target 后面，而是使用目标模型的中间
特征并遵守训练时的 token shift：

1. target prefill prompt，同时捕获第 `1,13,24` 层后的残差流；
2. 对每个目标位置拼接三层特征并经 `fc(3H -> H)` 融合；
3. 位置 `i` 的目标特征与 token `i+1` 的 embedding 配对；prompt 最后一行特征与
   target 已选出的 `pending` token 配对；
4. EAGLE3 recurrent layer 产生一条 greedy proposal chain；
5. target 一次验证 `[pending, proposals...]`，接受首个不一致前的连续前缀；
6. target KV 只保留 `pending + accepted`，拒绝点改用 target token；draft 丢弃未验证
   suffix，再用 target 已确认的隐藏状态重建接受前缀。

这里 `pending` 是 target 已选出、但尚未写入 target KV 的 token。这个状态约定确保
拒绝、全接受 bonus 和最后一个 tail token 都与普通 greedy 完全等价。

`--speculative-tokens` 在 EAGLE3 路径表示**整个 target 验证块宽度**，包含
`pending`：

| 参数 | 每块最多 proposal 数 | 含义 |
|---:|---:|---|
| 2 | 1 | 手机 CPU 当前推荐；草稿成本最低 |
| 3 | 2 | 更长链，只有前缀持续命中才划算 |
| 4 | 3 | 对应模型卡的 topk=1、3-step 单链设置 |

设置 `TINYQWEN_EAGLE3_TRACE=1` 会打印每块的 root、草稿 token、target token 和接受数，
仅用于对齐诊断，正式测速应关闭。

## 4. 运行

CPU：

```bash
TINYQWEN_MT_THREADS=6 ./build/runtime/tinyqwen \
  --model model_qwen3_06b_f16_ctx2048.tqwen \
  --eagle3-model eagle3_qwen3_06b_specforge_f16.tqwen \
  --tokens-json prompt.json --max-new-tokens 64 --max-seq-len 256 \
  --speculative-tokens 2 --matvec-impl neon_mt_kv_nt --ops-impl neon \
  --kv-f16 --speculative-stats-out eagle3.json
```

Android：

```bash
./scripts/build_android.sh
adb push build-android/runtime/tinyqwen /data/local/tmp/tinyqwen/tinyqwen
adb push eagle3_qwen3_06b_specforge_f16.tqwen \
  /data/local/tmp/tinyqwen/models/eagle3_qwen3_06b_specforge_f16.tqwen

adb shell "cd /data/local/tmp/tinyqwen && TINYQWEN_MT_THREADS=6 ./tinyqwen \
  --model models/model_qwen3_06b_f16_ctx2048.tqwen \
  --eagle3-model models/eagle3_qwen3_06b_specforge_f16.tqwen \
  --tokens-json prompt.json --max-new-tokens 64 --max-seq-len 256 \
  --speculative-tokens 2 --matvec-impl neon_mt_kv_nt --ops-impl neon \
  --kv-f16 --eos -1 --speculative-stats-out eagle3.json"
```

首次部署必须用完全相同的 target、prompt、生成长度和 EOS 设置再跑一次不带
`--eagle3-model` 的普通 greedy，并比较两行 `generated_ids`。接受率、调用次数或速度
都不能替代这个正确性门禁。

Host 上可直接运行同样的自动门禁：

```bash
./scripts/verify_eagle3.sh \
  model_qwen3_06b_f16_ctx2048.tqwen \
  eagle3_qwen3_06b_specforge_f16.tqwen \
  prompt.json 2 64
```

在 Android 上追加 `--backend vulkan` 可以让 target 的 FP16 matrix op 使用通用
Vulkan 后端；EAGLE3 drafter 仍明确使用 CPU。该路径每个矩阵算子都会 submit + wait，
主要用于功能/正确性 A/B，不代表已经完成 EAGLE3 整图 GPU 后端。

`--no-eagle3-batch-verify` 是专门用于因果消融的诊断开关。默认路径把
`[pending, proposals...]` 一次交给 target；打开该开关后，保持 proposal、接受/拒绝、
rollback 和输出语义不变，但让 target 每次只验证一个 token。这样可以在相同接受率和
相同 target 输入数下，直接测出多 token matrix tile、权重复用与提交合并的贡献。该模式
只用于测量，不是推荐部署配置。

Android 上的四路径消融可以自动运行：

```bash
./scripts/bench_eagle3_ablation_android.sh \
  model_qwen3_06b_f16_ctx2048.tqwen \
  eagle3_qwen3_06b_specforge_f16.tqwen \
  prompt.json 3 64 2
```

脚本默认跑完整四路径消融：先做短预热，再把 greedy/batched、batched/sequential 保持
相邻并逐轮反转顺序；它比较全部 `generated_ids`，同时报告 prefill、decode、两者之和及
同轮加速比。只复测实际部署的 Vulkan greedy/batched 两条路径时，使用
`EAGLE3_BENCH_MODE=pair`；此模式适合增加重复次数并缩小慢速 sequential 路径造成的热
干扰。每轮原始输出、统计和汇总 JSON 保存到 `artifacts/eagle3-ablation-*`。

## 5. PLK110 真机结果（2026-09-14）

设备：OnePlus PLK110 / Android 16 / Adreno 840。target 与 draft 权重均为 FP16，
FP16 KV，6 个 CPU worker，`neon_mt_kv_nt + neon`，`max_seq_len=256`。

### 5.1 英文 64-token

prompt：`Explain why the sky is blue in one concise sentence.`，chat template 后 41 token，
强制生成 64 token。宽度 2 跑了三轮相邻 A/B；所有 greedy/EAGLE/Vulkan 输出文件的
SHA-256 完全相同。

| 路径 | 宽度 | 接受情况 | target 调用/输入 | decode |
|---|---:|---:|---:|---:|
| CPU greedy | — | — | 63 次 / 63 token | 2857.92 / 2871.97 / 2976.24 ms |
| CPU EAGLE3 | 2 | 22/41（53.7%） | 41 次 / 82 token | 2962.18 / 3021.45 / 3052.65 ms |
| CPU EAGLE3 | 3 | 27/71（38.0%） | 36 次 / 107 token | 3346.40 ms |
| CPU EAGLE3 | 4 | 29/99（29.3%） | 34 次 / 133 token | 3969.14 ms |
| Vulkan target greedy | — | — | 63 次 / 63 token | 6279.38 ms |
| Vulkan target + CPU EAGLE3 | 2 | 22/41（53.7%） | 41 次 / 82 token | 4845.53 ms |

CPU 宽度 2 的 target verify 最好一轮为 2761.04 ms，确实比同场 greedy 最好一轮
2857.92 ms 少约 96.9 ms；但 draft 另花 200.78 ms，所以最终仍慢 3.6%。按三轮中位数
比较则慢 5.2%。这说明投机机制和 batch verification 都已经生效，尚未跨过端到端
盈亏点。

这次早期单轮里，Vulkan 混合路径相对逐算子 Vulkan greedy 的 decode 是 1.30x
（6279.38 -> 4845.53 ms），但它有 9253 次 GPU dispatch，并且仍比 CPU greedy 慢约
69%。因此它只能证明 GPU target A/B 路径正确且能从较少 target block 中受益，不能称为
手机最佳后端；测量修正后的多轮权威数字见 5.3 节。

### 5.2 中文 16-token

prompt：`你好，请用一句话介绍你自己。`，chat template 后 36 token，强制生成 16 token。
三条路径的全部输出 token 完全一致。

| 路径 | 宽度 | 接受情况 | decode |
|---|---:|---:|---:|
| CPU greedy | — | — | 825.20 ms |
| CPU EAGLE3 | 2 | 2/13（15.4%） | 878.85 ms |
| CPU EAGLE3 | 4 | 2/37（5.4%） | 1379.21 ms |

中文样例的低接受率不是 FP16 转换误差：用原始 BF16 checkpoint、官方 Qwen3-0.6B
target 和 SpecForge/EAGLE3 方程独立复算，前几块第一 proposal 与 C++ 路径一致。
例如首块 root 为 `151667` 时两边都提案 `151668`，而 target 选择 `198`。这更符合
checkpoint 训练域/单样本差异，而不是运行时把层号、拼接顺序或 token shift 写错。

### 5.3 投机调度与 token tile 因果消融（2026-09-15）

greedy 无法预知未来 token，因此“关闭投机、但仍让多 token tile 满载”在算法上不存在：
没有 drafter 提供候选，就没有第二个 token 可以与当前 token 一起验证。这里采用四路径
阶梯，而不伪造一个不存在的完全正交 2×2：

1. CPU greedy：手机当前实用基线；
2. Vulkan greedy：同一通用 Vulkan target 的单 token 基线；
3. Vulkan EAGLE3 sequential：开启同一 drafter，但用
   `--no-eagle3-batch-verify` 逐 token 验证；
4. Vulkan EAGLE3 batched：开启 drafter 与默认的两 token 批量 tile。

测试仍使用 41-token 英文 prompt、强制生成 64 token、width=2、FP16 target/draft/KV、
6 个 CPU worker。所有正式样本的 64 个 `generated_ids` 完全相同；两种 EAGLE3 路径
也都有相同的 22/41（53.66%）接受率和 82 个 target 输入，因此性能差异不是 proposal
或接受决策变化造成的。

#### 测量修正

首轮消融曾报告完整投机相对 Vulkan greedy 为 1.406x。继续复查后发现两项会污染短
进程计时的因素：

1. batched verification 的 `N>1` lm_head 会走 Vulkan matmul，而 greedy/sequential
   的 `N=1` lm_head 走 CPU。旧实现到第一个 speculative block 才创建并复制约
   296.8 MiB 的 tied lm_head GPU buffer，因此一次性准备成本被混进 decode；另一个
   warmup 进程无法替后续新进程保留这个 buffer。
2. 四路径旧顺序让极慢的 sequential 路径隔在主要对照之间，手机温度和频率漂移会直接
   进入配对比值。

现在 batched EAGLE3 在 target prefill 后显式调用 backend weight preparation：GPU
buffer 的创建/复制发生在 decode 计时前，但仍完整计入该请求的 prefill，而不是被隐藏。
脚本也让 `greedy↔batched` 与 `batched↔sequential` 相邻，并在偶数轮反转顺序。

修正后的三轮完整阶梯中位数如下；`total` 是每轮 `prefill + decode` 后再取中位，因此
不必等于前两列中位数之和：

| 路径 | prefill 中位（ms） | decode 中位（ms） | total 中位（ms） | target 调用 / 输入 |
|---|---:|---:|---:|---:|
| CPU greedy | 783.80 | 2605.78 | 3665.35 | 63 / 63 |
| Vulkan greedy | 1370.58 | 8429.86 | 9866.40 | 63 / 63 |
| Vulkan EAGLE3 sequential | 1524.85 | 11263.70 | 13190.54 | 82 / 82 |
| Vulkan EAGLE3 batched | 1970.66 | 6573.34 | 8421.69 | 41 / 82 |

按同轮比值取中位数：只有投机、没有批量验证为 **0.741x**；投机场景内打开 batch/tile
为 **1.799x**；完整 batched EAGLE3 相对 Vulkan greedy 的 decode 为 **1.292x**，
`prefill + decode` 为 **1.183x**。四路径绝对耗时仍有明显热状态波动，因此生产两路径
又单独做了 7 轮相邻配对：

```bash
EAGLE3_BENCH_MODE=pair ./scripts/bench_eagle3_ablation_android.sh \
  model_qwen3_06b_f16_ctx2048.tqwen \
  eagle3_qwen3_06b_specforge_f16.tqwen \
  prompt.json 7 64 2
```

| 路径 | prefill 中位（ms） | decode 中位（ms） | total 中位（ms） |
|---|---:|---:|---:|
| Vulkan greedy | 1756.20 | 8251.62 | 10003.54 |
| Vulkan EAGLE3 batched | 2065.13 | 6646.20 | 8708.91 |

7 轮逐对相除后，decode 加速中位为 **1.239x**，全部轮次范围 **1.237–1.246x**；
请求内 `prefill + decode` 加速中位为 **1.150x**，范围 **1.143–1.154x**。EAGLE3 的
`draft_ms` 中位为 571.11 ms，`target_verify_ms` 中位为 6077.74 ms。第二次独立的
7 轮手工配对也得到 decode 中位 1.241x，所以当前权威结论取 **约 1.24x decode、
约 1.15x 请求内端到端**。早先 1.406x 保留为测量修正前的历史结果，不再作为性能
结论；它与 llama.cpp 单轮约 1.41x 接近不足以证明两边具有相同的稳定加速倍率。

#### 宽度与 prompt 探索

继续测试了 width=3/4 和两个英文 prompt。除 width=2 sky 的 7 轮复测外，下表性能值
来自测量修正前的三轮探索，只能看趋势，不能和权威值做精确横向比较；接受统计和输出
一致性不受 lm_head 准备时机影响。

| prompt | width | 接受情况 | target 调用 / 输入 | 探索性 decode 加速 |
|---|---:|---:|---:|---:|
| sky | 2 | 22/41（53.7%） | 41 / 82 | **1.239x**（修正后 7 轮） |
| sky | 3 | 27/71（38.0%） | 36 / 107 | 1.211x |
| sky | 4 | 29/99（29.3%） | 34 / 133 | 1.175x |
| Fibonacci code | 2 | 25/38（65.8%） | 38 / 76 | 1.438x |
| TCP vs UDP | 2 | 24/38（63.2%） | 39 / 77 | 1.372x |

sky 样例中，width 变宽虽把 target 调用从 41 降到 36/34，但接受率持续下降，target
输入增至 107/133，最终反而不如 width=2。不同 prompt 的接受率差异也很明显，因此
单 prompt 不能代表业务分布；Fibonacci/TCP 的旧性能数字需要用新 pair 模式复测后才能
成为正式结论。

归因结论不变，但倍率修正为更保守的值：**drafter 的有效预测提供可并行验证的未来
token，投机调度把 63 个串行 target step 变成 41 个验证块，而 token tile 把每块两个
输入合并执行；二者是乘法交互。**禁用 tile 后投机为 0.741x 负收益，说明执行层面的
决定性收益来自 batch/tile；但没有 drafter 提案，greedy 路径也没有第二个未来 token
可以填入 tile。

## 6. 如何解读“是否符合论文预期”

当前结果分成三层：

1. **算法正确性符合预期**：greedy 输出在 CPU 宽度 2/3/4、中文/英文以及 Vulkan
   target 下都与普通 target 逐 token 一致；接受、拒绝、bonus、target/draft rollback
   都真实发生。
2. **drafter 质量在英文样例上有效**：首 proposal 接受率 53.7%，target 调用减少
   34.9%。所以不能简单归因成“drafter 完全不给力”。
3. **PLK110 端到端速度尚未达到服务器论文/模型卡场景**：Qwen3-0.6B target 本身已很
   小且 ARM FP16 matvec 已优化；EAGLE3 每个确认 token 仍要支付一层 recurrent network
   和 32K lm_head。服务器结果使用 SGLang、BF16 GPU、CUDA graph、批量 benchmark 与
   多种数据集，不能把其推荐宽度 4 直接套到手机 CPU。
4. **Vulkan 内部稳定约 1.24x decode / 1.15x 请求内端到端，是投机与 tile 的交互**：
   逐 token 消融证明 drafter 单独运行是负收益，而批量验证把相同的 82 个 target 输入
   从 82 次调用合并为 41 次，才兑现执行收益。它不属于 TinyLLM tile 或 drafter 任一方
   可独占的收益；早先 1.406x 是测量修正前的历史值。
5. **仍不等于复现服务器论文吞吐**：通用 Vulkan 每个 matrix op 都 submit/wait，绝对
   速度仍明显慢于手机 CPU；当前结果证明机制和相对 A/B 生效，不证明这是手机最佳后端，
   也不能直接与 SGLang/CUDA graph 的论文吞吐倍率等同。

当前手机建议使用宽度 2 做实验；正式决定是否启用前，应在目标业务数据集上报告
`generated_ids` 一致率、`draft_ms`、`target_verify_ms`、decode 和接受长度分布。下一步
若要稳定超过 CPU greedy，优先级是让 target + EAGLE3 drafter + 共享/裁剪后的输出头
整段常驻 GPU，并减少逐算子提交，而不是继续扩大 CPU 单链宽度。
