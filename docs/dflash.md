# DFlash / DFlare + Markov 投机解码

本项目支持一种不同于 `--draft-model` 自回归小模型的草稿路径：DFlash 一次接收
一个 anchor token 和若干 mask token，利用目标模型选定层的隐藏状态并行预测整个块；
DFlare 为每个草稿层学习目标层融合权重，DSpark Markov 头再把前一个草稿 token
的信息按低秩残差加到下一个位置的 logits。

当前范围刻意固定为：Qwen3 稠密目标、batch=1、greedy、单链、FP16 权重。目标
验证当前走 CPU；草稿可走 CPU，或在 Android 上用 Vulkan GPU-resident 执行器。
树搜索和采样式 rejection sampling 尚未接入。

## 1. 已验证 checkpoint

本次真机使用：

```text
target:  Qwen/Qwen3-0.6B
draft:   experiment/dflash-qwen3-0.6b/checkpoints/
         qwen3-0.6b-block8-9tli-ab-markov/epoch_2_step_18152/
```

草稿结构：3 层、hidden=1024、intermediate=3072、16 个 Q head、8 个 KV head、
head_dim=128、block_size=8；目标残差层为 `0,3,6,9,12,15,18,21,24`；DFlare
融合参数形状 `[3,9]`；Markov rank=128。

推理只需要 `config.json`、`dflash.py` 和 `model.safetensors`；
`training_state.pt` 是续训状态，不应部署到手机。

## 2. 导出

先按普通流程把 Qwen3-0.6B 导出为 FP16 目标文件，再导出 DFlash：

```bash
.venv/bin/python tools/export_qwen_to_tiny.py \
  --model models/Qwen3-0.6B \
  --out model_qwen3_06b_f16_ctx2048.tqwen \
  --dtype f16 --max-seq-len 2048

.venv/bin/python tools/export_dflash_to_tiny.py \
  --model models/dflash-qwen3-0.6b-block8-9tli-ab-markov-epoch2-step18152 \
  --out dflash_qwen3_06b_markov_f16.tqwen
```

DFlash 文件沿用 tiny header v2 与 tensor table。专用 loader 将 v2 扩展中的
`eos_token_id` 解释为 mask token id，并从额外的 `dflash.target_layer_ids` FP16
tensor 读取目标层列表；普通 `QwenModel` 不应加载该文件。块长由
`block_pos_embed.weight` 第一维推导，Markov rank 由 `markov_w1.weight` 推导。

## 3. 推理流程

1. 目标模型 prefill prompt，并捕获指定层在每个位置的残差流。层号 `L` 对应
   Hugging Face 的 `output_hidden_states[L+1]`。
2. 目标模型给出尚未写入 cache 的 anchor token。
3. DFlash 把新的目标残差融合并追加到自身 context KV；输入
   `[anchor, mask, ...] + block_position_embedding`。
4. 三层 DFlare 做非因果块注意力。只保留已确认 context 的 K/V，noise K/V 在提案后丢弃。
5. 每个草稿位置先用目标模型共享的 `lm_head` 得到 backbone logits，再按链顺序加入：

   ```text
   logits[k] += markov_w2(markov_w1[previous_draft_token])
   ```

6. 目标模型一次验证 `[anchor, proposals...]`。接受第一个不一致位置之前的前缀，
   拒绝点使用目标 token；目标 KV 裁到 `anchor + accepted`，保证结果与普通 greedy
   解码逐 token 一致。

运行：

```bash
./build/runtime/tinyqwen \
  --model model_qwen3_06b_f16_ctx2048.tqwen \
  --dflash-model dflash_qwen3_06b_markov_f16.tqwen \
  --tokens-json prompt.json \
  --speculative-tokens 2 \
  --max-new-tokens 16 --max-seq-len 128 \
  --matvec-impl neon_mt_kv_nt --ops-impl neon --kv-f16 \
  --speculative-stats-out dflash_stats.json
```

`--speculative-tokens` 表示整个验证块的输入 token 数，包含 anchor，必须在 `[2,8]`
内。诊断单块提案时可设置 `TINYQWEN_DFLASH_TRACE=1`；它会把草稿与目标 token 写到
stderr，正常部署不要开启。

## 4. Android 真机

```bash
./scripts/build_android.sh
adb push build-android/runtime/tinyqwen /data/local/tmp/tinyqwen/tinyqwen
adb push model_qwen3_06b_f16_ctx2048.tqwen \
  /data/local/tmp/tinyqwen/models/model_qwen3_06b_f16_ctx2048.tqwen
adb push dflash_qwen3_06b_markov_f16.tqwen \
  /data/local/tmp/tinyqwen/models/dflash_qwen3_06b_markov_f16.tqwen

adb shell "cd /data/local/tmp/tinyqwen && ./tinyqwen \
  --model models/model_qwen3_06b_f16_ctx2048.tqwen \
  --dflash-model models/dflash_qwen3_06b_markov_f16.tqwen \
  --tokens-json prompt.json --max-new-tokens 16 --max-seq-len 128 \
  --speculative-tokens 2 --matvec-impl neon_mt_kv_nt --ops-impl neon --kv-f16"
```

让 DFlash drafter 走 Vulkan，只需在同一命令追加 `--backend vulkan`。不要把它和
普通 Qwen 的逐算子 Vulkan 路径混为一谈：DFlash 专用执行器把整个 proposal block
录进一条 command buffer，权重和 KV 常驻 GPU，每块只等待一次 fence。

首次部署应分别跑普通 greedy 与 DFlash，比较两行 `generated_ids` 必须完全相同。
随后再看 `blocks`、`accepted`、`target_calls` 与 `decode`，不能只看接受率。

## 5. PLK110 实测（2026-09-14）

配置：OnePlus PLK110 / Android 16，Qwen3-0.6B target FP16，DFlash FP16，43-token
英文代码提示，生成 16 token，NEON matvec + ops，FP16 KV。

### 5.1 初始串行 FP16 路径

| 路径 | 块长 | 接受情况 | 主模型调用 | decode |
|---|---:|---:|---:|---:|
| 普通 greedy | — | — | 15 次单步 | 干净轮 412–422 ms |
| DFlash | 2 | 6/8 proposals（75%） | 9 次、17 个输入 | 507–555 ms |
| DFlash | 8 | 10/32 proposals（31.2%） | 5 次、37 个输入 | 1040 ms |

两条路径的 16 个 `generated_ids` 完全一致。DFlash 已经真实减少目标调用次数，
block=8 时平均每个验证块产出 `16/5 = 3.2` token；但目前在这台手机上没有延迟收益。
以同场较干净轮的最小值比较，block=2 约慢 20%–23%。设备后台负载会把普通解码
放大到 1.2–1.3 秒，因此被污染轮次中 DFlash 看似更快；这不是可归因的加速结论。

当时根因是 Android 的 FP16 `matmul` 尚未实现，目标块验证会回退为 N 次 FP16
matvec；草稿侧 block backbone 和共享 lm_head 也逐位置执行。

### 5.2 FP16 小块 GEMM 后

现已增加 FP16 weight-only `matmul` 分发、reference kernel 和 aarch64 NEON/常驻线程池
kernel。NEON kernel 一次加载权重、同时计算最多四个 token 列，并保持与既有 matvec
相同的四链累加顺序。目标 Transformer 验证、目标 verify lm_head、DFlash context/block
投影和 DFlash 共享 lm_head 全部改走批量路径；Markov 位置之间有真实 token 依赖，仍按
顺序执行。统计 JSON 新增 `draft_ms` 与 `target_verify_ms`，可以直接判断收益被哪一侧吃掉。

同一英文代码提示、同一模型与生成设置的更新结果：

| 生成长度 | 路径 | 块长 | 接受情况 | 主模型调用 | decode |
|---:|---|---:|---:|---:|---:|
| 16 | 普通 greedy | — | — | 15 次 | 较干净轮 296.5–306.0 ms |
| 16 | DFlash | 2 | 6/8（75.0%） | 9 次、17 输入 | 290.7–292.6 ms |
| 16 | DFlash | 8 | 10/32（31.2%） | 5 次、37 输入 | 最小 435.3 ms |
| 64 | 普通 greedy（6 线程） | — | — | 63 次 | 1361.7 ms |
| 64 | DFlash（6 线程） | 2 | 25/37（67.6%） | 38 次、75 输入 | 1629.8 ms |

16-token、block=2 已从负收益转为约 2%–5% 的小幅正收益，但仍接近设备噪声边界；
block=8 则从约 1040 ms 降到最小 435 ms，证明批量 kernel 确实生效，不过仍慢于普通
greedy。更能摊销固定成本的 64-token 配对实验中，两条路径的全部 64 个 token 逐位
一致：目标验证只用 1195.3 ms，比普通 decode 少约 166 ms，但 drafter 本身用了
434.0 ms，最终总耗时慢约 19.7%。另一次较干净 DFlash 轮为 1509.5 ms，仍比同线程
普通路径慢约 10.6%。因此不能把 16-token 的微弱领先宣称为稳定加速。

瓶颈已经发生转移：不再是“目标验证没有批量化”，而是这份 0.6B
DFlare + rank-128 Markov drafter 在 CPU 上的额外权重读取。尤其 Markov 链要求先得到
位置 `k-1` 的 token 才能构造位置 `k` 的 bias，不能像 backbone/lm_head 那样一次批量
完成。在当前 checkpoint 与手机 CPU 上，长序列仍没有复现论文 GPU 实验的端到端
加速；下一步要么优化/裁剪 drafter 与 Markov 输出头，要么使用更适合并行块解码的
加速器后端。

另一个实测现象：中文“用一句话介绍自己”样本的 draft acceptance 为 0%，而英文
问答/代码/数学样本为 19%–31%（block=8）。这说明单条 prompt 不能代表论文报告的
多数据集均值；评估 drafter 质量应复现相同数据集和 prompt 模板。

### 5.3 Android Vulkan FP16 drafter

Vulkan 实现分成两层：

1. `VulkanBackend` 提供通用 FP16 matvec/matmul，权重按 host 指针缓存到 GPU；其他
   算子复用 CPUBackend。它每个 matrix op 都会 submit + wait，真模型上正确但很慢，
   用于单算子测试和硬件 bring-up。
2. `DFlashVulkanEngine` 是实际 DFlash 路径。三层 DFlare、target residual fusion、
   context KV、非因果 block attention、共享 target lm_head、Markov bias 和 argmax
   全部留在同一 Vulkan command buffer；CPU 每块只上传新增 target hidden，并读回
   `block_size-1` 个 token id。

矩阵 shader 以 `uint` 成对读取 FP16 权重，用 `unpackHalf2x16` 转 FP32，并使用
16-lane `subgroupClusteredAdd` 做归约；一个 workgroup 同时计算 8 个输出行和最多 8 个
token 列。Markov W2 也使用打包 FP16 读取，但相邻位置有真实 token 依赖，仍必须顺序
执行。构建时 NDK `glslc` 把 shader 编译为 SPIR-V 并嵌入可执行文件。

当前 checkpoint 的 GPU 常驻权重约 473 MB（包含共享 lm_head），DFlash FP32 KV 在
`max_seq_len=256` 时约 6 MB。CPU 侧 target / draft 文件仍保留，所以这是额外内存而非
替代内存。Android 内存门禁读取 `/proc/meminfo` 的 `MemAvailable`，避免 Bionic 的
`_SC_AVPHYS_PAGES` 只统计完全空闲页而误拒绝可安全回收 page cache 的情况。

PLK110 / Adreno 840、同一 43-token 提示、生成 16 token、block=8、6 个 CPU worker
的相邻配对结果如下。所有路径输出完全相同的 16 个 token，接受统计也完全相同：
5 blocks、32 proposed、10 accepted（31.25%）、4 rollback、1 bonus。

| 路径 | draft | target verify | decode |
|---|---:|---:|---:|
| CPU DFlash，轮 1 | 160.73 ms | 387.70 ms | 548.54 ms |
| Vulkan DFlash，轮 1 | 155.12 ms | 484.37 ms | 639.64 ms |
| CPU DFlash，轮 2 | 166.23 ms | 404.76 ms | 571.11 ms |
| Vulkan DFlash，轮 2 | 185.71 ms | 486.25 ms | 672.11 ms |

GPU 确实执行并且一次块提交已生效；开发过程中的朴素 whole-block shader 在同一工作量
上约 579 ms，打包读取 + subgroup 归约的最好轮为 140 ms。但与已经高度优化的 ARM
CPU drafter 配对后，GPU draft 目前只是同量级，且 GPU 权重流会影响随后 CPU target
验证的频率/内存状态，端到端反而慢约 16%–18%。因此当前结论是：**GPU 后端功能与
正确性成立，论文式端到端加速尚未成立**。下一步的高价值工作是让 target verify 也
GPU-resident（并与 drafter 共用 device/lm_head），或提高 checkpoint 接受率以减少
block=8 的 37 个 target 输入；继续优化单个逐算子 submit 不会解决结构性瓶颈。

诊断时可用：

```bash
TINYQWEN_VULKAN_TRACE=1 ./tinyqwen ... --dflash-model draft.tqwen \
  --backend vulkan --speculative-tokens 8
```

每块会输出 `ctx`、`block`、录制命令数、fence wall time 和 proposal ids。它会增加
stderr I/O，正式 benchmark 不应开启。设备要求 Vulkan 1.2、FP16 shader、16-bit
storage、compute clustered subgroup；不满足时初始化会 fail-fast，而不会静默声称
正在使用 GPU。
