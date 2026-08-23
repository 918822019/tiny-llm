# 优化日志

> 每一次性能优化都在这里留一条记录：**改了什么、提升了多少、为什么提升**。
> 方法论见 `docs/optimization.md`（优化手册）。所有数字来自 `scripts/bench.sh` 的标准负载
> （固定 prompt、decode 32 token、丢弃预热 4、取稳态中位数），跨版本可比。

## 汇总表

主指标 = **decode 延迟中位数（ms/token）**，越小越好。

每行是一个**优化栈**（基线 + 已叠加的优化）。记**两个**加速比：
**vs 原始基线**（总共快多少）+ **vs 上一配置**（这一步贡献多少）。
多个优化叠加时的交互分析见 `docs/optimization.md` 第 7 节。

| 配置（优化栈）                                           | commit   | decode 中位 ms/tok |    p95 |            vs 原始基线 |                                          vs 上一配置 | 交互 / 归因                                                                                                                                   |
|---------------------------------------------------|----------|-----------------:|-------:|-------------------:|-------------------------------------------------:|-------------------------------------------------------------------------------------------------------------------------------------------|
| fp32-baseline（标量，无优化）                             | 5f679ea  |           230.36 | 252.35 |              1.00× |                                                — | fp32 标量、单线程、未量化，只求正确                                                                                                                      |
| fp32-double（对照：同配置复测）                             | 40b8e26  |           222.59 | 241.87 |          ≈1.0×（波动） |                                                — | 与基线同配置、换 commit 复测；当 float 的对照，顺带暴露 ~4% 运行波动                                                                                              |
| fp32-float（基线 + matvec float 累加，**已回退**）          | 40b8e26  |           204.12 | 269.79 |              1.13× |                                            1.09× | matvec 内层累加 double→float（Apple Silicon fp64 慢），计算类优化                                                                                      |
| restore-double（回退 fp32-float）                     | e1523c7  |           239.64 | 269.20 |              0.93× |                                            0.85× | 纪律性回退：_ref 恢复 double 累加，回到基线配置；float 累加将来以变体形式重做                                                                                          |
| double_2_float（fp32-baseline + matvec float 累加变体） | 9d1a572  |           219.84 | 245.48 |     1.01×（被机器波动掩盖） |                                            1.12× | matvec 内层累加 double→float，首次以**变体**形式合规落地；同场 A/B 才是真贡献                                                                                     |
| neon                                              | fd430db  |            28.25 |  30.14 |              7.88× |                                        8.05×（同场） | matvec NEON 向量化（4 路 FMA + 4 累加器展开）；decode 纯权重带宽瓶颈，有效带宽 8.7→72 GB/s，kernel 加速比几乎全额传导到端到端                                                   |
| acc4（阶梯 L2：隔离累加结构）                                | 2dfb2c8  |            89.32 |  99.92 |              2.49× |                                        2.65×（同场） | 标量 4 链并行累加，隐藏乘加延迟；阶梯口径 vs double_2_float = 2.41×                                                                                          |
| neon_nofma（阶梯 L3：隔离 SIMD 宽度）                      | 2dfb2c8  |            28.62 |  29.03 |              7.78× |                                        7.97×（同场） | NEON 向量化但故意不用 FMA；SIMD 是最大单项（vs acc4 = 3.02×），并撞带宽墙                                                                                       |
| neon_mt                                           | 56f51ba  |            10.50 |  11.26 |             21.19× |                                   2.69×（vs neon） | NEON + 常驻线程池行切分（默认 6 线程）：单核带宽 72 GB/s 打满后，多核接力整机带宽 ~199 GB/s；同场 vs ref 21.42×                                                             |
| neon_mt_bal                                       | a323eb7  |            10.07 |  14.42 |             22.11× |                             ≈1.0×（vs neon_mt，同场） | 自校准加权分块——带宽墙下无 E 核尾巴可消，证伪归档                                                                                                               |
| neon_mt_kv                                        | d17bc57  |            11.19 |  12.19 |             19.89× |              1.02–1.05×（vs neon_mt，同场 4 块 24 样本） | k/v 成对融合：48 次内联小 matvec 合并成 24 次 fork-join，摊薄+并行                                                                                          |
| neon_mt_kv_nt                                     | f4beeb7  |            10.06 |  11.13 |             22.12× |                     1.03–1.04×（vs neon_mt_kv，同场） | 权重 LDNP 流式加载（内联汇编 + 对齐兜底）：预期≈0 被证伪，提示真有效且尾部更稳                                                                                             |
| f16_neon_mt_kv_nt                                 | c8fba49  |             8.00 |   9.19 |             27.81× |                            1.75×（同场 vs fp32 最佳栈） | 权重 f16 量化（流量减半）+ fp32 满栈移植 + 线程甜蜜点 6→8；非 matvec 固定开销 Amdahl 稀释，未到 2×                                                                      |
| f16 + qkv/gate_up 融合（**fp16 路线终点**）               | 本次       |             6.37 |   6.77 |              36.2× |                      ≈1.1×（同场 A/B 乘积；跨窗绝对值受噪声污染） | qkv 三路融合 + gate/up 成对融合：fork-join 96→48 次/token。此后 fp16 路线正式关闭（见文末关闭小节）                                                                   |
| ops_neon                                          | 2cdbdd0  |             6.35 |   6.87 |             35.08× |                                        1.05×（同场） | 非 matvec 五算子 NEON 化 + ops dispatch：argmax 10×/attention 2.3×/swiglu 3.1×/rmsnorm 2.5×，profile 合计省 ~0.48ms；非 matvec 开销 694→200µs           |
| cuda（⚠️ 换机器：x86_64 + A10）                         | 7fc4a2d  |           333.75 | 356.47 | N/A（跨机器，基线是 M 系芯片） |                              1.75×（同场 vs 本机 ref） | matvec CUDA 参考变体（每次调用重传权重）：打通 CUDA 路径，朴素形态如实记录                                                                                            |
| cuda_resident（x86_64 + A10）                       | dd58fd2  |            47.40 |  48.07 | N/A（跨机器，基线是 M 系芯片） |            12.30×（同场 vs 本机 ref）；7.04× vs 朴素 cuda | matvec 权重常驻显存（按 host 指针缓存 device 副本），砍掉每 token ~2GB 的 PCIe 重传；GPU 从此反超本机最好 CPU 变体                                                         |
| cuda_resident_ws（x86_64 + A10）                    | aa30aa5  |            46.05 |  46.52 | N/A（跨机器，基线是 M 系芯片） | 1.03×（vs cuda_resident，≈无效）；12.67×（同场 vs 本机 ref） | x/y 常驻 workspace（删每调用 alloc/free）——证伪归档：CUDA 池化小分配，瓶颈实为 kernel 非合并访存                                                                      |
| cuda_resident_coal（x86_64 + A10）                  | 7356481  |             9.16 |   9.81 | N/A（跨机器，基线是 M 系芯片） |     5.17×（vs cuda_resident）；63.94×（同场 vs 本机 ref） | matvec kernel 改合并访存（block-per-row + warp 归约），warp 读权重从 32 条分散 cache line → 1 条满线；CUDA 路线首次逼近带宽地板                                          |
| f16_cuda_resident_coal（x86_64 + A10）              | 3408267  |             7.68 |   8.14 | N/A（跨机器，基线是 M 系芯片） |                       1.18×（vs f32_cuda_coal，同场） | 权重 fp16（流量减半）：理论 2× 被 Amdahl 稀释成 1.18×——时间大头已是固定开销而非带宽，诊断价值大于提速价值                                                                         |
| cuda_resident_coal_ws（x86_64 + A10）               | 716d63c  |             7.97 |   8.34 | N/A（跨机器，基线是 M 系芯片） |                  1.15×（vs cuda_resident_coal，同场） | 砍每调用开销：x/y 常驻 workspace（删 680 次 alloc/free）+ 删冗余 cudaDeviceSynchronize（同步 D2H 已保证完成）；kernel 未动                                            |
| f16_cuda_resident_coal_ws（x86_64 + A10）           | cf5c805  |             6.54 |   7.01 | N/A（跨机器，基线是 M 系芯片） |                       1.19×（vs f16_cuda_coal，同场） | f16 满栈 = f16_cuda_coal + x/y workspace + 删冗余 sync；A10 追平 Mac fp16 满栈量级（6.54 vs 6.35，跨机器仅参照）                                               |
| f16_cuda_fused（x86_64 + A10）                      | 47c5335  |             5.79 |   6.28 | N/A（跨机器，基线是 M 系芯片） |            1.07×（vs 部分融合，同场）；1.13×（vs 真不融合 6.54） | qkv 三合一 + gate_up 二合一 CUDA 融合（减 matvec 调用次数砍启动/拷贝），**5.79 压过 Mac fp16 满栈 6.35**（跨机器仅参照，目标达成）                                              |
| gpu_engine（x86_64 + A10）                          | 15e4653  |             4.89 |   5.23 | N/A（跨机器，基线是 M 系芯片） |                       1.19×（vs CPU 驱动 f16 满栈，同场） | GPU-resident 整段 forward（权重/激活/KV 常驻显存，单 stream），消灭逐 matvec 桥接；e2e 与 CPU 逐位一致，低于 Mac 6.35                                                  |
| fp16-neon-android（Android）                        | 54d05f4  |            21.16 |  24.33 |             32.87× |                                       32.88×（同场） | f16 满栈 kernel 落地 Android（带宽减半 + SIMD + 多线程），带宽类                                                                                           |
| 第二次测试（Android）                                    | 54d05f4  |           695.55 | 700.46 |              1.00× |                                                — | Android f16 标量 ref 基线（即后文各行引用的 android-fp16-ref-baseline 695.52），i4/跨模型 A/B 的对照锚点                                                        |
| i4-hqq-android（Android）                           | 54d05f4+ |            73.31 |  77.79 |              9.49× |                              8.12×（同场 vs i4 ref） | HQQ@64 量化 + i4 NEON matvec + lm_head 脱离标量 ref；但 **比 f16 满栈（21.16）慢**：i4 kernel 单线程 unpack 慢、fp32 lm_head 仍占 57% 流量，i4 kernel 优化是下一刀       |
| i4-neon_mt（macOS）                                 | 54d05f4+ |            10.99 |  12.29 |       —（i4 阶梯内部对比） |                   3.40×（vs i4 neon 单线程 37.40，同机） | i4 matvec 多线程行切分（RowPool 移植自 f16 neon_mt_kv_nt）；线程甜蜜点 8（5P+3，与 f16 时代一致），12 线程互踩回升                                                        |
| i4-lm_head（macOS）                                 | 54d05f4+ |            10.37 |  12.87 |       —（i4 阶梯内部对比） |                1.13×（vs lm_head-fp32 11.73，同机同场） | tied 模型导出独立 lm_head.weight i4 副本（embed 保持 fp32 lookup），流量 544→70MB/tok；Mac 带宽大收益温和，Android（42GB/s 墙）预期收益大                                 |
| i4-neon_lut（macOS，**证伪归档**）                       | 54d05f4+ |            38.78 |      — |                  — |                       0.31×（vs neon_mt 11.89，更慢） | per-group fp16 LUT 查表反量化：指令账漏算了"每组构造 16 项查找表"的标量开销——group=64 组数多，构造成本吞掉查表收益，反而慢 3.3×；且 fp16 激活+累加使单 matvec max_err ~0.02 超 1e-3 门禁。**弃用** |

| i4-hqq-lmh-mt-android（Android） | 54d05f4+ | 44.72 | 50.47 | 15.6×（vs android-fp16-ref-baseline 695.52） | 13.58×（同场 vs i4 ref 607） | HQQ@64 + lm_head i4 + neon_mt(2核) 全套落地 Android。**但仍比 f16 满栈(21.16)慢 2.1×**：i4 unpack 算力瓶颈 ~2.8GB/s/核，2 核仅 5.6GB/s，到不了 42 带宽墙。i4 慢不是 int4 之过（MNN int4 同机 3.68ms=68GB/s 为证），是我们 kernel 指令效率低 → 下一刀重写反量化内核 |
| i4-sdot_mt-android（Android，W4A8 SDOT） | 54d05f4+ | 22.35 | 22.50 | 31.1×（vs android-fp16-ref-baseline 695.52） | 2.00×（同场 vs i4 neon_mt 44.59） | **W4A8 SDOT**：权重解包 int8 + 激活 int8 对称量化 + SDOT 整数点积（vdotq_s32），unpack 指令/字节 3.5→~0.9。i4 从慢 f16 2.1× 收窄到 ~1.2-1.3×。**跨模型同场 A/B：i4 22.35 vs f16 满血 17.05 = 0.76×**（f16 带宽瓶颈随温度漂 17~21，i4 算力瓶颈稳定 22.3）——i4 优势是内存与热稳定，非绝对速度 |
| qwen35-f16-neon-android（Android，Qwen3.5-0.8B） | 54d05f4+ | 35.29 | 37.69 | —（跨模型，vs Qwen2.5-0.5B f16 ~17-21 = 慢 ~1.7-2×） | 30.19×（同场 vs qwen35 ref 1063） | Qwen3.5-0.8B 混合架构（GDN+full attn）f16 满栈首测。**lm_head 主导**：552.94ms/32=17.3ms/tok 占 49%（Qwen3.5 词表 248320→lm_head 508MB f16）。GDN 层高效 ~0.6ms/tok/层（O(1) seq）。peak RSS 1463MB（权重 1435 + GDN state 19.3 + KV 1.5） |
| i4-sdot2_mt（macOS，预计算+2-row） | 本次 | **3.67** | 4.79 | —（i4 阶梯内部对比） | **1.72×**（同场 vs sdot_mt 6.31，3轮中位） | **i4 首次反超 f16**：预计算 scale/zero 为 f32（消 per-group memcpy+half_to_float，省 ~39M inst/token ~10%）+ 2-row 并行内循环（2 条独立 SDOT 链 + 共享激活加载，ILP 翻倍）。同场 i4 sdot2 3.58 vs f16 满栈 5.68 = **1.59× i4 更快** |
| qwen35-4b-i4-sdot2_mt（macOS M4，4B 首测） | ea3aba3 | 50.30 | 52.38 | —（4B 首测，无历史基线） | — | Qwen3.5-4B i4（HQQ@64 + lm_head i4）M4 首测。有效带宽仅 ~57GB/s（含预计算缓存 +525MB/token 额外流量）远未及墙；**静态切分 E 核拖尾**：10 线程反慢于 4P（50.3 vs 45.9），6 线程最差（60.4） |
| qwen35-4b-i4-sdot3_mt（macOS M4） | 04788a5 | **46.73** | 48.57 | —（4B 阶梯内部对比） | **1.08×**（同场交替 3 轮 vs sdot2_mt 50.30） | sdot3：work-stealing 动态行调度（消异构核拖尾）+ 砍预计算缓存改内联组头硬件 FCVT（流量 −18%、RSS −525MB）+ 128 位解包（−37% 指令）。**副产物：TTFT 456→180ms（2.53×）**——Qwen3.5 prefill 逐 token 回退走 matvec，sdot2 的懒预计算原来记在首 token 上。0.8B 同场 10.74→10.53（1.02×，模型小收益温和）。数值与 sdot2 逐位一致 |
| qwen35-4b-i4-sdot4_mt（macOS M4） | 8d5447f | **36.50** | 38.51 | 1.38×（4B 内，vs sdot2_mt 50.30） | **1.29×**（同场交替 3 轮 vs sdot3_mt 46.91） | 修组头转换特性守卫：clang +fp16 根本不定义 `__ARM_FEATURE_FP16`（实际是 `__ARM_FEATURE_FP16_SCALAR_ARITHMETIC`），**sdot3 的硬件 FCVT 从未生效**，热循环静默跑软件转换（~4.6G 条指令/token，sample 第一大头）。改 `_Float16` cast（fmov+fcvt 2 条）。单核 160→119（1.35×）；有效带宽 50.5→64.8 GB/s；**E 核由负转正**：10 线程 36.0 < 4 线程 39.6。0.8B 同场 10.19→8.08（1.26×）。数值与 sdot3 逐位一致 |
| prefill_skip_logits（macOS M4，TTFT 专项） | 本次 | 36.49（TOPT 不变） | 38.42 | —（TTFT 专项，看右侧归因） | TOPT 1.00×；**TTFT 1.14×**（33-tok prompt 1320→1158ms） | Qwen3.5 prefill 逐 token 回退中，非末位 token 的 logits 被丢弃却全量计算 lm_head（4B 占单 token 16%）。forward_token 加 need_logits 门控跳过 final_norm+lm_head+argmax。微观证据：同 profile 内跳过位 29.96 vs 末位 35.77（Δ5.81≈lm_head 5.87ms）；0.8B TTFT ~21.4。对齐契约（--verbose 逐位置 dump）不受影响。TOPT 逐位不变 |
| backend_refactor | ebca4db | 234.96 | 263.50 | 0.95× | — | 纯后端抽象重构（非优化）：IBackend 虚分发开销在 ~4% 运行波动内不可辨识，带宽瓶颈路径上抽象零成本                                                                              |
<!-- 新的优化按时间顺序往上表追加行（优化栈 = 上一行 + 本次优化），并在下面补一个详细小节 -->

---

## 详细记录

### fp32-baseline（2026-08-12）

- **是什么**：v1 参考实现。FP32 标量 kernel、单线程、token-by-token、未做任何量化。
- **测量环境**：arm（Apple Silicon），commit `5f679ea`，Release `-O3`。
- **结果**：
    - decode 中位 **230.36 ms/token**（27 个稳态样本，mean 235.31，min 226.65，p95 252.35）
    - prefill：3 token 共 769 ms
- **时间花在哪**（top op）：`lm_head` ~2186ms（占比最大），其次是各层 `down_proj`。
  说明瓶颈在 matvec，且 decode 是**带宽瓶颈**（每 token 都要把全部权重读一遍）。
- **结论 / 下一步**：这是正确性基准，不追求快。优化主攻方向 = 量化 matvec
  （减少权重搬运）+ 后续多线程。复现命令：
  ```bash
  ./scripts/bench.sh fp32-baseline
  ```

---

### fp32-float（2026-08-13，**已回退**——见下方 restore-double）

- **优化栈**：fp32-baseline + matvec float 累加。
- **是什么**：`kernels/matvec_f32_ref.cpp` 内层点积累加器从 `double` 改回 `float`，
  去掉逐元素 double 强转。
- **假设**：计算类优化。matvec 内层每元素一次乘 + 一次加，decode 每 token 要跑
  几十次、每次几千迭代。Apple Silicon 的 fp64（double）吞吐远低于 fp32，
  用 double 累加等于在热点里塞了慢指令；改 fp32 应明显提速。
- **结果**：decode 中位 **204.12 ms/token**（27 样本，mean 215.00，min 199.28，p95 269.79）。
- **vs 原始基线**：1.13×（230.36 → 204.12）。
- **vs 上一配置**：1.09×（同 commit 对照 fp32-double 222.59 → 204.12）。
- **验证**：31/31 单测通过；canonical prompt 生成 token 与 double 版**逐位一致**
  （float 累加的舍入误差未翻转任何 greedy 决策）。
- **瓶颈转移**：top op 仍是 `lm_head` + 各层 `down_proj`（都是 matvec），
  说明方向没错——matvec 仍是主攻点，下一刀 = NEON 向量化 / 量化。
- **意外 / 教训**（最值钱的部分）：
    - **基线本身会波动**：同一配置在 5f679ea 测 230.36、在 40b8e26 复测 222.59，
      相差 ~4%。所以判断小幅优化时，要用**同 commit 的对照组**做 A/B，
      别直接跟历史数字比。
    - **p95 不稳**：float 这次 p95 269.79 反而比 double 的 241.87 高，
      这是单次 bench 的尾部噪声（热降频/后台），不代表回退；**以中位数为准**，
      要更稳就多跑几遍取中位数的中位数。
- **复现**：`./scripts/bench.sh fp32-float`

---

### restore-double（2026-08-13）

- **优化栈**：fp32-baseline（回退 fp32-float，回到基线配置）。
- **是什么**：**这不是优化，是纪律性回退**。撤销 `5e28c91` 对
  `matvec_f32_ref.cpp` 的 double→float 累加改动，`_ref` 恢复为 double 累加。
  原因：`_ref` 是"标准答案"（正确性基准 + 兜底），按
  `optimization.md` 的铁律**只增不改**；fp32-float 当时直接改了
  `_ref` 本体，违反这条纪律。float 累加的收益是真的（~9%），将来应以
  变体形式重新引入（或直接并入 NEON 变体——NEON float32x4 本来就是 fp32 累加）。
- **假设**：不适用（预期变慢 ~9%，换回的是 `_ref` 的数值可信度）。
- **结果**：decode 中位 **239.64 ms/token**（3 遍取中位，每遍 27 样本），p95 269.20。
- **vs 上一配置**：0.85×（vs fp32-float 204.12；跨 commit 比较，仅作参考）。
- **基线参照**：fp32-baseline（222.59 ms/tok @ 40b8e26），本次 vs 基线 = 0.93×
- **验证**：scripts/verify.sh（31 单测 + golden token 对照，逐位一致）。
- **瓶颈转移**：top op = `lm_head`、`layer_3.down_proj`、`layer_5.down_proj`，
  仍是 matvec 热点；下一刀 = NEON 变体（fp32 累加顺便把 float 的 ~9% 拿回来）。
- **意外 / 教训**：本次 239.64 比历史 double 测量（222.59/230.36）略高，
  在已知的 ~4% 基线波动 + 当日机器噪声范围内（本轮 3 遍散布 232.86–239.94）；
  再次印证"小幅差异看同 commit A/B，别跨 commit 比绝对值"。
  更大的教训就是本次回退本身：**优化必须走变体，不许碰 `_ref`**——
  否则"标准答案"自己变了，后面所有对齐都失去锚点。
- **复现**：`./scripts/bench.sh restore-double`

---

### double_2_float（2026-08-13）

- **优化栈**：fp32-baseline + matvec double_2_float 变体。
- **是什么**：被回退的 fp32-float 以**合规形式重做**：数值改动相同（内层累加
  double→float），但不再修改 `_ref`，而是新增 `kernels/matvec/matvec_f32_double_2_float.cpp`
  变体（kernel 函数体由用户手写、逐轮 review 补全），经 dispatch `kDouble2Float`
  可插拔选用，默认仍是 ref。配套：`--matvec-impl double_2_float` / 配置文件可选、
  bench/record 支持透传额外参数（9d1a572）。
- **假设**：计算类。Apple Silicon fp64 吞吐远低于 fp32，matvec 内层是绝对热点，
  去掉 double 累加的慢指令应提速 ~9%（fp32-float 已验证过方向）。
- **结果**：decode 中位 **219.84 ms/token**（3 遍取中位，每遍 27 样本），p95 245.48。
- **vs 上一配置**：**1.12×**——同场 A/B：同 commit（9d1a572）同场先测 ref 对照
  246.02 ms/token，再测变体 219.84。（vs 历史基线仅 1.01×，见教训 1。）
- **基线参照**：fp32-baseline（222.59 ms/tok @ 40b8e26），本次 vs 基线 = 1.01×
- **验证**：scripts/verify.sh（33 单测：新增 dispatch 切换测试 + 变体 vs ref
  对齐测试，in_dim=257、容差 5e-3；golden token 对照）；变体模式跑真模型，
  canonical prompt 16 个生成 token 与 ref **逐位一致**（float 舍入未翻转 greedy）。
- **瓶颈转移**：top op = `lm_head`、`layer_23.down_proj`、`layer_22.down_proj`，
  仍是 matvec；下一刀 = NEON 向量化（NEON float32x4 天然 fp32 累加，本变体是它的标量前奏）。
- **意外 / 教训**：
    1. **vs 历史基线只有 1.01×，差点误判"优化无效"**——当天机器整体偏慢
       （ref 同场对照 246.02 vs 历史 222.59），把真实收益淹没了。同场 A/B 才看出 1.12×。
       optimization.md 的纪律再次应验：小幅优化只认同场对照，不认历史数字。
    2. **纪律不吃亏**：同一个数值改动，直接改 `_ref` 时违反铁律被回退；以变体
       重做后收益相同（1.09× → 1.12×），还额外得到可切换、可 A/B、可兜底。
    3. 变体优化暴露了基建缺口（bench 传不进开关），先补基建再测——工具链也是
       优化 pipeline 的一部分。
- **复现**：`./scripts/bench.sh double_2_float --extra-args "--matvec-impl double_2_float"`

---

### neon（2026-08-13）

- **优化栈**：fp32-baseline + matvec neon 变体（与 double_2_float 互斥——dispatch
  单选，本变体替代它成为当前优化选择，默认仍是 ref）
- **是什么**：新增 `kernels/matvec/matvec_f32_neon.cpp`：NEON float32x4 四路 SIMD，
  `vfmaq_f32` 乘加融合，4 路展开 + 4 个独立累加器隐藏 FMA 延迟（每迭代 16 元素在飞），
  `vaddvq_f32` 横向归约，标量尾段补齐；自注册进 dispatch，`--matvec-impl neon` 选用。
  配套门禁单测 `matvec_neon_matches_ref`（in_dim=4103 全覆盖三级尾段，容差 5e-3）。
- **假设**：指令类 + 带宽类。decode 的 matvec 是纯权重流式读取的带宽瓶颈型负载
  （每 token 读约 2GB 权重），ref 标量只有 ~8.7 GB/s（double 累加 + 串行依赖，
  计算端也没喂饱带宽）；SIMD + FMA + 多累加器去掉计算瓶颈后，带宽利用率应大幅
  上升，且收益会几乎不打折地传导到端到端（matvec 占算子时间 98%）。
- **结果**：decode 中位 **28.25 ms/token**（3 遍取中位，每遍 27 样本），p95 30.14
- **vs 上一配置**：**8.05×（同场 A/B）**——对照（无额外参数，当前即 ref）中位 227.28（p95 236.63）→ 变体 28.25，同 binary
  同场交错测量。（vs 基线 7.88×）
- **基线参照**：fp32-baseline（222.59 ms/tok @ 40b8e26），本次 vs 基线 = 7.88×
- **验证**：scripts/verify.sh（34 单测全过，含新增 `matvec_neon_matches_ref`：
  in_dim=4103 = 256×16+7，主循环/向量/标量三级尾段全覆盖，vs ref 容差 5e-3，
  micro-bench 实测 max_err 3.3e-9；golden token 16 个与 ref 逐位一致——
  float 舍入未翻转 greedy）。
- **瓶颈转移**：top op = `lm_head`、`layer_17.up_proj`、`layer_22.down_proj`，
  但 profile 里 matvec 仍占算子时间 **98%**——没有新瓶颈冒头，只是同一个瓶颈
  从 227ms/tok 缩到 ~27.5ms/tok。lm_head 单算子 ~10.5ms/tok（545MB，~52 GB/s），
  全模型端到端有效带宽 ~72 GB/s（ref 只有 ~8.7 GB/s）。下一刀仍在 matvec：
  ① INT4 量化——带宽瓶颈型负载，权重流量直接砍到 1/8，理论还能再快数倍；
  ② 多线程——单核 ~72 GB/s 未吃满内存带宽，并行摊开读权重。
  其余算子（attention/rmsnorm/rope/swiglu 合计 <2%）不值得单独 NEON 化。
- **意外 / 教训**：
    1. **端到端 8.05× ≈ kernel micro-bench 7–10×，几乎零 Amdahl 稀释**——因为
       decode 是纯 matvec 带宽瓶颈（算子时间 98%），kernel 快多少端到端就快多少。
       对照 double_2_float 只有 1.12×：瓶颈在带宽时，标量计算微优化天花板很低，
       改变"读内存的方式"（向量化提高利用率、量化减少流量）才是量级收益。
    2. **micro-bench 惊现 2.8e9 误差，罪魁是 bench 自己不是 kernel**：数据生成
       里 `size_t` 减法下溢回绕，权重变成 ~1.8e16 的假大数，float/double 累加
       在该量级本来就会差出部分和的 ulp（~1e9）。教训：误差异常先看数据生成，
       再看误差量级——和 partial sum 的 ulp 同阶多半是精度现象而非逻辑错。
       换成合法数据后同一 kernel 以 3.3e-9 通过 4103 维门禁。
    3. 本次同场对照 227.28 与历史基线 222.59 接近，机器状态稳定，vs 基线 7.88×
       与同场 8.05× 罕见地都可信；但流程上仍然只以同场 A/B 为准。
- **复现**：`./scripts/bench.sh neon --extra-args "--matvec-impl neon"`

---

### acc4（2026-08-13）

- **优化栈**：fp32-baseline + double_2_float + acc4（**归因阶梯第 3 层**，
  隔离"累加结构"；阶梯全貌见下方 neon_nofma 条目）。
- **是什么**：新变体 `kernels/matvec/matvec_f32_acc4.cpp`：仍是标量、仍是
  float 累加，唯一变化——把一条自依赖的累加链拆成 4 条互不依赖的链
  （咖啡机类比见该文件头）。
- **假设**：指令/延迟类。浮点乘加延迟 ~3-4 周期，单链每步都等上一步的
  结果，流水线空转；4 条独立链让乱序执行把延迟重叠。
- **结果**：decode 中位 **89.32 ms/token**（3 遍取中位，每遍 27 样本），p95 99.92
- **vs 上一配置**：**2.65×（同场 A/B）**——对照（无额外参数，当前即 ref）中位 237.08（p95 256.26）→ 变体 89.32，同 binary
  同场交错测量。（vs 基线 2.49×）
  按阶梯口径 vs 真正的上一配置 double_2_float：同场阶梯扫描
  211.78 → 87.92 = **2.41×**。
- **基线参照**：fp32-baseline（222.59 ms/tok @ 40b8e26），本次 vs 基线 = 2.49×
- **验证**：scripts/verify.sh（36 单测：新增 acc4 vs ref 对齐测试，
  in_dim=4103 覆盖标量尾段；golden token 逐位一致）
- **瓶颈转移**：top op = `lm_head`、`layer_8.down_proj`、`layer_16.down_proj`，
  仍是 matvec；标量循环从延迟受限转向吞吐受限，下一层 = SIMD 向量化。
- **意外 / 教训**：**没想到纯结构改动（无 SIMD、无指令变化）值 2.4×**——
  标量单链的延迟受限程度远超直觉。这正是归因阶梯的价值：若只测
  ref→neon 的整包 8×，这份功劳会被错误归给 SIMD。
- **复现**：`./scripts/bench.sh acc4 --extra-args "--matvec-impl acc4"`

---

### neon_nofma（2026-08-13）

- **优化栈**：fp32-baseline + double_2_float + acc4 + neon_nofma
  （**归因阶梯第 4 层**，隔离"SIMD 宽度"——乘加刻意拆成
  vmulq+vaddq 两条指令，把 FMA 的贡献留给下一层单独量）。
- **是什么**：新变体 `kernels/matvec/matvec_f32_neon_nofma.cpp`：把 acc4
  的 4 条标量链升级为 NEON 向量链（一条指令 4 个 float），其余结构
  （4 链、尾段、归约）与 neon 完全一致，唯一差别是循环内不用 vfmaq。
- **假设**：指令效率/宽度类。一条指令处理 4 个元素，预期对标量 4 链
  接近 4×（若未提前撞带宽墙）。
- **结果**：decode 中位 **28.62 ms/token**（3 遍取中位，每遍 27 样本），p95 29.03
- **vs 上一配置**：**7.97×（同场 A/B）**——对照（无额外参数，当前即 ref）中位 228.00（p95 240.01）→ 变体 28.62，同 binary
  同场交错测量。（vs 基线 7.78×）
  按阶梯口径 vs 真正的上一配置 acc4：同场阶梯扫描
  87.92 → 29.10 = **3.02×**。
- **基线参照**：fp32-baseline（222.59 ms/tok @ 40b8e26），本次 vs 基线 = 7.78×
- **验证**：scripts/verify.sh（36 单测：新增 neon_nofma vs ref 对齐测试，
  in_dim=4103 覆盖三段尾段；golden token 逐位一致）
- **瓶颈转移**：top op = `lm_head`、`layer_19.gate_proj`、`layer_23.gate_proj`，
  仍是 matvec，但已到权重带宽墙（有效带宽 ~72 GB/s，见 neon 条目）——
  **算术指令数不再是瓶颈**，这直接预言了下一层 FMA ≈ 0（neon 条目同场
  复测 29.42 vs 本条 29.10，FMA 贡献在噪声内）。
- **意外 / 教训**：
    1. **SIMD 是最大的单项贡献（3.02×），但低于理论 4×**——加载指令与
       带宽/cache 已开始稀释收益；撞墙信号在下一层被 FMA≈0 坐实。
    2. **归因阶梯完整账本**（同场连测，每层只加一个技术）：

       | 阶梯层 | 实现 | 同场中位 ms/tok | vs 上一层 | 累计 vs ref | 归因 |
            |---|---|---:|---:|---:|---|
       | L0 | ref | 229.89 | — | 1.00× | 基线（标量 double 累加） |
       | L1 | double_2_float | 211.78 | 1.09× | 1.09× | float 累加（去 fp64 慢指令） |
       | L2 | acc4 | 87.92 | **2.41×** | 2.61× | 4 链并行累加（隐藏延迟） |
       | L3 | neon_nofma | 29.10 | **3.02×** | 7.90× | SIMD 向量化（一条指令 4 元素） |
       | L4 | neon | 29.42 | ≈1.00× | 7.81× | FMA 融合——**撞带宽墙后贡献≈0** |

    3. 若当初只做"ref vs neon"一个对比，会把 8× 全部归功于"NEON/FMA"；
       阶梯拆开后真相是：**累加结构 2.4× + SIMD 宽度 3.0× + FMA ≈ 0**。
       FMA 的价值暂时只在数值侧（单次舍入）与未来余量（量化减少搬运后
       若算术重新成为瓶颈，它才会显形）。
- **复现**：`./scripts/bench.sh neon_nofma --extra-args "--matvec-impl neon_nofma"`

---

### neon_mt（2026-08-13）

- **优化栈**：fp32-baseline + matvec neon_mt 变体（与 neon/double_2_float 互斥——
  dispatch 单选，本变体替代 neon 成为当前优化选择，默认仍是 ref）
- **是什么**：新增 `kernels/matvec/matvec_f32_neon_mt.cpp`：NEON 行点积之上叠加
  多线程行切分。常驻线程池（原子 generation 自旋唤醒，无 OS 锁），master 也
  参与算第 0 块后自旋等归位；权重 <1MB 的小矩阵（k/v_proj）内联单线程；
  默认并行度 = Apple P 核数 + 1，`TINYQWEN_MT_THREADS` 可调。
- **假设**：并行类 + 带宽类。单核 NEON 已打到 ~72 GB/s（单核份额天花板），
  但 decode 每 token ~2GB 权重流量是纯带宽瓶颈，整机统一内存带宽远未吃满；
  多核并行读权重应接近线性提速，直到撞上总带宽上限。
- **结果**：decode 中位 **10.50 ms/token**（3 遍取中位，每遍 27 样本），p95 11.26
- **vs 上一配置**：**2.69×（vs neon：28.25 → 10.50）**——这才是本步的真实贡献。
  同场 A/B 对照（无额外参数 = ref）中位 224.93（p95 236.09）→ 变体 10.50，
  即 vs ref 21.42×，同 binary 同场交错测量。
- **基线参照**：fp32-baseline（222.59 ms/tok @ 40b8e26），本次 vs 基线 = 21.19×
- **验证**：scripts/verify.sh（37 单测全过，含新增 `matvec_neon_mt_matches_ref`：
  并行路径 out_dim=301×in_dim=4103 不整除线程数 ×8 轮重复压线程池、内联路径
  小矩阵、行数 3 < 线程数的空块极端形；golden token 16 个与 ref 逐位一致）。
- **瓶颈转移**：top op = `lm_head`、`topk_argmax`、`layer_3.up_proj`。matvec
  占比 98% → **93.9%**，有效带宽 72 → **~199 GB/s**（1.97GB ÷ 9.9ms/tok）——
  整机带宽已吃了大半，matvec 单点继续堆核收益递减。标量小算子开始露头：
  attention 2.1%、topk_argmax 1.9%、swiglu 1.6%（合计 ~6%）。下一刀：
  ① INT4 量化——权重流量 ÷8，带宽瓶颈下最直接的量级收益，且能重新打开
  多线程的带宽空间；② topk_argmax/swiglu/attention 的 NEON 化（从 6% 里抠）。
  优先①。
- **意外 / 教训**：
    1. **线程甜蜜点是 6（5P+1E），不是越多越好**：扫了 TINYQWEN_MT_THREADS，
       5 线程 10.9ms、6 线程 10.2、8 线程起抖动变大（10.3~12.7）、10+ 稳定更差。
       带宽瓶颈下并行度被总带宽封顶：6 线程已到 ~199 GB/s，再加核只是互踩。
       默认值因此取 P 核数 + 1（E 核单核带宽低，但白送的那一份还是正的）。
    2. **实测 2.69× < 线程数 6 的理论值——这是带宽版的 Amdahl**：单核 72 →
       整机 199 GB/s 只有 ~2.8× 的空间，并行收益在带宽墙处饱和，与线程数无关。
       反过来看，这正预告了量化（流量 ÷8）才是下一阶段的主升浪。
    3. **同步机制是这个变体的生死线**：每 token ~170 次 matvec 调用，若每次
       现起线程或用 mutex/condvar，微秒级开销 ×170 会把收益吃光。本实现用
       原子 generation + 自旋（master release-store 发布、worker acquire 读参、
       done 计数归位），单次 fork-join 亚微秒，全程无系统调用。粒度阈值
       （<1MB 内联）同样关键：k/v_proj 只有 0.45MB，并行必亏。
- **复现**：`./scripts/bench.sh neon_mt --extra-args "--matvec-impl neon_mt"`

---

### neon_mt_bal（2026-08-14）

- **优化栈**：fp32-baseline + neon_mt_bal（= neon_mt + 自校准加权分块；与
  neon_mt 互斥。真上一配置是 neon_mt，record 的对照是 ref，见 vs 上一配置）
- **是什么**：新增 `kernels/matvec/matvec_f32_neon_mt_bal.cpp`：neon_mt 的均分
  行块改为**按实测吞吐加权分块**——每个 worker 记录块耗时，master 用 EMA
  （新样本 0.3）维护各线程速度估计，下一 job 按速度比例切行（前缀和取整），
  目标是快核多算、慢核少算、同时完工。热循环零原子开销。
- **假设**：并行/均衡类。本机 5 超级核 + 10 性能核，默认 6 线程里那 1 个会被
  调度到慢核；均分下慢核成尾巴、快核空转等它。按吞吐加权分块应消尾。
- **结果**：decode 中位 **10.07 ms/token**（3 遍取中位，每遍 27 样本），p95 14.42
- **vs 上一配置**：**≈1.0×（vs neon_mt，同场）——假设被证伪**。record 机制对照
  是 ref：232.71 → 10.07 = 23.11×（含 neon_mt 全部机制，不是本步贡献）。
  本步真贡献用 vs neon_mt 的同场阶梯测：6 轮交错单遍 + 2 个 runs=3 块状对照
  （含交换顺序抗顺序红利），两变体在 ±3% 内互换；块状对照里 mt 反而略快
  （9.76 vs 10.41、9.99 vs 10.38，bal 多出的 ~2-4% 疑为计时/EMA 开销）。
- **基线参照**：fp32-baseline（222.59 ms/tok @ 40b8e26），本次 vs 基线 = 22.11×
- **验证**：scripts/verify.sh（38 单测：新增 bal 的 4 组 shape 对齐——不均分
  取整、校准收敛全程、shape 交替、空块极端形；golden token 16 个逐位一致）
- **瓶颈转移**：top op 不变（`lm_head`、`topk_argmax`、各层 gate/up_proj），
  因为本步无效。下一刀仍按原计划：k/v_proj 小 matvec 融合并行（变体 2）。
- **意外 / 教训**：
    1. **假设被证伪，且机理说得通**：decode 已深陷共享带宽墙——每核带宽被整机
       额度限制（5 核合计才 ~181 GB/s，单核 ~36，远低于单核独占时的 72），
       线程间完成时间差本就很小，join 处的"尾巴"无从谈起。均分早已足够好。
       与 neon_nofma 条目"FMA≈0"同构：**撞墙之后，均衡/算术类优化天花板归零，
       只有改变流量（量化）或改变流量利用率才有收益**。
    2. **顺序红利差点造成假阳性**：首轮交错 bal 恒后测、机器恰在整体变快，
       得出 1.07-1.24× 的假优势；交换顺序后优势消失。教训：手工阶梯扫描必须
       双向交换顺序，或直接用块状 runs=N 对照。
    3. 变体按纪律保留在树中（可切换、可复测），但不进优化栈；下一步从
       neon_mt 继续。
- **复现**：`./scripts/bench.sh neon_mt_bal --extra-args "--matvec-impl neon_mt_bal"`

---

### neon_mt_kv（2026-08-14）

- **优化栈**：fp32-baseline + neon_mt_kv（= neon_mt + k/v 成对融合；与
  neon_mt/neon_mt_bal 互斥。真上一配置是 neon_mt，record 的对照是 ref）
- **是什么**：dispatch 新增成对入口 `matvec_pair_f32`（y1=W1@x、y2=W2@x 共享
  输入；未注册 pair 的 impl 兜底为调两次 matvec_f32，数值不变），runtime 的
  k/v_proj 合并为一次调用；新变体 `kernels/matvec/matvec_f32_neon_mt_kv.cpp`
  注册 pair 实现：两个 128×896 矩阵的行拼成 256 行的单个 fork-join job。
  行内点积与 neon_mt 逐位一致，改的是调用粒度。
- **假设**：并行/调度类。k/v_proj 各 0.45MB 够不着 1MB 并行阈值，每 token
  48 次 ~6µs 内联单线程 matvec 合计 ~0.3ms；合并后 0.9MB 过 pair 阈值
  （0.5MB），24 次 fork-join 摊薄同步开销且 6 线程并行，预期省一半左右。
- **结果**：decode 中位 **11.19 ms/token**（3 遍取中位，每遍 27 样本），p95 12.19
- **vs 上一配置**：**1.02–1.05×（vs neon_mt，同场）**。record 机制对照是 ref：
  231.52 → 11.19 = 20.69×（含 neon_mt 全部机制，不是本步贡献）。本步真贡献：
  4 个 runs=3 块状对照（AC 块 mt 先测、BD 块 kv 先测，抗顺序红利与慢漂移），
  逐块比值 1.17× / 1.00× / 1.04× / 1.02×，kv 无一败绩；合并 24 个 per-run
  样本中位数 mt 11.49 vs kv 11.27，差 0.22ms，与"0.3ms 内联时间砍半"的
  假设量级吻合。（vs 基线 19.89×）
- **基线参照**：fp32-baseline（222.59 ms/tok @ 40b8e26），本次 vs 基线 = 19.89×
- **验证**：scripts/verify.sh（40 单测：新增 pair 兜底中性门禁——ref/
  double_2_float 下 pair 入口与分开调逐位一致；neon_mt_kv 的 4 组 shape——
  单矩阵并行、pair 并行行映射、pair 内联、总行数 6 < 线程数的空块极端形；
  golden token 16 个与 ref 逐位一致——runtime 改调用方式未改变数值）
- **瓶颈转移**：top op = `lm_head`、`topk_argmax`、`layer_4.down_proj`，
  结构未变（k/v 本就不在 top）。下一刀：流式加载提示（neon_mt_kv_nt，
  实验性，预期≈0）；fp32 路线扫完后转量化（权重流量 ÷N）。
- **意外 / 教训**：
    1. **收益小但方向稳**：单看任一块都在噪声边缘（1.00–1.17×），靠 4 块
       双向换序 + 合并样本才把"从不输"立住。2% 级优化的判据不是单次比值，
       是多块一致性——这是对 bal 条目"顺序红利假阳性"教训的正向应用。
    2. **兜底设计让基建零风险**：pair 入口对未注册的 impl 等价于原调用，
       所以 runtime 可以无条件切换调用方式，golden token 在全部 impl 下
       保持逐位一致——基建改动与优化收益解耦，这是 dispatch 模式的又一例。
    3. 今日机器整体偏慢偏噪（变体中位 11.2 vs 历史 neon_mt 10.5），
       vs 基线比值参考性弱，一切以同场对照为准。
- **复现**：`./scripts/bench.sh neon_mt_kv --extra-args "--matvec-impl neon_mt_kv"`

---

### neon_mt_kv_nt（2026-08-14）

- **优化栈**：fp32-baseline + neon_mt_kv_nt（= neon_mt + k/v 融合 + 流式加载；
  归因阶梯顶层：neon_mt →(kv)→ +nt。bal 已证伪不进栈；与 neon_mt_kv 互斥。
  真上一配置是 neon_mt_kv，record 的对照是 ref）
- **是什么**：新增 `kernels/matvec/matvec_f32_neon_mt_kv_nt.cpp`：权重加载从
  vld1q 换成 **LDNP 内联汇编**（`ldnp q0, q1, [ptr]`，一条指令 32B 非时间
  加载），x 向量保持普通加载（它是重用的热数据）；LDNP 要求 16B 对齐，
  逐行检查、不对齐兜底普通路径。其余（4 链 FMA、线程池、pair 融合、阈值）
  与 neon_mt_kv 一致。
- **假设**：预期 **≈0，做的是排除法**——decode 权重是纯流式访问（每 token
  全量读一遍），理论上非时间提示能少污染 cache，但 Apple 硬件预取器极强、
  且已撞带宽墙（FMA≈0 的前科），加载侧微调大概率归零。测过才能把这块
  石头从 fp32 路线上搬走。
- **结果**：decode 中位 **10.06 ms/token**（3 遍取中位，每遍 27 样本），p95 11.13
- **vs 上一配置**：**1.03–1.04×（vs neon_mt_kv，同场）**——≈0 预期被证伪。
  4 个 runs=3 块状对照（AC 块 kv 先测、BD 块 nt 先测），逐块 1.08× / 1.01× /
  1.03× / 1.04×，nt 全胜；合并 24 个 per-run 样本中位数 kv 10.35 vs nt 10.00。
  record 机制对照是 ref：227.18 → 10.06 = 22.58×（含阶梯全部机制，不是本步
  贡献）。（vs 基线 22.12×）
- **基线参照**：fp32-baseline（222.59 ms/tok @ 40b8e26），本次 vs 基线 = 22.12×
- **验证**：scripts/verify.sh（41 单测：新增 5 组 shape——对齐 LDNP 主路径
  （in_dim=4104）、非对齐兜底路径（4103）、对齐 pair、对齐内联、6 行空块
  极端形；二进制 otool 确认 LDNP 真实发射（16 处）；golden token 16 个与
  ref 逐位一致）
- **瓶颈转移**：top op = `lm_head`、`topk_argmax`、`layer_20.gate_proj`，
  结构未变。fp32 路线至此扫完（均衡/调用粒度/加载指令三块石头全部测过）：
  下一刀 = **量化**（fp16 流量÷2 / int4 流量÷8，带宽瓶颈下唯一剩余的量级
  空间），次选 topk_argmax/swiglu/attention NEON 化（合计 ~6% 的零头）。
- **意外 / 教训**：
    1. **预期≈0 的优化测出 3-4%**：流式提示在 M5 Pro 上真实有效——可能来自
       L2 污染减少（x/激活的热区不再被 2GB 权重流冲刷）和加载 µop 减半的
       叠加。本次测量分不开"提示"与"循环重构"两个因素，如实记录不硬归因。
    2. **先验 codegen 再谈测量**：clang 的 `__builtin_nontemporal_load` 对本
       工具链的 NEON 向量加载静默丢弃提示（仍发普通 ldp）——若没先查 codegen
       就"测出≈0"，会把假 no-op 当成真结论归档。指令类优化的第一步永远是
       反汇编确认指令真的发射了。
    3. **nt 的尾部明显更稳**：24 样本里 kv 最差 12.46、nt 最差 10.43，p95
       系统性更低——流式加载对后台 cache 抢占的抗干扰是额外赠品。
- **复现**：`./scripts/bench.sh neon_mt_kv_nt --extra-args "--matvec-impl neon_mt_kv_nt"`

---

### f16_neon_mt_kv_nt（2026-08-14）

- **优化栈**：fp32-baseline +（fp32 路线全部机制）+ **权重 f16 量化**——本步
  切换赛道：不再优化"读得多快"，而是"少读一半"。实现名 neon_mt_kv_nt 在
  f16 注册表里是满栈移植版（NEON fp32 累加 + 线程池 + kv 融合 + LDNP），
  与 f32 同名实现按模型文件 dtype 解析（main 按 dtype 查表，选错 fail fast）。
- **是什么**：一条完整的 weight-only f16 链路：① exporter `--dtype f16` 导出
  model_f16.tqwen（942MB，原 1.98GB 的一半，RNE 量化与 C++ 端位操作转换
  逐位同款）；② loader 接受 f16（单一 dtype 校验）；③ dispatch 新增 f16
  注册表/入口/pair 兜底；④ runtime dtype 感知：大矩阵 f16 流式，norm/bias
  小向量 create 期升 fp32（热点路径不改），embed 查表逐行转换；
  ⑤ kernels：matvec_f16_ref（double 累加标准答案，全平台兜底）+
  matvec_f16_neon_mt_kv_nt（满栈移植，vld1q_f16/cvt/fp32 FMA）；
  ⑥ 默认线程数 6→8（P 核+3，见教训 2）。
- **假设**：减少搬运类。decode 是纯权重带宽瓶颈（fp32 时每 token 读 1.97GB），
  f16 流量减半，理论端到端 ~2×；且带宽墙后退，多线程空间重新打开。
- **结果**：decode 中位 **8.00 ms/token**（3 遍取中位，每遍 27 样本），p95 9.19。
  注：record 时窗机器噪声大（安静时窗同配置测过 6.29–6.99），绝对值参考性弱，
  以同场比值为准。
- **vs 上一配置**：**1.75×（同场 A/B）**——对照（model.tqwen fp32 + 最佳栈
  neon_mt_kv_nt）中位 14.04（p95 15.06）→ 变体 8.00，同 binary 同场交错，
  逐轮比值 1.64× / 1.76× / 1.76× 方向全一致。（vs 基线 27.81×）
- **基线参照**：fp32-baseline（222.59 ms/tok @ 40b8e26），本次 vs 基线 = 27.81×
- **验证**：scripts/verify.sh（47 单测：half↔float 全 65536 位形往返穷举、
  f16_ref vs 量化后精确值、满栈变体 5 组 shape vs f16_ref、loader f16/混
  dtype 拒绝、f16 dispatch 选择；golden token 对照 fp32 路径不变）。
  **量化数值验收**（optimization.md §5）：f16 真模型跑 canonical prompt，
  标量参考与优化 kernel 下 16 个生成 token 均与 fp32 参考**逐位一致**——
  量化未翻转任何 greedy 决策（且 top1-top2 logit 间距大）。注意仅覆盖该
  prompt，长文本生成需另行抽检。
- **瓶颈转移**：top op = `lm_head`、`topk_argmax`、`layer_10.up_proj`。
  matvec 占比从 93.9% 降到 ~85%——流量减半后**非 matvec 固定开销显形**
  （topk_argmax 标量扫 151936 维 ~0.26ms/tok 升到 3%+）。下一刀：
  topk_argmax/swiglu/attention NEON 化（从零头里抠）；再下一刀 int4
  （流量再减半，但 0.5B 小模型 int4 的 greedy 翻转风险需要专门验收）。
- **意外 / 教训**：
    1. **没到 2×，止于 1.75×——Amdahl 的量化版**：matvec 本体接近 2×，
       但非 matvec 开销（argmax/attention/rmsnorm/同步）不随流量缩小，
       占比被动放大。fp32 时代"matvec 占 98%"的舒适区结束了：从这一步起
       每个非 matvec 算子都值得看一眼。
    2. **线程甜蜜点右移 6→8**（预测兑现）：流量减半后单核更晚撞份额墙，
       多 2 个 E 核还能挤带宽；10 线程互踩（8.98 vs 8 线程 6.99）。实测
       扫描写进了变体代码注释；默认并行度从此是"按 dtype 分别标定"的量。
    3. **数值验收比预期顺**：f16 的 ~5e-4 相对扰动没翻转任何 greedy 决策，
       double_2_float 时代验证过的容差余量在这里继续有效。但别外推：
       更长 prompt/更敏感的生成任务需要抽检，int4 更是要专门过这道门。
    4. **基建缺口先补再测**（第三次应验）：record 的 A/B 原本只会"同文件
       vs 无参数对照"，跨文件对照要扩展 --control-model/--control-args
       （含漂移检查语义：跨配置对照跳过基线漂移检查）。工具链永远是
       pipeline 的一部分。
- **复现**：`MODEL=model_f16.tqwen ./scripts/bench.sh f16_neon_mt_kv_nt --extra-args "--matvec-impl neon_mt_kv_nt"`

---

### f16_qkv_gateup_fusion（2026-08-14）

- **优化栈**：fp32-baseline +（fp32 全部机制）+ 权重 f16 + **qkv 三路融合 + gate/up 成对融合**
- **是什么**：两处调用粒度融合，把每 token 的 fork-join 次数从 96 降到 48：
  ① **gate/up 成对融合**：FFN 的 gate_proj 和 up_proj 共享输入 normed_ 且维度
  相同 [4864,896]，合并为一次 `mv_pair`——零新基建，直接复用现有 pair 机制；
  ② **qkv 三路融合**：q_proj [896,896] 与 k/v [128,896] out_dim 不同，pair API
  不适用，新增 dispatch qkv 入口（`matvec_qkv_f32/f16`，兜底 = mv(q) + pair(k,v)，
  未注册的 impl 行为不变）；kernel 侧 RowPool 新增 kQkv 模式，总行数
  q_dim+2×kv_dim=1152 一次 fork-join 均分给线程池。A/B 开关：
  `--no-fuse-gate-up` / `--no-fuse-qkv`（配置项 fuse_gate_up / fuse_qkv）。
- **假设**：调度类。fork-join 自旋唤醒延迟实测 ~15-20µs/次（远超此前 <1µs
  的估计），每 token 省 48 次同步应省 ~0.7-1ms。
- **结果**：decode 中位 **6.37 ms/token**（安静窗口 runs=3 取中位，p95 6.77）。
- **vs 上一配置**：**≈1.1×（同场 A/B 乘积）**——两个融合各自做块状对照
  （4 轮双向换序）：gate_up ON 赢 7/8 轮（可靠轮 1.04-1.07×），qkv ON 赢
  5/8 轮（剔除漂移污染轮 1.05-1.06×）。注意：当天机器噪声大，跨窗绝对值
  不可比（同一配置跨窗散布 5.6-8.9）；融合前的安静窗数字（6.29-6.99）与
  融合后 6.37 有重叠，真实收益只认同场比值。
- **基线参照**：fp32-baseline（222.59 ms/tok @ 40b8e26），本次 vs 基线 = 36.2×
- **验证**：47 单测全过；三种融合开关组合（全开 / qkv 关 / 全关）golden
  token 16 个**逐位一致**——qkv 的行映射与 pair 的行内点积数学等价。
- **瓶颈转移**：top op = `lm_head`（29%）、`topk_argmax`（4.2%）、
  `gate_up_proj`（24%）。非 matvec 开销显形，见下方关闭小节。
- **意外 / 教训**：
    1. **同步开销比估计大一个量级**：原以为 fork-join <1µs、两个融合合计
       只值 0.15%；实测各值 4-7%。自旋唤醒 + cache-line 乒乓在 170+ 次/token
       的调用频率下是真实成本。教训：调用粒度类优化的预期要用实测同步成本
       算，别用理论值。
    2. **噪声窗的 A/B 只能靠多轮一致性立住**：单轮比值在 ±10% 里乱跳，
       靠"ON 从不系统性输"的方向一致性下结论——与 neon_mt_kv 条目同款方法。
- **复现**：`MODEL=model_f16.tqwen ./scripts/bench.sh f16_fusion --extra-args "--matvec-impl neon_mt_kv_nt"`（对照加
  `--no-fuse-gate-up --no-fuse-qkv`）

---

### fp16 路线关闭（2026-08-14，结论记录，非优化）

- **是什么**：**这不是优化，是路线关闭结论**。fp16 满栈（NEON fp32 累加 +
  线程池 + kv/qkv/gate_up 融合 + LDNP）实测 **6.37 ms/token**，判定已抵达
  带宽墙，fp16（不降 bit）路线的 decode 优化正式关闭。
- **带宽墙账本**：

  | 项 | 数值 |
    |---|---|
  | f16 权重流量/token | 942 MB |
  | 实测整机带宽（neon_mt 条目） | ~199 GB/s |
  | 理论下限 | 942 ÷ 199 ≈ **4.7 ms/tok** |
  | 当前实测 | 6.37 ms/tok |
  | 剩余 gap | ~1.7 ms（非 matvec 固定开销） |

  gap 构成（每 token）：topk_argmax 0.27ms（标量扫 151936 维）+
  attention/rmsnorm/rope/swiglu/bias ~0.7ms + 剩余同步/调度 ~0.7ms。
- **为什么没有大刀了**：decode 延迟 = 权重流量 ÷ 内存带宽。不降 bit 时
  分子（模型×dtype 固定）和分母（硬件带宽上限）都是常数，可优化面只剩
  常数旁边的 ~15% 开销。这是 batch=1 autoregressive decode 的物理形态
  （计算强度 ≈ 1 FLOP/byte，永远带宽受限），不是实现不努力。
- **剩余小刀与天花板**（列出来供将来决策，不建议现在做）：
  ① topk_argmax NEON 化（省 ~0.15ms）；② attention/rmsnorm/swiglu NEON 化
  （合计省 ~0.3-0.5ms）；③ 线程甜蜜点复扫（融合后未确认，当日扫描被
  噪声污染作废，默认 8 保持）。全部做完最乐观 ~5.3-5.5 ms/tok，然后
  撞死在 4.7 地板。
- **要继续提速必须打破某个假设**：量化（流量÷N，int4 理论 ~1.5ms/tok，
  需 greedy 翻转审计）、speculative decoding（少生成，v1 排除）、
  batched prefill（prefill 是 matmul 计算受限形态，另一片空间）、
  换硬件（GPU/ANE）。
- **意外 / 教训**：
    1. **能拿着数据说"到此为止"是测量纪律的产出**：归因阶梯把每层贡献
       拆清（FMA≈0、均衡≈0、nt 3-4% 都是测出来的证伪/证实），带宽墙有
       实测锚点，所以关闭结论是可辩护的，而不是"做不动了"。
    2. 线程复扫的教训：跨窗扫描（7/8/9/10 线程）散布 5.6-8.9ms，同一配置
       跨窗差 40%——**任何调参结论都必须在同窗块状对照里得出**，这次扫描
       整体作废，默认值不动。
- **复现**：`MODEL=model_f16.tqwen ./scripts/bench.sh f16_full_stack --extra-args "--matvec-impl neon_mt_kv_nt"`

---

### ops_neon（2026-08-14）

- **优化栈**：fp32-baseline +（fp32 全部机制）+ 权重 f16 + qkv/gate_up 融合 +
  **非 matvec 五算子 NEON 化**（fp16 路线关闭小节里列的"剩余小刀"一次性收完）
- **是什么**：新建 ops dispatch 层（rmsnorm/rope/attention/swiglu/argmax 五个
  小注册表共享一个实现名，未注册兜底各自 `_ref`，语义同 matvec_pair），新增
  NEON 变体：argmax 两遍法（NEON 求 max + 找首个相等，**与 ref 逐位一致**）、
  rmsnorm（4 路 float 累加平方和 + 向量化缩放）、rope（cos/sin 表与 ref 逐字
  相同、旋转循环向量化）、attention（online softmax 结构不变、点积 + oh 更新
  向量化）、swiglu（新融合 op：silu 与乘 up 合成单遍 + 多项式逼近 exp）。
  A/B 开关 `--ops-impl ref|neon`（配置项 ops_impl）。
- **假设**：指令类。这些算子全是逐元素/短归约的标量循环（argmax 串行 max 依赖、
  silu 逐个 expf），是 f16 路线关闭账本里 ~15% 非 matvec 开销的主体，SIMD 化
  应接近线性提速。
- **结果**：decode 中位 **6.35 ms/token**（3 遍取中位，每遍 27 样本），p95 6.87。
- **vs 上一配置**：**1.05×（同场 A/B）**——对照（同 binary + `--ops-impl ref`）
  中位 6.65 → 变体 6.35。注：本时段机器噪声大（单轮散布 6.2-9.1），手工 6 轮
  双向换序剔除 1 个异常轮后，neon 中位 ~6.42 vs ref ~6.83 ≈ **1.06×**，方向
  全胜。**算子级归因**（profile 逐算子计时，比端到端干净）：argmax 235→23µs
  （10.2×）、attention 215→94（2.3×）、swiglu 176→56（3.1×）、rmsnorm ×2
  36→14（2.5×）、rope 10.6→8.4，目标算子合计省 **~0.48 ms/tok**。（vs 基线 35.08×）
- **基线参照**：fp32-baseline（222.59 ms/tok @ 40b8e26），本次 vs 基线 = 35.08×
- **验证**：scripts/verify.sh（54 单测：新增 ops_neon 7 组门禁——argmax 下标
  逐位相等、rmsnorm/rope/attention/swiglu 容差对齐、swiglu_ref vs 原两步逐位
  一致）；golden token 在 ops ref/neon 两种配置下均与参考实现**逐位一致**。
- **瓶颈转移**：top op = `lm_head`、各层 `gate_up_proj`/`down_proj`（全是 matvec，
  带宽墙内）。非 matvec 开销从 ~694µs 压到 ~200µs。decode 进一步只剩 matvec
  带宽墙本身 + ~0.2ms 残余非 matvec（attention 尾、kv_append、调度），已无
  非量化大刀——与"fp16 路线关闭"结论闭环。
- **意外 / 教训**：
    1. **argmax 是隐藏大头**：bench 不传 --topk，走的是 greedy 的 argmax_ref
       分支——一个"看起来 O(n) 很简单"的标量扫描因串行 max 依赖 + 600KB 低吞吐，
       实测 235µs/tok，是非 matvec 最大单项。NEON 两遍法直接 10×。教训：别凭
       复杂度猜成本，串行依赖 + 带宽利用率才决定真实耗时。
    2. **跨窗 profile 对比会被 matvec 漂移污染**：首次对比 ops ref/neon 两份
       profile 时 matvec 各算子也"变快"了 10-16%——那是两次运行的机器状态差，
       不是本次改动。算子级归因要用同窗 A/B 或只看目标算子的相对变化。
    3. 机器噪声大的时段，端到端 A/B 只能给出方向（neon 全胜）+ 量级（~1.05×），
       精确值信算子级 profile。
- **复现**：
  `MODEL=model_f16.tqwen ./scripts/bench.sh ops_neon --extra-args "--matvec-impl neon_mt_kv_nt --ops-impl neon"`（对照加
  `--ops-impl ref`）

---

### pipeline 工具链（2026-08-14，infra，非优化）

- **是什么**：**这不是性能优化，是优化 pipeline 工具链本身的建设**。把
  "怎么测、怎么记录、怎么验证"固化成工具，让后续每个优化（含 Android
  端侧）都能低成本、可复现、可追溯地走完 verify→bench→record 全流程。
  commit 7213f49。
- **为什么是 infra 而非优化**：不改变任何 kernel 的数值行为或运行时性能，
  decode 延迟不受影响——这也是它不进汇总表的原因（汇总表只记优化栈）。
- **交付内容**（四块）：
    1. **Android 端侧 pipeline**（与本地对等）：verify_android.sh（golden token
       门禁，与本地同一 GOLDEN）/ bench_android.sh / record_android.sh /
       set_baseline_android.sh；底层 bench_android.py + record_android.py
       （A/B 同场对照 + 漂移警告 + 自动写日志）。基线独立
       （benchmarks/baseline_android.json，设备与 host 数字不可比）。
    2. **测量可靠性**：热门禁（测量前等设备降温，默认 45°C）+ 绑核（taskset
       绑大核，减少大小核迁移抖动）+ 回归检测（bench 对比 baseline 超阈值
       报警，--fail-on-regression 供 CI）。
    3. **profile_diff.py**：两次 profile op 级 delta + 分类汇总 + top 改善/回退。
    4. **visualize.py**：火焰图 + token 时序 + op 占比 + 优化历史趋势，
       独立 HTML 零依赖（内嵌 SVG + 原生 JS）。
- **与测量纪律的关系**：这些工具是 docs/optimization.md 各项纪律的代码化
  落地——固定负载（复用 bench.py 常量）、同场 A/B（复用 record 逻辑）、
  golden token 门禁（复用同一 GOLDEN）、数字可追溯（自动记 commit）。
  Android 侧此前只有手工 adb 命令，现在有了与本地一致的纪律。
- **验证**：visualize 用最小 DOM stub 冒烟测试四图全部渲染；profile_diff
  对同一 profile 自比 delta=0；全部 Python 工具 import 通过。Android 侧脚本
  待真机连接后跑 verify_android.sh 首测。
- **下一步**：连真机跑 set_baseline_android.sh 建立 Android 基线，此后每个
  kernel 优化可用 record_android.sh 一键记录端侧数字。

---

### gdn_neon（2026-08-17）

- **优化栈**：fp16 满栈 + ops_neon + **GDN 四算子 NEON 化**（Qwen3.5 专属）
- **是什么**：为 Qwen3.5-0.8B 的 Gated DeltaNet 层新增 4 个 NEON 变体，
  经过两轮优化迭代。四个算子共用 `--ops-impl neon` 开关（自注册进 GDN ops
  dispatch，兜底到各自 `_ref`）。具体改动：
    1. **gdn_step_neon**（delta rule 递归步，128×128 状态矩阵）：
        - 第一轮：沿 v_dim 向量化、分块 chunk=64（16 累加器）、三遍独立
        - 第二轮：融合 Pass 2+3 为单次遍历（省 64KB S 重读）、4× 循环展开、
          `__builtin_prefetch` 预取 2 行
    2. **causal_conv1d_update_neon**（dim=6144, kernel_size=4）：
        - vld3q/vld4q 跨通道解交错、双路 8 通道展开重叠 exp 延迟
    3. **l2norm_inplace_neon**（n=128, 每层 32 次调用）：
        - vrsqrteq_f32 + 2 步 Newton-Raphson 替代标量 sqrt+div、16 元素展开
    4. **rmsnorm_gated_neon**（n=128, SiLU 门控）：
        - vrsqrte 计算 scale、向量化 exp 多项式、双路 8 元素展开
    5. **dispatch 基建**：4 组注册表 + 4 通用入口 + 4 自注册宏 + runtime 调用点替换
- **假设**：指令效率 + 减少内存搬运（GDN 递归步的 S 矩阵 64KB 三遍读写是主要瓶颈）
- **结果**（A/B 基准，4 层 fake Qwen3.5 模型，真实 GDN 维度 128×128）：

  | 算子 | Ref (us/tok) | NEON (us/tok) | 加速比 |
    |------|-------------|--------------|-------|
  | gdn_recurrent | 560.5 | 214.3 | **2.62×** |
  | gdn_conv1d | 55.7 | 14.6 | **3.81×** |
  | gdn_l2norm | 6.0 | 2.9 | **2.06×** |
  | gdn_norm | 11.6 | 3.5 | **3.27×** |
  | **合计** | **633.7** | **235.4** | **2.69×** |

- **vs 上一配置**：GDN 层非投影算子总耗时 634→235 us/tok（2.69×）。
  实际端到端 decode（含 matvec 投影、FFN 等）从 40.3→38.8 ms/tok（GDN 算子
  仅占 1.6% 的 total decode，主瓶颈仍在 matvec 带宽墙）
- **验证**：78/78 单测全过（含 7 个 GDN NEON 专属 case：l2norm/rmsnorm_gated/
  conv1d/conv1d_inplace/gdn_step/gdn_step_multi_step/gdn_step_small_dims）。
  Token 输出与 ref 完全一致（greedy argmax 逐位相同）。
  数值容差：S 矩阵 ≤1e-5，output o ≤1e-4，其余 ≤1e-5。
- **瓶颈转移**：GDN 层内部 top op 仍为 gdn_recurrent（214 us，占 91%），
  理论极限 ~3.5×（受 L1/L2 带宽墙限制）。下一步真正的大头是 GDN 投影矩阵
  （gdn_proj/gdn_out_proj 占整体 ~90%），需 INT4 量化或更高效 matvec。
- **意外 / 教训**：
    1. **Apple Silicon vdivq_f32 很快**（~7 cycles）：用 vrecpeq+Newton 替代
       在 Cortex-A78 上是经典优化，但在 M 系列反而更慢——rmsnorm 从 3.28× 跌到
       3.11×。已回退为 vdivq。教训：microarch 差异大，必须实测。
    2. **融合遍是此量级 kernel 的主要收益来源**：gdn_step 从 2.10→2.62× 主要
       靠消除一遍 64KB 重读（减 33% 内存流量），循环展开和预取只贡献 ~10%。
    3. **l2norm 在 n=128 时受函数调用开销制约**：计算本身 ~80 cycles，
       sqrt+div 就占 30 cycles。vrsqrte 替代是性价比最高的单点优化。
- **复现**：
  ```bash
  # 需要 fake35_bench.tqwen（4 层真实 GDN 维度）
  python3 tools/make_fake_qwen35_bench.py  # 或用 /tmp/fake35_bench.tqwen
  ./build/runtime/tinyqwen --model /tmp/fake35_bench.tqwen \
      --tokens 3,7,11 --max-new-tokens 32 --eos -1 \
      --ops-impl ref --profile-out /tmp/ref.json     # A 组
  ./build/runtime/tinyqwen --model /tmp/fake35_bench.tqwen \
      --tokens 3,7,11 --max-new-tokens 32 --eos -1 \
      --ops-impl neon --profile-out /tmp/neon.json   # B 组
  ```

---

### cuda（2026-08-17）

- **⚠️ 首次换机器**：本条及之后的数字在 **x86_64 + NVIDIA A10**（8 CPU 核，
  gcc 10.2.1，CUDA 12.8，Ampere SM86）上测量，与汇总表前面所有条目
  （Apple Silicon）**不是同一台机器**——"vs 原始基线"一栏失去意义，
  只有同场 A/B 可比。baseline.json（222.59 @ M 系芯片）保持不动，
  本机跑 record.sh 会持续报漂移警告，属预期。
- **优化栈**：x86-fp32-baseline（本机标量 ref）+ matvec cuda 参考变体
- **是什么**：新增 `kernels/matvec/matvec_f32_cuda.cu`——matvec 的 CUDA
  **参考版**。两个刻意"不优化"的设计：host 包装每次调用都
  cudaMalloc + H2D 上传全部权重 + kernel + D2H + cudaFree（权重不常驻显存）；
  device kernel 一行一个 thread、行内串行点积（无 shared memory / warp
  归约）。目标不是快，是把 CUDA 接进现有 dispatch（`--matvec-impl cuda`
  即可用，CMake 用 check_language(CUDA) 可选接入，无 nvcc 平台零影响），
  并如实量出"朴素形态"的真实成本，给下一步优化提供基准。
- **假设**：带宽类。A10 HBM 带宽 ~600 GB/s 远大于 CPU；decode 是纯权重
  带宽瓶颈（~2 GB/token 流量），理论上 GPU 应大幅领先。但朴素包装每 token
  要经 PCIe 重传 ~2 GB 权重——传输税会吃掉大部分理论收益，预期被显著稀释。
- **结果**：decode 中位 **333.75 ms/token**（3 遍取中位，每遍 27 样本），p95 356.47
- **vs 上一配置**：**1.75×（同场 A/B，本机 ref）**——对照（无额外参数 = 标量 ref）
  中位 583.50（p95 588.56）→ 变体 333.75，同 binary 同场交错测量。
  本机对照补充：acc4（跨平台最快 CPU 变体）单机单遍 269.21 ms/tok——
  **cuda 参考版比本机最好的 CPU 标量变体还慢**，只快于裸 ref。
- **基线参照**：fp32-baseline（222.59 ms/tok @ 40b8e26）是 M 系芯片数字，
  跨机器不可比，本次 vs 基线 = 0.67× 无意义（仅存档）。
- **验证**：scripts/verify.sh（55 单测全过，含新增 `matvec_cuda_matches_ref`：
  常规形状 + grid-stride 覆盖的大 out_dim 形状；golden token 16 个逐位一致）
- **瓶颈转移**：top op 仍是 `lm_head`、qkv_proj、o_proj（matvec 结构不变），
  但时间构成变了：计算本身可忽略（A10 fp32 ~31 TFLOPS，2 GFLOP/token 是零头），
  成本几乎全在**权重上传**（~2 GB/token，PCIe Gen4 有效 ~25 GB/s 量级 →
  传输本身 ~80 ms/token 量级）+ 每调用一次的 cudaMalloc/cudaFree/kernel
  启动（170 次/token，其余 ~250 ms 的大头）。下一刀：**权重常驻显存**
  （decode 期权重不变，按 host 指针缓存 device 副本，把每 token 的
  1.98 GB 上传降为 x 的几 KB + y 的回读），然后再谈 kernel 侧优化
  （block-per-row + shared memory 归约，喂饱小矩阵）。
- **意外 / 教训**：
    1. **"GPU 一定快"在 batch=1 小模型上被证伪**——朴素 CUDA 变体输给本机
       最好的 CPU 标量变体（333 vs 269 ms/tok）。输的原因不是 GPU 算力，
       是传输税：每调用重传权重的包装形态把 PCIe 当成了主瓶颈。这正是
       "先落参考版量真实成本"策略的价值——没有这个 333 的锚点，下一步
       权重常驻的收益（预期一个量级）就无从归因。
    2. **换机器 = 换尺子**：跨机器的"vs 基线"彻底失效（0.67× 是荒谬的，
       x86 标量 ref 本来就比 M 系慢 2.6×），纪律上只认同场 A/B；汇总表里
       机器切换必须显式标注（本条已标）。
- **复现**：`./scripts/bench.sh cuda --extra-args "--matvec-impl cuda"`

---

### cuda_resident（2026-08-17）

- **机器**：与 cuda 条目同一台（x86_64 + NVIDIA A10）。跨机器的"vs 基线"
  无意义（基线 222.59 是 M 系芯片），只认同场 A/B 与本机对照。
- **优化栈**：x86-fp32-baseline（本机标量 ref）+ matvec cuda_resident 变体
  （= cuda 参考版 + 权重常驻，kernel 逐字不变，只改 host 包装）
- **是什么**：新增 `kernels/matvec/matvec_f32_cuda_resident.cu`。唯一改动：
  权重不再每次调用重传——按 host 指针缓存 device 副本（首次见到上传一次，
  之后命中缓存零拷贝）。device kernel 与 cuda 参考版**逐字相同**（一行一
  thread、行内串行点积），刻意只改"权重常驻"这一个变量，便于归因。
- **假设**：减少搬运类。cuda 条目已定位瓶颈是"每 token 经 PCIe 重传 ~2GB
  权重"（传输税）；权重在 decode 期内容不变，理应只传一次。预期砍掉这
  部分后 GPU 的 HBM 带宽（~600 GB/s）才开始真正派上用场。
- **结果**：decode 中位 **47.40 ms/token**（3 遍取中位，每遍 27 样本），p95 48.07
- **vs 上一配置**：**7.04×（vs 朴素 cuda：333.75 → 47.40，同机同场）**——
  这才是"权重常驻"这一步的真实贡献。同场 A/B 对照（无额外参数 = 标量 ref）
  中位 583.01（p95 589.21）→ 变体 47.40，即 vs 本机 ref 12.30×，同 binary
  同场交错测量。
- **本机 CPU 对照**：acc4（跨平台最快 CPU 标量变体）269.21 ms/tok——
  cuda_resident 比本机最好 CPU 变体快 **5.68×**。**逆转了 cuda 条目的结论**：
  朴素 cuda 输给 acc4（333 vs 269），权重常驻后 GPU 大幅反超。输赢不在
  GPU 算力，在有没有把传输税砍掉。
- **基线参照**：fp32-baseline（222.59 ms/tok @ 40b8e26）是 M 系芯片数字，
  跨机器不可比，本次 vs 基线 = 4.70× 无意义（仅存档）。
- **验证**：scripts/verify.sh（56 单测全过，含新增 `matvec_cuda_resident_matches_ref`：
  冷路径 + 连跑 12 遍压缓存命中 + 两块不同指针同形状权重交替 8 轮防串线；
  golden token 16 个逐位一致）
- **瓶颈转移**：top op 仍是 `lm_head`、qkv_proj、down_proj（matvec 结构不变）。
  权重读 now 走 HBM：decode 每 token 仍需把 ~2GB 权重从 HBM 读一遍，
  A10 ~600 GB/s 带宽下限 ≈ **3–4 ms/token**——当前 47.40 离带宽地板还有
  ~12× 空间。剩余成本大头在**每调用的 cudaMalloc/cudaFree（x/y，~680 次/token）
    + 170 次 cudaDeviceSynchronize + 朴素 kernel 喂不带宽**。下一刀：
      ① x/y 用常驻 workspace，删掉每调用 alloc/free；② kernel 改 block-per-row
    + shared memory/warp 归约，把 HBM 带宽吃满（目标逼近 3–5 ms/tok）。
      （后记：① 被 cuda_resident_ws 条目证伪——CUDA 分配器池化小缓冲，
      alloc/free 不是大头；真正的瓶颈是 kernel 内非合并访存，见该条。）
- **意外 / 教训**：
    1. **一步归因干净利落**：kernel 一个字没改，只把权重改成常驻，就拿到
       7.04×——精确兑现了 cuda 条目"下一刀砍传输税"的预判。这再次印证
       "先落参考版量出真实成本、再逐个变量拆"的价值：没有 333 那个锚点，
       就说不清这 7× 是谁的功劳。
    2. **GPU 反超 CPU 的开关是"少搬数据"，不是"算得快"**：本机 acc4 269，
       朴素 cuda 333（输），常驻 cuda 47（赢 5.68×）。对带宽瓶颈型负载，
       数据搬运路径的设计比算力选型更决定成败——与 CPU 侧 fp16 量化
       （流量减半）带来主升浪是同一个规律。
- **复现**：`./scripts/bench.sh cuda_resident --extra-args "--matvec-impl cuda_resident"`

---

### cuda_resident_ws（2026-08-17）

- **机器**：与 cuda / cuda_resident 条目同一台（x86_64 + NVIDIA A10）。
  跨机器的"vs 基线"无意义，只认同场 A/B 与本机对照。
- **优化栈**：x86-fp32-baseline（本机标量 ref）+ cuda_resident + x/y 常驻 workspace
- **是什么**：新增 `kernels/matvec/matvec_f32_cuda_resident_ws.cu`。在
  cuda_resident 之上再改一处：x/y 不再每次调用 cudaMalloc/cudaFree，改用
  grow-only 常驻 workspace（只在遇到更大 shape 时扩容，decode 稳态零
  分配）。权重缓存与 device kernel 仍然逐字不变——继续一次只动一个变量。
- **假设**：cuda_resident 条目推断剩余大头是"每调用 680 次 cudaMalloc/
  cudaFree（~170 次 matvec/token × 4）"，预期删掉后有明显收益。
- **结果**：decode 中位 **46.05 ms/token**（3 遍取中位，每遍 27 样本），p95 46.52
- **vs 上一配置**：**1.03×（vs cuda_resident：47.40 → 46.05，≈无效）**——
  假设被证伪：CUDA 分配器对小缓冲做池化缓存，小块的每调用 alloc/free
  远比预想便宜，省下的 ~1.35 ms/tok（约 3%）基本在噪声边缘。同场 A/B
  对照（无额外参数 = 标量 ref）中位 583.27（p95 589.64）→ 变体 46.05，
  即 vs 本机 ref 12.67×，同 binary 同场交错测量。
- **基线参照**：fp32-baseline（222.59 ms/tok @ 40b8e26）是 M 系芯片数字，
  跨机器不可比，本次 vs 基线 = 4.83× 无意义（仅存档）。
- **验证**：scripts/verify.sh（57 单测全过，含新增
  `matvec_cuda_resident_ws_matches_ref`：小 shape→大 shape→回到小 shape
  交替，覆盖 workspace 冷启动/扩容/复用大缓冲跑小 shape 三条路径；
  golden token 16 个逐位一致）
- **瓶颈转移**：top op 不变（lm_head / qkv_proj / down_proj）。排除了
  alloc/free 之后，46 ms/tok 的构成指向**朴素 kernel 的非合并访存**：
  一行一个 thread 意味着同一 warp 的 32 个线程在读 32 条相隔
  in_dim×4 字节的权重行——每步 warp 读 32 条分散 cache line 只用回
  128 字节，HBM 有效利用率 ~3%，1.98GB/token 的权重读被放大成百倍
  流量；外加每调用一次 cudaDeviceSynchronize。下一刀：**合并访存的
  matvec kernel**（block-per-row：一个 block 算一行、线程沿 in_dim
  连续读权重、shared memory/warp 归约求点积），把 HBM 带宽吃满，
  目标逼近 3–5 ms/tok 的带宽地板。
  （后记：已被 cuda_resident_coal 条目兑现——合并访存一步 47.40→9.16，
  5.17×，诊断完全正确；而本条的 workspace 思路在 kernel 降下来后重新
  变成下一刀，见该条瓶颈转移。）
- **意外 / 教训**：
    1. **归因猜测被测量打脸是常态，测了才知道**：上一条信誓旦旦"剩余大头
       是 680 次 alloc/free"，实测只值 3%。教训：CUDA 运行时对小分配有
       池化，想当然的成本模型不可靠——每个"下一刀"都要像本条一样落地实测，
       证伪也是成果（它把矛头准确指向了 kernel 访存模式）。
    2. **变体保留的价值**：cuda_resident_ws 虽然≈无效，但留在树里当
       "排除项"——后人不用再猜 alloc/free 是不是瓶颈。与 neon_mt_bal
       归档同款处理（实测收益≈0，归档保留，不删）。
- **复现**：`./scripts/bench.sh cuda_resident_ws --extra-args "--matvec-impl cuda_resident_ws"`

---

### cuda_resident_coal（2026-08-17）

- **机器**：与 cuda 系列条目同一台（x86_64 + NVIDIA A10）。跨机器的
  "vs 基线"无意义，只认同场 A/B 与本机对照。
- **优化栈**：x86-fp32-baseline（本机标量 ref）+ cuda_resident + 合并访存 kernel
- **是什么**：新增 `kernels/matvec/matvec_f32_cuda_resident_coal.cu`。权重
  常驻逻辑沿用 cuda_resident，唯一变化是 **kernel 访存/归约方式**：从
  "一行一个 thread（warp 读 32 条分散 cache line）"改成 **block-per-row**
  ——一个 block（256 线程）合算一行，线程沿 in_dim 跨步取数（warp 内任一
  时刻读连续 32 个 float = 一条满 128B cache line），部分和经两级归约
  （warp 内 `__shfl_down_sync` + warp 间 shared memory）合成行结果。
- **假设**：访存模式类。ws 条目已定位朴素 kernel 的非合并访存把 HBM 流量
  放大 ~32 倍（warp 每步读 32 条分散 cache line 只用回 128 字节）；改成
  合并访存后权重读取 100% 有效，预期直逼 HBM 带宽地板（~3–4 ms/tok）。
- **结果**：decode 中位 **9.16 ms/token**（3 遍取中位，每遍 27 样本），p95 9.81
- **vs 上一配置**：**5.17×（vs cuda_resident：47.40 → 9.16，同机）**——
  这是"合并访存"这一步的真实贡献。同场 A/B 对照（无额外参数 = 标量 ref）
  中位 585.86（p95 592.07）→ 变体 9.16，即 vs 本机 ref 63.94×，同 binary
  同场交错测量。
- **本机 CPU 对照**：acc4（跨平台最快 CPU 标量变体）269.21 ms/tok——
  cuda_resident_coal 比本机最好 CPU 变体快 **29.4×**。至此 CUDA 路线从
  "输给 CPU"（朴素 cuda 333）一路打到"碾压 CPU"（9.16），三步完成逆转。
- **基线参照**：fp32-baseline（222.59 ms/tok @ 40b8e26）是 M 系芯片数字，
  跨机器不可比，本次 vs 基线 = 24.29× 无意义（仅存档）。
- **验证**：scripts/verify.sh（58 单测全过，含新增
  `matvec_cuda_resident_coal_matches_ref`：覆盖 in_dim 非 256 倍数、
  in_dim<256（高编号线程空转）、大 out_dim block-per-row 等边界；
  golden token 16 个逐位一致）。**对齐门禁还真抓到过一个 bug**——见教训 3。
- **瓶颈转移**：top op 不变（lm_head / qkv_proj / gate_up_proj）。9.16 已
  逼近但未到 HBM 地板（~3–4 ms/tok，权重 1.98GB/token ÷ ~600 GB/s）：
  kernel 本体 ~3–4ms，其余 ~5ms 是**每调用的 x/y alloc/free（680 次/token）
    + kernel 启动（170 次）+ cudaDeviceSynchronize（170 次）**。注意：ws 条目
      曾证伪"alloc/free 是大头"——但那是在 kernel 还占 40+ms 时；如今 kernel
      降到 ~4ms，这些固定开销的**相对**占比反升，ws 的思路此刻才真正值钱。
      下一刀：x/y 常驻 workspace + 削减同步/启动开销（或 CUDA Graph 一次提交
      整段 decode），目标逼近 3–5 ms/tok。
- **意外 / 教训**：
    1. **访存模式是最大的单项杠杆**：FLOPs 没变、搬运的字节数（原理上）没变，
       只改"线程怎么读权重"就拿到 5.17×。GPU 上"算得快"之前先"读得对"——
       合并访存是第一课，也是本次从 47 到 9 的全部原因。
    2. **优化的时序性**：ws（删 alloc/free）在 47ms 时≈无效、在 9ms 时却成了
       下一刀——不是 ws 错了，是它当时被更大的瓶颈（kernel）淹没。先砍
       主导瓶颈、再回头收固定开销，这个顺序本身是方法论。
    3. **对齐门禁抓到真 bug（且是熟人）**：首版测试挂在 shape(8,67) row6，
       排查发现罪魁不是 kernel，而是测试数据生成用了 `size_t i` 导致
       `(i*37%29)-14` 无符号下溢成 ~2.3e18——与 neon 条目 micro-bench 那次
       是**同一类坑**。教训：误差异常先查数据生成；`size_t` 减法永远警惕。
       kernel 本身一次就写对了。
- **复现**：`./scripts/bench.sh cuda_resident_coal --extra-args "--matvec-impl cuda_resident_coal"`

---

### f16_cuda_resident_coal（2026-08-17）

- **机器**：与 cuda 系列条目同一台（x86_64 + NVIDIA A10）。跨机器的
  "vs 基线"无意义，只认同场 A/B 与本机对照。
- **优化栈**：x86-fp32-baseline + cuda_resident_coal + 权重 fp16（weight-only）
- **是什么**：新增 `kernels/matvec/matvec_f16_cuda_resident_coal.cu`——f32
  cuda_resident_coal 的 fp16 权重版。host 包装、合并访存 kernel 的归约结构
  全部照搬，唯一变化：权重按 `uint16_t` 加载、`__half2float` 转 fp32 再乘加。
  与 f32 版**同名注册**（"cuda_resident_coal"），按模型 dtype 解析（对齐
  CPU 侧 f16 族的做法）。配套导出 `model_f16.tqwen`（--dtype f16，942MB，
  恰为 fp32 的一半）。
- **假设**：减少搬运类。decode 是纯搬权重负载，权重 fp32→fp16 每 token
  HBM 流量减半（1.98GB→0.99GB），带宽地板 ~3.3ms→~1.7ms，理论应接近 2×。
- **结果**：decode 中位 **7.68 ms/token**（3 遍取中位，每遍 27 样本），p95 8.14
- **vs 上一配置**：**1.18×（同场 A/B，vs f32_cuda_coal）**——对照
  （model.tqwen + cuda_resident_coal，即 fp32 版）中位 9.05（p95 9.49）→
  变体（model_f16.tqwen + 同名实现）7.68，同 binary 同场交错测量。
  **远低于理论的 2×**——见教训 1。
- **基线参照**：fp32-baseline（222.59 ms/tok @ 40b8e26）是 M 系芯片数字，
  跨机器不可比，本次 vs 基线 = 28.98× 无意义（仅存档）。
- **验证**：scripts/verify.sh（59 单测全过，含新增
  `matvec_f16_cuda_resident_coal_matches_ref`：对齐 f16 族 matvec_f16_ref，
  覆盖 in_dim 非 256 倍数 / in_dim<256 / 大 out_dim 等边界）；f16 golden
  token 16 个逐位一致（canonical prompt，与 fp32 参考相同）。
- **瓶颈转移**：top op 出现 `topk_argmax`（非 matvec 算子开始冒头，说明
  matvec 占比进一步下降）。7.68 离 f16 带宽地板（~1.7ms）还有 ~4.5×，
  但差的不是带宽——是**固定开销**：每调用 x/y alloc/free + kernel 启动 +
  cudaDeviceSynchronize（~170 次/token），在 fp32 版就占了 ~5ms，fp16 只
  减了权重读那部分。下一刀：**砍固定开销**——x/y 常驻 workspace（此时才
  真正值钱，ws 条目埋的伏笔）+ 削同步/启动，或 CUDA Graph 一次提交整段
  decode，目标把 fp32/f16 都推向各自带宽地板。
- **意外 / 教训**：
    1. **理论 2× 被 Amdahl 稀释成 1.18×——这本身就是诊断结果**：当时间大头
       已从"带宽"转移到"固定开销"时，任何只减带宽的优化都只能作用于剩下的
       小部分。fp16 的 1.18× 精确告诉我们：权重读只占总时间的 ~1/3，主矛盾
       是开销。想吃到 fp16 的全部红利，必须先砍开销——顺序不能反。
    2. **对齐基准要跟着 dtype 走**：f16 变体对齐的是 matvec_f16_ref（权重已
       半精度量化），不是 f32 的 ref——跨族对齐没有意义。测试里用
       float_to_half 构造 0.125 倍数权重（half 可精确表示），隔离量化噪声。
- **复现**：
  `MODEL=model_f16.tqwen ./scripts/bench.sh f16_cuda_resident_coal --extra-args "--matvec-impl cuda_resident_coal"`

---

### cuda_resident_coal_ws（2026-08-17）

- **机器**：与 cuda 系列条目同一台（x86_64 + NVIDIA A10）。跨机器的
  "vs 基线"无意义，只认同场 A/B 与本机对照。
- **优化栈**：x86-fp32-baseline + cuda_resident_coal + 砍每调用开销
  （x/y workspace + 删冗余 sync）
- **是什么**：新增 `kernels/matvec/matvec_f32_cuda_resident_coal_ws.cu`。
  kernel 与 cuda_resident_coal **逐字相同**，只改 host 包装两处：
  ① x/y 改 grow-only 常驻 workspace（删掉每调用 cudaMalloc/cudaFree×2，
  ~680 次/token）；② 删掉冗余的 `cudaDeviceSynchronize()`——后面的 D2H
  `cudaMemcpy` 本就同步、会等 kernel 完成，显式 sync 是重复保险。
- **假设**：削减固定开销类。f16 条目已诊断主矛盾是每调用开销；此刀直指它。
- **结果**：decode 中位 **7.97 ms/token**（3 遍取中位，每遍 27 样本），p95 8.34
- **vs 上一配置**：**1.15×（同场 A/B，vs cuda_resident_coal）**——对照
  （model.tqwen + cuda_resident_coal，即 fp32 版）中位 9.18（p95 9.61）→
  变体 7.97，同 binary 同场交错测量。
- **归因拆分**：workspace 单独上次已证≈0（cuda_resident_ws 47.40→46.05），
  所以这 ~1.2ms 的增量**主要来自删冗余 sync**（~170 次/token 的全设备同步
  调用开销），workspace 贡献趋近 0 但为后续融合铺了地基。
- **基线参照**：fp32-baseline（222.59 ms/tok @ 40b8e26）是 M 系芯片数字，
  跨机器不可比，本次 vs 基线 = 27.92× 无意义（仅存档）。
- **验证**：scripts/verify.sh（60 单测全过，含新增
  `matvec_cuda_resident_coal_ws_matches_ref`：小→大→回小 shape 压 workspace
  扩容/复用，多 shape 交替连跑压"删 sync 后的调用时序"无竞态；golden token
  16 个逐位一致）
- **瓶颈转移**：top op 不变（lm_head / qkv_proj / gate_up_proj）。7.97 里
  权重读 ~3.3ms（f32），剩 ~4.7ms 是**每次调用都逃不掉的三件套**：kernel
  启动 + H2D x + D2H y（各 ~170 次/token，每次合 ~9µs）。逐调用削已经削到
  头了，要再降只能**减少调用次数**——下一刀 B：qkv 三合一、gate_up 二合一
  的 CUDA 融合（把 ~170 次 matvec 砍到 ~100 次），或更彻底的 CUDA Graph。
- **意外 / 教训**：
    1. **"冗余 sync"是隐蔽的固定开销**：它看起来只是"保险"，但 170 次/token
       累加成 ~1ms+。判断能否删的依据是"后续是否已有同步点"——这里同步 D2H
       已保证 kernel 完成，显式 sync 即纯开销。删 sync 前务必确认有等价的
       同步语义兜底，否则会引入读未就绪的竞态。
    2. **逐调用优化有天花板**：alloc/free、sync 都削过之后，剩下的启动 + 两次
       拷贝是"每调用必付"的底价，单靠优化单次调用无法再降——必须从"减少
       调用次数"或"批量提交"破局。这为 B（融合）定好了方向。
- **复现**：`./scripts/bench.sh cuda_resident_coal_ws --extra-args "--matvec-impl cuda_resident_coal_ws"`

---

### f16_cuda_resident_coal_ws（2026-08-17）

- **机器**：与 cuda 系列条目同一台（x86_64 + NVIDIA A10）。跨机器的
  "vs 基线"无意义，只认同场 A/B 与本机对照。
- **优化栈**：x86-fp32-baseline + f16_cuda_resident_coal + 砍每调用开销
  （x/y workspace + 删冗余 sync）= **f16 CUDA 满栈**
- **是什么**：新增 `kernels/matvec/matvec_f16_cuda_resident_coal_ws.cu`——
  f32 `cuda_resident_coal_ws` 的 f16 对应版。kernel 与 f16_cuda_resident_coal
  逐字相同，只把 host 包装的两处开销（每调用 alloc/free、冗余 sync）砍掉。
  与 f32 版同名注册（"cuda_resident_coal_ws"），按模型 dtype 解析。
- **假设**：A（砍开销）在 f32 已证 1.15×；f16 配置才是要压过 Mac 的那条，
  把同款手法移植过来，预期把 7.68 进一步压低。
- **结果**：decode 中位 **6.54 ms/token**（3 遍取中位，每遍 27 样本），p95 7.01
- **vs 上一配置**：**1.19×（同场 A/B，vs f16_cuda_coal）**——对照
  （model_f16.tqwen + cuda_resident_coal）中位 7.76（p95 8.56）→ 变体 6.54，
  同 binary 同场交错测量。
- **与 Mac 对照（跨机器，仅参照）**：Mac fp16 满栈（f16+qkv/gate_up 融合+ops
  NEON）≈ 6.35 ms/tok。A10 这条 CUDA f16 满栈 6.54，**已追平其量级**——
  且 A10 还没上融合、非 matvec 算子还在 CPU，仍有明确空间。注意两机器不同，
  此对比只做方向参照，不作结论。
- **基线参照**：fp32-baseline（222.59 ms/tok @ 40b8e26）是 M 系芯片数字，
  跨机器不可比，本次 vs 基线 = 34.01× 无意义（仅存档）。
- **验证**：scripts/verify.sh（61 单测全过，含新增
  `matvec_f16_cuda_resident_coal_ws_matches_ref`：对齐 matvec_f16_ref，
  workspace 扩容/复用 + 删 sync 后多 shape 交替时序无竞态）；f16 golden
  token 16 个逐位一致。
- **瓶颈转移**：top op 不变（lm_head / qkv_proj / topk_argmax）。6.54 里
  权重读（f16 ~0.99GB）≈1.7ms，剩 ~4.8ms 仍是**每调用三件套**（启动 + H2D +
  D2H，各 ~170 次/token）——与 f32 版同一个天花板。下一刀 B：**CUDA 融合
  qkv（三合一）+ gate_up（二合一）**，把 ~170 次 matvec 调用降到 ~100 次，
  直接减少启动/拷贝次数，目标把 f16 压到 6.35 以下。
- **意外 / 教训**：
    1. **移植已验证的优化是高确定性收益**：A 在 f32 证过 1.15×，移植 f16 实测
       1.19×，量级一致——好优化是可迁移的，不用每次重新怀疑。
    2. **追平≠超越，跨机器对比要克制**：6.54 vs Mac 6.35 是不同机器的参照，
       说明"方向对、量级到"，但真正的超越要靠下一步融合在本机实测兑现。
- **复现**：
  `MODEL=model_f16.tqwen ./scripts/bench.sh f16_cuda_resident_coal_ws --extra-args "--matvec-impl cuda_resident_coal_ws"`

---

### f16_cuda_fused（2026-08-17）

- **机器**：与 cuda 系列条目同一台（x86_64 + NVIDIA A10）。跨机器的
  "vs 基线"无意义，只认同场 A/B 与本机对照。
- **优化栈**：x86-fp32-baseline + f16_cuda_resident_coal_ws + qkv/gate_up 融合
  = **f16 CUDA 满栈（含融合）**
- **是什么**：新增 `kernels/matvec/matvec_f16_cuda_fused.cu`——给 qkv / pair
  两个 f16 注册表各注册一个融合 kernel（名字与主表共享 cuda_resident_coal_ws，
  model 的 mv_qkv / mv_pair 自动命中）：qkv 三合一（一个 kernel 按全局行号把
  q/k/v 三段一起算）、gate_up 二合一。每组从"N 次 matvec 调用"变"1 次"：
  共享一次 H2D x、只启动一个 kernel，每层 matvec 调用 7→4 次。
- **假设**：减少调用次数类。cuda_resident_coal_ws 条目已定位剩余开销是
  "每调用三件套"（启动 + H2D + D2H），融合直接砍调用数，是它的正面解法。
- **结果**：decode 中位 **5.79 ms/token**（3 遍取中位，每遍 27 样本），p95 6.28
- **vs 上一配置**：**1.07×（同场 A/B）**——对照（model_f16 + cuda_resident_coal_ws
    + `--no-fuse-qkv --no-fuse-gate-up`）中位 6.22（p95 6.95）→ 变体 5.79，同 binary
      同场交错测量。**口径注**：该对照的 k/v 仍走 pair 融合（非融合 qkv 路径里
      k/v 本来就调 mv_pair），属"部分融合"，故 1.07× 偏保守；对真正全不融合
      （上一条的 6.54）融合总收益 = 6.54→5.79 = **1.13×**。
- **达成目标**：**5.79 < Mac fp16 满栈 6.35**——"压到 Mac 以下"兑现。
  （跨机器对比仅作方向参照：A10 vs M 系芯片；但这是设定的目标，数字上达成。）
- **基线参照**：fp32-baseline（222.59 ms/tok @ 40b8e26）是 M 系芯片数字，
  跨机器不可比，本次 vs 基线 = 38.45× 无意义（仅存档）。
- **验证**：scripts/verify.sh（62 单测全过，含新增
  `matvec_f16_cuda_fused_pair_qkv_match_ref`：pair/qkv 融合输出逐一对齐
  matvec_f16_ref，用 Qwen 真实形状 q_dim=896/kv_dim=128/in_dim=896）；
  f16 golden token 16 个逐位一致。
- **瓶颈转移**：top op 不变（lm_head / qkv_proj / topk_argmax）。5.79 里
  权重读 ~1.7ms，剩 ~4ms：融合后调用数虽降，单次调用的启动/拷贝仍在，且
  **非 matvec 算子（rmsnorm/rope/attention/swiglu/argmax）还在 CPU**，
  topk_argmax 已进 top op。下一刀候选：① 非 matvec 算子上 GPU（或整段
  forward 常驻显存 + CUDA Graph）；② 更大模型/batch——那才是 A10 主场。
- **意外 / 教训**：
    1. **融合的收益随优化深入而递减**：resident 7×、coalesce 5×、融合只有
       1.13×——越往后剩余开销越分散，单刀收益越小，这是优化后期的常态。
       但融合是把"调用数"这个结构性开销砍掉的必要一步，积少成多。
    2. **A/B 对照要选准**：`--no-fuse` 开关的对照仍含 pair 融合（k/v 走
       mv_pair），直接拿它当"上一配置"会低估收益；用历史记录的真·不融合
       （6.54）做锚点才看得全 1.13×。对照的"纯度"本身是要审查的对象。
- **复现**：`MODEL=model_f16.tqwen ./scripts/bench.sh f16_cuda_fused --extra-args "--matvec-impl cuda_resident_coal_ws"`

---

### gpu_engine（2026-08-17）

- **机器**：与 cuda 系列条目同一台（x86_64 + NVIDIA A10）。跨机器的
  "vs 基线"无意义，只认同场 A/B 与本机对照。
- **优化栈**：x86-fp32-baseline + f16 权重 + **GPU-resident 整段 forward**
  （权重/激活/KV 常驻显存，单 stream 跑完，每步仅 4B argmax 过 PCIe）
- **是什么**：新增 `kernels/cuda/gpu_engine.cu`（engine 本体，自注册 "cuda"）
    + `kernels/cuda/gpu_kernels.cu`（device 指针形态的 embed/rmsnorm/rope/
      kv_append/attention/swiglu/argmax/bias/residual + coalesced matvec
      single/pair/qkv）。dispatch 加 decode-engine registry，main 加 `--engine cuda`。
      CPU `forward_token` 原样保留作参考。这是**架构级**改动，不是单 kernel。
- **假设**：消灭桥接类。此前 forward 是 CPU 驱动，每 matvec 都 H2D x + D2H y、
  非 matvec 算子在 CPU——这些桥接 + 同步是离带宽地板 ~1.6ms 还差 ~4ms 的主因。
  整段常驻后应逼近地板。
- **结果**：decode 中位 **4.89 ms/token**（3 遍取中位，每遍 27 样本），p95 5.23
- **vs 上一配置**：**1.19×（同场 A/B，vs CPU 驱动 f16 满栈）**——对照
  （model_f16 + cuda_resident_coal_ws，即上条 CPU 驱动融合版）中位 5.80
  （p95 6.06）→ 变体 4.89，同 binary 同场交错测量。
- **达成目标**：**4.89 < Mac fp16 满栈 6.35**（跨机器仅参照），且比本机 CPU
  驱动 f16 满栈再快 1.19×。fp32 模型同样验证 golden 逐位一致。
- **基线参照**：fp32-baseline（222.59 ms/tok @ 40b8e26）是 M 系芯片数字，
  跨机器不可比，本次 vs 基线 = 45.50× 无意义（仅存档）。
- **验证**：scripts/verify.sh（71 单测全过，CPU 默认路径 golden 逐位一致）；
  **engine e2e：fp32 与 f16 模型 16-token golden 逐位一致，32-token 与 CPU
  forward 逐位一致**——24 层 × 12 算子全链路一次写对。算子层另有 9 个单测
  逐一对齐 *_ref（attention online-softmax+GQA、kv_append 索引逐位）。
- **瓶颈转移**：engine 绕过了 op 级 profiler 作用域，top op 不再可见。4.89
  离 f16 带宽地板（942MB÷~600GB/s ≈ 1.6ms）仍差 ~3.3ms，主要是**每 token
  ~300 次 kernel 启动的开销 + 小矩阵（k/v_proj 128 行）block-per-row 低占用**。
  下一刀：① CUDA Graph 把整段 decode 的启动批成一次（最大头）；② 小 matvec
  改用更细并行（split-K / 多行每 block）喂满 SM。
- **意外 / 教训**：
    1. **整段常驻 e2e 一次写对**：靠的是算子层先逐个对齐 *_ref（9 个单测），
       组装只是"按 forward_token 顺序串起来"。分层验证把 e2e 调试成本降到近零
       ——这是"先算对再谈快"纪律在大改动上的复利。
    2. **消灭桥接的收益（1.19×）小于预期**：因为上一版 CPU 驱动已经把 matvec
       融合 + 权重常驻做好，桥接里"权重上传"那部分早被砍掉，剩下的主要是
       x/y 小拷贝和同步——所以整段常驻的增量集中在启动/调度层，真正的启动
       开销大头要靠 CUDA Graph 才能吃掉。
- **复现**：`MODEL=model_f16.tqwen ./scripts/bench.sh gpu_engine --extra-args "--engine cuda"`

---

### gpu_engine_cudagraph（2026-08-17）

- **机器**：与 cuda 系列条目同一台（x86_64 + NVIDIA A10）。
- **优化栈**：gpu_engine（f16，GPU-resident 整段 forward）+ **CUDA Graph
  capture/replay**
- **是什么**：`engine_step` 首次调用时把整段 launch 序列（embed→24 层→
  final norm/lm_head→argmax，~300 次 launch）`cudaStreamBeginCapture` 成
  一个 `cudaGraphExec_t`；之后每步只更新两个 device int（`d_pos`/
  `d_token`）的内容，再 `cudaGraphLaunch` 复用同一份图。前提改动：
  `embed_lookup`/`rope`/`kv_append`/`attention_decode` 的 `token`/`pos`
  参数从"按值传参"改成"传 device 指针、kernel 内解引用"——因为一份 graph
  一旦 capture，node 的参数值就固定了，每步变化的标量必须来自固定地址才能
  被 replay 时读到新值，不需要逐 node 用 `cudaGraphExecKernelNodeSetParams`
  打补丁。`engine_reset`（`seq_len=0`）不触发重新 capture，因为 graph 结构
  本身不依赖 pos 的值。
- **假设**：启动开销类。上一条目诊断"4.89ms 离带宽地板还差 ~3.3ms，主要是
  每 token ~300 次 kernel 启动的开销"，CUDA Graph 把整段启动批成一次
  `cudaGraphLaunch`，预期是"最大头"。
- **结果**：decode 中位 **4.63ms/token**（4 遍同场 A/B 交替测：A1 4.88/A2
  4.90（无 graph）、B1 4.62/B2 4.64（有 graph），各 27 个稳态 decode
  token），p95 ~4.97
- **vs 上一配置**：**1.06×**（同场 A/B 交替，4.89→4.63，两种顺序下结果一致，
  排除了机器漂移/顺序红利）
- **vs 原始基线**：跨机器（Mac fp32-baseline 是 M 系芯片），无意义，仅存档，
  同 `gpu_engine` 条目的处理方式。
- **验证**：71 单测全过（新增：`test_gpu_ops.cpp` 里 rope/kv_append/
  attention/embed 4 个测试因签名改动同步改为 device 指针调用，数值断言
  不变）；`--engine cuda` 在 fp32/f16 模型上 16-token 与 golden（CPU ref）
  逐位一致，f16 32-token 与 CPU forward 逐位一致；`scripts/verify.sh`
  （CPU 路径门禁）全过，未受影响。
- **瓶颈转移**：engine 路径无 op 级 profiler（同 gpu_engine 条目的既有限制），
  下一刀转向 kernel 内部效率（见 `gpu_engine_warprow`）。
- **意外 / 教训**：
    1. **收益（1.06×）远小于"最大头"的预期**：日志诊断的 ~3.3ms 缺口不能
       简单归因于"launch 次数多"。本机是 x86 高速主机，单次 kernel enqueue
       到 stream 的 CPU 侧开销本就很低，graph 省下来的主要是 driver 侧重复
       的参数校验/队列管理，而不是能大幅压缩的部分——"诊断听起来合理"和
       "测出来真值钱"是两件事，这里又是一次提醒（同构于 `neon_mt_bal` 条目
       "假设被证伪"）。
    2. **不是白费**：1.06× 虽小但方向稳（两次交替测都在同一水平），且改动
       本身（device 指针间接引用）是后续任何"减少每步开销"优化的基础设施，
       不是一次性的。
- **复现**：`MODEL=model_f16.tqwen ./scripts/bench.sh gpu_engine_cudagraph --extra-args "--engine cuda"`

---

### gpu_engine_warprow（2026-08-17）

- **机器**：与 cuda 系列条目同一台（x86_64 + NVIDIA A10）。
- **优化栈**：gpu_engine_cudagraph + **matvec 改 warp-per-row（多行每
  block）**
- **是什么**：`kernels/cuda/gpu_kernels.cu` 的 `matvec_single_kernel`/
  `matvec_pair_kernel`/`matvec_qkv_kernel`（engine 内部 `gpu::` namespace
  自带一份，不影响 `matvec_f*_cuda_resident_coal*.cu` 那批独立 dispatch
  变体）从 block-per-row（256 线程/block 归约一行，两级 shared mem + 两次
  `__syncthreads`）改成 **warp-per-row**：一个 warp（32 线程）独立归约一整
  行（stride 32，全程只用 warp-shuffle，无 shared mem/无同步），一个 block
  装 8 个 warp、同时算 8 行；grid 从"总行数"降到"总行数/8"。选 warp-per-row
  而不是 split-K：k/v_proj 融合进 qkv kernel 后是 1152 行里的一段，行数（乃
  至 pair 的 9728、single 的 896/151936）从不稀缺到需要 split-K 补 SM，真正
  浪费的是 block 内 256 线程只分到 in_dim=896/256≈3.5 个元素的低利用率。
- **假设**：并行粒度/占用类，直接对应上一条目诊断"k/v_proj 128 行
  block-per-row 占用低"。
- **结果**：decode 中位 **3.44ms/token**（3 遍同场测：3.44/3.44/3.45），
  p95 ~3.84
- **vs 上一配置**：**1.35×**（vs gpu_engine_cudagraph 的 4.63ms，同机同场）
- **vs 原始基线**（gpu_engine，无 graph 无 warprow，4.89ms）：**1.42×**
- **交互**：两步优化实测组合 1.42× ≈ 1.06×（cudagraph）× 1.35×（warprow）
  ≈ 1.43×，乘积近似成立，无明显负交互——两者分别作用在"host 侧调度开销"
  和"device 侧 kernel 内部归约效率"两个不同维度，互不干扰。
- **验证**：71 单测全过（含 `gpu_matvec_matches_ref` 覆盖 out_dim=128,
  in_dim=896 的 k/v_proj 真实形状，5e-3 容差覆盖归约顺序变化带来的浮点误
  差）；`--engine cuda` 在 fp32/f16 模型上 16-token 与 golden 逐位一致，f16
  32-token 与 CPU forward 逐位一致（reduction 顺序变了但贪心 argmax 结果
  未翻转）；`scripts/verify.sh` 全过。
- **瓶颈转移**：engine 路径仍无 op 级 profiler 覆盖，留待下次引入 profiling
  hook 再细分；合理猜测 lm_head（151936 行，本次同样吃到 warp-per-row 的
  收益）和层内 matvec 仍是耗时大头，继续压缩需要降权重流量（量化）或用
  tensor core，而不是再切并行粒度。
- **意外 / 教训**：
    1. **真正的"大头"在这一步，不在 CUDA Graph**：1.35× 远超上一步的 1.06×，
       证实"诊断里排第二的手段"实测反而是第一大贡献——日志诊断的顺序
       （① graph ② 小 matvec 并行）不等于实际收益的顺序，两个都做、都测，
       才知道哪个真的值钱。
    2. **warp 内 32 路 shuffle 归约 + 无 shared mem** 对 in_dim~896 这种"短"
       归约特别合适：省掉两次 `__syncthreads` 和跨 warp 的 shared mem 读写，
       且每线程从 3.5 个元素提到 28 个，算术强度显著提升——这类"归约粒度
       和数据规模错配"的问题，比"block 数够不够喂饱 SM"更常见也更容易被
       忽视（本例里 block 数从来没缺过）。
- **复现**：`MODEL=model_f16.tqwen ./scripts/bench.sh gpu_engine_warprow --extra-args "--engine cuda"`

---

### gpu_engine_attn_blocksize（2026-08-17）

- **机器**：与 cuda 系列条目同一台（x86_64 + NVIDIA A10）。
- **优化栈**：gpu_engine_warprow + **attention_kernel block 收窄到 head_dim**
- **是什么**：先用 `nsys profile --cuda-graph-trace=node --stats=true` 跑了
  一遍 `--engine cuda`（64 decode token），按 kernel 聚合耗时——这才是本条目
  的关键：**先测再猜**，没有再凭直觉排"下一刀"的优先级。结果
  `attention_kernel` 占 ~31-35%/token（单项第一，比 matvec 加起来还多），
  `matvec_single`（o_proj+down_proj+lm_head）~26-27%，`matvec_pair`
  （gate+up）~20-21%，其余算子合计 ~20%。据此把 `attention_decode` 的
  launch block size 从固定 `kBlock=256` 收紧成
  `ceil(head_dim/kWarp)*kWarp`（本模型 head_dim=64 → 正好 64，2 个 warp），
  这样 `attention_kernel` 里每个 timestep 都要做的 block 级双精度归约
  （`block_sum_d`，两次 `__syncthreads`）从 8-warp 树缩到 2-warp，且
  256 线程里原本白跑的 192 个（head_dim=64 时 `active` 恒假）全部消失。
- **假设**：并行粒度/占用类，与 `gpu_engine_warprow` 同一诊断模式（block
  内线程利用率低）。预期：kernel 内部耗时应随归约树变浅、无效线程消失而
  明显下降，进而拉动 decode 总时延。
- **结果**：decode 中位 **3.39ms/token**（3 遍同场测：3.39/3.39/3.40），
  p95 ~3.77；**假设部分证伪**——重新 profile 确认 `attention_kernel` 平均
  耗时只从 49.8us 降到 46.0us（降 ~8%，远不到"归约树变浅 4 倍"该有的降幅），
  真实 decode 时延也只降了 ~1.5%（3.44→3.39ms），比 profile 里的 35% 占比
  暗示的"biggest lever"小得多。
- **vs 上一配置**：**1.015×**（vs gpu_engine_warprow 的 3.44ms，同场）
- **vs 原始基线**（gpu_engine，无 graph/无 warprow/无本条，4.89ms）：**1.44×**
- **验证**：71 单测全过（`gpu_attention_matches_ref` 覆盖 seq_len∈{1,3,16}，
  launch 配置变了但数值断言不变，5e-3 容差下通过）；`--engine cuda` 在
  fp32/f16 模型上 16-token 与 golden 逐位一致，f16 32-token 与 CPU forward
  逐位一致；`scripts/verify.sh` 全过。
- **瓶栈转移**：仍是 `attention_kernel` 第一大头（重新 profile 后从
  ~35%→~29%，占比降了但仍居首），`matvec_single`/`matvec_pair` 相对占比
  升到并列第一梯队。下一刀如果还要继续压 attention，方向应该是"消灭
  timestep 间的串行依赖链"（online softmax 的 m/l/oh_i 状态跨迭代必须
  串行更新），而不是再切并行粒度——这需要更大的重构（比如整体换成
  flash-decoding 式的 split-KV + 最终归并），成本远高于本条目，暂不做。
- **意外 / 教训**：
    1. **这次真的"先测再改"，但预测力还是有限**：profile 告诉了我们
       "attention 是第一大头"（这个结论后来站得住——重新 profile 后依然
       第一），但没告诉我们"改 block size 能省多少"——kernel 内部的真实瓶颈
       是 timestep 间的**串行内存延迟依赖链**（每次迭代要等上一轮 m/l/oh_i
       更新完才能发下一轮的全局内存读取+归约），不是"归约树宽度"，所以把
       归约树从 8-warp 砍到 2-warp 只省了树本身的开销（~8%），没碰到真正
       的延迟大头。**profile 定位"哪个 kernel"很可靠，但定位"kernel 内部
       哪个子机制"仍然需要更细粒度的工具**（如 ncu 的 stall reason 分析），
       这是本条目和 `gpu_engine_cudagraph`/`gpu_engine_warprow` 两条对比后
       补上的认知缺口。
    2. **依然是净正、不撞回归**：1.015× 虽小，但方向稳（3 遍一致）、改动
       风险低（只改 launch 配置、不改数值语义），符合"能测出正收益就
       记账"的纪律，不因为小就不记录。
- **复现**：`MODEL=model_f16.tqwen ./scripts/bench.sh gpu_engine_attn_blocksize --extra-args "--engine cuda"`

---

### gpu_engine_argmax_2pass（2026-08-17）

- **机器**：与 cuda 系列条目同一台（x86_64 + NVIDIA A10）。
- **优化栈**：gpu_engine_attn_blocksize + **argmax 改两阶段归约**
- **是什么**：profile（同上条目的 nsys 方法）显示 `argmax_kernel` 单次
  ~220us——它是 `<<<1, 256>>>` 单 block 跨步扫全 vocab（151936 个 float），
  256 个线程摸近 15 万元素，A10 72 个 SM 只吃到 1 个，并行度严重不足。
  改成两阶段：①`argmax_partial_kernel`，`kArgmaxBlocks=288`（~4×SM 数）个
  block 各用 grid-stride 扫一段、block 内归约出局部 (value,idx) 最优写进
  固定大小的 `partial` 缓冲；②`argmax_final_kernel`，grid=1 合并 288 个
  partial 得到全局最优。两个 kernel 共享抽出来的 `block_reduce_best` 归约
  helper，tie-break（严格 > 取最小下标）逐位对齐原实现。partial 缓冲是
  进程内单例、固定大小（不随 n 增长），仿 `matvec_f32_cuda_resident*.cu`
  的常驻 workspace 写法。
- **踩坑**：partial 缓冲最初写成"首次调用时懒分配"（`static` 指针，首次
  `cudaMalloc`），实测直接炸：engine 第一次 `engine_step`（= CUDA Graph
  capture 那一步）里调 argmax 时触发分配，报
  `cudaMalloc ... operation not permitted when stream is capturing`——
  cudaMalloc 在 stream capture 期间确实不允许调用（和上一条目"以为
  block_sum_d 归约树宽度是瓶颈"一样，这次是"以为 cudaMalloc 不挂 stream
  就总是安全"，两次都是先跑了才知道对不对）。修法：新增
  `gpu::argmax_init()`，`engine_create` 里在所有其它 cudaMalloc 之后、
  第一次 `engine_step` 之前显式调用一次，把分配挪到 capture 开始前完成；
  测试直接调 `argmax()` 不涉及 capture，走懒分配即可，不用改。
- **假设**：并行度类，与 `gpu_engine_warprow`/`gpu_engine_attn_blocksize`
  同一诊断模式（block/thread 利用率低），但这次是"单 block 对整个 vocab"
  这种更极端的形式。
- **结果**：decode 中位 **3.19ms/token**（3 遍同场测：3.19/3.19/3.19），
  p95 ~3.57；重新 profile 确认 argmax 从 ~220.9us/次砍到 ~6.1us/次
  （partial 3.8us + final 2.3us），**36×**，与"这次真的是并行度瓶颈"的
  诊断吻合——不同于上一条目 attention 的"诊断对但预测力有限"。
- **vs 上一配置**：**1.06×**（vs gpu_engine_attn_blocksize 的 3.39ms，同场，
  与 argmax 原占比~6.5% 吻合，说明这一刀几乎把该项开销压到了地板）
- **vs 原始基线**（gpu_engine，无 graph/无 warprow/无 attn 收窄/无本条，
  4.89ms）：**1.53×**
- **验证**：71 单测全过（`gpu_argmax_matches_ref` 覆盖 n=151936 大数组、
  平局取首个、全等取 0 等边界，两阶段版数值与单 block 版逐位一致）；
  `--engine cuda` 在 fp32/f16 模型上 16-token 与 golden 逐位一致，f16
  32-token 与 CPU forward 逐位一致；`scripts/verify.sh` 全过。
- **瓶栈转移**：`attention_kernel`（~31%）和 `matvec_single`
  （o_proj+down_proj+lm_head，~28%）并列第一梯队，`matvec_pair`
  （gate+up，~22%）紧随。argmax/embed/kv_append/rope/swiglu 这类小算子
  合计已 <10%，边际收益越来越薄——继续压需要回到之前两条目分析过的大项：
  attention 的串行依赖链（需要 flash-decoding 式重构）或量化（权重流量
  ÷4/÷8，同时压 matvec_single/pair/qkv 三块）。
- **意外 / 教训**：
    1. **"单 block 处理超大数组"是本次发现的第三种低并行度模式**：先是
       matvec 的"block-per-row 线程摸太少"，再是 attention 的"block 内
       归约树太宽但真正瓶颈是延迟"，这次是"干脆就一个 block"——三种模式
       长得像但根因和修法都不同，profile 才能分清哪个 kernel 属于哪种。
    2. **cudaMalloc 在 capture 期间被禁止，这是本 session 第二次"以为安全
       实际不安全"的教训**：和 `gpu_engine_attn_blocksize` 条目的"以为
       block size 是瓶颈"一样，都是先做了假设、写了代码、跑起来才发现假设
       不成立——区别是这次是直接报错（好发现），上次是数字对但不及预期
       （更隐蔽）。两次都验证了"写完就跑一遍"这个纪律的价值。
- **复现**：`MODEL=model_f16.tqwen ./scripts/bench.sh gpu_engine_argmax_2pass --extra-args "--engine cuda"`

---

### fp16-neon-android（2026-08-18，Android）

- **设备**：PLK110 / canoe / Android 16（6×3.63GHz + 2×4.61GHz，绑 2 超大核，TINYQWEN_MT_THREADS=2）
- **优化栈**：android-fp16-ref-baseline（f16 标量 ref）+ f16 满栈（neon_mt_kv_nt + ops neon）
- **是什么**：macOS 上已验证的 f16 满栈 kernel（NEON fp32 累加 + 常驻线程池行切分 +
  k/v 成对融合 + LDNP 流式加载，另含 rmsnorm/rope/attention/swiglu/argmax 五个非
  matvec 算子的 NEON 变体）首次在 Android arm64 真机落地测量
- **假设**：带宽类——decode 是纯权重带宽瓶颈，f16 流量减半 + SIMD + 多线程拉高有效带宽
- **结果**：decode 中位 **21.16 ms/token**（3 遍取中位，每遍 27 样本），p95 24.33
- **vs 上一配置**：**32.88×（同场 A/B）**——对照（无额外参数，当前即 ref）中位 695.64（p95 696.63）→ 变体 21.16，同 binary
  同场交错测量。（vs 基线 32.87×）
- **基线参照**：android-fp16-ref-baseline（695.52 ms/tok @ PLK110），本次 vs 基线 = 32.87×
- **验证**：scripts/verify_android.sh（golden token 逐位对照）
- **瓶颈转移**：top op = `lm_head`（139ms/35 token，单算子最大流量：151936×896 f16）、
  其后是各层 `gate_up_proj`。下一刀砍哪：`lm_head` —— INT4 量化（流量÷4）是既定路线
- **意外 / 教训**：① 热敏"全 zone 取 max"会撞上 `cpu-hw-trip-*`（硬件关机阈值，
  恒定 95°C）把热门禁永久卡死——只能信 type 以 cpu 开头的 zone；② toybox taskset
  不认 `0x` 前缀与 CPU 列表，只认裸十六进制 mask；③ 多线程 kernel 默认线程数 =
  全部核数，绑核后必须对齐 TINYQWEN_MT_THREADS，否则线程挤在绑定的核上互相打架
- **复现**：
  `MODEL=model_f16.tqwen ./scripts/bench_android.sh fp16-neon-android --extra-args "--matvec-impl neon_mt_kv_nt --ops-impl neon"`

---

### 第二次测试（2026-08-18，Android）

- **设备**：PLK110 / canoe / Android 16
- **优化栈**：<基线 + 本次优化，如 android-baseline + XXX>
- **是什么**：<本次改了哪个 kernel / 数据结构 / 调度，一两句话>
- **假设**：<为什么预期会快：带宽 / 计算 / 并行 / 指令 哪一类>
- **结果**：TTFT **1490.48 ms**（prefill 3 tok）；TOPT 中位 **695.55 ms/token**（1 遍取中位；decode 共 32 tok，丢预热，稳态样本
  27），p95 700.46；forward 总耗时 23186.4 ms
- **端侧资源**：端侧内存：peak RSS 949.4 MB（avg 939.1，94 个采样点），权重 942.3 MB，KV cache 1.5 MB；推理期间大核频率：883200 ~
  3398400 kHz（中位 2438400）
- **vs 上一配置**：<填：vs 上一配置>（vs 基线 1.00×）
- **基线参照**：android-fp16-ref-baseline（695.52 ms/tok @ PLK110），本次 vs 基线 = 1.00×
- **验证**：scripts/verify_android.sh（golden token 逐位对照）
- **瓶颈转移**：top op = `lm_head`、`layer_3.gate_up_proj`、`layer_0.gate_up_proj`，下一刀砍哪：<填>
- **意外 / 教训**：<填——往往最值钱>
- **复现**：`MODEL=model_f16.tqwen ./scripts/bench_android.sh 第二次测试`

---

### dataset-smoke（2026-08-19，Android，数据集负载）

- **设备**：PLK110 / canoe / Android 16（绑大核 cpu6,7，TINYQWEN_MT_THREADS=2）
- **配置**：f16 满栈（neon_mt_kv_nt + ops neon），与 fp16-neon-android 同配方
- **是什么**：**数据集负载首测**（非优化，换尺子）——诗词数据集 test.csv 取 16 条
  真实 prompt（token 长度 30~49，中位 44），runtime 批量模式
  （`--batch-tokens-jsonl`：一次进程逐条 reset→prefill→decode）测 TTFT 分布与 TOPT。
  测量侧新工具：`tools/tokenize_batch.py` + `tools/bench_dataset.py`，面板可发起
- **结果**：
    - TTFT 中位 **832.88 ms**（p95 1249.44，归一 23.09 ms/tok）；分桶：16-32 tok
      （n=3）642 ms / 20.7 ms/tok，32-64 tok（n=13）997 ms / 25.7 ms/tok——
      prefill 随长度近线性
    - TOPT 中位 **30.75 ms/tok**（各 prompt 中位的中位），全量 p95 33.44（496 decode 步）
    - 总耗时 29.0 s（16 条 × decode 32 tok）
- **为什么比 canonical 的 21.16 慢（负载不同，不直接可比）**：
  ① 设备带 61.6°C 积热开跑（热门禁等满 180s 超时放行），全程峰 79.8°C，大核中位
  频率被压到 **1632 MHz**（正常 2438+）；② prompt 30~49 tok，KV 比 canonical 的
  3 tok 长。逐条 TOPT 从首条 22.2 涨到末条 33.1 ms/tok——热降频过程在资源曲线
  上直接可见
- **资源**：peak RSS 953.2 MB（权重 942.3 + KV cache 3.0 @ seq 128），
  扛算核均值利用率 99.4%（双线程喂满两大核）
- **正确性**：批量 vs 逐条单跑 generated_ids 逐位一致（fake Qwen2.5/Qwen3.5 双验证）
- **曲线**：`benchmarks/series/20260819-105819_dataset-smoke{,.ttft}.json`
  （控制台历史表「曲线」按钮：资源四联图 + TTFT vs prompt 长度散点）
- **复现**：
  `python tools/tokenize_batch.py --num 16 --out benchmarks/poetry16.jsonl && python tools/bench_dataset.py --label dataset-smoke --jsonl benchmarks/poetry16.jsonl --model model_f16.tqwen --extra-args "--matvec-impl neon_mt_kv_nt --ops-impl neon"`

---

---

### i4-hqq-android（2026-08-19，Android，INT4 路线开工）

- **设备**：PLK110 / canoe / Android 16（绑大核 cpu6,7，TINYQWEN_MT_THREADS=2）
- **优化栈**：android-i4-ref（HQQ 模型 + ref kernel）+ i4 NEON matvec + lm_head f32 实现联动
- **是什么**：INT4 路线落地第一步——量化算法从 RTN 换成 **HQQ**（hqq 0.2.8，
  MSE 最优迭代、免校准，`--method hqq --group-size 64`，产物 `model_i4_hqq.tqwen`
  746MB）；修复三处 i4 链路 bug（verify 脚本 DTYPE_I4=2→3、quant_group_size
  offset 44→40、header/tensor-table 布局整个读错）；补齐 12 个 i4 单测；
  修复 i4 模型下 **lm_head 走标量 f32 ref** 的问题（main.cpp 现在给 f32
  注册表联动选实现）
- **假设**：带宽类——权重流量 ÷4 应带来接近 4× 提速（f16 已撞 42 GB/s 墙）
- **结果**：TOPT 中位 **73.31 ms/tok**（3 遍，p95 77.79），同场 A/B vs i4 ref
  594.92 → 73.31 = **8.12×**
- **精度**：HQQ@64 逐层 MSE 均值 4.40e-06（RTN@128 为 6.12e-06，**-28%**）；
  token diff vs fp32 = 1/16（RTN 为 2/16），生成文本与 fp32 几乎一致
- **PMU 流量账**：bus_access 7.87 亿 vs f16 的 10.5 亿——**流量只降 25% 而非 ÷4**，
  因为 tied lm_head 是 fp32（544MB/token = 57% 流量）；DRAM 带宽仅用 6~12 GB/s
- **为什么比 f16 满栈（21.16）慢**：① i4 matvec kernel 只有单线程 neon 版，
  nibble unpack 慢，层部分实测仅 ~3 GB/s 有效带宽；② lm_head fp32 15 ms/tok。
  当前 73 ms 的构成 ≈ lm_head 15 + i4 层 ~55 + 其余
- **意外 / 教训**：① verify_i4_accuracy.py 此前因 DTYPE bug 一直空跑 PASS；
  ② i4 文件的 tied lm_head=embed(fp32) 是隐形流量大头，INT4 的 ÷4 红利要先
  解决它（独立量化 lm_head，+~70MB 文件，流量 544→70MB/token）；③ HQQ 的
  dequant 语义 (q-zero)×scale 与现有 kernel 完全一致，C++ 零改动
- **下一刀**：i4 kernel 优化（多线程 + 快 unpack/LUT，层部分 ~55→10 ms/tok）+
  lm_head i4 量化（15→2 ms/tok）→ 目标 ~6-10 ms/tok
- **复现**：
  `python tools/export_qwen_to_tiny_i4.py --model models/Qwen2.5-0.5B --out model_i4_hqq.tqwen --method hqq --group-size 64 && ./scripts/bench_android.sh i4-hqq --model model_i4_hqq.tqwen --extra-args "--matvec-impl neon --ops-impl neon"`

---

---

### i4-neon_mt（2026-08-19，macOS）

- **优化栈**：i4-hqq（HQQ@64 + i4 neon）+ i4 matvec 多线程行切分
- **是什么**：`matvec_i4_neon_mt`（`kernels/matvec/matvec_i4_neon_mt.cpp`）——
  RowPool 常驻线程池（spin barrier）按输出行切分，数值部分与 i4 neon 相同；
  小矩阵（<256K 元素）自动单线程。注册名 `neon_mt`（i4 注册表）
- **结果**（M5 Pro，TINYQWEN_MT_THREADS 扫描）：
  | 线程 | 2 | 4 | 6 | 8 | 12 | 15（默认） |
  |---|---|---|---|---|---|---|
  | ms/tok | 22.39 | 14.37 | 12.95 | **11.95** | 14.37 | 13.07 |

  8 线程 runs=3 复测：**10.99 ms/tok**（p95 12.29），vs i4 neon 单线程
  37.40 = **3.40×**
- **构成**（8 线程）：lm_head（fp32，走 f32 neon_mt）2.55 ms/tok；i4 层部分
  ~8.4 ms/tok（180MB/token → 21 GB/s，离 Mac 带宽墙仍远——i4 unpack 还是
  compute-bound，继续优化的空间在更快的 unpack/LUT 而非加线程）
- **验证**：单测 3 例（多线程路径 g64/g128 + 小矩阵单线程分支）；Mac 全链路
  16 token 输出与 neon/ref 逐位一致
- **待办**：Android 真机同场 A/B（设备暂离线）。注意 Android 绑 2 大核场景
  TINYQWEN_MT_THREADS=2，收益取决于 2 核分摊 unpack 的效率
- **复现**：
  `TINYQWEN_MT_THREADS=8 python tools/bench.py --model model_i4_hqq.tqwen --label i4-neon-mt8 --runs 3 --extra-args "--matvec-impl neon_mt --ops-impl neon"`
- **为什么 i4 在 macOS 仍比 f16 慢（10.37 vs 5.85）——unpack 计算墙诊断**：
  profile 算有效带宽：f16 各 op 120~180 GB/s（撞 DRAM 墙），i4 只有 7~30 GB/s。
  线程扫描 4/5/8 = 17.08/13.55/10.57 ms——持续改善，是 compute-bound 曲线而非
  调度病理。根因：i4 内循环每 16B 权重 ~56 条 NEON 指令（nibble 拆分 vzip +
  两级展宽 vmovl×12 + vcvt×8 + dequant/累加 FMA×16 + x 加载×8）≈ 3.5 条/字节，
  f16 kernel（LDNP+FMLA）≈ 0.2 条/字节，差 ~19×。M5 P 核 ~10 G NEON ops/s ÷
  3.5 ≈ 单核 dequant 上限 ~3 GB/s——8 核合力 ~25 GB/s，离 Mac 的 ~150 GB/s
  DRAM 能力差 6×。**i4 的 ÷4 流量红利被 unpack 指令流吃掉了**。
  ⚠️ 推论修正：同一计算墙在 Android 同样存在，i4 在那边也到不了"带宽÷4"的
  理论 TOPT——下一刀必须是 dequant 提速（vqtbl LUT 查表，~4-6 条指令/16B，
  上限 3-4×；或 i4→i8 dotprod 路径），而非继续加线程

---

### i4-lm_head（2026-08-19，macOS）

- **优化栈**：i4-neon_mt（HQQ@64 + i4 neon_mt）+ lm_head 独立 i4 量化
- **是什么**：tied 模型导出时把 lm_head 也量化——`export_qwen_to_tiny_i4.py`
  追加 `lm_head.weight`（embed 的独立 HQQ 副本），embed_tokens 本身保持 fp32
  供 token lookup。runtime（qwen_model.cpp）tied 时优先绑定文件里的
  lm_head.weight，把每 token **544MB fp32 → ~70MB i4** 流量。文件 +73MB。
  开关 `--no-lm-head-i4` 可导出旧行为（A/B 对照）
- **假设**：带宽类——lm_head 曾是 INT4 路径最大单项流量（占 57%），量化后
  流量 ÷8 该层
- **结果**（M5 Pro，8 线程，同场先后跑 runs=3）：
    - lm_head-fp32（对照）：11.73 ms/tok
    - lm_head-i4（变体）：**10.37 ms/tok** → **1.13×**
    - Mac 带宽充裕（fp32 lm_head 也能快速 stream），收益温和；Android 42GB/s
      墙下该层 ~13ms → ~2ms，预期收益显著（待设备验证）
- **精度**：lm_head.weight (vs embed) MSE 1.94e-06 / cos 0.9960（优于层均值）；
  token diff vs fp32 = 2/16（其中 1 处是层量化既有差异），文本语义一致
  （仅多一个句号）
- **验证**：audit 新增 tied lm_head 对照（vs embed）；Mac 全链路输出与预期一致
- **复现**：`python tools/export_qwen_to_tiny_i4.py --model models/Qwen2.5-0.5B --out model_i4_hqq_lmh.tqwen`（默认开
  lm_head i4）+
  `python tools/bench.py --model model_i4_hqq_lmh.tqwen --extra-args "--matvec-impl neon_mt --ops-impl neon"`

---

### i4-neon_lut（2026-08-19，macOS，证伪归档）

- **优化栈**：i4-neon_mt + per-group fp16 LUT 查表反量化（`matvec_i4_neon_lut`）
- **假设**：把整条点积留在 fp16——x 每次 matvec 预转一次 f16、权重按组生成
  16 项 fp16 查找表经 2×vqtbl+vzip 直出、vfmaq_f16 累加——预期替代三级展宽
  （vmovl_u8→vmovl_u16→vcvtq）拿到 ~3×
- **结果**：**38.78 ms/tok，比 neon_mt（11.89）慢 3.3×**；单 matvec max_err ~0.02
  超 1e-3 门禁 → 弃用，代码已移除
- **为什么慢（指令账错在哪）**：只算了内循环的查表指令，漏算了**每组构造查找表**
  的标量开销（16 项 fp16 计算 + 字节拆分循环）。group=64 时每行组数多
  （lm_head 一行 14 组、共 151936 行 → ~213 万次表构造），per-group 构造成本
  完全吞掉查表省下的指令。**LUT 只在 group_size 大（组少）时才可能摊平。**
- **为什么精度差**：为提速把激活 x 也转 fp16（10-bit 尾数），单 matvec 误差
  ~0.02。若保 fp32 激活则查表无速度优势（展宽链本就接近 fp32 反量化的指令下限）。
- **教训**：① 微基准的指令账必须计入 per-group/per-call 的**准备开销**，不能只数
  内循环；② int4→fp32 反量化在 NEON 上，展宽链（vmovl×2+vcvt）已接近指令下限，
  纯"查表换展宽"在 fp32 精度下无收益、在 fp16 下用精度换且被表构造开销抵消。
  i4 层剩余空间在**减少 kernel 调用数/改善访存**而非反量化指令本身。
- **复现**：已移除；如需复现见 git 历史该小节对应 commit

---

### i4-hqq-lmh-mt-android（2026-08-19，Android）

- **优化栈**：i4-hqq（HQQ@64）+ lm_head i4 + i4 neon_mt（2 大核）+ ops neon 全套
- **结果**（PLK110 / canoe，绑 cpu6,7，TINYQWEN_MT_THREADS=2）：
  **TOPT 44.72 ms/tok**（p95 50.47，3 遍），同场 vs i4 ref 607 = **13.58×**；
  vs android-fp16-ref-baseline(695.52) = 15.6×；TTFT 1078ms；peak RSS 791MB
- **关键发现——但仍比 f16 满栈(21.16)慢 2.1×**，瓶颈归因：
  - i4 流量 ~250MB/token，44.72ms → 有效带宽仅 **5.6 GB/s**，离 42 带宽墙差 7×
  - 即 **i4 是算力瓶颈不是带宽瓶颈**：unpack 链 ~3.5 指令/字节，单核 dequant
    上限 ~2.8 GB/s（Mac/Android 都测出这个数），2 核 × 2.8 = 5.6 GB/s
  - **纠正上一轮错误预测**：曾说"低于 86GB/s 临界点 i4 赢"，那默认了 i4 能跑满
    带宽；实际 i4 永远到不了带宽墙（算力封顶），核数不够多就追不上 f16 的带宽
- **i4 慢不是 int4 之过**：`mnn_comparison.json` 里 MNN int4(HQQ) 在 M5 Pro
  3.68ms = 68 GB/s（带宽态），是我们 i4 kernel（23GB/s@Mac）的 3×。**差的是
  kernel 指令效率，不是量化方案**
- **结论**：i4 路线剩最后一刀 = **重写反量化内核**，把指令/字节从 3.5 打下来
  （方向：i8mm/dotprod 的 W4A8 Marlin 式，或大 group 共享反量化），而非调线程/
  再量化别的层。已排期
- **复现**：`python tools/bench_android.py --label i4-hqq-lmh-mt-android --model model_i4_hqq_lmh.tqwen --extra-args "--matvec-impl neon_mt --ops-impl neon" --runs 3`

---

### i4-sdot_mt（2026-08-19，Android，W4A8 SDOT——i4 追平 f16）

- **优化栈**：i4-hqq-lmh + W4A8 SDOT kernel（`matvec_i4_sdot` / `sdot_mt`）
- **是什么**：把 i4 反量化从"展宽转 fp32"（3.5 指令/字节）换成 **W4A8 整数点积**：
  权重 q_s=q−8 解包成 signed int8，激活 x 每次 matvec 对称量化成 int8（scale_x），
  用 armv8.2 SDOT（clang 的 `vdotq_s32`，Apple/NDK clang 21 通用）1 条指令做 16 个
  int8×int8 乘累加；非对称 zero 修正走前缀和：
  `y[o]=Σ_g A_g·DOT_g − A_g·(zero_w−8)·XQSUM_g`（DOT_g=SDOT，XQSUM_g=Σx_q 前缀和）。
  unpack 指令/字节 3.5→~0.9
- **结果**（PLK110，绑 cpu6,7，TINYQWEN_MT_THREADS=2）：
  **TOPT 22.30 ms/tok**（p95 22.44，3 遍），同场 vs i4 neon_mt(44.59) = **2.00×**
- **校正后的准确结论（跨模型同场 A/B）**：i4 22.35 vs **f16 满血 17.05 = 0.76×**
  （i4 仍慢 ~24%）。此前"追平 f16"是拿了过热降频的 f16（21.16）做参照的过头表述。
  本质：f16 是带宽瓶颈、随温度漂（凉 17.05 ↔ 热 21.16）；i4 是算力瓶颈、稳定 22.3。
  所以 i4 追平的是"过热的 f16"，比"满血 f16"仍慢 ~24%。i4 在这台高带宽设备上的
  优势是**内存**（784 vs 942MB、matmul 流量÷4）与**热稳定**，不是绝对速度。
  i4 的速度优势要在带宽更低的设备上才兑现
- **为什么 Mac 上 sdot 反而只快 1.23×（9.18 vs neon_mt 11.30）且不敌 f16(6.35)**：
  Mac 带宽 ~150GB/s 太高，f16 带宽墙在高处，i4 的算力优势追不上——**i4 的价值
  主场就是带宽紧的端侧**
- **代价**：激活 int8 量化引入量化误差（业界 W4A8 标准做法），token diff vs fp32
  在既有门禁内容忍；精度审计与 neon 版同级
- **剩余差距**（22.30 vs 21.16 的 5%）：lm_head 仍占 5.5ms/token（虽已 i4），
  per-group 标量开销（A/C 系数、前缀和）——后续可 fuse/减少组开销再压
- **复现**：`python tools/bench_android.py --label i4-sdot-mt-android --model model_i4_hqq_lmh.tqwen --extra-args "--matvec-impl sdot_mt --ops-impl neon" --control-args "--matvec-impl neon_mt --ops-impl neon" --runs 3`

---

### i4 路线复盘（2026-08-19，HQQ→lm_head→neon_mt→LUT（证伪）→W4A8 SDOT→f16 对照）

**路线全景**：INT4 推理从"HQQ 导出+仅 ref kernel 能跑"到"W4A8 SDOT 满栈达
f16 七成+，内存/热稳定优势明确"，全程 3 天、5 个台阶、3 个证伪/校正。

**各台阶真机 Android TOpt（PLK110，canonical 负载，2 大核）**：

| 台阶 | 配置 | TOpt |
|------|------|------|
| ① 起点 | i4 ref（RTN 量化，标量 ref kernel，lm_head fp32） | ~607 ms |
| ② HQQ + neon + lm_head 脱离标量 | i4 neon（HQQ@64，NEON unpack，lm_head→f32 联动） | 73.3 ms |
| ③ 多线程 | i4 neon_mt（RowPool 行切分，2 核） | 44.7 ms |
| ④ lm_head 量化 | i4 neon_mt + lm_head i4（tied 独立副本，流量 544→70MB/tok） | 44.7 ms（同③，lm_head 量化的收益被多线程覆盖，但 Mac 侧 1.13×） |
| ⑤ LUT 查表反量化 | **证伪**：per-group fp16 LUT 组内标量建表开销吞掉查表收益，实测 38.78ms 比③慢 3.3×，且 fp16 激活 max_err~0.02 超门禁 | 38.78 ms（弃用） |
| ⑥ W4A8 SDOT | **收窄**：激活对称 int8 量化 + SDOT 整数点积，指令/字节 3.5→~0.9 | **22.35 ms** |

**跨模型同场 A/B（i4 ⑥ vs f16 满血）**：i4 22.35 vs f16 17.05 = **0.76×**。
f16 带宽瓶颈，随温度漂（凉 17 ↔ 热 21）；i4 算力瓶颈，稳定 22.3。校正：
此前"追平 f16"是拿了过热降频的 f16（21.16）做参照的过头表述，已在日志中
修正。

**i4 在这台设备上的价值定位**：
- 速度：比满血 f16 慢 ~24%，但比过热 f16 快或持平，且热稳定（22.3ms 几乎不漂）
- 内存：784MB vs f16 的 942MB（matmul 流量÷4，但 embed 仍 fp32 占大头）
- 速度优势要到带宽更低的设备兑现（临界 BW ≈ 22.3× f16 可用带宽 / 0.76…）

**Mac 侧的教训（i4 全程慢于 f16，不应作为 i4 判据）**：
- Mac 带宽 ~150GB/s，f16 撞墙在 6.35ms——太远，i4 的算力优势追不上
- **i4 vs f16 的快慢排序取决于硬件带宽，不是模型大小**——这个判断是正确的
- 但"Android 上 i4 会反超 f16"的预测偏乐观（低估了算力天花板），实测是
  收窄差距而非反超

**LUT 证伪的教训**（通用方法论）：
- 微基准的指令账必须计入 per-group/per-call 的**准备开销**（标量建表、
  前缀和、A/C 系数），不能只数内循环
- NEON 上 int4→fp32 展宽链（vmovl×2+vcvt）已接近指令下限，纯"查表换展宽"
  在 fp32 精度下无收益、在 fp16 下用精度换且被表构造开销抵消
- 真正有效的提速是换整数点积（SDOT 路线），而不是在 fp32 反量化内部抠指令

**SDOT 的教训（编译器适配）**：
- Apple clang 21 与 NDK clang 21 的 SDOT intrinsic 名都是 `vdotq_s32`（不是
  ARM 官方文档的 `vsdotq_s32`），`vdotq_s32` 带 `target("dotprod")` 属性，
  无需全局 `-march` flag 也能编译——但文件级 `-march=armv8.2-a+dotprod` 仍是
  最稳妥的守卫

**剩余方向**（i4 路线到此超预期交付，以下为后续选项）：
- lm_head 进一步优化（5.5ms/token 仍在总时间中占大头，可 fuse 或更大 tile）
- per-group 开销压缩（A/C 系数、前缀和预计算）
- 换设备验证（带宽更低的端侧，i4 速度优势可能兑现）

---

### qwen35-f16-neon-android（2026-08-19，Android，Qwen3.5-0.8B 混合架构首测）

- **是什么**：Qwen3.5-0.8B（Gated DeltaNet + full attention 3:1 混合）f16 满栈
  首次导出 + macOS 连贯性验证 + Android 真机测速。HF 权重在
  `~/.cache/modelscope/models/Qwen--Qwen3.5-0.8B`，exporter 已支持 qwen3_5
  （字段映射 linear_num_key_heads→linear_num_qk_heads 等；attn_output_gate
  由 runtime sigmoid 门支持）。导出 `model_qwen35_f16.tqwen` 1435MB
- **macOS 连贯性**：prompt "中国的首都是" → "中国的首都是**北京**。北京位于中国
  东部，是中华人民共和国的行政中心，也是全国…" 连贯且正确
- **Android 结果**（PLK110，f16 满栈 neon_mt_kv_nt + neon ops，绑 cpu6,7）：
  **TOPT 35.29 ms/tok**（p95 37.69），同场 vs qwen35 ref(1063) = **30.19×**；
  TTFT 231ms（prefill 3 tok）；peak RSS 1463.7MB（权重 1435 + GDN 19.3 + KV 1.5）
- **关键发现——lm_head 主导**：lm_head 552.94ms/32 = **17.3 ms/token，占 49%**。
  根因：Qwen3.5 词表 248320（Qwen2.5 是 151936），lm_head = 248320×1024 f16
  = **508MB/token 流量**，是 Qwen2.5 lm_head（272MB）的 1.9×。GDN 层很高效
  （~0.6ms/token/层，O(1) seq，无 KV 增长）
- **vs Qwen2.5-0.5B**：Qwen3.5 TOPT 35.29 vs Qwen2.5 f16 ~17-21 = **慢 ~1.7-2×**，
  主因是模型更大（0.8B vs 0.5B）+ 词表/lm_head 更大
- **优化方向**：lm_head 是最大单一成本（49%），若要提速，优先量化 lm_head
  （i4 可把 508MB→127MB/token）；GDN 层已经很快不是瓶颈
- **复现**：`python tools/export_qwen_to_tiny.py --model <Qwen3.5 dir> --out model_qwen35_f16.tqwen --dtype f16`
  + `python tools/bench_android.py --label qwen35-f16-neon-android --model model_qwen35_f16.tqwen --extra-args "--matvec-impl neon_mt_kv_nt --ops-impl neon" --runs 3`

---

### i4-sdot2_mt（2026-08-19，macOS，预计算 + 2-row 并行——i4 首次反超 f16）

- **优化栈**：i4-sdot_mt（HQQ@64 + W4A8 SDOT）+ **预计算 scale/zero f32 + 2-row 并行**
- **是什么**：新增 `kernels/matvec/matvec_i4_sdot2.cpp`，两项优化叠加：
  ① **预计算 scale/zero 为 f32 数组**：首次见到某权重指针时，把 interleaved
  `[scale_fp16|zero_fp16|packed]` 中的 scale/zero 解包成连续 f32 数组缓存
  （按权重指针索引）。消除 per-group 的 2×memcpy + 2×half_to_float + 1×FSUB
  = 5 inst/group，省 ~39M inst/token（~10%）。
  ② **2-row 并行内循环**：同时处理 2 行权重，2 条独立 SDOT 链（acc_a/acc_b）
  + 共享激活加载（省 1 条 vld1q_s8/迭代）。两条独立链提高 ILP，隐藏 unpack 延迟。
- **假设**：指令效率 + ILP。指令分析显示 per-group 开销占 34.7%（131M/378M），
  内循环 SDOT 链有 serial dependency（acc_dot 串行）。预计算消前者，2-row 解后者。
- **结果**（M5 Pro，3 轮同场 A/B）：
  - sdot_mt 原版：median **6.31ms**/tok（3 轮：6.31/7.64/5.59）
  - sdot2_mt 优化：**3.67ms**/tok（3 轮：3.67/3.58/4.33）
  - **vs 上一配置：1.72×（同场）**
  - **同场 vs f16 满栈（neon_mt_kv_nt 5.68ms）：i4 sdot2 3.58ms = 1.59× i4 更快**
  - **i4 首次在同机上反超 f16**
- **验证**：scripts/verify.sh 全过；sdot2_mt vs sdot_mt 生成 token 16 个逐位一致
  （greedy argmax 结果未翻转）。
- **瓶颈转移**：top op 仍是 `lm_head` + 各层 `gate_up_proj`（结构不变），但时间
  大头从"unpack 指令"转向"剩余的 per-group 标量开销 + cache miss"。
  下一刀候选：① 4-row 并行（寄存器压力可能限制收益）；② 权重布局重排
  （scale/zero 分离到连续数组，进一步改善 cache locality）；③ Android 真机验证
  （2 大核场景预期 ~13ms，超过 f16 满血 17ms）。
- **意外 / 教训**：
  1. **i4 的算力墙不是 SDOT 的锅，是 per-group 标量开销**——指令账显示 34.7%
     的指令花在 scale/zero 的 memcpy+half_to_float+系数计算上，而不是内循环
     的 SDOT 点积。这解释了为什么 sdot（0.9 inst/byte）没有预期的快。
  2. **2-row 的收益超出预期**——预期 ~1.3×（纯 ILP），实测与预计算叠加得
     1.72×。两行共享激活加载（省 1 条 vld1q_s8/迭代）只是小头，大头是两条
     独立 SDOT 链消除了 serial dependency 的流水线停顿。
  3. **预计算的内存开销可控**：7.72M groups × 8B = ~62MB，相对模型 823MB
     不到 8%。首 token 有一次性 precompute 成本（~25ms），后续 token 零开销。
- **复现**：
  `./build/runtime/tinyqwen --model model_i4_hqq_lmh.tqwen --tokens 105538,59975,100132 --max-new-tokens 32 --matvec-impl sdot2_mt --ops-impl neon`

### backend_refactor（2026-08-20）

- **优化栈**：fp32-baseline + IBackend/CPUBackend 抽象（**纯结构重构，非性能优化**，
  按纪律留档占位）
- **是什么**：抽出 `IBackend` 接口（`runtime/backend.h`），QwenModel 所有算子改走
  后端虚函数；`qwen_model.cpp` 拆为 create / forward_token / forward_prefill
  三个文件；新增 `WeightTensor` 量化抽象。计算路径本身一行未改。
- **假设**：无提速预期，目标是**零开销**。decode 是权重带宽瓶颈，每 matvec 多一层
  虚函数间接调用是 ns 级，相对 ms 级 kernel 应完全淹没在噪声里。
- **结果**：TTFT **577.57 ms**（prefill 3 tok）；TOPT 中位 **234.96 ms/token**（3 遍取中位；decode 共 32 tok，丢预热，稳态样本 27），p95 263.50；forward 总耗时 7965.0 ms
- **vs 上一配置**：0.95×（vs fp32-baseline 222.59），落在已知 ~4% 运行波动内
  （参照 fp32-double 复测条目的同配置波动）——判定为无性能变化
- **基线参照**：fp32-baseline（222.59 ms/tok @ 40b8e26），本次 vs 基线 = 0.95×
- **验证**：scripts/verify.sh 全过（含新增 test_backend / test_quantization，共 110 测试）；
  fp32 greedy 输出与重构前一致
- **瓶颈转移**：op 结构不变，top op 仍是 `lm_head`、`layer_4.gate_up_proj`、
  `layer_3.gate_up_proj`；重构不开新优化方向，下一刀仍按 i4/端侧路线走
- **意外 / 教训**：实测确认"虚函数抽象在带宽瓶颈路径上零成本"——此后新增后端
  （CUDA/Metal）不需要为调用开销做特殊设计。另注意本条 commit 同时携带了
  i4 sdot/Android 工具链等无关改动，bench 用的是 fp32 ref 路径，不受影响。
- **复现**：`./scripts/bench.sh backend_refactor`

---

### qwen35-4b-i4-sdot2_mt（2026-08-23，macOS M4，4B 首测——非优化，建档）

- **背景**：4B 模型刚接入（`ea3aba3`：导出器提速 + `f2d5c46`：sdot in_dim>8192
  截断修复），此前无任何 4B 性能数字。本条为 M4 上的基线建档。
- **机器**：Apple M4（Mac16,10，4P+6E，16GB）。DRAM 墙：峰值 114 / P 簇 74 /
  持续 74.6 GB/s（`benchmarks/machine_ceiling/m4_ceiling.md`）。
- **模型**：Qwen3.5-4B i4（HQQ@64，lm_head 独立 i4 副本）。每 token 权重流量
  = i4 2365MB；**但 sdot2 另有预计算 scale/zero f32 缓存 = +525MB/token
  （+22%）+ 525MB 常驻内存**（65.7M 组 × 8B）。
- **结果**（冷态，bench 标准负载，3 轮）：
  - decode 中位 **50.30 / 50.59 / 51.64 ms/tok**（三个会话各自 3 遍中位），
    p95 ~52–64。有效带宽 ~57 GB/s（含缓存流量）——远低于墙，**不是带宽瓶颈**。
  - 线程扫描（关键诊断）：4 线程（纯 P 簇）**45.93** < 10 线程 50.31 <
    6 线程 60.43（最差）。静态等分行切分下 E 簇拿同样多的行变拖尾，
    P 簇自旋等待——加核反而变慢，证伪"多线程=更快"的朴素假设。
  - TTFT **456 ms**——其中 ~276ms 是 sdot2 懒预计算（首次见到权重指针时
    unpack 65.7M 组）。Qwen3.5 的 prefill 走逐 token 回退（GDN 无批量
    prefill），matvec 在 prefill 即被调用，预计算开销记在 TTFT 上。
- **瓶颈定位**：单线程仅 ~15 GB/s（单核流式能力 68.8），10 核聚合 ~57——
  kernel 是**单核效率 + 调度**问题，不是墙。下一刀：sdot3。
- **复现**：`MODEL=model_qwen35_4b_i4.tqwen ./scripts/bench.sh qwen35-4b-i4-sdot2_mt --extra-args "--matvec-impl sdot2_mt --ops-impl neon"`

---

### qwen35-4b-i4-sdot3_mt（2026-08-23，macOS M4，work-stealing + 内联组头 + 128 位解包）

- **优化栈**：qwen35-4b-i4-sdot2_mt（HQQ@64 + lm_head i4 + W4A8 SDOT + 预计算 + 2-row）
  的内核替换为 `kernels/matvec/matvec_i4_sdot3.cpp`（新变体，sdot2 原样保留供 A/B）。
- **是什么**：三项叠加（数值全部逐位不变）：
  ① **动态行调度（work-stealing）**：原子行计数器按需取块（块 = 总行/4p，
  下限 16），替代 sdot2 的静态等分。快核多做、慢核少做，消除异构核拖尾。
  ② **砍掉预计算 f32 缓存**：改回内联读 4B 组头（与 packed 同一缓存行，
  零额外流量），fp16→fp32 用硬件 FCVT（`+fp16` 编译，1~2 指令）替代
  sdot v1 的软件 half_to_float（当初 34.7% 指令开销的元凶）。
  −525MB/token 流量（−18%）、−525MB 常驻内存、无首 token 预计算。
  ③ **128 位解包**：`vld1q_u8` 16B=32 权重，`vandq/vshrq/vzipq` 重排 + 2×SDOT，
  解包指令 ~16→~10 / 32 权重（−37%）。
- **假设**：① 消拖尾让 10 核可用（目标逼近整机墙 114）；② 减 18% 流量；
  ③ 减指令提升单核效率。三者都指向"有效带宽 57 GB/s → 更高"。
- **结果**（M4，冷态，同场交替 3 轮 A/B，每轮 27 稳态样本）：
  - 4B decode：sdot2_mt [49.73, 50.64, 50.30] 中位 **50.30**；
    sdot3_mt [46.73, 47.88, 46.65] 中位 **46.73** → **1.076×**。
    另两个独立会话：50.59 vs 46.84（1.080×）、51.64 vs 47.04（1.098×）。
    三会话一致 **≈1.08×**。
  - **TTFT：456.30 → 180.21 ms（2.53×）**——懒预计算消失的直接兑现。
  - p95：~63.9 → ~48.6（拖尾消除，方差同步收窄）。
  - 0.8B 同场（2 轮）：10.74 → 10.53 = **1.02×**（模型小，缓存流量占比低，收益温和）。
  - sdot3 线程扫描：4 核 49.98 / 8 核 43.58 / 10 核 46.56（热态单轮，未冷态复核）。
    10 核已无拖尾惩罚（≈ sdot2 的 4P 最优值 45.9），但 E 簇净增益仍有限。
- **验证**：
  - 117 单测全过（新增 7 个：sdot3/sdot3_mt × g64/g128/尾部组/g48 边界
    + **sdot3 vs sdot2 逐位一致测试**（容差 0））。
  - 端到端：4B 与 0.8B 上 sdot3_mt vs sdot2_mt 贪心 32 token **逐位一致**
    （整数点积与累加顺序无关；fp16→fp32 无损；浮点还原同序）。
- **瓶颈转移**：有效带宽 57 → ~50.5 GB/s（流量降了，时间也降了，但仍远低于
  P 簇墙 74）。单核 ~15 GB/s vs 流式能力 68.8——**单核效率仍是主限**，
  不是带宽也不是调度。下一刀候选：
  ① 4-row 内循环（更多独立 SDOT 链 / 摊薄组头开销）；
  ② i4 的 gate_up/qkv 融合（dispatch 已有 pair/qkv 入口，无 i4 实现，
  现全部兜底为单发——每发一次付一遍激活量化 + fork-join）；
  ③ 激活量化 SIMD 化（amax/quant/prefix 三遍标量，串行在 master）。
- **意外 / 教训**：
  1. **prefetch 证伪**：内循环加 `__builtin_prefetch`（前 6 组）实测中性
     （47.04 vs 46.84，噪声内）——限制不在加载延迟暴露，已回退删除。
  2. **work-stealing 没赚到 E 核**：消了拖尾（10 核不再比 4 核慢），
     但也没让 10 核快过 4 核多少——E 簇每字节吞吐太低，值得做的
     前提是先降低单核指令数（与下一刀 ①③ 同源）。
  3. **零值不是整数**：曾想把预计算缓存压成 fp16（流量再减半），
     但导出器 `zero = -vmin/scale` 是 fp16 实数（非整数零点），
     `zero−8` 存 fp16 有损 → 放弃，改走内联读（反而更省：零额外流量）。
  4. **RSS 实测不可靠**：`/usr/bin/time -l` 下 sdot2 的 ru_maxrss 反而比
     sdot3 低（4.73 vs 4.91GB，均低于模型文件 4.91GB）——macOS 页回收/
     压缩干扰。内存节省只以代码事实（不再分配 525MB）记录，不以实测断言。
- **复现**：`MODEL=model_qwen35_4b_i4.tqwen ./scripts/bench.sh qwen35-4b-i4-sdot3_mt --extra-args "--matvec-impl sdot3_mt --ops-impl neon"`

> **后续更正（2026-08-23，见下一条）**：本节声称"硬件 FCVT 替代软件
> half_to_float"，实际**未生效**——守卫宏 `__ARM_FEATURE_FP16` 在 clang
> 下不存在（+fp16 定义的是 `__ARM_FEATURE_FP16_SCALAR_ARITHMETIC`），
> 热循环一直跑软件转换。sdot3 的实测数字本身有效（收益来自调度 +
> 流量削减 + 128 位解包），但"硬件转换"一项由 sdot4 才真正兑现。

---

### qwen35-4b-i4-sdot4_mt（2026-08-23，macOS M4，修正组头转换守卫——硬件 _Float16 FCVT）

- **优化栈**：qwen35-4b-i4-sdot3_mt（work-stealing + 内联组头 + 128 位解包）
  内核替换为 `kernels/matvec/matvec_i4_sdot4.cpp`（sdot3 原样保留供 A/B）。
- **是什么**：**唯一改动 = 修正 `half_bits_to_float` 的特性守卫**。
  sdot3 写 `#if defined(__ARM_FEATURE_FP16)`——clang `-march=...+fp16`
  从不定义这个宏（定义的是 `__ARM_FEATURE_FP16_SCALAR_ARITHMETIC` /
  `_VECTOR_ARITHMETIC`），守卫永远为假，**静默回退软件版转换**
  （分支 + `clz` 循环，每次 ~15-20 条指令）。sdot4 改用
  `__ARM_FEATURE_FP16_SCALAR_ARITHMETIC` 守卫 + `_Float16` 标量 cast
  （编译为 `fmov s0,w0` + `fcvt s0,h0` 两条指令）。
- **假设来源（先诊断后动刀）**：单线程 4B decode 用 `sample` 指令级采样，
  97% 时间在 `dot_2rows` 内，热点却散布在**软件转换的分支指令**上
  （+400~+760 偏移处成片 `clz/ubfx/cmp #0x1f/b.eq`），而非预想的解包/
  归约。反汇编对照确认。每组 4 次转换（2 行 × scale+zero）× 65.7M
  组/token ≈ **4.6G 条多余指令**——单核仅 ~15 GB/s（流式能力 68.8）的
  第一解释。
- **结果**（M4，冷态，同场交替 3 轮，每轮 27 稳态样本）：
  - 4B decode：sdot3_mt [46.86, 46.91, 48.63] 中位 **46.91**；
    sdot4_mt [36.50, 36.44, 36.65] 中位 **36.50** → **1.285×**
    （样本极稳：全距 0.21ms）。p95：~48-98 → ~38。
  - **单核**：160.2 → 118.6 ms/tok = **1.35×**——验证"单核效率"诊断。
  - **有效带宽**：50.5 → **64.8 GB/s**（4B 权重 2365MB/36.5ms）。
  - **E 核翻案**：sdot3 时代 4P 最优（45.9 < 10 线程 50.3）；sdot4 后
    10 线程 **36.0** < 4 线程 39.6 ≈ 8 线程 39.0——单核效率上来后
    work-stealing 让 E 簇真正贡献带宽（整机墙 114 开始有感）。
  - 0.8B 同场（2 轮）：10.19 → **8.08** = **1.26×**（0.8B i4 vs f16 满栈
    17.87 → **2.21×**）。
  - TTFT 无变化（~170-180）：~~prefill 走 matmul_i4_ref~~ **更正**：Qwen3.5
    prefill 实际逐 token 回退走 forward_token（含 lm_head），当时两次测量
    都含完整 lm_head 且热态相近故无差；此误判由下一条（prefill_skip_logits）
    的 profile 拆解纠正。
  - 4B vs sdot2_mt 基线（50.30）累计 **1.38×**。
- **验证**：
  - 124 单测全过（新增 7 个：sdot4/sdot4_mt × g64/g128/尾部组/g48 边界
    + **sdot4 vs sdot3 逐位一致测试**）。
  - 反汇编确认：`dot_2rows_i4_sdot4` 热路径 63 条 `fcvt`、**0 条**软件
    转换特征指令（`clz`）；sdot3 同位置是成片软件转换。
  - 4B 端到端贪心 32 token：sdot4_mt vs sdot3_mt 逐位一致
    （两种转换都是无损的 fp16→fp32）。
- **瓶颈转移**：有效带宽 64.8 GB/s——已越过 P 簇墙（74）的 87%，
  开始进入"全机带宽题"区间；剩余单核空间（118.6ms ≈ 20 GB/s/核 vs
  能力 68.8）指向下一刀仍是内核效率：**4-row 内循环 / 激活量化 SIMD /
  i4 gate_up·qkv 融合**；或转攻 prefill（matmul_i4 目前还是 ref 实现！
  4B TTFT 175ms 里它占大头）。
- **意外 / 教训**：
  1. **守卫宏要验尸，不要想当然**：`__ARM_FEATURE_FP16` 听起来天经地义，
     实际不存在；一条 `clang -dM -E | grep` 就能避免一轮无效优化。
     凡是"条件编译切换快慢路径"，都要反汇编确认快路径真的被选中。
  2. **sample 指令级采样是端侧 kernel 的一级工具**：本轮若按原计划直接
     做 4-row，最多赚 ~10%；profile 把 1.35× 的单核空间直接指了出来。
     方法论已补进 `docs/optimization.md` 的候选（见该文件更新）。
  3. 单核效率与调度交互：同一套 work-stealing，sdot3 时 E 核为负、
     sdot4 时为正——**线程策略的最优解依赖内核效率**，调优顺序应该是
     先单核、后并行。
- **复现**：`MODEL=model_qwen35_4b_i4.tqwen ./scripts/bench.sh qwen35-4b-i4-sdot4_mt --extra-args "--matvec-impl sdot4_mt --ops-impl neon"`

---

### prefill_skip_logits（2026-08-24，macOS M4，TTFT 专项——非末位 prefill 跳过 logits）

- **优化栈**：qwen35-4b-i4-sdot4_mt + `forward_token(need_logits)` 门控。
- **是什么**：Qwen3.5 prefill 走逐 token 回退，**每个** token 都算 final_norm
  + lm_head + argmax，但非末位 token 的 logits 会被丢弃（下一个 token 只由
  末位 logits 决定）。给 `forward_token` 加 `need_logits` 参数（默认 `true`），
  回退循环对非末位传 `false`，整段跳过；末位与 decode 不受影响。
  KV cache / GDN state 照常更新，返回 -1 仅被中间迭代覆盖。
- **假设**：省掉一次 lm_head（4B 357.6MB i4 ≈ 单 token 流量 16%、0.8B 34%）
  + final_norm + argmax。prefill 是逐 token 带宽题，省一份权重搬运就快一份。
- **结果**（M4，冷态）：
  - **33-token 真实中文 prompt**（梯度下降主题，tokenize 自 Qwen3.5-4B 分词器）：
    OLD 1317.2/1322.9（中位 **1320**）→ NEW 1144.9/1170.4（中位 **1158**）
    = **1.14×**。与理论吻合：33 个里 32 个跳过、每个省 ~16% → 33/(33−32×0.16)≈1.135。
  - **微观证据**（单次 profile 内拆每个 prefill token）：
    跳过位 tok1=29.96ms vs 末位 tok2=35.77ms，Δ=**5.81ms ≈ lm_head 实测 5.87ms**——
    省的就是 lm_head。OLD（commit 8d5447f）同拆：tok1=33.45≈tok2=35.93（未跳过）。
  - **0.8B**：TTFT 21.4/22.0ms（未跳过理论 ~24-25），TOPT 7.91-8.44 无回归。
  - **decode TOPT 不变**（本刀只动 prefill）：4B 35.7-36.5 vs 基线 36.5；逐位一致。
  - canonical 3-token TTFT 绝对值受热噪声污染（冷 107.6，热态漂到 ~200），
    故以"长 prompt + 同 profile 内逐 token 拆分"为权威证据。
- **验证**：
  - 124 单测全过；4B 端到端贪心 32 token 与优化前**逐位一致**。
  - **对齐契约**：`tools/align_fake_qwen35_model.py` 通过（worst 1.16e-6 < 1e-5）——
    它走 `--verbose` 逐位置 `--dump-logits`，由 main 直接调 `forward_token`
    （need_logits 恒 true），不受回退循环门控影响。
- **瓶颈转移**：Qwen3.5 prefill 仍是逐 token 串行（GDN 状态需顺序更新）。
  真正的下一刀是 **GDN 架构的批量 prefill**：投影/FFN 部分 GEMM 化（权重
  流量从"每 token 一遍"摊薄为"整批一遍"），仅 GDN 递归态保留顺序扫描——
  长 prompt 预期量级级改善。
- **意外 / 教训**：
  1. **bench 的 TTFT 是热污染重灾区**：3-token canonical 冷态 107.6，热态能漂
     到 ~200（175/3≈58ms/tok≈热态 decode 速度）。TTFT 比 TOPT 更吃测量时的
     热状态；小幅 TTFT 优化要用长 prompt + 同 profile 逐 token 拆分来证明。
  2. **先 profile 再下结论**：上一轮日志曾断言"prefill 走 matmul_i4_ref"，
     实际 Qwen3.5 逐 token 回退、根本不走 matmul。本条已一并更正该误判。
- **复现**：长 prompt 用 `tools/tokenize_prompt.py --model models/Qwen3.5-4B`
  生成，`--tokens-json` 喂给 runtime；对比 `--profile-out` 的 `first_token_ms`。

---

<!-- 模板：复制下面这段，填好后追加。注意优化栈 = 上一配置 + 本次优化。 -->
<!--
### <优化名>（<日期>）

- **优化栈**：<基线 + 已叠加的全部优化，如 fp32-baseline + NEON>
- **是什么**：<本次改了哪个 kernel / 数据结构 / 调度，一两句话>
- **假设**：<为什么预期会快，落在带宽/计算/并行/指令哪一类>
- **结果**：decode 中位 __ ms/token（样本 __），p95 __
- **vs 原始基线**：__×        ← 总共快了多少（用户视角）
- **vs 上一配置**：__×        ← 本次优化贡献了多少（归因视角）
- **交互**：<若叠加，实测组合加速比 vs 各优化乘积，交互系数 __，是否撞新瓶颈>
- **验证**：<和 reference 的对齐误差，确认没算错>
- **瓶颈转移**：<重新 profile 后 top op 变成了谁，下一刀砍哪>
- **意外 / 教训**：<如果和预期不符，分析为什么——这往往是最值钱的部分>
- **复现**：`./scripts/bench.sh <优化名>`
-->
