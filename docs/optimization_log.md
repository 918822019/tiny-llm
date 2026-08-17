# 优化日志

> 每一次性能优化都在这里留一条记录：**改了什么、提升了多少、为什么提升**。
> 方法论见 `docs/optimization.md`（优化手册）。所有数字来自 `scripts/bench.sh` 的标准负载
> （固定 prompt、decode 32 token、丢弃预热 4、取稳态中位数），跨版本可比。

## 汇总表

主指标 = **decode 延迟中位数（ms/token）**，越小越好。

每行是一个**优化栈**（基线 + 已叠加的优化）。记**两个**加速比：
**vs 原始基线**（总共快多少）+ **vs 上一配置**（这一步贡献多少）。
多个优化叠加时的交互分析见 `docs/optimization.md` 第 7 节。

| 配置（优化栈）               | commit  | decode 中位 ms/tok |    p95 | vs 原始基线 | vs 上一配置 | 交互 / 归因              |
|-----------------------|---------|-----------------:|-------:|--------:|--------:|----------------------|
| fp32-baseline（标量，无优化） | 5f679ea |           230.36 | 252.35 |   1.00× |       — | fp32 标量、单线程、未量化，只求正确 |
| fp32-double（对照：同配置复测） | 40b8e26 |           222.59 | 241.87 | ≈1.0×（波动） |       — | 与基线同配置、换 commit 复测；当 float 的对照，顺带暴露 ~4% 运行波动 |
| fp32-float（基线 + matvec float 累加，**已回退**） | 40b8e26 |           204.12 | 269.79 |   1.13× |   1.09× | matvec 内层累加 double→float（Apple Silicon fp64 慢），计算类优化 |
| restore-double（回退 fp32-float） | e1523c7 | 239.64 | 269.20 | 0.93× | 0.85× | 纪律性回退：_ref 恢复 double 累加，回到基线配置；float 累加将来以变体形式重做 |
| double_2_float（fp32-baseline + matvec float 累加变体） | 9d1a572 | 219.84 | 245.48 | 1.01×（被机器波动掩盖） | 1.12× | matvec 内层累加 double→float，首次以**变体**形式合规落地；同场 A/B 才是真贡献 |
| neon | fd430db | 28.25 | 30.14 | 7.88× | 8.05×（同场） | matvec NEON 向量化（4 路 FMA + 4 累加器展开）；decode 纯权重带宽瓶颈，有效带宽 8.7→72 GB/s，kernel 加速比几乎全额传导到端到端 |
| acc4（阶梯 L2：隔离累加结构） | 2dfb2c8 | 89.32 | 99.92 | 2.49× | 2.65×（同场） | 标量 4 链并行累加，隐藏乘加延迟；阶梯口径 vs double_2_float = 2.41× |
| neon_nofma（阶梯 L3：隔离 SIMD 宽度） | 2dfb2c8 | 28.62 | 29.03 | 7.78× | 7.97×（同场） | NEON 向量化但故意不用 FMA；SIMD 是最大单项（vs acc4 = 3.02×），并撞带宽墙 |
| neon_mt | 56f51ba | 10.50 | 11.26 | 21.19× | 2.69×（vs neon） | NEON + 常驻线程池行切分（默认 6 线程）：单核带宽 72 GB/s 打满后，多核接力整机带宽 ~199 GB/s；同场 vs ref 21.42× |
| neon_mt_bal | a323eb7 | 10.07 | 14.42 | 22.11× | ≈1.0×（vs neon_mt，同场） | 自校准加权分块——带宽墙下无 E 核尾巴可消，证伪归档 |
| neon_mt_kv | d17bc57 | 11.19 | 12.19 | 19.89× | 1.02–1.05×（vs neon_mt，同场 4 块 24 样本） | k/v 成对融合：48 次内联小 matvec 合并成 24 次 fork-join，摊薄+并行 |
| neon_mt_kv_nt | f4beeb7 | 10.06 | 11.13 | 22.12× | 1.03–1.04×（vs neon_mt_kv，同场） | 权重 LDNP 流式加载（内联汇编 + 对齐兜底）：预期≈0 被证伪，提示真有效且尾部更稳 |
| f16_neon_mt_kv_nt | c8fba49 | 8.00 | 9.19 | 27.81× | 1.75×（同场 vs fp32 最佳栈） | 权重 f16 量化（流量减半）+ fp32 满栈移植 + 线程甜蜜点 6→8；非 matvec 固定开销 Amdahl 稀释，未到 2× |
| f16 + qkv/gate_up 融合（**fp16 路线终点**） | 本次 | 6.37 | 6.77 | 36.2× | ≈1.1×（同场 A/B 乘积；跨窗绝对值受噪声污染） | qkv 三路融合 + gate/up 成对融合：fork-join 96→48 次/token。此后 fp16 路线正式关闭（见文末关闭小节） |
| ops_neon | 2cdbdd0 | 6.35 | 6.87 | 35.08× | 1.05×（同场） | 非 matvec 五算子 NEON 化 + ops dispatch：argmax 10×/attention 2.3×/swiglu 3.1×/rmsnorm 2.5×，profile 合计省 ~0.48ms；非 matvec 开销 694→200µs |
| cuda（⚠️ 换机器：x86_64 + A10） | 7fc4a2d | 333.75 | 356.47 | N/A（跨机器，基线是 M 系芯片） | 1.75×（同场 vs 本机 ref） | matvec CUDA 参考变体（每次调用重传权重）：打通 CUDA 路径，朴素形态如实记录 |
| cuda_resident（x86_64 + A10） | dd58fd2 | 47.40 | 48.07 | N/A（跨机器，基线是 M 系芯片） | 12.30×（同场 vs 本机 ref）；7.04× vs 朴素 cuda | matvec 权重常驻显存（按 host 指针缓存 device 副本），砍掉每 token ~2GB 的 PCIe 重传；GPU 从此反超本机最好 CPU 变体 |
| cuda_resident_ws（x86_64 + A10） | aa30aa5 | 46.05 | 46.52 | N/A（跨机器，基线是 M 系芯片） | 1.03×（vs cuda_resident，≈无效）；12.67×（同场 vs 本机 ref） | x/y 常驻 workspace（删每调用 alloc/free）——证伪归档：CUDA 池化小分配，瓶颈实为 kernel 非合并访存 |
| cuda_resident_coal（x86_64 + A10） | 7356481 | 9.16 | 9.81 | N/A（跨机器，基线是 M 系芯片） | 5.17×（vs cuda_resident）；63.94×（同场 vs 本机 ref） | matvec kernel 改合并访存（block-per-row + warp 归约），warp 读权重从 32 条分散 cache line → 1 条满线；CUDA 路线首次逼近带宽地板 |
| f16_cuda_resident_coal（x86_64 + A10） | 3408267 | 7.68 | 8.14 | N/A（跨机器，基线是 M 系芯片） | 1.18×（vs f32_cuda_coal，同场） | 权重 fp16（流量减半）：理论 2× 被 Amdahl 稀释成 1.18×——时间大头已是固定开销而非带宽，诊断价值大于提速价值 |
| cuda_resident_coal_ws（x86_64 + A10） | 716d63c | 7.97 | 8.34 | N/A（跨机器，基线是 M 系芯片） | 1.15×（vs cuda_resident_coal，同场） | 砍每调用开销：x/y 常驻 workspace（删 680 次 alloc/free）+ 删冗余 cudaDeviceSynchronize（同步 D2H 已保证完成）；kernel 未动 |
| f16_cuda_resident_coal_ws（x86_64 + A10） | cf5c805 | 6.54 | 7.01 | N/A（跨机器，基线是 M 系芯片） | 1.19×（vs f16_cuda_coal，同场） | f16 满栈 = f16_cuda_coal + x/y workspace + 删冗余 sync；A10 追平 Mac fp16 满栈量级（6.54 vs 6.35，跨机器仅参照） |
| f16_cuda_fused（x86_64 + A10） | 47c5335 | 5.79 | 6.28 | N/A（跨机器，基线是 M 系芯片） | 1.07×（vs 部分融合，同场）；1.13×（vs 真不融合 6.54） | qkv 三合一 + gate_up 二合一 CUDA 融合（减 matvec 调用次数砍启动/拷贝），**5.79 压过 Mac fp16 满栈 6.35**（跨机器仅参照，目标达成） |
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
- **vs 上一配置**：**8.05×（同场 A/B）**——对照（无额外参数，当前即 ref）中位 227.28（p95 236.63）→ 变体 28.25，同 binary 同场交错测量。（vs 基线 7.88×）
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
- **vs 上一配置**：**2.65×（同场 A/B）**——对照（无额外参数，当前即 ref）中位 237.08（p95 256.26）→ 变体 89.32，同 binary 同场交错测量。（vs 基线 2.49×）
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
- **vs 上一配置**：**7.97×（同场 A/B）**——对照（无额外参数，当前即 ref）中位 228.00（p95 240.01）→ 变体 28.62，同 binary 同场交错测量。（vs 基线 7.78×）
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
- **复现**：`MODEL=model_f16.tqwen ./scripts/bench.sh f16_fusion --extra-args "--matvec-impl neon_mt_kv_nt"`（对照加 `--no-fuse-gate-up --no-fuse-qkv`）

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
- **复现**：`MODEL=model_f16.tqwen ./scripts/bench.sh ops_neon --extra-args "--matvec-impl neon_mt_kv_nt --ops-impl neon"`（对照加 `--ops-impl ref`）

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
- **复现**：`MODEL=model_f16.tqwen ./scripts/bench.sh f16_cuda_resident_coal --extra-args "--matvec-impl cuda_resident_coal"`

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
- **复现**：`MODEL=model_f16.tqwen ./scripts/bench.sh f16_cuda_resident_coal_ws --extra-args "--matvec-impl cuda_resident_coal_ws"`

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
