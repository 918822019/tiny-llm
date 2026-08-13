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
