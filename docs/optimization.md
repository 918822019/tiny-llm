# 优化手册（kernel 怎么加、性能怎么测）

> 本项目优化工作的一站式文档：**参考实现放哪、改进版往哪加、怎么证明算对了、
> 怎么科学测速、怎么记录**。核心信条：没有测量就没有优化——数字才算数。
>
> 优化历史与当前数字见 `optimization_log.md`；测量产物的字段含义见
> `profiling_schema.md`。

---

## 1. 原则与术语

三条铁律：

- **`_ref` 是标准答案**。任何优化版都要先和它对齐（误差在容差内）才算"算对了"，
  然后才谈"快不快"；它也是出问题时的一键兜底。
- **优化版只增不删、绝不修改 `_ref`**。改 ref 本体 = 标准答案自己变了，
  后续所有对齐失去锚点（fp32-float 曾因此被回退，见优化日志）。
- **每个算子一个文件夹**，参考实现和所有优化版都放里面——加新 kernel
  不做结构决策，放进对应文件夹即可。

术语约定（全仓库统一，避免同物异名）：

| 术语 | 同义词（旧文档可能出现） | 含义 |
|---|---|---|
| 参考实现 | `_ref` / base / 标准答案 / reference | 正确性基准 + 兜底，永不改 |
| 变体 | variant / 优化版 / 改进版 / 改进位 | 优化实现，只增不删 |
| 基线 | baseline / fp32-baseline | `benchmarks/baseline.json` 记录的参照配置 |

## 2. 目录与命名约定

```text
kernels/
├── ref_ops.h                     # 所有 kernel 的签名契约（共用）
├── dispatch.h / dispatch.cpp     # 分发层 + 注册表：model 只调通用入口；
│                                 # 变体用 TINYQWEN_MATVEC_VARIANT 宏自注册（共用）
└── <op>/                         # 每个算子一个文件夹
    ├── <op>_<dtype>_ref.cpp      # 参考实现（base）：永不删、永不覆盖
    └── <op>_<dtype>_<variant>.cpp  # 优化版（改进位）：只增不删，末尾一行自注册
```

命名 `<算子>_<数据类型>_<优化手段>`，可叠加：

| 文件名                  | 含义                      |
|----------------------|-------------------------|
| `matvec_f32_ref`     | fp32 参考实现（base）         |
| `matvec_f32_neon`    | fp32 + NEON SIMD        |
| `matvec_f32_neon_mt` | fp32 + NEON + 多线程       |
| `matvec_i8_neon`     | INT8 weight-only + NEON |
| `rmsnorm_neon`       | RMSNorm 的 NEON 版        |

## 3. dispatch 自注册：model 不感知具体实现

model（`qwen_model.cpp`）只调**通用入口** `matvec_f32(...)`，不直接调
`matvec_f32_ref`。实现按名字注册进 dispatch 的注册表，运行时查表调用：

```text
qwen_model.cpp ──> matvec_f32()  ──dispatch──> matvec_f32_ref()            (默认)
                                             └──> matvec_f32_double_2_float() / 将来的 neon 等
```

好处：

- **加优化不用动 model，也不用动 dispatch/main/conf**——只加实现文件
  （含一行自注册宏）+ CMake 一行；
- **可 A/B**：同一负载切换实现直接对比（`record.sh --extra-args` 内置同场 A/B）；
- **可兜底**：未知实现名报错并列出所有可用；切回 `ref` 永远有正确结果。

选择方式优先级 **CLI 开关 > 配置文件 > 默认**：

- CLI：`--matvec-impl double_2_float`（benchmark 时随手切换）；
- 配置文件：`tinyqwen.conf` 里写 `matvec_impl = ref`（用 `--config` 指定）。

**为什么 kernels/runtime 必须是 OBJECT 库**（改 CMake 前先读这段）：
变体靠文件里的静态注册器接入 dispatch。STATIC 库归档是"被引用才提取"，
没人引用的变体 .o（注册器所在）会被链接器丢弃，注册悄悄丢失——而且
症状隐蔽（兜底到 ref，看起来"能跑但没生效"）。OBJECT 库配合可执行
"显式直链"（见 `runtime/CMakeLists.txt` 注释）把所有 .o 直接放上链接行，
注册器必然生效。

## 4. 加一个新变体：标准流程

以"给 matvec 加 NEON 版"为例：

1. **写实现 + 注册**：新建 `kernels/matvec/matvec_f32_neon.cpp`，函数签名
   和 `_ref` 完全一致；`#include "dispatch.h"`，在文件末尾（namespace 内）加一行：
   `TINYQWEN_MATVEC_VARIANT(matvec_f32_neon, "neon");`
2. **CMake**：把新 `.cpp` 加进 `kernels/CMakeLists.txt`。
   到此 `--matvec-impl neon` / `tinyqwen.conf` 即可选用。
3. **正确性门禁**（第 5 节）：单测对齐 ref——**先证明算对了**。
4. **测速 + 记录**（第 6、8 节）：
   `./scripts/record.sh neon --extra-args "--matvec-impl neon"`，
   人工补全日志归因后 `commit_opt.sh` 提交。

## 5. 正确性门禁（不可跳过）

优化版在宣称"快"之前，必须先过对齐：

```text
对同一组输入：max | optimized(x) - ref(x) | < 容差
```

- fp32 之间的优化（如 NEON）：容差可紧些（~1e-4），因为只是计算顺序不同；
- 量化（INT8/INT4）：容差放宽，但要在 `dump_qwen_reference` / 真模型上
  确认最终 logits top-k 不变。

对齐工具：`tests/`（小 shape 单测）+ `tools/align_fake_model.py`（整模型）。
整机层面由 `scripts/verify.sh` 把关：build + 单测 + golden token 对照
（canonical prompt 的 16 个生成 token 必须与参考实现逐位一致）。

测试里切换实现必须断言成功：`EXPECT_TRUE(set_matvec_impl_by_name("neon"))`
——set 失败时 dispatch 会静默兜底到 ref，不断言就是假通过。

## 6. 测量方法论

### 6.1 固定负载：改负载 = 换尺子

比较两次性能必须用**完全相同**的输入。`tools/bench.py` 把负载写死了：

| 项          | 固定值                                 | 为什么                          |
|------------|-------------------------------------|------------------------------|
| prompt     | `[105538, 59975, 100132]`（"中国的首都是"） | 写死 token id，连 tokenizer 都不依赖 |
| decode 数量  | 32 个                                | 样本量足够算统计                     |
| 预热丢弃       | 前 4 个                               | 冷启动有抖动，丢掉才是稳态                |
| batch / 采样 | batch=1、greedy                      | 排除随机性                        |

### 6.2 中位数为主指标

单次测量受 CPU 频率波动、后台进程、温度降频影响。

- **中位数（median）**：抗离群点，代表"典型情况"，主指标。
- **p95**：尾部有多差。**注意 p95 对单次 bench 的噪声很敏感**，
  判断回退与否以中位数为准。
- **min**：接近硬件理论上限。

`record_optimization.py` 每遍取 27 个稳态 token 的中位数，再对多遍取
"中位数的中位数"抗单次波动。

### 6.3 同场 A/B：小幅优化的唯一可信判据

**机器状态会漂移**（实测同配置跨天差 ~4–10%），所以：

- **vs 历史基线**（baseline.json）只是"用户视角"参考，会被漂移掩盖；
- **vs 同场对照**才是真贡献：`record.sh` 在 `--extra-args` 非空时自动
  升级 A/B 模式——每遍先测对照（同 binary、无额外参数，当前即 ref）
  再测变体，交错进行抗慢漂移，日志自动填 A/B 比值；
- 日志同时记**两个**加速比：vs 原始基线（总共快多少）+ vs 上一配置
  （这一步贡献多少）。叠加优化不能简单相乘（第 7 节）。

配套机制：

- **基线漂移警告**：参照中位偏离 baseline.json >5% 时自动提示
  ——此时考虑 `scripts/set_baseline.sh` 重建基线；
- **`record.sh --skip-verify`**：刚跑过门禁、快速迭代调参时跳过重复验证
  （默认每次都跑，别滥用）。

### 6.4 测量纪律：一次只改一个变量

想验证"INT8 量化让 matvec 变快"，就**只**改量化这一件事。同时改量化、
线程数、布局，测出变快也不知道是谁的功劳——**无法归因，等于白做**。

### 6.5 数字可追溯到代码版本

1. **每个优化单独一个 commit**（`commit_opt.sh` 把代码 + 日志放一个 commit）；
2. bench.py 自动在日志记录 git commit；
3. 里程碑打 tag（如 `v0.1-fp32-baseline`）。

### 6.6 解释"为什么提升"：归因四分类

| 类别   | 例子                                  |
|------|-------------------------------------|
| 减少搬运 | INT4 量化让权重体积 /8，decode 是带宽瓶颈，少搬数据就快 |
| 减少计算 | double 累加改 float（fp64 慢指令）、在线 softmax |
| 并行   | 多线程把 matvec 切到多个核                   |
| 指令效率 | NEON 一条指令算 4 个 float                |

找原因看 bench 输出的 **top op**：fp32 基线里 `lm_head` 和 `down_proj` 是大头，
说明 matvec 是主攻方向。top op 是指南针——每加一个优化重新 profile，
瓶颈会转移（Amdahl 的另一面）。

## 7. 进阶：多个优化叠加时怎么评测

### 7.1 最大的坑：加速比不能直接相乘

"NEON 2× × 多线程 4× = 8×"实测往往达不到，因为多个优化可能抢同一资源。

```text
交互系数 = 实测的组合加速比 / (各优化单独加速比的乘积)
  > 1   → 协同（少见）    ≈ 1   → 独立（理想）    < 1   → 竞争（很常见！）
```

例子：NEON 让每线程算得更快 → 更快撞到**内存带宽**墙 → 再加线程也搬不进
更多数据 → 多线程收益打折。**组合收益必须实测，不能拍脑袋相乘。**

### 7.2 用"优化栈"记录

每个配置是"基线 + 若干优化"组成的栈。只测增量序列：
基线 → +A → +A+B → +A+B+C，每步记 vs-上一配置（这正是最终的真实叠加顺序）。
只有**怀疑某两个优化打架**时才专门测那一对算交互系数。
不要为"全面"去测所有 2^N 组合。

### 7.3 每个配置可独立复现

- 每个配置一个独立 commit/tag，能 checkout 回去复测；
- 换配置**必须重新干净编译**，别让旧二进制混入别的优化；
- 配置命名清晰：`fp32-scalar` / `fp32-neon` / `fp32-neon-mt4` / `int8-neon-mt4`。

## 8. 工具速查

| 脚本 / 工具 | 职责 | 关键行为 |
|---|---|---|
| `scripts/verify.sh` | 正确性门禁 | build + 单测 + golden token 对照，失败即中止 |
| `scripts/bench.sh <label> [额外参数]` | 快速单遍测速 | 开发迭代用；额外参数透传给 runtime |
| `scripts/record.sh <label> [--skip-verify] [--extra-args "..."]` | 正式记录 | 门禁 + 3 遍测速 + 自动写日志；带 extra-args 自动同场 A/B |
| `scripts/set_baseline.sh <label>` | 确立/更新基线 | 写 `benchmarks/baseline.json` |
| `scripts/commit_opt.sh <label> "总结"` | 规范化提交 | 代码+日志一个 commit；日志最新小节有 `<填...>` 占位会拒绝提交 |
| `tools/bench.py` | 底层测速 | 固定负载、丢预热、`--runs` 多遍取中位、自动记 commit |
| `tools/record_optimization.py` | 底层记录 | A/B、漂移警告、写 optimization_log.md |

## 9. 当前状态

- 已接入 dispatch 的算子：**matvec**（唯一热点，优化主攻方向），
  已注册实现：`ref`（默认）、`double_2_float`。
- 其余算子（rmsnorm/rope/attention/...）目前直接调 `_ref`；将来要优化哪个，
  照第 4 节给它也加一个通用入口即可。
- 优化记录与数字：`optimization_log.md`。
