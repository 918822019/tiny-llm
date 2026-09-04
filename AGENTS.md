# AGENTS.md

给在本仓库工作的 AI agent 的指引：构建/测试命令、环境、核心工作流与**已踩过的坑**。
面向人类的完整文档见 `README.md` 与 `docs/`（阅读路线见 README「文档」一节）。

## 项目一句话

tinyqwen：面向 **Qwen2.5-0.5B / Qwen3.5-0.8B / Qwen3.5-4B**（decoder-only，Qwen3.5 为 Gated
DeltaNet + full attention 3:1 混合架构）的端侧推理实验 runtime。C++17 + CMake 主线，
Python 工具链（导出 / 对齐 / 测速）。刻意**不做**通用推理框架、graph executor、C++ tokenizer。

## 构建 & 测试

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure     # 等价 ./build/tests/tinyqwen_tests
```

- CUDA 自动探测；无 nvcc（如 Mac）自动跳过，不影响 CPU 主线。
- **Apple Metal prefill（`--engine metal`）**：APPLE 构建时自动编译 `runtime/metal_prefill.mm`
  （ObjC++，需顶层 `enable_language(OBJCXX)`）；非 Apple 走 `metal_prefill_stub.cpp` 占位。
  GEMM 走 MPS、其余算子走内嵌 shader 的 Metal compute kernel。引擎自持 GPU KV cache，
  支持接续调用（投机解码 verify pass），新序列前须 `metal_prefill_reset_kv()`。
  改这个文件后除了 `verify.sh`，还要跑 KV 接续等价性测试（一次喂 N vs 分两段喂，
  末位 logits 误差必须 < 1e-3）—— fresh prefill 测不出接续路径的 bug。
  现成工具：`./build/benchmarks/test_metal_continuation --model <f16.tqwen> --tokens <csv> --split K`
  （会打印归因分解，把 attention/接续偏差与 lm_head 舍入拆开）。
   **判据为什么是容差不是"逐位相同"**：见坑 #10。
   **支持混合架构（Qwen3.5 GDN）**：GDN 四算子有 Metal 实现，每层一个 kernel、
   token 循环在 kernel 内部（逐 token dispatch 会有 9216 次 launch，光开销就 460 ms）。
   跑 Qwen3.5 时 `metal_prefill_run` **必须传 `GdnState*`**，否则 decode 发散（见坑 #14）。
   实测 seq=512 提速 1.377×，但 seq≤128 反而比 CPU 慢 —— GDN 递归的 GPU 占用率极低
   （只有 2048 线程）且 S[128,128]=64KB 超过 32KB 片上上限、每 token 两遍读写设备内存，
   是当前瓶颈；数字与下一步方向见 `docs/optimization_log.md`。
- 正确性门禁：`./scripts/verify.sh`（编译 + 单测 + golden token 对照；无真模型 `model.tqwen` 时第 3 步跳过）。

## Python 环境（非显而易见，关键）

- 用仓库根 `.venv`，由 brew **python@3.12** 建（系统自带 3.9 不满足 transformers main 的 ≥3.10）。
  调 Python 工具一律 `.venv/bin/python tools/...`。
- **Qwen3.5 依赖 `Qwen3_5ForCausalLM`，只在 transformers 的 GitHub main 分支（未发正式版）**；
  PyPI 最新 4.57.6 **没有**。装法：`curl -L https://github.com/huggingface/transformers/archive/refs/heads/main.tar.gz`
  后 `.venv/bin/pip install <tarball>`。遇到 `ImportError: Qwen3_5ForCausalLM` = transformers 过旧或 Python 过旧。
- pip 默认阿里源（`~/Library/Application Support/pip/pip.conf`）。已装：torch、numpy、pyyaml、tiktoken、modelscope。

## 核心工作流

- **接入新模型（一条命令）**：`./scripts/add_model.sh <HF 模型目录> [--dtype i4|f16] [--method hqq|rtn]`
  —— 自动走完：编译 → 导出 → 文件头校验 → 冒烟生成+解码验证。i4 默认 HQQ（质量优先），
  快速试跑用 `--method rtn`（快数倍）。导出器支持 `--workers` 并行量化 + 流式写盘，
  4B 量级内存占用有界（旧版会把全模型 fp32 驻留内存）。
- **导出真实权重**：先把 HF 权重下到本地目录，再
  `.venv/bin/python tools/export_qwen_to_tiny.py --model <本地目录> --out model_xxx.tqwen --dtype f16`。
  Qwen3.5 可用阿里 ModelScope 下载（`modelscope download --model Qwen/Qwen3.5-0.8B --local-dir models/Qwen3.5-0.8B`）。
- **数值对齐（无需真模型）**：`tools/align_fake_model.py`（Qwen2）/ `tools/align_fake_qwen35_model.py`（Qwen3.5）。
  通过标准：worst max_abs_err ~1e-6/1e-7（tol 1e-5）。
- **测速**：`MODEL=<file.tqwen> ./scripts/bench.sh <label> --extra-args "..."`
  （固定 prompt、decode 32、丢预热 4、取稳态中位）。
- **正式记录一条优化**：`./scripts/record.sh <label> [--extra-args ...]`（门禁 + 稳定测速 + 自动写日志）。
  历史数字与当前基线的唯一权威：`docs/optimization_log.md`。

## 常见坑（都已在本环境实际踩过）

1. **测速默认 ref kernel，数字不是真速度**：`--matvec-impl` 默认 `ref`（标量兜底），Qwen3.5-0.8B 实测
   ~595 ms/tok；必须显式传优化实现（见各 `model*.yaml` 的 `recipe` 字段）。当前最佳 i4 配方：
   `--extra-args "--matvec-impl sdot4_mt --ops-impl neon"`（4B ≈36.5、0.8B ≈8.1 ms/tok，M4）。
   f16 满栈例：`--matvec-impl neon_mt_kv_nt --ops-impl neon` → 0.8B ~17.65 ms/tok（M4）。
2. **Qwen3.5 对齐要 transformers main**：见上「Python 环境」。
3. **align 脚本依赖逐位置 dump**：已加 `--verbose` 强制逐 token prefill 恢复契约（`tools/align_fake_*.py`），勿删。
4. **fake Qwen3.5 的 tied lm_head**：`make_fake_qwen35_model.py` 对 tied 模型**不写**独立 `lm_head.weight`
   （与真实导出器 `if not tie_word_embeddings` 守卫一致）。若写了与 embed 不同的独立 lm_head，runtime 的 tied
   绑定会拿错权重，C++ vs HF logits 偏差 ~1.9，对齐直接失败。
5. **批量 prefill 有 token 阈值**：Qwen3.5 prompt ≥`kBatchPrefillMinQwen35`(=32) 才走 GEMM 批量路径，
   否则回退逐 token。短 prompt 测速看不到批量收益，别误判"批量没用"。`--no-batch-prefill` 可关。
6. **`--kv-f16` 是内存特性不是提速**：KV 存 fp16 + 融合 attention，KV 内存减半、长上下文可用，
   但解码慢 ~8%（寄存器内 fp16→fp32 转换抵消读带宽减半）。短上下文用 fp32，长上下文内存不够才开。
7. **Metal prefill 测速必须看离散度**：这台 M4 的 prefill 计时跨进程波动可达 **2×**
   （MPS kernel 每进程重新 JIT + 连续测速热降频）。用 `./scripts/bench_metal_prefill.sh`，
   输出带离散度列（max/min）——**离散度 >1.5 的行不可用于归因**。短 seq（16/64）噪声尤其大。
   CPU arm 已内置 f16 满栈配方（坑 #1），别手改成 ref，否则 speedup 会被放大十几倍。
   **两个补充教训（都是实测踩到的）**：
   - **测性能前先 `uptime`**。agent 运行时自己会抢 CPU（曾见 load 5.67、两个 opencode
     进程各占 125%/79%），端到端计时被污染到 CPU arm 离散度 **37×**，一度得出完全相反
     的优化结论并误回退。
   - **污染是加性的，所以取 min 比取中位数稳**。比较 Metal 版本时用引擎自带的
     `TINYQWEN_METAL_TIMING=1`（只量 GPU 侧 exec，不含 CPU 编排）跑 5 次取 min，
     比端到端中位数可靠得多。
   - **加一道物理合理性检查：省下不可能比被优化对象本身还大**。曾同场 A/B 测出
     "lm_head 末行 vs 全行省 220–289 ms"，而隔离微基准测得 lm_head 全行总共才 ~63 ms
     —— 差 3.5–4.6×，**物理上不可能**，于是判为噪声（配对差值区间 -60.9 到 +597.8，
     有一轮反而更慢）。凡是"收益大于被砍掉的那部分成本"的结论，先怀疑测量再怀疑代码。
     agent 自身持续占 CPU 时，这台机器的端到端 prefill 计时**根本无法归因**，
     只能靠隔离微基准（见坑 #11）。
8. **GPU 上并行度比 dispatch 次数更值钱**：曾试图融合 Metal compute kernel 减少 dispatch
   （每层 10 次 → 5 次），**结果全线变慢**（seq=512 1531 → 1791 ms），已回退。原因：RoPE 原本
   一个线程一对（524288 线程），融合后一个线程一个头（12288 线程）、每个串行跑 64 轮，
   **并行度掉 43×**。"减少 dispatch"这个直觉在 GPU 上不成立，要按并行度算账。
9. **MSL 里 `half` 是内建类型名**：不能拿它当参数名/变量名，否则整个 kernel 解析失败，
   报错还很误导（指向别的 kernel 的 `{`）。用 `n_half`。同理注意 `float`/`thread`/`device`。
10. **MPS 按矩阵尺寸选内部 kernel，所以"逐位相同"不能当数值判据**。实测同一行 logits：
    `resultRows=12` vs `resultRows=1` 有 144139/151936 个元素逐位不同（`max_abs_err=2.19e-05`），
    而 `resultRows=7` vs `1` **逐位完全相同** —— MPS 在某个行数阈值上换 tiling，fp32 累加顺序随之变。
    所以任何"两条路径 n 不同"的对比（如一次喂 N vs 分两段喂）都不可能逐位相同。
    **判据改用容差 1e-3**：良性舍入实测 ~6.6e-05，而历史上真的接续 bug
    （dispatch_2d grid 轴写反、RoPE 用批次内下标而非绝对位置）偏差是 **2.14 绝对值**，
    两边差 32000×，1e-3 卡得很干净。
    **另一个同源教训**：`TINYQWEN_METAL_TIMING` 的 `exec` 曾经只在 28 层循环里累加，
    **漏掉循环外的 lm_head command buffer** —— 拿它归因 lm_head 优化会得出完全错误的结论
    （曾测出"省 180 ms"，实际微基准是 63 ms，差的那部分是噪声）。已修，标签是 `(28 层 + lm_head)`。
    **用现成计时器做归因前，先确认它覆盖了你改的那一段。**
11. **对组件下结论前必须隔离测量它，不能从端到端时间倒推**。曾据端到端数字判断
    "MPS 只有峰值 57%"，据此投入写自写 GEMM；隔离重测发现 MPS 实际到 **60–73% 峰值**，
    自写 tiled GEMM 反而落后 9–17%，整条路证伪（见 `docs/optimization_log.md`）。
    顺带：**Apple GPU 没有 tensor core**，`simdgroup_multiply_accumulate` 不给硬件加速，
    只是 8×8 tile 摊到 32 lane 的寄存器布局约定，底层仍是标量 FMA —— 照搬 ggml 的
    simdgroup 结构不会凭空拿到矩阵单元吞吐。现成对照工具：`benchmarks/bench_metal_gemm.mm`。
12. **`dispatch_2d(cols < 线程组上限)` 的 kernel 必须检查 `gid.x`**。
   `dispatch_2d` 的 `threadsPerThreadgroup` 取的是 kernel 上限（1024），而 cols 可能
   远小于它。`kv_append` 最初只检查 `gid.y`（token），没检查 `gid.x`：
   Qwen3-0.6B 的 `NKV*HD` 恰好 = 1024 所以从未暴露，Qwen3.5 是 `2*256 = 512`，
   于是 `e ∈ [512,1024)` 的线程算出 `kvh = 2,3`（只有 2 个 kv 头），
   `dst` **越界写到下一层的 KV cache 槽**。凡是 cols 小于线程组上限的都要检查。
13. **读 Shared 存储的 GPU buffer 必须在 `commit + waitUntilCompleted` 之后**。
   调试插桩最初放在 commit 之前，读到的是上一层的旧值，制造出"每层都发散"的假象，
   导致一度怀疑已经验证正确的 GDN 路径，白绕一大圈。
   **插桩位置错了结论就全错 —— 先确认插桩读的是完成后的状态。**
14. **Metal prefill 跑 Qwen3.5 必须把 GDN 状态写回 CPU 的 `GdnState`**。
   引擎在 GPU 上算完递归后状态只在自己的 buffer 里，后续 decode 走 CPU 路径读的是
   `GdnState`。不写回的症状很迷惑：**prefill 首 token 完全正确，但 decode 立刻发散**。
   `metal_prefill_run` 的 `GdnState*` 参数就是为此加的；两边布局一致，是整块 memcpy。
15. **`sigmoid` 与 `silu` 一字之差会让输出门全错**。
   `apply_gate` 曾写成 `g/(1+exp(-g))` = `g·sigmoid(g)` = **silu(g)**，而 CPU 用
   `sigmoidf32`。定位靠**闭式检验 + 比值反推**：token 0 只有 1 个 KV 位置，
   softmax 权重必为 1，输出必须 = `v[0]·sigmoid(gate)`；实测 `Xa/v[0]` 全 256 维
   都是负的（sigmoid 恒正，数学上不可能），而比值 ≈ gate 原值 → 反推出多乘了一个 `g`。
   比逐行读代码快得多。
16. **`ModelConfig::full_layer_cache_index()` 在稠密模型下会除零**。
   `full_attention_interval = 0`（v1 格式）时 `(idx+1)/interval` 除零，返回值变垃圾。
   `n_full_layers()` 与 `is_linear_layer()` 都有 `<= 1` 守卫，但这个函数没有 ——
   **同一结构体里三个函数的守卫不一致**，混合架构代码里用它之前先确认 interval。

## 权重 / 数据位置（均已被 .gitignore 忽略，不入库）

- `models/<repo>/`：HF 原始权重（modelscope / huggingface-cli 下载）。
- `*.tqwen`：导出权重，如 `model_qwen35_f16.tqwen`（~1435 MB）。
- `model*.yaml`：模型注册表（架构 / dtype / recipe / 实测数字），`tools/model_registry.py` 读取。

## 代码约定

- `kernels/`：每算子一个目录，变体自注册 + 未注册兜底 ref；导读见 `kernels/README.md`。
- 优化纪律：优化前测基线、优化后复测、结果记 `docs/optimization_log.md`；方法论见 `docs/optimization.md`。
- commit 风格：中文 + Conventional 前缀（feat/fix/perf/refactor/build/docs）。
