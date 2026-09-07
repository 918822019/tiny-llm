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
- **MoE SSD 卸载对齐（fake MoE）**：`.venv/bin/python tools/align_fake_qwen35_moe_model.py`
  —— 生成 fake MoE 模型（GPTQ 专家），跑 resident vs SSD（slots=0/4）三遍，
  逐字节比对 logits。验证 ExpertStore pread+LRU 与 resident 指针逐位一致。
  cache 抖动扫描：`./scripts/bench_moe.sh`。机制见 `docs/moe_offload.md`。
- **导出真 MoE GPTQ 模型**：`.venv/bin/python tools/export_qwen_moe_to_tiny.py
  --model models/Qwen3-30B-A3B-GPTQ-Int4 --out model_qwen3_30b_moe_i4.tqwen`
  （48 层 / 18867 tensors / 17.16 GB）。运行时加 `--moe-ssd`（专家字节不进 RAM，
  resident 只剩 2884 MB）。**不开稀疏加载跑不起来**——文件 17.16 GB > 物理内存 16 GB。
- **测速**：`MODEL=<file.tqwen> ./scripts/bench.sh <label> --extra-args "..."`
  （固定 prompt、decode 32、丢预热 4、取稳态中位）。
- **正式记录一条优化**：`./scripts/record.sh <label> [--extra-args ...]`（门禁 + 稳定测速 + 自动写日志）。
  历史数字与当前基线的唯一权威：`docs/optimization_log.md`。

## 常见坑（都已在本环境实际踩过）

1. **测速默认 ref kernel，数字不是真速度**：`--matvec-impl` 默认 `ref`（标量兜底），Qwen3.5-0.8B 实测
   ~595 ms/tok；必须显式传优化实现（见各 `model*.yaml` 的 `recipe` 字段）。当前最佳 i4 配方：
   `--extra-args "--matvec-impl sdot4_mt --ops-impl neon"`（4B ≈36.5、0.8B ≈8.1 ms/tok，M4）。
   f16 满栈例：`--matvec-impl neon_mt_kv_nt --ops-impl neon` → 0.8B ~17.65 ms/tok（M4）。
   **GPTQ 例外：变体族只有一个标量实现，传什么都没用**。`kernels/matvec/` 下 GPTQ 只有
   `matvec_gptq_ref`，`--matvec-impl` 的可用列表（acc4/neon/neon_mt/sdot*/ref…）全是
   HQQ/i4 interleaved 打包的实现，**不适用于 GPTQ 列主序布局**。所以 Qwen3-30B-A3B-GPTQ
   实测 9400 ms/tok 是标量吞吐的真实反映，不是配置错误。别浪费时间调参，写 GPTQ
   NEON+MT kernel 才是唯一出路（预估 37–374×，见 `docs/moe_offload.md`）。
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
17. **MoE SSD 卸载不开稀疏加载等于没卸载**。`ModelFile::load()` 历史上整文件读进
   `data_`，所以即便走 ExpertStore 的 pread 路径，专家权重照样全量常驻 RAM ——
   pread 只是演示机制。必须 `load(path, err, offload_experts=true)`（`--moe-ssd`
   自动传），判据是名字含 `.mlp.experts.`（路由门 `mlp.gate.weight`、共享专家
   `mlp.shared_experts.*` 都不含，别把判据放宽到 `.mlp.` 否则误伤常驻部分）。
   两个连带坑：**(a) `(view.data - file.base())` 不再是文件偏移**——稀疏把 resident
   tensor 紧凑重排了，一律改用 `TensorView::file_offset`；**(b) 卸载 tensor 的
   `data` 是 `nullptr`**，任何无条件解引用它的校验都会段错误（GPTQ in-band magic
   校验就中招过，改成只 pread 前 4 字节）。紧凑打包时每个 tensor 起点仍须 64B 对齐
   （kernel 有按对齐选路的分支）。内存是否真降看启动行 `[init] weights: resident ..
   offloaded ..`，logits 逐位一致只证明算对了、不证明内存省了。
18. **MoE 实现原本只适配 Qwen3.5-MoE，真模型 Qwen3-MoE 会静默算错**。三个坑都是
   拿真 checkpoint（`Qwen3-30B-A3B-GPTQ-Int4`）才暴露的：
   **(a) HF `model_type=qwen3_moe` 不是 `qwen3_5_moe`**，且它是 dense attention
   （无 GDN 混合）、**没有共享专家**。原来 loader 硬性要求
   `shared_expert_intermediate_size != 0 && n_shared_experts != 0`，直接拒绝。
   现共享专家改为可选，用 `ModelConfig::has_shared_expert()` 判断，两个字段必须
   同为 0 或同非 0（只填一个 = 导出器写错，fail fast）。新增 `ModelType::kQwen3MoE=3`。
   **(b) `qwen_forward_prefill.cpp` 的 MoE 逐 token 分支原本在 `if (is_qwen35)`
   里面**，`kQwen3MoE` 不属于 `is_qwen35`，会掉进 Qwen2 批量 prefill 路径 ——
   那条算 dense SwiGLU 而非 MoE，**不报错但结果全错**。已提到架构判断之前。
   **(c) `full_layer_cache_index()` 除零**（坑 #16）：dense-attention MoE 每层都是
   full attention，`full_attention_interval` 为 0/1 时该函数每层都会被调用。已补守卫。
   **教训：只在 fake 模型上验证过的架构假设，遇到真模型基本都会破。**
19. **matvec 必须按张量自身 dtype 路由，不能用模型级 dtype**。`QwenModel::mv()`
   原先用 `quant_type_of(dtype_)`（文件级 master dtype）。GPTQ MoE 模型里
   **router（`mlp.gate.weight`）与非 tied 的 `lm_head` 是 fp32，而 master 是
   kGPTQ4** —— fp32 数据被当 GPTQ in-band 块解析，router logits 全错 →
   选错专家 → 输出完全乱码，**且不报任何错**。修法：新增 `mv_typed(w,x,y,out,in,dtype,gs)`，
   绑定时记录张量自身 dtype。
   **为什么 fake 模型测不出**：fake 生成器把 router 也做成了 GPTQ
   （`add_quant(p + "mlp.gate.weight", ...)`），恰好掩盖了这个 bug。
   而 `align_fake_qwen35_moe_model.py` 只比 **resident vs SSD（都是 C++）**，
   从不与独立参考比 —— 所以 MoE FFN 的数值正确性此前从未被验证过。
   **定位方法**：单 token + pos=0 做参考前向（此时 RoPE 是恒等、attention 只有
   一个位置故 softmax 权重必为 1、输出=v，参考实现极简），再 `--dump-logits`
   逐位比。CosSim 从 -0.0016 修到 1.00000012。
   **两个自己踩的坑**：(a) 用 `safetensors.torch.load_file` 后 `.float()` 会把
   int32 的 `qweight/qzeros/g_idx` 转成 float32，再 `.view(np.uint32)` 就是把浮点
   位模式当整数读 → 全是垃圾（qzeros 本该恒为 7，结果算出 mean=13.88）。
   读 GPTQ 张量必须用 `safe_open(framework='numpy')` 保留 dtype。
   (b) SwiGLU 写成 `sigmoid(gate)*up` 而非 `gate*sigmoid(gate)*up`，就是坑 #15。
20. **MoE 测速必须扫 cache slots，不能固定一个大值**。反直觉：**小 cache 更快**。
   Qwen3-30B-A3B-GPTQ 实测（3 次取 min）：`slots=4`（仓库默认）**616 ms/tok** 最优，
   `slots=4096` **1126 ms/tok 慢 1.8×**——而 4096 是命中最多的配置（hits=3361）。
   `slots=16` 几乎全 miss（hits=0）依然比全命中的 4096 快。
   **原因不是命中率，是工作集能否常驻 CPU cache**：13.48 GB 顺序读进 ~40 MB 热
   缓冲，胜过读 5.05 GB 进 ~5 GB 缓冲反复 thrash。铁证：同一份数学的
   `expert_ffn` 耗时随 slots 变 **2.5×**（2.16 s vs 5.35 s）——计算时间不该随
   cache 配置变，变了就说明是局部性问题。
   **我自己浪费了一整轮测速**：全程用 `slots=4096`，那是最差配置。
   连带坑：**大 slots 会把机器推进重度换页**（实测 `vm.swapusage used = 11.3 GB
   / 12 GB`、pageouts 132 万），此后所有计时都不可信。测速前后都查
   `sysctl vm.swapusage`。cache 应按**字节预算**而非槽数限界。
21. **`--matvec-impl` 的名字在多个注册表里同名但语义不同**。GPTQ 的
   `neon`/`neon_mt`/`ref` 与 f32 注册表同名，但 **f32 的 `neon` 是单线程版**。
   把用户的 impl 名传播到 f32 注册表会让 lm_head（fp32、1187 MB、每 token 全读）
   落到单线程内核。修法：GPTQ 分支不传播 impl 名，f32 侧一律 `neon_mt_kv_nt`。
   **教训：跨注册表复用同一个 CLI 参数时，先确认名字在各表的语义是否一致。**
22. **kernel 级多线程对 MoE 无效**。专家矩阵 `[768,2048]` 只有 `768/64 = 12` 个
   o_block 给 10 线程（负载不均），且每 token 有 **3 matvec × 8 专家 × 48 层
   = 1152 次 fork-join**，同步开销压过收益。提高粒度阈值到 4M 反而更差。
   **MoE 要并行得在专家层并行**（top-8 彼此独立，48 次 fork-join/token），
   是 runtime 级改动不是 kernel 级。加 MT 变体前先算 fork-join 次数。
23. **导出器不要硬编码 dtype——保留源 dtype**。源 checkpoint 的 lm_head /
   embed_tokens / layernorm **本来就是 fp16**，`export_qwen_moe_to_tiny.py` 曾
   硬编码 `DTYPE_F32` 把它们升 fp32，**白白翻倍且零收益**（lm_head 594→1187 MB）。
   实测 fp16 相对 RMS 误差 0.0000%、argmax 一致率 100%、CosSim 1.0（就是原值）。
   连带要改三处，漏一处就报错或静默变慢：loader 的 dtype 白名单、
   `qwen_model.cpp` 的 `mixed_ok` 校验、`main.cpp` 对应 dtype 注册表的 impl 选择
   （漏设 f16 表会让 fp16 lm_head 落到标量 ref，坑 #21 同类陷阱）。
24. **卸载张量的行步长必须按 dtype 算，不能硬编码 `sizeof(float)`**。embed 卸载
   的 pread 曾写 `token_id * hidden * sizeof(float)`，而 fp16 embed 的行步长是
   `hidden*2` —— 读到错误偏移的数据。实测症状：**CosSim 掉到 0.871、
   max|Δ| 10.47、argmax 全错**，且不报任何错。与坑 #17(b)（卸载 tensor 的
   `data` 是 nullptr）同源：卸载路径的每一处尺寸/偏移计算都要按真实 dtype 走。
25. **量化收益必须用 argmax 一致率衡量，不能只看 CosSim**。i4 RTN 量化 lm_head
   的 CosSim 是 0.995（看起来"很好"），但 **top-1 argmax 一致率只有 75%** ——
   greedy decode 每 4 个 token 就有 1 个分叉，长序列累积偏离。权重相对 RMS 误差
   9.989% 不是 bug，是 RTN i4 的理论值（step/√12 ÷ 权重 RMS ≈ 11.5%），但对
   **直接产生 logits 的 lm_head** 偏高。实测 200 个随机 hidden 向量：
   fp16 → 100% 一致（无损），i4 → 75%。所以 lm_head 默认保留 fp16，
   量化（`--quant-lm-head`）仅作 opt-in。
26. **内存校验要用"可用内存"而非物理总量**。wired（内核与不可换出部分）+ 其他
   进程已占掉大半：实测 16 GB 机器 wired 就有 8 GB、可用只剩 1.5 GB。按物理
   总量校验会宽松 **7×**，放行后照样把机器推进换页（实测 swap used 11.3 GB /
   12 GB、pageouts 132 万）。**swap 是写操作、消耗 SSD 寿命**，所以宁可
   fail-fast。macOS 用 `host_statistics64` 取 free+inactive+purgeable，留 10% 余量。
27. **并发代码两条铁律（都是实测踩到的死锁）**。定位手段：后台跑 + `sample <pid> 1`
   抓调用栈，两个线程都停在 `__psynch_cvwait` 就是互等。
   **(a) 不同等待条件必须用不同 CV。** 主线程等"某 key 就绪"、worker 等"队列有活"，
   共用一个 CV 时 `notify_one` 可能唤醒**错误的等待者**（主线程），它重查条件
   不满足又睡回去，worker 永不被唤醒 → 双方永久互等。拆成 `pf_work_cv_` /
   `pf_ready_cv_` 才修好。
   **(b) 等待条件必须把"等待对象消失"也算作满足。** 预取槽被复用给别的 key 后，
   主线程重查自己的 key 得到 nullptr；若条件写成 `(p != nullptr && p->ready)`
   就会永远为假 → 永等。必须写成 `p == nullptr || p->ready`，让"对象没了"走
   回退路径。同理：**槽/缓冲选择绝不能覆盖"在飞"（未就绪）的对象**。
28. **估算 overlap 收益要在实际重叠的粒度上比较，不能在上层粒度算**。B-2 预估
   1.83× 是按"整 token 级" `max(I/O, 计算)` 算的，实测只有 **1.17×**。根因：
   重叠发生在**单专家级**，而单专家 I/O 0.50 ms（2.5 MB ÷ 5 GB/s）**大于**单专家
   计算 0.34 ms —— 与预估假设相反，预取只能把 I/O 提前、不能消除它。
   实测预取命中率 hits 7% / waits 24% / **fallbacks 70%** 就是证据。
29. **decode 已达 NVMe 带宽下限，CPU/IO 侧优化到顶**。每 token 读 1.682 GB，
   有效带宽 5.46 GB/s（M4 NVMe 峰值 **84%**），理论下限 259 ms/tok，距当前
   308 ms/tok 仅 **49 ms** 空间。要再快只能**读更少数据**（更激进量化、按热度
   常驻高频专家），不是继续优化计算或调度。
30. **批量路径的 matmul 必须有 NEON 版，否则 I/O 收益会被计算反噬吃掉**。
   MoE 批量 prefill 把 I/O 降了 **96%**（n=128：230 GB → 8.4 GB），但初版
   `matmul_gptq` 只有标量 ref，结果 **prefill 反而慢 3.5×**（expert_ffn
   25.14s vs 逐 token 4.50s）。补 NEON 版才转正到 1.44×。
   **教训：做"减少 I/O"的优化前，先确认计算侧不会因为换实现而变慢。**
   逐 token 路径用的是优化过的 `matvec_gptq_neon`，批量路径若退回标量就等于
   用 5× 计算变慢换 96% I/O 降幅 —— 净亏。
31. **列主序矩阵不能对"列"向量化**。`matmul_gptq` 的 X 是列主序 `[K, N]`，
   元素 (row r, col c) 在 `c*K + r`。**固定 r、变化 c 的步长是 K —— 不连续**。
   初版误以为 `x[k*N + c]` 对固定 k 连续，实测 generated_ids **完全错**。
   正确的向量化方向是 **o**（qweight 是 `[(K/8), M]`，固定 c8 变化 o 才连续），
   与 `matvec_gptq_neon` 同构，只是外面多一层列循环。
   **写 GEMM kernel 前先画清布局：哪个维度在内存里连续，就沿它向量化。**
32. **批量路径的每个张量都要按自身 dtype 读**。MoE 批量 prefill 的 router 初版
   按 fp32 读，但真 checkpoint 的 router 是 **fp16** → 把 fp16 字节当 fp32 解析
   → 垃圾 → NaN → 输出全 0。坑 #24 同类。
   **定位手段：逐层插 NaN 检查**（`std::isfinite` 扫描各中间缓冲），首个 NaN
   出现在 `gate_logits` 就直接锁定 router。比逐行读代码快得多。

## 权重 / 数据位置（均已被 .gitignore 忽略，不入库）

- `models/<repo>/`：HF 原始权重（modelscope / huggingface-cli 下载）。
- `*.tqwen`：导出权重，如 `model_qwen35_f16.tqwen`（~1435 MB）、
  `model_qwen3_30b_moe_i4.tqwen`（~17.16 GB，MoE GPTQ，须配 `--moe-ssd`）。
- `model*.yaml`：模型注册表（架构 / dtype / recipe / 实测数字），`tools/model_registry.py` 读取。

## 代码约定

- `kernels/`：每算子一个目录，变体自注册 + 未注册兜底 ref；导读见 `kernels/README.md`。
- 优化纪律：优化前测基线、优化后复测、结果记 `docs/optimization_log.md`；方法论见 `docs/optimization.md`。
- commit 风格：中文 + Conventional 前缀（feat/fix/perf/refactor/build/docs）。
