# Android 端侧指南（编译 → push → 真机 decode → 拉回 profile）

v1 使用命令行 binary，不做 App / JNI。端到端流程：
本地导出权重 → NDK 交叉编译 → adb push → 真机 decode → 拉回 profile。

## 0. 准备

```bash
./scripts/doctor_android.sh      # 一键预检：设备/NDK/模型/构建/基线一次列全
```

doctor 会把整条链路的阻塞点逐项列出（PASS/WARN/FAIL + 修复命令），
退出码 0=就绪 / 1=仅警告 / 2=有阻塞错误。手动确认时等价于：

```bash
adb devices                      # 确认设备连接，已开开发者模式（unauthorized 时去手机上点允许）
export ANDROID_NDK=/path/to/ndk  # r25+ 测试通过即可；也可不设置，见下
```

NDK 自动探测：`build_android.sh` 与 doctor 共用同一套级联——
`$1` → `$ANDROID_NDK` → `~/Library/Android/sdk/ndk/<最新版本>` →
`/opt/homebrew/share/android-ndk`（`brew install --cask android-ndk`）→
`/usr/local/share/android-ndk`。探测到即直接用，无需手动 export。

目标 ABI：`arm64-v8a`（NEON/INT4 kernel 的主战场；NEON 是 aarch64 基线指令，
无需额外编译选项）。

## 1. 导出权重（本机，一次）

```bash
python tools/export_qwen_to_tiny.py --model /path/to/Qwen2.5-0.5B --out model.tqwen
python tools/tokenize_prompt.py --model /path/to/Qwen2.5-0.5B \
  --prompt "你好" --chat --out prompt_tokens.json
```

注意：v1 FP32 权重约 2 GB，push 耗时正常；`run_android.sh` 会按文件大小跳过重复 push。

## 2. NDK 交叉编译

```bash
./scripts/build_android.sh       # arm64-v8a, android-28, Release
```

等价手工命令：

```bash
cmake -S . -B build-android \
  -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
  -DANDROID_ABI=arm64-v8a \
  -DANDROID_PLATFORM=android-28 \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build-android -j
```

产物：`build-android/runtime/tinyqwen`。

编译注意：

- `ANDROID_PLATFORM` 需要 ≥ 28（C++17 运行时 + posix_memalign 行为稳定）；
- 不引入第三方依赖；profiler 的 JSON 为手写输出，无需 JSON 库；
- `libc++_shared.so` 相关问题：确认 NDK toolchain 默认 `ANDROID_STL=c++_static`
  （本项目无额外要求，保持默认）；
- Release 默认 `-O3`；FP32 reference 不额外开 fast-math，保证与 PyTorch 对齐。

## 3. 真机运行

```bash
./scripts/run_android.sh model.tqwen prompt_tokens.json --max-new-tokens 16 --topk 5
```

等价手工命令（便于排查）：

```bash
adb push build-android/runtime/tinyqwen /data/local/tmp/tinyqwen/
adb push model.tqwen /data/local/tmp/tinyqwen/
adb push prompt_tokens.json /data/local/tmp/tinyqwen/tokens.json
adb shell "cd /data/local/tmp/tinyqwen && chmod +x tinyqwen && \
  ./tinyqwen --model model.tqwen --tokens-json tokens.json \
             --max-new-tokens 16 --profile-out profile.json"
```

### stdout 输出格式

```text
topk <id>:<val> ...   # logits 分布；第一行在 prefill 结束后输出
gen <step> <token_id> # 生成 token；每个 topk 行描述下一个 gen 行的 token
generated_ids: ...    # 末尾汇总全部生成 ids
```

需要全量 logits 做数值比对时加 `--dump-logits PATH`（每次 forward 写一行
vocab 个 fp32，行序 = 位置序；完整语义见 README「CLI 参考」）。

## 4. 拉回 profile

```bash
./scripts/pull_profile.sh profile_android.json
```

字段含义见 `profiling_schema.md`。

## 5. 常见坑

| 现象                       | 处理                                                                  |
|--------------------------|---------------------------------------------------------------------|
| `CANNOT LINK EXECUTABLE` | 确认 ABI 与设备匹配（`adb shell getprop ro.product.cpu.abi`），用 `c++_static` |
| push 后权限报错               | `adb shell chmod +x /data/local/tmp/tinyqwen/tinyqwen`              |
| OOM / 被杀                 | 0.5B fp32 权重约 2GB + KV cache；先降 `--max-seq-len`，或等 INT8/INT4        |
| 输出乱码 token id            | 确认 prompt 用了 chat template，且 eos 设置正确（默认 151645）                    |
| 时延抖动大                    | 手机热降频/大小核迁移；v1 不绑核，解读 profile 时看 p50/p95 而不是单次值                     |

## 6. 性能基准与优化 pipeline

Android 侧有一套完整的优化 pipeline，与本地（macOS）对等：

```bash
# 一键预检（设备/NDK/模型/构建/基线，阻塞点一次列全）
./scripts/doctor_android.sh

# 正确性门禁（golden token 对照）
./scripts/verify_android.sh

# 快速单遍测速
./scripts/bench_android.sh <label> [--extra-args "..."]

# 正式记录：门禁 + 3 遍测速 + 自动写 optimization_log.md
./scripts/record_android.sh <label> [--skip-verify] [--extra-args "..."]

# 建立 Android 基线（label 之后的参数透传给 bench，基线要带上被基线化的配置）
./scripts/set_baseline_android.sh <label> [--extra-args "..."]

# 对比两次 profile（定位 op 级变化）
python tools/profile_diff.py before.json after.json
```

### fp16 满栈参考流程（当前主测配置）

```bash
MODEL=model_f16.tqwen ./scripts/verify_android.sh                 # 门禁：golden 逐位
MODEL=model_f16.tqwen ./scripts/set_baseline_android.sh \
    android-fp16-baseline --extra-args "--matvec-impl neon_mt_kv_nt --ops-impl neon"
MODEL=model_f16.tqwen ./scripts/record_android.sh fp16-neon-android \
    --extra-args "--matvec-impl neon_mt_kv_nt --ops-impl neon"
```

`neon_mt_kv_nt` 是 f16 满栈变体（NEON + 线程池 + k/v 融合 + LDNP，仅 aarch64），
与 f32 阶梯顶层同名、按模型 dtype 解析；`--ops-impl neon` 同时启用
rmsnorm/rope/attention/swiglu/argmax/GDN 的 NEON 变体。

### INT4（HQQ 量化 + W4A8 SDOT）

```bash
# 导出（默认 HQQ@64 + lm_head 独立 i4 副本；--no-lm-head-i4 / --method rtn 对照）
python tools/export_qwen_to_tiny_i4.py --model models/Qwen2.5-0.5B \
    --out model_i4_hqq_lmh.tqwen --method hqq --group-size 64

# 精度审计（对 fp32 逐层 MSE/cos）+ token 门禁
python tools/verify_i4_accuracy.py --model-i4 model_i4_hqq_lmh.tqwen --model-fp32 model.tqwen

# 测速：i4 kernel 用 W4A8 SDOT（sdot/sdot_mt，armv8.2 dotprod）
MODEL=model_i4_hqq_lmh.tqwen ./scripts/bench_android.sh i4-sdot \
    --extra-args "--matvec-impl sdot_mt --ops-impl neon"
```

**现状**：W4A8 SDOT kernel 把 i4 反量化从"展宽转 fp32"（3.5 指令/字节）换成
整数点积（权重解包 int8 + 激活 int8 量化 + SDOT，~0.9 指令/字节）。Android
实测 **TOPT 22.35 ms/tok**，同场 vs i4 neon_mt 提速 **2.00×**（44.59→22.35），
比最初的 i4（44.7，慢 f16 2.1×）大幅收窄。
**准确定位（跨模型同场 A/B）**：i4 22.35 vs **f16 满血 17.05 = 0.76×**——i4
比满血 f16 仍慢 ~24%，但追平了过热降频的 f16（21.16）。本质：f16 带宽瓶颈、
随温度漂（17~21ms）；i4 算力瓶颈、稳定 22.3ms。i4 在这台高带宽设备上的优势
是**内存**（784 vs 942MB、matmul 流量÷4）与**热稳定**，不是绝对速度；速度
优势要在带宽更低的设备兑现。代价：激活 int8 量化（业界标准，token 门禁内）。
剩余差距在 lm_head（~5.5ms/token）+ per-group 开销。详见优化日志 i4-sdot_mt。

### 延迟指标口径（TTFT / TOPT）

bench/record 的结果（JSON、history jsonl、optimization_log 记录）都带标准
LLM 推理指标，口径与本地 `bench.py` 一致（数字来自 profiler JSON）：

| 指标          | 含义                    | 口径                                               |
|-------------|-----------------------|--------------------------------------------------|
| prefill 长度  | prompt token 数        | 标准负载固定 3 tok（改负载 = 换尺子，见 `optimization.md` §6.1） |
| decode 长度   | 生成 token 数            | 标准负载固定 32 tok，丢前 4 预热，稳态样本 28                    |
| TTFT        | 首 token 延迟            | `first_token_ms` = prefill 总耗时                   |
| TOPT        | 每输出 token 延迟          | 稳态 decode **中位数**为主指标，p95 看抖动                    |
| forward 总耗时 | prefill + decode 全部时间 | `total_ms`                                       |

注意：

- 默认 CPU 路径是**批量 GEMM prefill**：整个 prompt 一条记录，
  `first_token_ms` 即真 TTFT；verbose / GPU engine 走逐 token 路径时它是
  各 prefill token 之和。两种路径的 TTFT 不直接可比。
- TTFT 只在 prompt 长度相同时可比；且每次运行都是新进程，TTFT 含首次
  forward 的冷启动成本（workspace/KV 首触）。
- `generated_tokens` = 实际生成数；decode **步数** = 生成数 - 1
  （最后一个 token 不需要再 forward），稳态样本数另计预热丢弃。

### 热门禁（Thermal Gate）

Android 设备热降频是测量抖动的最大来源。bench 默认在每次测量前检查设备温度：

- 超过阈值（默认 45°C）时等待降温，每 10s 轮询一次
- 超时 180s 后打印警告继续（不阻塞 pipeline）
- `--no-thermal-gate` 跳过（快速迭代时用）
- `--thermal-max 40000` 自定义阈值（单位 millidegree）

### 绑核（Core Pinning）

默认绑大核（`--pin-cores big`），自动检测频率最高的那组 CPU：

- `--pin-cores all` 不绑核
- `--pin-cores "4,5,6,7"` 手动指定 CPU 编号
- 在 adb shell 里通过 `taskset <mask>` 实现

### 线程数对齐

多线程 matvec kernel（`neon_mt*` 族）读 `TINYQWEN_MT_THREADS` 环境变量，
默认按 `hardware_concurrency()`（全部核）开线程。绑核后 bench 自动注入
`TINYQWEN_MT_THREADS=<绑核数>`——否则线程会在绑定的少量大核上过度竞争，
测速失真。`--threads N` 显式覆盖（`--threads 0` = 不注入，用 kernel 默认；
不绑核时默认也不注入）。实际注入值会记进结果 JSON 的 `mt_threads` 字段。

### 资源采样（内存 / 频率 / 温度 / 利用率）

时延之外，每遍测量还会在设备端后台采样推理进程的资源占用（默认每 100ms
一次，直到进程退出），补上静态环境信息给不了的运行时数据：

- **进程内存（RSS）**：读 `/proc/<pid>/status` 的 `VmRSS`（当前常驻）和
  `VmHWM`（峰值，内核记录——采样再粗也不会漏峰）。权重是 fread 全量读进
  进程内存的（非 mmap），所以 **RSS 就是"权重 + KV cache + workspace"的
  真实常驻占用**。同时解析 runtime `[init]` 行报告的 KV cache / GDN state
  分配大小和权重文件大小，便于对账：RSS ≈ 权重 + KV cache + 其余开销。
- **各核实实时频率**：读 `scaling_cur_freq`。静态的 `cpuinfo_max_freq`
  只说明"最高能跑多快"，这里采的才是推理期间"实际跑多快"——热降频的
  直接证据。绑大核时只按大核聚合（小核低频会掩盖降频），字段标
  `scope: big`；未绑核时聚合全部核（`scope: all`）。
- **CPU 温度**：全部 thermal zone，host 侧按 `cpu*` 且非 `trip` 过滤取 max
  （与热门禁同口径）；看推理把设备烤热多少、何时触发热降频。
- **各核利用率**：`/proc/stat` 相邻样本做差；看算力到底压没压满。

聚合值写入结果 JSON：`resources.mem`（peak/avg RSS、样本数、权重/KV cache/GDN
大小）、`resources.freq`（min/median/max kHz）、`resources.temp`
（起/峰/止 °C）、`resources.util`（全核/扛算核均值 %）。跨模型 A/B 时对照组
在 `ab_control_resources`——对比 dtype 时内存占用差异（如 i4 权重约是 fp32 的
1/4）和时延同样重要。`record_android.py` 会把资源行写进 optimization_log.md。

**时序曲线**：完整采样序列另存 `benchmarks/series/<时间>_<label>.json`
（RSS / 温度 / 各核频率 / 各核利用率 vs 时间），结果与 history 行里只放文件名
引用。Web 控制台「历史」表的「曲线」按钮读它画四联图——能直观看到 RSS 加载
爬升、温度上涨、大核热降频的整个过程。采样循环只用 shell 内建（不 fork
grep/cat），避免采样本身拖慢循环、拉低分辨率。

`--no-resource-sample` 跳过采样（快速迭代）。

### PMU 内存带宽测量（--pmu）

decode 是内存带宽瓶颈，此前只能从"权重字节 ÷ TOPT"估算有效带宽；
现在可以用硬件计数器真测：

```bash
python tools/bench_android.py --label <label> --model model_f16.tqwen --pmu \
    --extra-args "--matvec-impl neon_mt_kv_nt --ops-impl neon"
```

- **原理**：NDK 自带 simpleperf（非 root 可用）统计 `raw-bus-access-shared`
  事件（CPU 发出的总线访问，流式 matvec 下 ≈ DRAM 行填充），按每次 32~64B
  （SoC 总线宽度决定，设备侧不可知）给出流量/带宽**上下界**；另收
  cpu-cycles/instructions（IPC）与 L1D/L2D refill。
- **约束**：单遍单配置测量（不支持 A/B）；与资源采样互斥（simpleperf 裹住
  进程后 `/proc/<pid>` 读到的是 simpleperf，自动关闭采样）；计数区间含模型
  加载（约 2% 流量）。
- **设备适配**：PLK110 上 `l3d-cache` 系列事件不受支持（计数恒 0，simpleperf
  有告警），故用 bus-access；换设备先 `simpleperf list` 核实事件。
- **实测自证**（PLK110，f16，32 decode）：NEON 满栈 bus_access=10.52 亿
  （14.8~29.6 GB/s）vs ref 标量 10.74 亿（1.46~2.92 GB/s）——同一负载数据量
  差 <2%（计数器测的是"搬了多少"，与快慢无关），速率差 20×；ref 测量值与
  "权重 ÷ TOPT"估算（1.4 GB/s）互相印证。

### 回归检测

bench 完成后自动对比 `benchmarks/baseline_android.json`：

- 本次中位数比基线慢超过 5%（可调 `--regression-threshold`）时打印警告
- `--fail-on-regression`：回归时非零退出（CI 集成用）
- `--no-check-regression`：跳过

### A/B 同场对照

`--extra-args` 非空时自动启用——每遍先测对照（ref）再测变体，
交错抗热降频漂移。方法论与本地一致，见 `optimization.md` §6.3。

基线独立于本地：`benchmarks/baseline_android.json`（设备和 host 数字不可比）。

### 数据集负载测试（真实 prompt 的 TTFT / TOPT 分布）

canonical 负载是 3-token prompt：测 TOPT 够用，但 **TTFT / prefill 行为在
3-token 下没有意义**。数据集模式用真实文本补这块——runtime 批量模式
（`--batch-tokens-jsonl`）一次进程顺序跑多条 prompt，每条独立
reset → prefill → decode 逐条计时（KV cache O(1) 重置；与逐条单跑的
generated_ids 逐位一致，fake 模型已验证）。

```bash
# 1. 数据集 → batch 输入（确定性取前 N，可复现；默认诗词数据集 test.csv）
python tools/tokenize_batch.py --num 16 --out benchmarks/poetry16.jsonl

# 2. 端侧批量测量（热门禁/绑核/资源采样与 bench_android 同一套）
python tools/bench_dataset.py --label poetry-neon \
    --jsonl benchmarks/poetry16.jsonl --model model_f16.tqwen \
    --extra-args "--matvec-impl neon_mt_kv_nt --ops-impl neon"
```

输出：

- **TTFT**：median/p95 + 按 prompt 长度分桶（8-16/16-32/32-64/64-96 tok），
  每桶给 ms 和归一的 ms/tok（prefill 吞吐口径）；
- **TOPT**：各 prompt decode 中位的中位（headline）+ 全量 decode 步 p95；
- **资源**：整场一条采样曲线（`benchmarks/series/`，控制台可画）；
- 结果进 history（`source="dataset"`，与 canonical 行区分）。

纪律：**数据集数字与 canonical 数字不可直接互比**（负载不同 = 尺子不同）。
TOPT 口径相近可参考；TTFT 只有数据集模式能测。默认 decode=32/条、
`--max-seq-len 128`（装长 prompt）。批量模式与 `--topk/--dump-logits/
--verbose/--engine/--profile-out` 互斥（输出契约是纯时序测量）。

Web 控制台同样支持：「发起测试」选 `dataset` 类型、选 JSONL 输入即可；
跑完历史表出现 dataset 行（紫色 badge），点「曲线」除资源四联图外还有
**TTFT vs prompt 长度散点**（带中位 ms/tok 斜率线，prefill 扩展性一眼看）。

## 7. 性能解读注意

v1 是单线程 FP32 reference，真机上会很慢——这是预期行为。
profiling 的目的是建立 op 级基线和占比结构，为后续 INT4/KronQ kernel、
多线程策略和 spec decode 提供对照，而不是追求 v1 的绝对时延。
真机是优化效果的终极裁判；测量方法见 `optimization.md`。

## 8. Web 控制面板（浏览器管理端侧测试）

不想敲命令时，用零依赖的本地控制台（Python 标准库）：

```bash
python3 tools/web_console.py [--port 8765] [--dev]
# 浏览器打开 http://127.0.0.1:8765
# --dev：开发模式，改 tools/webui/*.css|*.js 直接刷新生效（不缓存）
```

四个页面与命令行一一对应（基线已合并到设备状态页脚）：

| 页面    | 等价命令                                                                                       |
|-------|--------------------------------------------------------------------------------------------|
| 设备状态  | `scripts/doctor_android.sh`（检查项同款，8s 自动刷新，含基线摘要）                                         |
| 发起测试  | `verify_android.sh` / `bench_android.py` / `record_android.sh` / `set_baseline_android.sh` |
| 任务与日志 | 终端输出（1s 轮询流式，可终止运行中任务，日志 100KB 上限 + auto-scroll gate）                                 |
| 历史与趋势 | `docs/optimization_log.md` 汇总表 + `benchmarks/history_android.jsonl`（趋势图 + 资源曲线面板）     |

前端共享源码在 `tools/webui/`（CSS 设计 token + SVG 图表原语 + 色板），`/assets/*` 路由提供静态资源。

约定与限制：

- 只监听 127.0.0.1；设备是独占资源，**同一时刻只允许一个任务**（忙时 409）。
- 每次 bench/record/baseline 完成会往 `benchmarks/history_android.jsonl`
  追加一行结构化结果（`--no-history` 跳过）；`optimization_log.md` 仍是权威叙述账本。
- 任务日志存 `benchmarks/jobs/<id>.log`，完成记录存 `benchmarks/jobs/index.jsonl`
  （服务重启后历史任务仍可见）。
- 深色模式：跟随系统 `prefers-color-scheme`，无需手动切换。
