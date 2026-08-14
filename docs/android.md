# Android 端侧指南（编译 → push → 真机 decode → 拉回 profile）

v1 使用命令行 binary，不做 App / JNI。端到端流程：
本地导出权重 → NDK 交叉编译 → adb push → 真机 decode → 拉回 profile。

## 0. 准备

```bash
adb devices                      # 确认设备连接，已开开发者模式
export ANDROID_NDK=/path/to/ndk  # r25+ 测试通过即可
```

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
# 正确性门禁（golden token 对照）
./scripts/verify_android.sh

# 快速单遍测速
./scripts/bench_android.sh <label> [--extra-args "..."]

# 正式记录：门禁 + 3 遍测速 + 自动写 optimization_log.md
./scripts/record_android.sh <label> [--skip-verify] [--extra-args "..."]

# 建立 Android 基线
./scripts/set_baseline_android.sh <label>

# 对比两次 profile（定位 op 级变化）
python tools/profile_diff.py before.json after.json
```

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

### 回归检测

bench 完成后自动对比 `benchmarks/baseline_android.json`：
- 本次中位数比基线慢超过 5%（可调 `--regression-threshold`）时打印警告
- `--fail-on-regression`：回归时非零退出（CI 集成用）
- `--no-check-regression`：跳过

### A/B 同场对照

`--extra-args` 非空时自动启用——每遍先测对照（ref）再测变体，
交错抗热降频漂移。方法论与本地一致，见 `optimization.md` §6.3。

基线独立于本地：`benchmarks/baseline_android.json`（设备和 host 数字不可比）。

## 7. 性能解读注意

v1 是单线程 FP32 reference，真机上会很慢——这是预期行为。
profiling 的目的是建立 op 级基线和占比结构，为后续 INT4/KronQ kernel、
多线程策略和 spec decode 提供对照，而不是追求 v1 的绝对时延。
真机是优化效果的终极裁判；测量方法见 `optimization.md`。
