#!/usr/bin/env bash
# ============================================================================
# set_baseline_android.sh — 建立 / 更新 Android 设备性能基线
# ============================================================================
# 用途：在 Android 设备上稳定跑 3 遍基准测试，将结果写入
#       benchmarks/baseline_android.json。之后 record_android.sh 会自动读取
#       此基线，计算"vs 基线加速比"。
#
# 用法：
#   ./scripts/set_baseline_android.sh <label> [透传给 bench_android.py 的参数...]
#
# 示例：
#   ./scripts/set_baseline_android.sh android-fp32-baseline
#   # fp16 满栈基线：基线必须带上被基线化配置的 extra-args
#   MODEL=model_f16.tqwen ./scripts/set_baseline_android.sh android-fp16-baseline \
#       --extra-args "--matvec-impl neon_mt_kv_nt --ops-impl neon"
#
# 环境变量：
#   MODEL     — 模型文件路径（默认 model.tqwen）
#   BUILD_DIR — 交叉编译产物目录（默认 build-android）
#
# 产物：
#   benchmarks/baseline_android.json — 精简后的 Android 基线数据
#     包含：label、decode_median_ms、decode_p95_ms、commit、device_model、soc
#     可选：prefill_tokens、generated_tokens、ttft_ms、total_ms、peak_rss_mb、kv_cache_mb
#
# 前置条件：
#   - adb 可用且设备已连接
#   - scripts/build_android.sh 已完成交叉编译
#   - 模型已导出
# ============================================================================

# set -e: 任何命令失败立即退出；-u: 引用未定义变量报错；-o pipefail: 管道中任一命令失败则整体失败
set -euo pipefail

# 切换到项目根目录
cd "$(dirname "$0")/.."

# 第一个参数为 label，缺失时打印用法并退出
LABEL="${1:?usage: set_baseline_android.sh <label> [bench_android.py 参数...]}"
# shift 移除已消费的 label，剩余参数透传给 bench_android.py
shift

# 确保 benchmarks 目录存在
mkdir -p benchmarks

# 交叉编译产物目录
BUILD_DIR="${BUILD_DIR:-build-android}"
# 本地 binary 路径
BIN="$BUILD_DIR/runtime/tinyqwen"
# 模型文件路径
MODEL="${MODEL:-model.tqwen}"

# 检查 binary 是否存在
if [[ ! -f "$BIN" ]]; then
  echo "error: 找不到 ${BIN}（先 scripts/build_android.sh）" >&2
  exit 1
fi

# 检查模型文件是否存在
if [[ ! -f "$MODEL" ]]; then
  echo "error: 找不到模型 $MODEL" >&2
  exit 1
fi

# 调用 bench_android.py 跑 3 遍取中位，输出完整 JSON 到临时文件
# ${@+"$@"} 安全展开剩余参数（无参数时不报错）
python3 tools/bench_android.py --label "$LABEL" --runs 3 \
    --binary "$BIN" --model "$MODEL" ${@+"$@"} \
    --json benchmarks/.baseline_android_full.json

# ---------------------------------------------------------------------------
# 用内联 Python 脚本从完整结果中提取关键字段，生成精简的 baseline_android.json
# ---------------------------------------------------------------------------
python3 - "$LABEL" <<'PYEOF'
import json, sys
# 从命令行获取 label 参数
label = sys.argv[1]
# 读取完整测速结果
d = json.load(open("benchmarks/.baseline_android_full.json"))
# 构造精简的基线对象
out = {
    "label": label,                                          # 基线标签
    "decode_median_ms": round(d["decode_median_ms"], 2),     # decode 中位耗时（ms/token）
    "decode_p95_ms": round(d["decode_p95_ms"], 2),           # decode P95 耗时
    "commit": d["host_env"]["git_commit"],                   # 宿主机 git commit hash
    "device_model": d["device_env"].get("device_model", "unknown"),  # 设备型号
    "soc": d["device_env"].get("soc", "unknown"),            # SoC 型号
    "note": "Android 端侧基线；换基线时重跑 scripts/set_baseline_android.sh",
}

# 阶段指标参照：prefill/decode 长度 + TTFT + 总耗时
# 旧格式可能缺少这些字段，缺省时跳过
for k in ("prefill_tokens", "generated_tokens", "ttft_ms", "total_ms"):
    if d.get(k) is not None:
        out[k] = d[k]

# 内存口径参照：基线配置的常驻内存足迹
# 资源采样关闭/失败时 resources 或 mem 可能缺省
# 换 dtype / 改 KV cache 结构后重基线，这里的数字变化应能被解释
mem = (d.get("resources") or {}).get("mem") or {}
if mem.get("peak_rss_mb") is not None:
    out["peak_rss_mb"] = mem["peak_rss_mb"]          # 峰值 RSS（MB）
    if mem.get("kv_cache_mb") is not None:
        out["kv_cache_mb"] = mem["kv_cache_mb"]      # KV cache 占用（MB）

# 写入精简基线文件
json.dump(out, open("benchmarks/baseline_android.json", "w"), ensure_ascii=False, indent=2)
print("[baseline-android] benchmarks/baseline_android.json =", json.dumps(out, ensure_ascii=False))
PYEOF

# 清理临时文件
rm -f benchmarks/.baseline_android_full.json
