#!/usr/bin/env bash
# 建立 / 更新 Android 设备基线：稳定跑 3 遍，写进 benchmarks/baseline_android.json。
# 之后 record_android.sh 会自动用它算"vs 基线加速比"。
#
#   ./scripts/set_baseline_android.sh <label> [透传给 bench_android.py 的参数...]
#   例：./scripts/set_baseline_android.sh android-fp32-baseline
#       # fp16 满栈基线：基线必须带上被基线化配置的 extra-args
#       MODEL=model_f16.tqwen ./scripts/set_baseline_android.sh android-fp16-baseline \
#           --extra-args "--matvec-impl neon_mt_kv_nt --ops-impl neon"
set -euo pipefail
cd "$(dirname "$0")/.."

LABEL="${1:?usage: set_baseline_android.sh <label> [bench_android.py 参数...]}"
shift
mkdir -p benchmarks

BUILD_DIR="${BUILD_DIR:-build-android}"
BIN="$BUILD_DIR/runtime/tinyqwen"
MODEL="${MODEL:-model.tqwen}"

if [[ ! -f "$BIN" ]]; then
  echo "error: 找不到 ${BIN}（先 scripts/build_android.sh）" >&2
  exit 1
fi
if [[ ! -f "$MODEL" ]]; then
  echo "error: 找不到模型 $MODEL" >&2
  exit 1
fi

python3 tools/bench_android.py --label "$LABEL" --runs 3 \
    --binary "$BIN" --model "$MODEL" ${@+"$@"} \
    --json benchmarks/.baseline_android_full.json

python3 - "$LABEL" <<'PYEOF'
import json, sys
label = sys.argv[1]
d = json.load(open("benchmarks/.baseline_android_full.json"))
out = {
    "label": label,
    "decode_median_ms": round(d["decode_median_ms"], 2),
    "decode_p95_ms": round(d["decode_p95_ms"], 2),
    "commit": d["host_env"]["git_commit"],
    "device_model": d["device_env"].get("device_model", "unknown"),
    "soc": d["device_env"].get("soc", "unknown"),
    "note": "Android 端侧基线；换基线时重跑 scripts/set_baseline_android.sh",
}
# 阶段指标参照：prefill/decode 长度 + TTFT + 总耗时（旧格式缺省时跳过）
for k in ("prefill_tokens", "generated_tokens", "ttft_ms", "total_ms"):
    if d.get(k) is not None:
        out[k] = d[k]
# 内存口径参照（资源采样关闭/失败时缺省）：基线配置的常驻内存足迹，
# 换 dtype / 改 KV cache 结构后重基线，这里的数字变化应能被解释。
mem = (d.get("resources") or {}).get("mem") or {}
if mem.get("peak_rss_mb") is not None:
    out["peak_rss_mb"] = mem["peak_rss_mb"]
    if mem.get("kv_cache_mb") is not None:
        out["kv_cache_mb"] = mem["kv_cache_mb"]
json.dump(out, open("benchmarks/baseline_android.json", "w"), ensure_ascii=False, indent=2)
print("[baseline-android] benchmarks/baseline_android.json =", json.dumps(out, ensure_ascii=False))
PYEOF

rm -f benchmarks/.baseline_android_full.json
