#!/usr/bin/env bash
# 建立 / 更新 Android 设备基线：稳定跑 3 遍，写进 benchmarks/baseline_android.json。
# 之后 record_android.sh 会自动用它算"vs 基线加速比"。
#
#   ./scripts/set_baseline_android.sh <label>
#   例：./scripts/set_baseline_android.sh android-fp32-baseline
set -euo pipefail
cd "$(dirname "$0")/.."

LABEL="${1:?usage: set_baseline_android.sh <label>}"
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
    --binary "$BIN" --model "$MODEL" --json benchmarks/.baseline_android_full.json

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
json.dump(out, open("benchmarks/baseline_android.json", "w"), ensure_ascii=False, indent=2)
print("[baseline-android] benchmarks/baseline_android.json =", json.dumps(out, ensure_ascii=False))
PYEOF

rm -f benchmarks/.baseline_android_full.json
