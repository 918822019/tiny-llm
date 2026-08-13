#!/usr/bin/env bash
# 建立 / 更新基线（baseline）：稳定跑 3 遍，写进 benchmarks/baseline.json。
# 之后 record.sh 会自动用它算"vs 基线加速比"。
#
#   ./scripts/set_baseline.sh fp32-baseline
set -euo pipefail
cd "$(dirname "$0")/.."

LABEL="${1:?usage: set_baseline.sh <label>}"
mkdir -p benchmarks

python3 tools/bench.py --label "$LABEL" --runs 3 --json benchmarks/.baseline_full.json

python3 - "$LABEL" <<'PYEOF'
import json, sys
label = sys.argv[1]
d = json.load(open("benchmarks/.baseline_full.json"))
out = {
    "label": label,
    "decode_median_ms": round(d["decode_median_ms"], 2),
    "decode_p95_ms": round(d["decode_p95_ms"], 2),
    "commit": d["env"]["git_commit"],
    "note": "fp32 参考基线；换基线时重跑 scripts/set_baseline.sh",
}
json.dump(out, open("benchmarks/baseline.json", "w"), ensure_ascii=False, indent=2)
print("[baseline] benchmarks/baseline.json =", out)
PYEOF

rm -f benchmarks/.baseline_full.json
