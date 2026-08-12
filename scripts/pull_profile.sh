#!/usr/bin/env bash
# 拉回 run_android.sh 运行产生的 profile JSON。
#
#   ./scripts/pull_profile.sh [local_out.json]
set -euo pipefail

DEVICE_DIR="${DEVICE_DIR:-/data/local/tmp/tinyqwen}"
OUT="${1:-profile_android.json}"

adb pull "$DEVICE_DIR/profile.json" "$OUT"
echo "saved -> $OUT"
