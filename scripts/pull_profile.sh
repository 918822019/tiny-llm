#!/usr/bin/env bash
# Pull the profile JSON produced by run_android.sh.
#
#   ./scripts/pull_profile.sh [local_out.json]
set -euo pipefail

DEVICE_DIR="${DEVICE_DIR:-/data/local/tmp/tinyqwen}"
OUT="${1:-profile_android.json}"

adb pull "$DEVICE_DIR/profile.json" "$OUT"
echo "saved -> $OUT"
