#!/usr/bin/env bash
# 一键跑 Android 设备标准基准测试。
#
#   ./scripts/bench_android.sh <label> [--extra-args "..."]
#   例：./scripts/bench_android.sh android-baseline
#       ./scripts/bench_android.sh neon-android --extra-args "--matvec-impl neon"
#
# label 是这次测量的名字。额外参数原样透传给设备端 runtime。
# 测量方法论与纪律：docs/optimization.md §6。
set -euo pipefail
cd "$(dirname "$0")/.."

LABEL="${1:?usage: bench_android.sh <label>   例如 bench_android.sh android-baseline}"
shift
MODEL="${MODEL:-model.tqwen}"

BUILD_DIR="${BUILD_DIR:-build-android}"
BIN="$BUILD_DIR/runtime/tinyqwen"

if [[ ! -f "$BIN" ]]; then
  echo "error: 找不到 $BIN（先 scripts/build_android.sh）" >&2
  exit 1
fi
if [[ ! -f "$MODEL" ]]; then
  echo "error: 找不到模型 $MODEL（先用 tools/export_qwen_to_tiny.py 导出）" >&2
  exit 1
fi

python3 tools/bench_android.py --model "$MODEL" --binary "$BIN" --label "$LABEL" "$@"
