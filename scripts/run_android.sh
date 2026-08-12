#!/usr/bin/env bash
# 把 tinyqwen binary + 模型 push 到设备并运行 decode。
#
#   ./scripts/run_android.sh model.tqwen prompt_tokens.json [tinyqwen 的额外参数...]
#
# 前置：PATH 里有 adb、已连接一台设备、scripts/build_android.sh 已编译。
set -euo pipefail

cd "$(dirname "$0")/.."

MODEL="${1:?usage: run_android.sh <model.tqwen> <tokens.json> [tinyqwen args...]}"
TOKENS="${2:?usage: run_android.sh <model.tqwen> <tokens.json> [tinyqwen args...]}"
shift 2

BUILD_DIR="${BUILD_DIR:-build-android}"
DEVICE_DIR="${DEVICE_DIR:-/data/local/tmp/tinyqwen}"
BIN="$BUILD_DIR/runtime/tinyqwen"

[[ -f "$BIN" ]] || { echo "error: $BIN not found; run scripts/build_android.sh first" >&2; exit 1; }
[[ -f "$MODEL" ]] || { echo "error: model file not found: $MODEL" >&2; exit 1; }
[[ -f "$TOKENS" ]] || { echo "error: tokens file not found: $TOKENS" >&2; exit 1; }

adb shell "mkdir -p $DEVICE_DIR"
adb push "$BIN" "$DEVICE_DIR/tinyqwen" >/dev/null
adb shell "chmod +x $DEVICE_DIR/tinyqwen"

# 仅当设备上缺少模型或大小不一致时才重新 push（模型约 2GB，避免重复传输）。
LOCAL_SIZE=$(stat -f%z "$MODEL" 2>/dev/null || stat -c%s "$MODEL")
REMOTE_SIZE=$(adb shell "stat -c%s $DEVICE_DIR/model.tqwen 2>/dev/null || echo 0" | tr -d '\r')
if [[ "$LOCAL_SIZE" != "$REMOTE_SIZE" ]]; then
  echo "pushing model ($LOCAL_SIZE bytes)..."
  adb push "$MODEL" "$DEVICE_DIR/model.tqwen"
else
  echo "model already on device ($REMOTE_SIZE bytes), skipping push"
fi

adb push "$TOKENS" "$DEVICE_DIR/tokens.json" >/dev/null

echo "--- run ---"
adb shell "cd $DEVICE_DIR && ./tinyqwen \
  --model $DEVICE_DIR/model.tqwen \
  --tokens-json $DEVICE_DIR/tokens.json \
  --profile-out $DEVICE_DIR/profile.json \
  $*"
