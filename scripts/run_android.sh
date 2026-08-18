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
# 按文件名分开存，fp32/fp16/i4 可共存，切 dtype 不用重传整个模型。
DEVICE_MODEL="$DEVICE_DIR/models/$(basename "$MODEL")"

[[ -f "$BIN" ]] || { echo "error: $BIN not found; run scripts/build_android.sh first" >&2; exit 1; }
[[ -f "$MODEL" ]] || { echo "error: model file not found: $MODEL" >&2; exit 1; }
[[ -f "$TOKENS" ]] || { echo "error: tokens file not found: $TOKENS" >&2; exit 1; }

adb shell "mkdir -p $DEVICE_DIR/models"
adb push "$BIN" "$DEVICE_DIR/tinyqwen" >/dev/null
adb shell "chmod +x $DEVICE_DIR/tinyqwen"

# 仅当设备上缺少该模型或大小不一致时才重新 push（模型可达 2GB，避免重复传输）。
LOCAL_SIZE=$(stat -f%z "$MODEL" 2>/dev/null || stat -c%s "$MODEL")
REMOTE_SIZE=$(adb shell "stat -c%s $DEVICE_MODEL 2>/dev/null || echo 0" | tr -d '\r')
if [[ "$LOCAL_SIZE" != "$REMOTE_SIZE" ]]; then
  echo "pushing $(basename "$MODEL") ($LOCAL_SIZE bytes)..."
  adb push "$MODEL" "$DEVICE_MODEL"
else
  echo "$(basename "$MODEL") already on device ($REMOTE_SIZE bytes), skipping push"
fi

adb push "$TOKENS" "$DEVICE_DIR/tokens.json" >/dev/null

echo "--- run ---"
adb shell "cd $DEVICE_DIR && ./tinyqwen \
  --model $DEVICE_MODEL \
  --tokens-json $DEVICE_DIR/tokens.json \
  --profile-out $DEVICE_DIR/profile.json \
  $*"
