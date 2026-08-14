#!/usr/bin/env bash
# Android 正确性门禁：在设备上跑 canonical prompt，验证 golden token 逐位一致。
#
#   ./scripts/verify_android.sh
#
# 前置：adb 已连接设备、scripts/build_android.sh 已编译、model.tqwen 已导出。
# 做两件事，任一失败就非零退出：
#   1. 确保 binary + model 已 push 到设备
#   2. golden token 对照：canonical prompt 跑真模型，生成 token 必须与
#      参考实现逐位一致（fp32 double 累加，确定性结果）
set -euo pipefail
cd "$(dirname "$0")/.."

BUILD_DIR="${BUILD_DIR:-build-android}"
BIN="$BUILD_DIR/runtime/tinyqwen"
MODEL="${MODEL:-model.tqwen}"
DEVICE_DIR="${DEVICE_DIR:-/data/local/tmp/tinyqwen}"

GOLDEN="generated_ids: 2130 198 32 13 94305 245 46553 198 33 13 64118 55135 198 34 13 66521"

if [[ ! -f "$BIN" ]]; then
  echo "[verify-android] binary not found, building..."
  ./scripts/build_android.sh
fi

if [[ ! -f "$MODEL" ]]; then
  echo "error: model file not found: $MODEL（先用 tools/export_qwen_to_tiny.py 导出）" >&2
  exit 1
fi

echo "[1/2] Push binary & model to device..."
adb shell "mkdir -p $DEVICE_DIR"
adb push "$BIN" "$DEVICE_DIR/tinyqwen" >/dev/null
adb shell "chmod +x $DEVICE_DIR/tinyqwen"

LOCAL_SIZE=$(stat -f%z "$MODEL" 2>/dev/null || stat -c%s "$MODEL")
REMOTE_SIZE=$(adb shell "stat -c%s $DEVICE_DIR/model.tqwen 2>/dev/null || echo 0" | tr -d '\r')
if [[ "$LOCAL_SIZE" != "$REMOTE_SIZE" ]]; then
  echo "  pushing model ($LOCAL_SIZE bytes)..."
  adb push "$MODEL" "$DEVICE_DIR/model.tqwen"
else
  echo "  model already on device, skipping push"
fi

echo "[2/2] Golden token 对照..."
ACTUAL=$(adb shell "cd $DEVICE_DIR && ./tinyqwen \
  --model $DEVICE_DIR/model.tqwen \
  --tokens 105538,59975,100132 \
  --max-new-tokens 16 --max-seq-len 32 --eos -1 2>/dev/null" | grep '^generated_ids:' | tr -d '\r')

if [[ "$ACTUAL" == "$GOLDEN" ]]; then
  echo "PASS  生成 token 与参考实现逐位一致（Android）"
else
  echo "FAIL  生成 token 与参考不一致！"
  echo "  期望: $GOLDEN"
  echo "  实际: $ACTUAL"
  echo "  （若本次改动有意改变数值，如量化，请人工确认后更新本脚本的 GOLDEN）"
  exit 1
fi
