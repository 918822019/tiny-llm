#!/usr/bin/env bash
# Android 正确性门禁：在设备上跑 canonical prompt，验证生成 token 符合预期。
#
#   ./scripts/verify_android.sh                      # fp32/fp16：逐位对 golden
#   MODEL=model_i4.tqwen ./scripts/verify_android.sh # 量化：容差对照
#
# 前置：adb 已连接设备、scripts/build_android.sh 已编译、模型已导出。
# 做两件事，任一失败就非零退出：
#   1. 确保 binary + model 已 push 到设备
#   2. token 对照。分两种口径：
#      - 非量化模型：与参考实现逐位一致（fp32 double 累加，确定性结果）
#      - 量化模型（--quantized，或文件名含 _i4）：量化必然改变数值，逐位
#        一致是错误的期望。改为允许至多 MAX_TOKEN_DIFF 个 token 不同，
#        与本地 scripts/verify_i4.sh 同一口径。
set -euo pipefail
cd "$(dirname "$0")/.."

BUILD_DIR="${BUILD_DIR:-build-android}"
BIN="$BUILD_DIR/runtime/tinyqwen"
MODEL="${MODEL:-model.tqwen}"
DEVICE_DIR="${DEVICE_DIR:-/data/local/tmp/tinyqwen}"
MAX_TOKEN_DIFF="${MAX_TOKEN_DIFF:-2}"

GOLDEN_IDS="2130 198 32 13 94305 245 46553 198 33 13 64118 55135 198 34 13 66521"

# 量化模型走容差口径：显式 --quantized 或文件名含 _i4 都算。
QUANTIZED=0
for a in "$@"; do
  [[ "$a" == "--quantized" ]] && QUANTIZED=1
done
[[ "$(basename "$MODEL")" == *_i4* ]] && QUANTIZED=1

if [[ ! -f "$BIN" ]]; then
  echo "[verify-android] binary not found, building..."
  ./scripts/build_android.sh
fi

if [[ ! -f "$MODEL" ]]; then
  echo "error: model file not found: ${MODEL}（先用 tools/export_qwen_to_tiny.py 导出）" >&2
  exit 1
fi

# 按文件名分开存，fp32/fp16/i4 可共存，切 dtype 不用重传整个模型。
DEVICE_MODEL="$DEVICE_DIR/models/$(basename "$MODEL")"

echo "[1/2] Push binary & model to device..."
adb shell "mkdir -p $DEVICE_DIR/models"
adb push "$BIN" "$DEVICE_DIR/tinyqwen" >/dev/null
adb shell "chmod +x $DEVICE_DIR/tinyqwen"

LOCAL_SIZE=$(stat -f%z "$MODEL" 2>/dev/null || stat -c%s "$MODEL")
REMOTE_SIZE=$(adb shell "stat -c%s $DEVICE_MODEL 2>/dev/null || echo 0" | tr -d '\r')
if [[ "$LOCAL_SIZE" != "$REMOTE_SIZE" ]]; then
  echo "  pushing $(basename "$MODEL") ($LOCAL_SIZE bytes)..."
  adb push "$MODEL" "$DEVICE_MODEL"
else
  echo "  $(basename "$MODEL") already on device, skipping push"
fi

echo "[2/2] Token 对照（$(basename "$MODEL")）..."
ACTUAL_IDS=$(adb shell "cd $DEVICE_DIR && ./tinyqwen \
  --model $DEVICE_MODEL \
  --tokens 105538,59975,100132 \
  --max-new-tokens 16 --max-seq-len 32 --eos -1 2>/dev/null" \
  | grep '^generated_ids:' | tr -d '\r' | sed 's/generated_ids: //')

if [[ -z "$ACTUAL_IDS" ]]; then
  echo "FAIL  设备端没有产出 generated_ids（运行失败？）" >&2
  exit 1
fi

if [[ "$QUANTIZED" -eq 0 ]]; then
  if [[ "$ACTUAL_IDS" == "$GOLDEN_IDS" ]]; then
    echo "PASS  生成 token 与参考实现逐位一致（Android）"
  else
    echo "FAIL  生成 token 与参考不一致！"
    echo "  期望: $GOLDEN_IDS"
    echo "  实际: $ACTUAL_IDS"
    echo "  （若本次改动有意改变数值，如量化，用 MODEL=model_i4.tqwen 或 --quantized）"
    exit 1
  fi
else
  IFS=' ' read -ra GOLDEN_ARR <<< "$GOLDEN_IDS"
  IFS=' ' read -ra ACTUAL_ARR <<< "$ACTUAL_IDS"
  DIFF_COUNT=0
  for i in "${!GOLDEN_ARR[@]}"; do
    if [[ "${GOLDEN_ARR[$i]:-}" != "${ACTUAL_ARR[$i]:-}" ]]; then
      ((DIFF_COUNT++)) || true
      echo "  pos $i: golden=${GOLDEN_ARR[$i]:-?} vs 设备=${ACTUAL_ARR[$i]:-?}"
    fi
  done
  if [[ "$DIFF_COUNT" -le "$MAX_TOKEN_DIFF" ]]; then
    echo "PASS  量化容差口径：token diff $DIFF_COUNT/${#GOLDEN_ARR[@]}（上限 ${MAX_TOKEN_DIFF}）"
  else
    echo "FAIL  量化容差口径：token diff $DIFF_COUNT/${#GOLDEN_ARR[@]}（上限 ${MAX_TOKEN_DIFF}）"
    echo "  golden: $GOLDEN_IDS"
    echo "  设备:   $ACTUAL_IDS"
    exit 1
  fi
fi
