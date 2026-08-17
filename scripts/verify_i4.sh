#!/usr/bin/env bash
# INT4 quantization correctness gate.
#
#   ./scripts/verify_i4.sh [model_i4.tqwen] [model.tqwen]
#
# Steps:
#   1. Build (Release)
#   2. Run all unit tests (including INT4 kernel tests)
#   3. Token comparison: fp32 vs I4 on same prompt, allow <=2 token diffs
#   4. Accuracy audit via verify_i4_accuracy.py
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL_I4="${1:-model_i4.tqwen}"
MODEL_FP32="${2:-model.tqwen}"
BIN="build/runtime/tinyqwen"
PROMPT_TOKENS="105538,59975,100132"
MAX_NEW=16
MAX_SEQ=32

echo "[1/4] Build..."
cmake --build build -j"$(sysctl -n hw.ncpu 2>/dev/null || nproc 2>/dev/null || echo 4)" >/dev/null

echo "[2/4] Unit tests..."
./build/tests/tinyqwen_tests | tail -1

if [[ ! -f "$MODEL_I4" ]]; then
  echo "[3/4] SKIP token comparison ($MODEL_I4 not found)"
  echo "  Hint: run python tools/export_qwen_to_tiny_i4.py --out $MODEL_I4"
  exit 0
fi

echo "[3/4] Token comparison (fp32 vs INT4)..."
COMMON_ARGS="--tokens $PROMPT_TOKENS --max-new-tokens $MAX_NEW --max-seq-len $MAX_SEQ --eos -1"

if [[ -f "$MODEL_FP32" ]]; then
  FP32_IDS=$("$BIN" --model "$MODEL_FP32" $COMMON_ARGS 2>/dev/null | grep '^generated_ids:' | sed 's/generated_ids: //')
else
  FP32_IDS="2130 198 32 13 94305 245 46553 198 33 13 64118 55135 198 34 13 66521"
fi

I4_IDS=$("$BIN" --model "$MODEL_I4" $COMMON_ARGS 2>/dev/null | grep '^generated_ids:' | sed 's/generated_ids: //')

IFS=' ' read -ra FP32_ARR <<< "$FP32_IDS"
IFS=' ' read -ra I4_ARR <<< "$I4_IDS"

DIFF_COUNT=0
MAX_DIFF=2
for i in "${!FP32_ARR[@]}"; do
  if [[ "${FP32_ARR[$i]:-}" != "${I4_ARR[$i]:-}" ]]; then
    ((DIFF_COUNT++)) || true
    echo "  pos $i: fp32=${FP32_ARR[$i]:-?} vs i4=${I4_ARR[$i]:-?}"
  fi
done

if [[ $DIFF_COUNT -le $MAX_DIFF ]]; then
  echo "PASS  token diff $DIFF_COUNT/$MAX_NEW (max allowed: $MAX_DIFF)"
else
  echo "FAIL  token diff $DIFF_COUNT/$MAX_NEW (max allowed: $MAX_DIFF)"
  echo "  fp32: $FP32_IDS"
  echo "  i4:   $I4_IDS"
  exit 1
fi

echo "[4/4] Accuracy audit..."
ACCURACY_SCRIPT="tools/verify_i4_accuracy.py"
if [[ -f "$ACCURACY_SCRIPT" && -f "$MODEL_FP32" ]]; then
  python3 "$ACCURACY_SCRIPT" --model-i4 "$MODEL_I4" --model-fp32 "$MODEL_FP32"
else
  echo "  SKIP ($ACCURACY_SCRIPT or $MODEL_FP32 not found)"
fi

echo ""
echo "=== INT4 verification complete ==="
