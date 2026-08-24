#!/usr/bin/env bash
# ============================================================================
# measure_vq2.sh — 测量一个 .tqwen 模型：体积 + 解码速度 + 生成冒烟
# ============================================================================
# 用法:
#   ./scripts/measure_vq2.sh <model.tqwen> [label] [matvec_impl] [prompt_tokens]
# 例:
#   ./scripts/measure_vq2.sh model_qwen25_05b_vq2.tqwen vq2 neon
#   ./scripts/measure_vq2.sh model_qwen35_f16.tqwen f16 ""          # 默认实现
# ============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL="${1:?usage: measure_vq2.sh <model.tqwen> [label] [matvec_impl] [prompt_tokens]}"
LABEL="${2:-$(basename "$MODEL" .tqwen)}"
IMPL="${3:-}"
PROMPT="${4:-105538,59975,100132}"
BIN="${BIN:-build/runtime/tinyqwen}"
PY="${PY:-.venv/bin/python}"
DECODE_TOKENS="${DECODE_TOKENS:-32}"
MAX_NEW="${MAX_NEW:-24}"

echo "════ 模型: $MODEL  (label=$LABEL impl=${IMPL:-default}) ════"

# ---- 1. 体积 ----
if [[ -f "$MODEL" ]]; then
    bytes=$(stat -f%z "$MODEL" 2>/dev/null || stat -c%s "$MODEL")
    echo "[size] $bytes bytes  ($(echo "scale=1; $bytes/1048576" | bc) MB)"
else
    echo "[size] 文件不存在: $MODEL"
fi

# ---- 2. 解码速度（bench.py 标准负载） ----
echo ""
echo "[speed] decode=$DECODE_TOKENS tok"
EXTRA=""
[[ -n "$IMPL" ]] && EXTRA="--matvec-impl $IMPL"
"$PY" tools/bench.py --binary "$BIN" --model "$MODEL" --label "$LABEL" \
    --extra-args "$EXTRA" --no-check-regression 2>&1 | \
    grep -iE "TTFT|TOPT|error" || true

# ---- 3. 生成冒烟（定性） ----
echo ""
echo "[gen] prompt=$PROMPT  max_new=$MAX_NEW"
EXTRA2=""
[[ -n "$IMPL" ]] && EXTRA2="--matvec-impl $IMPL"
# shellcheck disable=SC2086
"$BIN" --model "$MODEL" --tokens "$PROMPT" --max-new-tokens "$MAX_NEW" \
    --eos -1 $EXTRA2 2>/dev/null | grep "generated_ids:" || echo "(无生成输出)"
