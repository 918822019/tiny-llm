#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-$ROOT/build}"
BIN="$BUILD_DIR/runtime/tinyqwen"

if [[ $# -lt 3 || $# -gt 5 ]]; then
  echo "usage: $0 <target.tqwen> <eagle3.tqwen> <tokens.json> [verify-width=2] [max-new=32]" >&2
  exit 2
fi
if [[ ! -x "$BIN" ]]; then
  echo "error: missing $BIN (configure and build the project first)" >&2
  exit 2
fi

TARGET_MODEL="$1"
EAGLE_MODEL="$2"
TOKENS_JSON="$3"
VERIFY_WIDTH="${4:-2}"
MAX_NEW="${5:-32}"
for path in "$TARGET_MODEL" "$EAGLE_MODEL" "$TOKENS_JSON"; do
  if [[ ! -f "$path" ]]; then
    echo "error: missing input: $path" >&2
    exit 2
  fi
done
if (( VERIFY_WIDTH < 2 || MAX_NEW < 2 )); then
  echo "error: verify width and max-new must both be >= 2" >&2
  exit 2
fi

TMP_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/tinyqwen-eagle3.XXXXXX")"
trap 'rm -rf "$TMP_ROOT"' EXIT

COMMON=(--model "$TARGET_MODEL" --tokens-json "$TOKENS_JSON"
        --max-new-tokens "$MAX_NEW" --max-seq-len 256 --eos -1)
if [[ "$(uname -m)" == "arm64" || "$(uname -m)" == "aarch64" ]]; then
  COMMON+=(--matvec-impl neon_mt_kv_nt --ops-impl neon --kv-f16)
fi

"$BIN" "${COMMON[@]}" >"$TMP_ROOT/greedy.out" 2>"$TMP_ROOT/greedy.err"
"$BIN" "${COMMON[@]}" --eagle3-model "$EAGLE_MODEL" \
  --speculative-tokens "$VERIFY_WIDTH" \
  --speculative-stats-out "$TMP_ROOT/eagle3.json" \
  >"$TMP_ROOT/eagle3.out" 2>"$TMP_ROOT/eagle3.err"
"$BIN" "${COMMON[@]}" --eagle3-model "$EAGLE_MODEL" \
  --speculative-tokens "$VERIFY_WIDTH" --no-eagle3-batch-verify \
  --speculative-stats-out "$TMP_ROOT/eagle3_sequential.json" \
  >"$TMP_ROOT/eagle3_sequential.out" 2>"$TMP_ROOT/eagle3_sequential.err"

greedy_ids="$(grep '^generated_ids:' "$TMP_ROOT/greedy.out")"
eagle3_ids="$(grep '^generated_ids:' "$TMP_ROOT/eagle3.out")"
sequential_ids="$(grep '^generated_ids:' "$TMP_ROOT/eagle3_sequential.out")"
if [[ -z "$greedy_ids" || "$greedy_ids" != "$eagle3_ids" ||
      "$greedy_ids" != "$sequential_ids" ]]; then
  echo "EAGLE3 exact-greedy output: FAIL" >&2
  diff -u "$TMP_ROOT/greedy.out" "$TMP_ROOT/eagle3.out" >&2 || true
  diff -u "$TMP_ROOT/greedy.out" "$TMP_ROOT/eagle3_sequential.out" >&2 || true
  exit 1
fi
grep -q '"draft_proposed": [1-9]' "$TMP_ROOT/eagle3.json"
grep -q '"target_verify_mode": "batched"' "$TMP_ROOT/eagle3.json"
grep -q '"target_verify_mode": "sequential"' "$TMP_ROOT/eagle3_sequential.json"
batch_calls="$(sed -n 's/.*"target_verify_calls": \([0-9][0-9]*\).*/\1/p' "$TMP_ROOT/eagle3.json")"
sequential_calls="$(sed -n 's/.*"target_verify_calls": \([0-9][0-9]*\).*/\1/p' "$TMP_ROOT/eagle3_sequential.json")"
if [[ -z "$batch_calls" || -z "$sequential_calls" ||
      "$sequential_calls" -le "$batch_calls" ]]; then
  echo "EAGLE3 verifier call-count ablation: FAIL" >&2
  exit 1
fi

echo "EAGLE3 batched/sequential exact-greedy output: PASS"
grep '^\[timing\]' "$TMP_ROOT/greedy.err"
grep '^\[speculative\]' "$TMP_ROOT/eagle3.err"
grep '^\[speculative\]' "$TMP_ROOT/eagle3_sequential.err"
