#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-$ROOT/build}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
BIN="$BUILD_DIR/runtime/tinyqwen"

if [[ ! -x "$BIN" ]]; then
  echo "error: missing $BIN (configure and build the project first)" >&2
  exit 2
fi

TMP_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/tinyqwen-spec.XXXXXX")"
trap 'rm -rf "$TMP_ROOT"' EXIT

"$PYTHON_BIN" "$ROOT/tools/make_fake_model.py" \
  --out "$TMP_ROOT/target.tqwen" --seed 0 --arch qwen3 >/dev/null
"$PYTHON_BIN" "$ROOT/tools/make_fake_model.py" \
  --out "$TMP_ROOT/draft.tqwen" --seed 1 --arch qwen3 >/dev/null

COMMON=(--model "$TMP_ROOT/target.tqwen" --tokens 3,7,11,2 \
        --max-new-tokens 24 --max-seq-len 32)
"$BIN" "${COMMON[@]}" >"$TMP_ROOT/baseline.out" 2>"$TMP_ROOT/baseline.err"
"$BIN" "${COMMON[@]}" --draft-model "$TMP_ROOT/target.tqwen" \
  --speculative-tokens 4 --speculative-stats-out "$TMP_ROOT/all.json" \
  >"$TMP_ROOT/all.out" 2>"$TMP_ROOT/all.err"
"$BIN" "${COMMON[@]}" --draft-model "$TMP_ROOT/draft.tqwen" \
  --speculative-tokens 4 --speculative-stats-out "$TMP_ROOT/reject.json" \
  >"$TMP_ROOT/reject.out" 2>"$TMP_ROOT/reject.err"

baseline="$(grep '^generated_ids:' "$TMP_ROOT/baseline.out")"
all_accept="$(grep '^generated_ids:' "$TMP_ROOT/all.out")"
with_reject="$(grep '^generated_ids:' "$TMP_ROOT/reject.out")"
[[ "$baseline" == "$all_accept" ]]
[[ "$baseline" == "$with_reject" ]]

"$PYTHON_BIN" - "$TMP_ROOT/all.json" "$TMP_ROOT/reject.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as f:
    all_accept = json.load(f)
with open(sys.argv[2], encoding="utf-8") as f:
    rejected = json.load(f)

assert all_accept["draft_proposed"] > 0
assert all_accept["draft_accepted"] == all_accept["draft_proposed"]
assert all_accept["bonus_tokens"] > 0
assert rejected["corrections"] > 0
assert rejected["rollbacks"] > 0
PY

echo "speculative all-accept output: PASS"
echo "speculative rejection rollback output: PASS"
