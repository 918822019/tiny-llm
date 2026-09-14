#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ADB_BIN="${ADB_BIN:-$(command -v adb 2>/dev/null || true)}"
if [[ -z "$ADB_BIN" && -x /opt/homebrew/bin/adb ]]; then
  ADB_BIN=/opt/homebrew/bin/adb
fi

if [[ $# -lt 3 || $# -gt 6 ]]; then
  echo "usage: $0 <target.tqwen> <eagle3.tqwen> <tokens.json> [repeats=3] [max-new=64] [verify-width=2]" >&2
  exit 2
fi
if [[ -z "$ADB_BIN" || ! -x "$ADB_BIN" ]]; then
  echo "error: adb was not found; set ADB_BIN explicitly" >&2
  exit 2
fi

TARGET_MODEL="$1"
EAGLE_MODEL="$2"
TOKENS_JSON="$3"
REPEATS="${4:-3}"
MAX_NEW="${5:-64}"
VERIFY_WIDTH="${6:-2}"
BIN="${BIN:-$ROOT/build-android-vulkan/runtime/tinyqwen}"
REMOTE_ROOT="${REMOTE_ROOT:-/data/local/tmp/tinyqwen}"
STAMP="$(date +%Y%m%d-%H%M%S)"
REMOTE_RUN="$REMOTE_ROOT/eagle3-ablation-$STAMP"
OUT_DIR="${OUT_DIR:-$ROOT/artifacts/eagle3-ablation-$STAMP}"

for path in "$TARGET_MODEL" "$EAGLE_MODEL" "$TOKENS_JSON" "$BIN"; do
  if [[ ! -f "$path" ]]; then
    echo "error: missing input: $path" >&2
    exit 2
  fi
done
if (( REPEATS < 1 || MAX_NEW < 2 || VERIFY_WIDTH < 2 )); then
  echo "error: repeats must be >= 1; max-new and verify-width must be >= 2" >&2
  exit 2
fi

mkdir -p "$OUT_DIR"
"$ADB_BIN" get-state >/dev/null
"$ADB_BIN" shell "mkdir -p '$REMOTE_ROOT/models' '$REMOTE_RUN'"

host_size() {
  stat -f '%z' "$1" 2>/dev/null || stat -c '%s' "$1"
}

ensure_remote_file() {
  local source="$1"
  local destination="$2"
  local expected actual
  expected="$(host_size "$source")"
  actual="$("$ADB_BIN" shell "stat -c '%s' '$destination' 2>/dev/null" | tr -d '\r' || true)"
  if [[ "$actual" != "$expected" ]]; then
    "$ADB_BIN" push "$source" "$destination" >/dev/null
  fi
}

REMOTE_BIN="$REMOTE_ROOT/tinyqwen_ablation"
TARGET_NAME="$(basename "$TARGET_MODEL")"
EAGLE_NAME="$(basename "$EAGLE_MODEL")"
TOKENS_NAME="$(basename "$TOKENS_JSON")"
for name in "$TARGET_NAME" "$EAGLE_NAME" "$TOKENS_NAME"; do
  if [[ ! "$name" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "error: input basenames may contain only letters, digits, '.', '_', and '-'" >&2
    exit 2
  fi
done
REMOTE_TARGET="$REMOTE_ROOT/models/$TARGET_NAME"
REMOTE_EAGLE="$REMOTE_ROOT/models/$EAGLE_NAME"
REMOTE_TOKENS="$REMOTE_RUN/$TOKENS_NAME"
"$ADB_BIN" push "$BIN" "$REMOTE_BIN" >/dev/null
ensure_remote_file "$TARGET_MODEL" "$REMOTE_TARGET"
ensure_remote_file "$EAGLE_MODEL" "$REMOTE_EAGLE"
ensure_remote_file "$TOKENS_JSON" "$REMOTE_TOKENS"
"$ADB_BIN" shell "chmod 755 '$REMOTE_BIN'"

run_case() {
  local label="$1"
  local repetition="$2"
  local warmup="${3:-false}"
  local run_max="$MAX_NEW"
  local suffix="${repetition}"
  if [[ "$warmup" == true ]]; then
    run_max=8
    suffix=warmup
  fi

  local extra=""
  case "$label" in
    cpu_greedy)
      ;;
    vulkan_greedy)
      extra="--backend vulkan"
      ;;
    vulkan_eagle_sequential)
      extra="--backend vulkan --eagle3-model '$REMOTE_EAGLE' --speculative-tokens '$VERIFY_WIDTH' --no-eagle3-batch-verify --speculative-stats-out '$REMOTE_RUN/${label}_${suffix}.json'"
      ;;
    vulkan_eagle_batched)
      extra="--backend vulkan --eagle3-model '$REMOTE_EAGLE' --speculative-tokens '$VERIFY_WIDTH' --speculative-stats-out '$REMOTE_RUN/${label}_${suffix}.json'"
      ;;
    *)
      echo "error: unknown case $label" >&2
      exit 2
      ;;
  esac

  echo "run $label $suffix"
  "$ADB_BIN" shell "cd '$REMOTE_ROOT' && TINYQWEN_MT_THREADS=6 '$REMOTE_BIN' \
    --model '$REMOTE_TARGET' --tokens-json '$REMOTE_TOKENS' \
    --max-new-tokens '$run_max' --max-seq-len 256 --eos -1 \
    --matvec-impl neon_mt_kv_nt --ops-impl neon --kv-f16 $extra \
    > '$REMOTE_RUN/${label}_${suffix}.out' \
    2> '$REMOTE_RUN/${label}_${suffix}.err'"
}

# Warm file cache, CPU kernels, Vulkan pipeline creation, and both verifier modes.
for label in cpu_greedy vulkan_greedy vulkan_eagle_sequential vulkan_eagle_batched; do
  run_case "$label" 0 true
done

# Reverse every second round so slow thermal drift does not always favor one mode.
for ((rep = 1; rep <= REPEATS; ++rep)); do
  if (( rep % 2 == 1 )); then
    order=(cpu_greedy vulkan_greedy vulkan_eagle_sequential vulkan_eagle_batched)
  else
    order=(vulkan_eagle_batched vulkan_eagle_sequential vulkan_greedy cpu_greedy)
  fi
  for label in "${order[@]}"; do
    run_case "$label" "$rep"
  done
done

"$ADB_BIN" pull "$REMOTE_RUN/." "$OUT_DIR" >/dev/null

PYTHON_BIN="${PYTHON_BIN:-$ROOT/.venv/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="$(command -v python3)"
fi
"$PYTHON_BIN" - "$OUT_DIR" "$REPEATS" <<'PY'
import json
import pathlib
import re
import statistics
import sys

root = pathlib.Path(sys.argv[1])
repeats = int(sys.argv[2])
labels = (
    "cpu_greedy",
    "vulkan_greedy",
    "vulkan_eagle_sequential",
    "vulkan_eagle_batched",
)

def generated_ids(path):
    for line in path.read_text().splitlines():
        if line.startswith("generated_ids:"):
            return line
    raise RuntimeError(f"missing generated_ids in {path}")

def decode_ms(label, repetition):
    if label.startswith("vulkan_eagle"):
        return json.loads((root / f"{label}_{repetition}.json").read_text())["decode_ms"]
    text = (root / f"{label}_{repetition}.err").read_text()
    match = re.search(r"\[timing\].*?decode=([0-9.]+) ms", text)
    if not match:
        raise RuntimeError(f"missing decode timing in {label}_{repetition}.err")
    return float(match.group(1))

reference = generated_ids(root / "cpu_greedy_1.out")
for label in labels:
    for repetition in range(1, repeats + 1):
        current = generated_ids(root / f"{label}_{repetition}.out")
        if current != reference:
            raise RuntimeError(f"output mismatch: {label} repetition {repetition}")

medians = {}
runs = {}
print("\nEAGLE3 Android ablation (decode latency; lower is better)")
for label in labels:
    values = [decode_ms(label, repetition) for repetition in range(1, repeats + 1)]
    runs[label] = values
    medians[label] = statistics.median(values)
    joined = ", ".join(f"{value:.2f}" for value in values)
    print(f"{label:27s} median={medians[label]:8.2f} ms  runs=[{joined}]")

seq_stats_runs = [
    json.loads((root / f"vulkan_eagle_sequential_{repetition}.json").read_text())
    for repetition in range(1, repeats + 1)
]
batch_stats_runs = [
    json.loads((root / f"vulkan_eagle_batched_{repetition}.json").read_text())
    for repetition in range(1, repeats + 1)
]
for repetition, (seq, batch) in enumerate(zip(seq_stats_runs, batch_stats_runs), 1):
    if seq["target_verify_mode"] != "sequential" or batch["target_verify_mode"] != "batched":
        raise RuntimeError(f"wrong verifier mode in repetition {repetition}")
    for key in ("draft_proposed", "draft_accepted", "target_input_tokens"):
        if seq[key] != batch[key]:
            raise RuntimeError(f"sequential/batched {key} differs in repetition {repetition}")
seq_stats = seq_stats_runs[0]
batch_stats = batch_stats_runs[0]

def paired_speedup(baseline, candidate):
    return statistics.median(
        before / after for before, after in zip(runs[baseline], runs[candidate])
    )

speedups = {
    "gpu_vs_cpu_anchor": paired_speedup("cpu_greedy", "vulkan_greedy"),
    "speculation_without_token_batching": paired_speedup(
        "vulkan_greedy", "vulkan_eagle_sequential"
    ),
    "batch_tile_inside_speculation": paired_speedup(
        "vulkan_eagle_sequential", "vulkan_eagle_batched"
    ),
    "combined_speculation_vs_gpu_greedy": paired_speedup(
        "vulkan_greedy", "vulkan_eagle_batched"
    ),
}
print(f"\nGPU backend vs CPU anchor:          {speedups['gpu_vs_cpu_anchor']:.3f}x")
print("speculation without token batching: "
      f"{speedups['speculation_without_token_batching']:.3f}x")
print("batch/tile gain inside speculation: "
      f"{speedups['batch_tile_inside_speculation']:.3f}x")
print("combined speculation vs GPU greedy: "
      f"{speedups['combined_speculation_vs_gpu_greedy']:.3f}x")
print(f"acceptance: {batch_stats['draft_accepted']}/{batch_stats['draft_proposed']} "
      f"({100.0 * batch_stats['acceptance_rate']:.2f}%)")
print(f"target calls sequential/batched: "
      f"{seq_stats['target_verify_calls']}/{batch_stats['target_verify_calls']}")
print("all generated_ids: exact match")
print(f"artifacts: {root}")

summary = {
    "repeats": repeats,
    "decode_ms": runs,
    "decode_median_ms": medians,
    "paired_speedup_median": speedups,
    "draft_proposed": batch_stats["draft_proposed"],
    "draft_accepted": batch_stats["draft_accepted"],
    "acceptance_rate": batch_stats["acceptance_rate"],
    "target_input_tokens": batch_stats["target_input_tokens"],
    "target_verify_calls_sequential": seq_stats["target_verify_calls"],
    "target_verify_calls_batched": batch_stats["target_verify_calls"],
    "generated_ids_exact_match": True,
}
(root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
PY
