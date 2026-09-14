#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ADB_BIN="${ADB_BIN:-$(command -v adb 2>/dev/null || true)}"
if [[ -z "$ADB_BIN" && -x /opt/homebrew/bin/adb ]]; then
  ADB_BIN=/opt/homebrew/bin/adb
fi

if [[ $# -lt 3 || $# -gt 6 ]]; then
  echo "usage: $0 <target.tqwen> <eagle3.tqwen> <tokens.json> [repeats=3] [max-new=64] [verify-width=2]" >&2
  echo "       EAGLE3_BENCH_MODE=full|pair (default: full)" >&2
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
BENCH_MODE="${EAGLE3_BENCH_MODE:-full}"
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
case "$BENCH_MODE" in
  full)
    LABELS=(cpu_greedy vulkan_greedy vulkan_eagle_sequential vulkan_eagle_batched)
    ;;
  pair)
    LABELS=(vulkan_greedy vulkan_eagle_batched)
    ;;
  *)
    echo "error: EAGLE3_BENCH_MODE must be 'full' or 'pair'" >&2
    exit 2
    ;;
esac

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

# Each measured process owns a fresh Vulkan device and weight cache. This warmup
# exercises kernels and file cache; the EAGLE path separately prepares the tied
# lm_head inside its measured-process prefill phase.
for label in "${LABELS[@]}"; do
  run_case "$label" 0 true
done

# Reverse every second round so slow thermal drift does not always favor one mode.
for ((rep = 1; rep <= REPEATS; ++rep)); do
  if [[ "$BENCH_MODE" == pair ]]; then
    if (( rep % 2 == 1 )); then
      order=(vulkan_greedy vulkan_eagle_batched)
    else
      order=(vulkan_eagle_batched vulkan_greedy)
    fi
  else
    if (( rep % 2 == 1 )); then
      # Keep B-D and D-C adjacent: production speedup and tile contribution each
      # see the smallest possible thermal gap.
      order=(cpu_greedy vulkan_greedy vulkan_eagle_batched vulkan_eagle_sequential)
    else
      order=(vulkan_eagle_sequential vulkan_eagle_batched vulkan_greedy cpu_greedy)
    fi
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
"$PYTHON_BIN" - "$OUT_DIR" "$REPEATS" "$BENCH_MODE" <<'PY'
import json
import pathlib
import re
import statistics
import sys

root = pathlib.Path(sys.argv[1])
repeats = int(sys.argv[2])
mode = sys.argv[3]
labels = (
    ("cpu_greedy", "vulkan_greedy", "vulkan_eagle_sequential",
     "vulkan_eagle_batched")
    if mode == "full"
    else ("vulkan_greedy", "vulkan_eagle_batched")
)

def generated_ids(path):
    for line in path.read_text().splitlines():
        if line.startswith("generated_ids:"):
            return line
    raise RuntimeError(f"missing generated_ids in {path}")

def timings(label, repetition):
    if label.startswith("vulkan_eagle"):
        data = json.loads((root / f"{label}_{repetition}.json").read_text())
        return data["prefill_ms"], data["decode_ms"]
    text = (root / f"{label}_{repetition}.err").read_text()
    match = re.search(r"\[timing\] prefill=([0-9.]+) ms decode=([0-9.]+) ms", text)
    if not match:
        raise RuntimeError(f"missing timing in {label}_{repetition}.err")
    return float(match.group(1)), float(match.group(2))

reference = generated_ids(root / f"{labels[0]}_1.out")
for label in labels:
    for repetition in range(1, repeats + 1):
        current = generated_ids(root / f"{label}_{repetition}.out")
        if current != reference:
            raise RuntimeError(f"output mismatch: {label} repetition {repetition}")

prefill_medians = {}
decode_medians = {}
total_medians = {}
prefill_runs = {}
decode_runs = {}
total_runs = {}
print("\nEAGLE3 Android ablation (latency; lower is better)")
for label in labels:
    pairs = [timings(label, repetition) for repetition in range(1, repeats + 1)]
    prefill_values = [pair[0] for pair in pairs]
    decode_values = [pair[1] for pair in pairs]
    total_values = [sum(pair) for pair in pairs]
    prefill_runs[label] = prefill_values
    decode_runs[label] = decode_values
    total_runs[label] = total_values
    prefill_medians[label] = statistics.median(prefill_values)
    decode_medians[label] = statistics.median(decode_values)
    total_medians[label] = statistics.median(total_values)
    joined = ", ".join(f"{value:.2f}" for value in decode_values)
    print(f"{label:27s} prefill={prefill_medians[label]:8.2f} ms  "
          f"decode={decode_medians[label]:8.2f} ms  total={total_medians[label]:8.2f} ms  "
          f"decode_runs=[{joined}]")

batch_stats_runs = [
    json.loads((root / f"vulkan_eagle_batched_{repetition}.json").read_text())
    for repetition in range(1, repeats + 1)
]
batch_stats = batch_stats_runs[0]
for repetition, batch in enumerate(batch_stats_runs, 1):
    if batch["target_verify_mode"] != "batched":
        raise RuntimeError(f"wrong batched verifier mode in repetition {repetition}")
    for key in ("draft_proposed", "draft_accepted", "target_input_tokens",
                "target_verify_calls"):
        if batch[key] != batch_stats[key]:
            raise RuntimeError(f"batched {key} differs in repetition {repetition}")
seq_stats_runs = []
if mode == "full":
    seq_stats_runs = [
        json.loads((root / f"vulkan_eagle_sequential_{repetition}.json").read_text())
        for repetition in range(1, repeats + 1)
    ]
    for repetition, (seq, batch) in enumerate(zip(seq_stats_runs, batch_stats_runs), 1):
        if seq["target_verify_mode"] != "sequential":
            raise RuntimeError(f"wrong sequential verifier mode in repetition {repetition}")
        for key in ("draft_proposed", "draft_accepted", "target_input_tokens"):
            if seq[key] != batch[key]:
                raise RuntimeError(
                    f"sequential/batched {key} differs in repetition {repetition}"
                )

def paired_speedups(values, baseline, candidate):
    return [
        before / after
        for before, after in zip(values[baseline], values[candidate])
    ]

speedup_runs = {
    "combined_speculation_vs_gpu_greedy": paired_speedups(
        decode_runs, "vulkan_greedy", "vulkan_eagle_batched"
    ),
    "combined_e2e_vs_gpu_greedy": paired_speedups(
        total_runs, "vulkan_greedy", "vulkan_eagle_batched"
    ),
}
if mode == "full":
    speedup_runs.update({
        "gpu_vs_cpu_anchor": paired_speedups(
            decode_runs, "cpu_greedy", "vulkan_greedy"
        ),
        "speculation_without_token_batching": paired_speedups(
            decode_runs, "vulkan_greedy", "vulkan_eagle_sequential"
        ),
        "batch_tile_inside_speculation": paired_speedups(
            decode_runs, "vulkan_eagle_sequential", "vulkan_eagle_batched"
        ),
    })
speedups = {key: statistics.median(values) for key, values in speedup_runs.items()}
speedup_min = {key: min(values) for key, values in speedup_runs.items()}
speedup_max = {key: max(values) for key, values in speedup_runs.items()}

if mode == "full":
    print(f"\nGPU backend vs CPU anchor:          {speedups['gpu_vs_cpu_anchor']:.3f}x")
    print("speculation without token batching: "
          f"{speedups['speculation_without_token_batching']:.3f}x")
    print("batch/tile gain inside speculation: "
          f"{speedups['batch_tile_inside_speculation']:.3f}x")
print("combined speculation vs GPU greedy: "
      f"{speedups['combined_speculation_vs_gpu_greedy']:.3f}x "
      f"[{speedup_min['combined_speculation_vs_gpu_greedy']:.3f}, "
      f"{speedup_max['combined_speculation_vs_gpu_greedy']:.3f}]")
print("combined end-to-end vs GPU greedy:   "
      f"{speedups['combined_e2e_vs_gpu_greedy']:.3f}x "
      f"[{speedup_min['combined_e2e_vs_gpu_greedy']:.3f}, "
      f"{speedup_max['combined_e2e_vs_gpu_greedy']:.3f}]")
print(f"acceptance: {batch_stats['draft_accepted']}/{batch_stats['draft_proposed']} "
      f"({100.0 * batch_stats['acceptance_rate']:.2f}%)")
if mode == "full":
    print(f"target calls sequential/batched: "
          f"{seq_stats_runs[0]['target_verify_calls']}/{batch_stats['target_verify_calls']}")
else:
    print(f"target calls batched: {batch_stats['target_verify_calls']}")
print("all generated_ids: exact match")
print(f"artifacts: {root}")

summary = {
    "mode": mode,
    "repeats": repeats,
    "prefill_ms": prefill_runs,
    "decode_ms": decode_runs,
    "total_ms": total_runs,
    "prefill_median_ms": prefill_medians,
    "decode_median_ms": decode_medians,
    "total_median_ms": total_medians,
    "paired_speedup_runs": speedup_runs,
    "paired_speedup_median": speedups,
    "paired_speedup_min": speedup_min,
    "paired_speedup_max": speedup_max,
    "draft_proposed": batch_stats["draft_proposed"],
    "draft_accepted": batch_stats["draft_accepted"],
    "acceptance_rate": batch_stats["acceptance_rate"],
    "target_input_tokens": batch_stats["target_input_tokens"],
    "target_verify_calls_batched": batch_stats["target_verify_calls"],
    "generated_ids_exact_match": True,
}
if mode == "full":
    summary["target_verify_calls_sequential"] = seq_stats_runs[0]["target_verify_calls"]
(root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
PY
