#!/usr/bin/env bash
# =============================================================================
# bench_metal_prefill.sh —— Metal prefill vs CPU prefill 同场 A/B 测速
# =============================================================================
# 只测 prefill（TTFT）：--max-new-tokens 1 让 decode 不参与，
# 时间取自 profiler JSON 的 first_token_ms（不是 wall clock，
# 排除模型加载/权重上传/进程启动的噪声）。
#
# 纪律同 scripts/bench.sh：同场 A/B（同一次调用里两个 arm 都测）、
# 预热丢弃、取中位数。
# =============================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

BUILD_DIR="${BUILD_DIR:-build}"
BIN="./${BUILD_DIR}/runtime/tinyqwen"
MODEL="${MODEL:-model_qwen3_06b_f16.tqwen}"
RUNS="${RUNS:-9}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-1024}"
# 预热次数：MPS kernel 每个进程都要重新 JIT，单次预热不够。
# 实测单次预热会让首个样本偏高几十个百分点，污染中位数。
WARMUP="${WARMUP:-3}"
# 位置参数覆盖默认扫描点，方便只测少数几个长度
SEQS="${*:-${SEQS:-16 32 64 128 256 512}}"

[[ -x "$BIN" ]] || { echo "error: $BIN 不存在，先编译" >&2; exit 1; }
[[ -f "$MODEL" ]] || { echo "error: $MODEL 不存在" >&2; exit 1; }

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

# CPU arm 必须显式传优化 kernel：默认 ref 是标量兜底（AGENTS.md 坑 #1），
# 拿 ref 当基线会把 speedup 放大几十倍，毫无意义。f16 模型满栈配方见 AGENTS.md。
CPU_IMPL="${CPU_IMPL:---matvec-impl neon_mt_kv_nt --ops-impl neon}"

# 生成确定性伪随机 token 序列（不需要 tokenizer）
make_tokens() {
    .venv/bin/python - "$1" <<'PY'
import sys, random
n = int(sys.argv[1])
random.seed(1234)
print(",".join(str(random.randrange(0, 151936)) for _ in range(n)))
PY
}

# 跑一次，取 profiler JSON 里的 first_token_ms
one_run() {
    local engine="$1" tokens="$2" out="$3"
    local extra=()
    if [[ -n "$engine" ]]; then
        extra=(--engine "$engine")
    else
        # CPU arm：显式优化 kernel（metal arm 不吃这两个开关）
        read -r -a extra <<< "$CPU_IMPL"
    fi
    "$BIN" --model "$MODEL" --tokens "$tokens" --max-new-tokens 1 \
        --max-seq-len "$MAX_SEQ_LEN" --profile-out "$out" ${extra[@]+"${extra[@]}"} \
        >/dev/null 2>&1
    .venv/bin/python -c "
import json,sys
print(json.load(open('$out'))['first_token_ms'])
"
}

median() {
    .venv/bin/python -c "
import statistics,sys
print('%.4f' % statistics.median([float(x) for x in sys.argv[1:]]))
" "$@"
}

# 样本离散度：max/min。用来判断这一行的数字可不可信 ——
# 实测这台 M4 的 prefill 计时跨进程波动可达 2×，离散度大时中位数没有意义。
spread() {
    .venv/bin/python -c "
import sys
v=[float(x) for x in sys.argv[1:]]
print('%.2f' % (max(v)/min(v)))
" "$@"
}

printf "%-8s %10s %10s %9s %12s %12s %8s %8s\n" \
    "seq" "cpu_ms" "metal_ms" "speedup" "cpu_ms/tok" "metal_ms/tok" "cpu散" "metal散"
printf "%-8s %10s %10s %9s %12s %12s %8s %8s\n" \
    "---" "------" "--------" "-------" "----------" "------------" "-----" "-------"

for n in $SEQS; do
    tokens="$(make_tokens "$n")"

    # 预热：两个 arm 各跑 WARMUP 次，丢弃
    for _ in $(seq "$WARMUP"); do
        one_run "" "$tokens" "$TMP/warm_cpu.json" >/dev/null || true
        one_run metal "$tokens" "$TMP/warm_metal.json" >/dev/null || true
    done

    cpu_samples=()
    metal_samples=()
    for _ in $(seq "$RUNS"); do
        cpu_samples+=("$(one_run "" "$tokens" "$TMP/cpu.json")")
        metal_samples+=("$(one_run metal "$tokens" "$TMP/metal.json")")
    done

    cpu_ms="$(median "${cpu_samples[@]}")"
    metal_ms="$(median "${metal_samples[@]}")"
    cpu_sp="$(spread "${cpu_samples[@]}")"
    metal_sp="$(spread "${metal_samples[@]}")"

    .venv/bin/python -c "
cpu=float('$cpu_ms'); metal=float('$metal_ms'); n=$n
print('%-8d %10.4f %10.4f %9.3f %12.4f %12.4f %8s %8s'
      % (n, cpu, metal, cpu/metal, cpu/n, metal/n, '$cpu_sp', '$metal_sp'))
"
done
