#!/usr/bin/env bash
# Android pipeline 一条龙：正确性门禁 → 稳定测速 → 自动写优化日志。
#
#   ./scripts/record_android.sh <label> [--skip-verify] [--extra-args "..."]
#   例：./scripts/record_android.sh android-baseline
#       ./scripts/record_android.sh neon-android --extra-args "--matvec-impl neon"
#       ./scripts/record_android.sh neon-tune --skip-verify --extra-args "--matvec-impl neon"
#       # i4 vs fp32 跨 dtype 同场对比（MODEL 同时作用于门禁和测速）
#       MODEL=model_i4.tqwen ./scripts/record_android.sh i4-android \
#           --control-model model.tqwen
#
# 流程：
#   1. scripts/verify_android.sh   正确性门禁（不过就中止，不测速）
#   2. tools/record_android.py     跑 3 遍取中位 + 算 vs 基线 + 写日志
# 方法论与纪律：docs/optimization.md §6/§8。
set -euo pipefail
cd "$(dirname "$0")/.."

LABEL="${1:?usage: record_android.sh <label> [--skip-verify]   例如 record_android.sh android-baseline}"
shift

MODEL="${MODEL:-model.tqwen}"

SKIP_VERIFY=false
PASS_ARGS=()
for arg in "$@"; do
    if [[ "$arg" == "--skip-verify" ]]; then
        SKIP_VERIFY=true
    else
        PASS_ARGS+=("$arg")
    fi
done

if [[ "$SKIP_VERIFY" == "true" ]]; then
    echo "=== [1/2] 正确性门禁（Android）：跳过（--skip-verify）==="
else
    echo "=== [1/2] 正确性门禁（Android）==="
    MODEL="$MODEL" ./scripts/verify_android.sh
fi

echo ""
echo "=== [2/2] 稳定测速 + 写优化日志（Android）==="
python3 tools/record_android.py --label "$LABEL" --runs 3 --model "$MODEL" \
        ${PASS_ARGS[@]+"${PASS_ARGS[@]}"}

echo ""
echo "下一步：补全 docs/optimization_log.md 里的 <填...>，然后 ./scripts/commit_opt.sh '$LABEL' \"一句话总结\""
