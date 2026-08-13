#!/usr/bin/env bash
# pipeline 第②③④步一条龙：正确性门禁 → 稳定测速 → 自动写优化日志。
#
#   ./scripts/record.sh <label> [额外 runtime 参数...]
#   例：./scripts/record.sh fp32-float
#       ./scripts/record.sh double_2_float --extra-args "--matvec-impl double_2_float"
#
# 额外参数原样透传给 runtime（测"需要开关才生效"的优化变体时用）。
# 流程：
#   1. scripts/verify.sh           正确性门禁（不过就中止，不测速）
#   2. tools/record_optimization.py 跑 3 遍取中位 + 算 vs 基线 + 写日志
set -euo pipefail
cd "$(dirname "$0")/.."

LABEL="${1:?usage: record.sh <label>   例如 record.sh fp32-float}"
shift

echo "=== [1/2] 正确性门禁 ==="
./scripts/verify.sh

echo ""
echo "=== [2/2] 稳定测速 + 写优化日志 ==="
python3 tools/record_optimization.py --label "$LABEL" --runs 3 "$@"

echo ""
echo "下一步：补全 docs/optimization_log.md 里的 <填...>，然后 ./scripts/commit_opt.sh '$LABEL' \"一句话总结\""
