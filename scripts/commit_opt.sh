#!/usr/bin/env bash
# pipeline 第⑤步：规范化提交（代码 + 优化日志一起进一个 commit）。
#
#   ./scripts/commit_opt.sh fp32-float "matvec 累加 double→float"
set -euo pipefail
cd "$(dirname "$0")/.."

LABEL="${1:?usage: commit_opt.sh <label> \"一句话总结\"}"
SUMMARY="${2:-$LABEL}"

git add -A
git commit -m "perf: $SUMMARY

label: $LABEL
测量: scripts/record.sh（3 遍取中位），详见 docs/optimization_log.md
正确性: scripts/verify.sh 通过"

echo ""
echo "[commit] 完成：$(git log --oneline -1)"
echo "        如需回退：git revert HEAD"
