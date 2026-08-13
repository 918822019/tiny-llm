#!/usr/bin/env bash
# pipeline 第⑤步：规范化提交（代码 + 优化日志一起进一个 commit）。
#
#   ./scripts/commit_opt.sh fp32-float "matvec 累加 double→float"
set -euo pipefail
cd "$(dirname "$0")/.."

LABEL="${1:?usage: commit_opt.sh <label> \"一句话总结\"}"
SUMMARY="${2:-$LABEL}"

# 门禁：最新一个日期小节里还有 <填...> 占位就拒绝提交。
# 归因/教训是优化日志的灵魂——宁可此刻报错，不让半成品记录混进历史。
LOG="docs/optimization_log.md"
LATEST=$(awk '
    /^### .*[0-9]{4}-[0-9]{2}-[0-9]{2}/ { buf = ""; in_sec = 1 }
    in_sec { buf = buf $0 "\n" }
    in_sec && /^---$/ { last = buf; in_sec = 0 }
    END { printf "%s", last }
' "$LOG")
if printf '%s' "$LATEST" | grep -q '<填'; then
    echo "❌ 拒绝提交：$LOG 最新小节仍有未填占位："
    printf '%s' "$LATEST" | grep '<填'
    echo "   补全后再跑 commit_opt.sh（归因/教训由人填，这是最需要判断的部分）"
    exit 1
fi

git add -A
git commit -m "perf: $SUMMARY

label: $LABEL
测量: scripts/record.sh（3 遍取中位），详见 docs/optimization_log.md
正确性: scripts/verify.sh 通过"

echo ""
echo "[commit] 完成：$(git log --oneline -1)"
echo "        如需回退：git revert HEAD"
