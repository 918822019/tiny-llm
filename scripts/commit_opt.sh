#!/usr/bin/env bash
# ============================================================================
# commit_opt.sh — 规范化提交优化结果（pipeline 第⑤步）
# ============================================================================
# 用途：将代码改动 + 优化日志一起打包为一个规范的 git commit。
#       确保 optimization_log.md 中最新小节的占位符已全部填写后才允许提交。
#
# 用法：
#   ./scripts/commit_opt.sh <label> "一句话总结"
#
# 示例：
#   ./scripts/commit_opt.sh fp32-float "matvec 累加 double→float"
#
# 参数说明：
#   label   — 优化标签，写入 commit message 的 label: 字段
#   summary — 一句话描述本次优化内容（缺省时取 label 的值）
#
# 门禁规则：
#   如果 docs/optimization_log.md 最新日期小节中仍有 "<填...>" 占位符，
#   则拒绝提交——归因/教训是优化日志的灵魂，必须由人手动填写。
#
# commit message 格式：
#   perf: <summary>
#
#   label: <label>
#   测量: scripts/record.sh（3 遍取中位），详见 docs/optimization_log.md
#   正确性: scripts/verify.sh 通过
# ============================================================================

# set -e: 任何命令失败立即退出；-u: 引用未定义变量报错；-o pipefail: 管道中任一命令失败则整体失败
set -euo pipefail

# 切换到项目根目录
cd "$(dirname "$0")/.."

# 第一个参数为 label，缺失时打印用法并退出
LABEL="${1:?usage: commit_opt.sh <label> \"一句话总结\"}"
# 第二个参数为 summary，缺省时使用 label 作为默认值
SUMMARY="${2:-$LABEL}"

# ---------------------------------------------------------------------------
# 门禁检查：optimization_log.md 最新小节是否还有未填占位符
# ---------------------------------------------------------------------------
LOG="docs/optimization_log.md"

# 用 awk 提取最新的日期小节内容：
#   - 匹配以 "### " 开头且含日期格式的标题行时，开始收集
#   - 遇到 "---" 分隔线时，保存当前小节并停止收集
#   - END 时输出最后收集到的小节
LATEST=$(awk '
    /^### .*[0-9]{4}-[0-9]{2}-[0-9]{2}/ { buf = ""; in_sec = 1 }
    in_sec { buf = buf $0 "\n" }
    in_sec && /^---$/ { last = buf; in_sec = 0 }
    END { printf "%s", last }
' "$LOG")

# 检查最新小节是否包含 "<填" 占位符
if printf '%s' "$LATEST" | grep -q '<填'; then
    echo "❌ 拒绝提交：$LOG 最新小节仍有未填占位："
    # 打印所有包含占位符的行，帮助用户定位需要补全的内容
    printf '%s' "$LATEST" | grep '<填'
    echo "   补全后再跑 commit_opt.sh（归因/教训由人填，这是最需要判断的部分）"
    exit 1
fi

# ---------------------------------------------------------------------------
# 执行 git 提交
# ---------------------------------------------------------------------------

# git add -A 暂存所有改动（包括新增、修改、删除的文件）
git add -A

# 构造多行 commit message：
#   第一行：perf: <summary>（符合 conventional commits 规范）
#   空行
#   正文：label、测量方法、正确性验证信息
git commit -m "perf: $SUMMARY

label: $LABEL
测量: scripts/record.sh（3 遍取中位），详见 docs/optimization_log.md
正确性: scripts/verify.sh 通过"

# 打印提交结果和回退提示
echo ""
echo "[commit] 完成：$(git log --oneline -1)"
echo "        如需回退：git revert HEAD"
