#!/usr/bin/env bash
# ============================================================================
# record.sh — pipeline 第②③④步一条龙：正确性门禁 → 稳定测速 → 自动写优化日志
# ============================================================================
# 用途：将 verify（正确性验证）和 bench（性能测量）合并为一个原子操作，
#       确保每次记录的优化数据都附带正确性证明。是日常优化迭代的主入口。
#
# 用法：
#   ./scripts/record.sh <label> [--skip-verify] [额外 runtime 参数...]
#
# 示例：
#   ./scripts/record.sh fp32-float
#   ./scripts/record.sh double_2_float --extra-args "--matvec-impl double_2_float"
#   ./scripts/record.sh neon-tune --skip-verify --extra-args "--matvec-impl neon"
#
# 参数说明：
#   label          — 本次优化的标签名
#   --skip-verify  — 跳过正确性门禁（刚跑过 verify、快速迭代调参时用，别滥用）
#   额外参数       — 原样透传给 tools/record_optimization.py
#
# 流程：
#   1. scripts/verify.sh           — 正确性门禁（不过就中止，不测速）
#   2. tools/record_optimization.py — 跑 3 遍取中位 + 算 vs 基线 + 写日志
#
# 方法论与纪律详见：docs/optimization.md §6/§8
# ============================================================================

# set -e: 任何命令失败立即退出；-u: 引用未定义变量报错；-o pipefail: 管道中任一命令失败则整体失败
set -euo pipefail

# 切换到项目根目录
cd "$(dirname "$0")/.."

# 第一个参数为 label，缺失时打印用法并退出
LABEL="${1:?usage: record.sh <label> [--skip-verify]   例如 record.sh fp32-float}"
# shift 移除已消费的 label 参数
shift

# ---------------------------------------------------------------------------
# 解析命令行参数：剥离 --skip-verify，其余参数收集到 PASS_ARGS 数组
# ---------------------------------------------------------------------------
SKIP_VERIFY=false     # 是否跳过正确性门禁的标志
PASS_ARGS=()          # 存放需要透传给 record_optimization.py 的参数
for arg in "$@"; do
    if [[ "$arg" == "--skip-verify" ]]; then
        # 识别到 --skip-verify 标志，设置跳过标记
        SKIP_VERIFY=true
    else
        # 其他参数原样保留，后续透传
        PASS_ARGS+=("$arg")
    fi
done

# ---------------------------------------------------------------------------
# 步骤 1/2：正确性门禁
# ---------------------------------------------------------------------------
if [[ "$SKIP_VERIFY" == "true" ]]; then
    # 用户指定了 --skip-verify，跳过门禁
    echo "=== [1/2] 正确性门禁：跳过（--skip-verify）==="
else
    # 执行完整正确性门禁（编译 + 单测 + golden token 对照）
    echo "=== [1/2] 正确性门禁 ==="
    ./scripts/verify.sh
fi

echo ""

# ---------------------------------------------------------------------------
# 步骤 2/2：稳定测速 + 写优化日志
# ---------------------------------------------------------------------------
echo "=== [2/2] 稳定测速 + 写优化日志 ==="
# 调用 Python 记录脚本：
#   --label  标签名
#   --runs 3 跑 3 遍取中位数（减少噪声）
#   ${PASS_ARGS[@]+"${PASS_ARGS[@]}"} 安全展开数组（空数组时不报错）
python3 tools/record_optimization.py --label "$LABEL" --runs 3 \
        ${PASS_ARGS[@]+"${PASS_ARGS[@]}"}

echo ""
# 提示下一步操作：补全日志中的占位符后提交
echo "下一步：补全 docs/optimization_log.md 里的 <填...>，然后 ./scripts/commit_opt.sh '$LABEL' \"一句话总结\""
