#!/usr/bin/env bash
# ============================================================================
# record_android.sh — Android pipeline 一条龙：正确性门禁 → 稳定测速 → 写优化日志
# ============================================================================
# 用途：Android 端的 record.sh 等价物。将设备端 verify 和 bench 合并为原子操作，
#       确保每次记录的 Android 优化数据都附带正确性证明。
#
# 用法：
#   ./scripts/record_android.sh <label> [--skip-verify] [--extra-args "..."]
#
# 示例：
#   ./scripts/record_android.sh android-baseline
#   ./scripts/record_android.sh neon-android --extra-args "--matvec-impl neon"
#   ./scripts/record_android.sh neon-tune --skip-verify --extra-args "--matvec-impl neon"
#   # i4 vs fp32 跨 dtype 同场对比（MODEL 同时作用于门禁和测速）
#   MODEL=model_i4.tqwen ./scripts/record_android.sh i4-android \
#       --control-model model.tqwen
#
# 环境变量：
#   MODEL — 模型文件路径（默认 model.tqwen），同时传给门禁和测速
#
# 流程：
#   1. scripts/verify_android.sh   — 设备端正确性门禁（不过就中止，不测速）
#   2. tools/record_android.py     — 跑 3 遍取中位 + 算 vs 基线 + 写日志
#
# 方法论与纪律详见：docs/optimization.md §6/§8
# ============================================================================

# set -e: 任何命令失败立即退出；-u: 引用未定义变量报错；-o pipefail: 管道中任一命令失败则整体失败
set -euo pipefail

# 切换到项目根目录
cd "$(dirname "$0")/.."

# 第一个参数为 label，缺失时打印用法并退出
LABEL="${1:?usage: record_android.sh <label> [--skip-verify]   例如 record_android.sh android-baseline}"
# shift 移除已消费的 label 参数
shift

# MODEL 环境变量可覆盖默认模型路径
MODEL="${MODEL:-model.tqwen}"

# ---------------------------------------------------------------------------
# 解析命令行参数：剥离 --skip-verify，其余参数收集到 PASS_ARGS 数组
# ---------------------------------------------------------------------------
SKIP_VERIFY=false     # 是否跳过正确性门禁的标志
PASS_ARGS=()          # 存放需要透传给 record_android.py 的参数
for arg in "$@"; do
    if [[ "$arg" == "--skip-verify" ]]; then
        # 识别到 --skip-verify 标志
        SKIP_VERIFY=true
    else
        # 其他参数原样保留
        PASS_ARGS+=("$arg")
    fi
done

# ---------------------------------------------------------------------------
# 步骤 1/2：Android 端正确性门禁
# ---------------------------------------------------------------------------
if [[ "$SKIP_VERIFY" == "true" ]]; then
    echo "=== [1/2] 正确性门禁（Android）：跳过（--skip-verify）==="
else
    echo "=== [1/2] 正确性门禁（Android）==="
    # MODEL 通过环境变量传给 verify_android.sh，使其使用正确的模型进行验证
    MODEL="$MODEL" ./scripts/verify_android.sh
fi

echo ""

# ---------------------------------------------------------------------------
# 步骤 2/2：Android 端稳定测速 + 写优化日志
# ---------------------------------------------------------------------------
echo "=== [2/2] 稳定测速 + 写优化日志（Android）==="
# 调用 Python Android 记录脚本：
#   --label  标签名
#   --runs 3 跑 3 遍取中位数
#   --model  模型文件路径
#   ${PASS_ARGS[@]+"${PASS_ARGS[@]}"} 安全展开数组（空数组时不报错）
python3 tools/record_android.py --label "$LABEL" --runs 3 --model "$MODEL" \
        ${PASS_ARGS[@]+"${PASS_ARGS[@]}"}

echo ""
# 提示下一步操作
echo "下一步：补全 docs/optimization_log.md 里的 <填...>，然后 ./scripts/commit_opt.sh '$LABEL' \"一句话总结\""
