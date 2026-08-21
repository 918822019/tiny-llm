#!/usr/bin/env bash
# ============================================================================
# bench_android.sh — 一键跑 Android 设备标准基准测试
# ============================================================================
# 用途：将已交叉编译的 tinyqwen binary 和模型 push 到 Android 设备，执行
#       标准化性能测量，并将结果记录到 optimization_log.md。
#
# 用法：
#   ./scripts/bench_android.sh <label> [--extra-args "..."]
#
# 示例：
#   ./scripts/bench_android.sh android-baseline
#   ./scripts/bench_android.sh neon-android --extra-args "--matvec-impl neon"
#   # i4 vs fp32 跨 dtype 同场对比
#   MODEL=model_i4.tqwen ./scripts/bench_android.sh i4-android \
#       --control-model model.tqwen
#
# 参数说明：
#   label    — 本次测量的标签名
#   额外参数 — 原样透传给设备端 runtime
#
# 环境变量：
#   MODEL      — 模型文件路径（默认 model.tqwen）
#   BUILD_DIR  — 交叉编译产物目录（默认 build-android）
#
# 前置条件：
#   - adb 可用且设备已连接
#   - scripts/build_android.sh 已完成交叉编译
#   - 模型已通过 tools/export_qwen_to_tiny.py 导出
#
# 测量方法论与纪律详见：docs/optimization.md §6
# ============================================================================

# set -e: 任何命令失败立即退出；-u: 引用未定义变量报错；-o pipefail: 管道中任一命令失败则整体失败
set -euo pipefail

# 切换到项目根目录
cd "$(dirname "$0")/.."

# 第一个参数为 label，缺失时打印用法并退出
LABEL="${1:?usage: bench_android.sh <label>   例如 bench_android.sh android-baseline}"
# shift 移除已消费的 label 参数
shift

# MODEL 环境变量可覆盖默认模型路径
MODEL="${MODEL:-model.tqwen}"

# BUILD_DIR 环境变量可覆盖交叉编译产物目录
BUILD_DIR="${BUILD_DIR:-build-android}"
# 拼接 Android binary 的完整路径
BIN="$BUILD_DIR/runtime/tinyqwen"

# 检查交叉编译产物是否存在
if [[ ! -f "$BIN" ]]; then
  echo "error: 找不到 ${BIN}（先 scripts/build_android.sh）" >&2
  exit 1
fi

# 检查模型文件是否存在
if [[ ! -f "$MODEL" ]]; then
  echo "error: 找不到模型 ${MODEL}（先用 tools/export_qwen_to_tiny.py 导出）" >&2
  exit 1
fi

# 调用 Python Android 测速脚本，传入模型、二进制、标签及所有剩余参数
python3 tools/bench_android.py --model "$MODEL" --binary "$BIN" --label "$LABEL" "$@"
