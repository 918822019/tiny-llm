#!/usr/bin/env bash
# ============================================================================
# bench.sh — 一键跑标准基准测试（桌面端 / 本机）
# ============================================================================
# 用途：对当前构建的 tinyqwen runtime 执行标准化性能测量，并将结果记录到
#       optimization_log.md。是优化 pipeline 中"测速"环节的统一入口。
#
# 用法：
#   ./scripts/bench.sh <label> [额外 runtime 参数...]
#
# 示例：
#   ./scripts/bench.sh fp32-baseline
#   ./scripts/bench.sh int8-w8a8
#   ./scripts/bench.sh double_2_float --extra-args "--matvec-impl double_2_float"
#
# 参数说明：
#   label    — 本次测量的标签名，会出现在输出和 optimization_log.md 的记录里
#   额外参数 — 原样透传给 runtime，用于测"需要开关才生效"的优化变体
#
# 约定：每做完一个优化、提交代码后，就用那个优化命名跑一次，记进日志。
# 测量方法论与纪律详见：docs/optimization.md §6
#
# 依赖环境：
#   - python3 + tools/bench.py
#   - 已导出的模型文件（默认 model.tqwen）
#   - 已编译的可执行文件（默认 build/runtime/tinyqwen）
# ============================================================================

# set -e: 任何命令失败立即退出；-u: 引用未定义变量报错；-o pipefail: 管道中任一命令失败则整体失败
set -euo pipefail

# 切换到项目根目录（脚本所在目录的上一级），确保相对路径一致
cd "$(dirname "$0")/.."

# 第一个参数为 label（标签名），缺失时打印用法并退出（${1:?...} 语法）
LABEL="${1:?usage: bench.sh <label>   例如 bench.sh fp32-baseline}"
# shift 移除已消费的 label 参数，剩余参数留给后续透传
shift

# MODEL 环境变量可覆盖默认模型路径；:- 表示若未设置或为空则取默认值
MODEL="${MODEL:-model.tqwen}"
# BIN 环境变量可覆盖默认可执行文件路径
BIN="${BIN:-build/runtime/tinyqwen}"

# 检查模型文件是否存在，不存在则报错并提示导出方法
if [[ ! -f "$MODEL" ]]; then
  echo "error: 找不到模型 $MODEL（先用 tools/export_qwen_to_tiny.py 导出）" >&2
  exit 1
fi

# 检查可执行文件是否存在且可执行（-x 判断），不存在则报错并提示编译
if [[ ! -x "$BIN" ]]; then
  echo "error: 找不到可执行文件 $BIN（先 cmake --build build）" >&2
  exit 1
fi

# 调用 Python 测速脚本，传入模型路径、二进制路径、标签名及所有剩余参数（"$@"）
python3 tools/bench.py --model "$MODEL" --binary "$BIN" --label "$LABEL" "$@"
