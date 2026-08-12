#!/usr/bin/env bash
# 一键跑标准基准测试。
#
#   ./scripts/bench.sh <label>
#   例：./scripts/bench.sh fp32-baseline
#       ./scripts/bench.sh int8-w8a8
#
# label 是这次测量的名字，会出现在输出和 optimization_log.md 的记录里。
# 约定：每做完一个优化、提交代码后，就用那个优化命名跑一次，记进日志。
set -euo pipefail
cd "$(dirname "$0")/.."

LABEL="${1:?usage: bench.sh <label>   例如 bench.sh fp32-baseline}"
MODEL="${MODEL:-model.tqwen}"
BIN="${BIN:-build/runtime/tinyqwen}"

if [[ ! -f "$MODEL" ]]; then
  echo "error: 找不到模型 $MODEL（先用 tools/export_qwen_to_tiny.py 导出）" >&2
  exit 1
fi
if [[ ! -x "$BIN" ]]; then
  echo "error: 找不到可执行文件 $BIN（先 cmake --build build）" >&2
  exit 1
fi

python3 tools/bench.py --model "$MODEL" --binary "$BIN" --label "$LABEL"
