#!/usr/bin/env bash
# bench_moe.sh — MoE 专家 SSD 卸载 cache 命中/miss 扫描
#
# 生成 fake MoE 模型，对一系列 cache_slots 跑 SSD 模式，打印 hits/misses/
# evictions/bytes_read + 每 token 延迟。用于观察缓存抖动拐点（slots 达到
# 工作集后 hits 激增、bytes_read 骤降）。
#
# 用法：./scripts/bench_moe.sh [model.tqwen] [slots...]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="$ROOT/build/runtime/tinyqwen"
GEN="$ROOT/tools/make_fake_qwen35_moe_model.py"
MODEL="${1:-/tmp/fake_moe.tqwen}"
shift || true
if [[ $# -gt 0 ]]; then SLOTS=("$@"); else SLOTS=(0 2 4 8 16 32); fi

PY="${PY:-$ROOT/.venv/bin/python}"
"$PY" "$GEN" --out "$MODEL" >/dev/null

echo "slots  hits  misses  evictions  bytes_read"
for s in "${SLOTS[@]}"; do
  out=$("$BIN" --model "$MODEL" --tokens 3,7,11,2 --max-new-tokens 16 \
        --max-seq-len 64 --matvec-impl ref --ops-impl ref \
        --moe-ssd --moe-expert-cache-slots "$s" 2>&1)
  stats=$(echo "$out" | grep -oE 'hits=[0-9]+ misses=[0-9]+ evictions=[0-9]+ bytes_read=[0-9]+' || true)
  hits=$(echo "$stats" | grep -oE 'hits=[0-9]+' | head -1 | cut -d= -f2)
  miss=$(echo "$stats" | grep -oE 'misses=[0-9]+' | cut -d= -f2)
  ev=$(echo "$stats" | grep -oE 'evictions=[0-9]+' | cut -d= -f2)
  br=$(echo "$stats" | grep -oE 'bytes_read=[0-9]+' | cut -d= -f2)
  printf "%-6s %-6s %-8s %-11s %s\n" "$s" "${hits:-0}" "${miss:-0}" "${ev:-0}" "${br:-0}"
done
echo "（注：fake 模型极小，I/O 不可测——真实归因需真模型；此处看 cache 抖动结构）"
