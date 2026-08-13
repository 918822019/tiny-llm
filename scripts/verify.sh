#!/usr/bin/env bash
# 正确性门禁（pipeline 第②步）：在测速之前，先证明"没算错"。
#
#   ./scripts/verify.sh
#
# 做三件事，任一失败就非零退出：
#   1. 编译（Release）
#   2. 跑全部单元测试
#   3. golden token 对照：用 canonical prompt 跑真模型，生成的 token 必须和
#      参考实现逐位一致。若不一致，说明本次改动改变了数值行为——这未必是错
#      （如量化），但必须人工确认，所以先拦下来。
set -euo pipefail
cd "$(dirname "$0")/.."

BIN="build/runtime/tinyqwen"
MODEL="model.tqwen"
# canonical prompt（"中国的首都是"）+ 16 个 greedy token，参考实现（fp32）产出。
GOLDEN="generated_ids: 2130 198 32 13 94305 245 46553 198 33 13 64118 55135 198 34 13 66521"

echo "[1/3] 编译..."
cmake --build build -j"$(sysctl -n hw.ncpu 2>/dev/null || nproc 2>/dev/null || echo 4)" >/dev/null

echo "[2/3] 单元测试..."
./build/tests/tinyqwen_tests | tail -1

if [[ ! -f "$MODEL" ]]; then
  echo "[3/3] 跳过 golden token 对照（找不到 $MODEL，先 export）"
  exit 0
fi

echo "[3/3] golden token 对照..."
ACTUAL="$("$BIN" --model "$MODEL" --tokens 105538,59975,100132 \
  --max-new-tokens 16 --max-seq-len 32 --eos -1 2>/dev/null | grep '^generated_ids:')"
if [[ "$ACTUAL" == "$GOLDEN" ]]; then
  echo "PASS  生成 token 与参考实现逐位一致"
else
  echo "FAIL  生成 token 与参考不一致！"
  echo "  期望: $GOLDEN"
  echo "  实际: $ACTUAL"
  echo "  （若本次改动有意改变数值，如量化，请人工确认后更新本脚本的 GOLDEN）"
  exit 1
fi
