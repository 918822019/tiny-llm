#!/usr/bin/env bash
# ============================================================================
# verify.sh — 正确性门禁（pipeline 第②步）
# ============================================================================
# 用途：在测速之前，先证明"没算错"。是优化 pipeline 的质量守门员。
#       任一环节失败就非零退出，阻止后续的测速和记录。
#
# 用法：
#   ./scripts/verify.sh
#
# 做三件事（按顺序）：
#   1. 编译（Release 模式）— 确保代码能编过
#   2. 跑全部单元测试     — 验证各算子的数值正确性
#   3. golden token 对照  — 用 canonical prompt 跑真模型，生成的 token 必须和
#                           参考实现逐位一致。若不一致，说明本次改动改变了数值
#                           行为——这未必是错（如量化），但必须人工确认。
#
# 依赖环境：
#   - cmake + C++ 编译器
#   - model.tqwen（可选，缺失时跳过 golden token 对照）
#   - build/runtime/tinyqwen（由步骤 1 自动编译）
# ============================================================================

# set -e: 任何命令失败立即退出；-u: 引用未定义变量报错；-o pipefail: 管道中任一命令失败则整体失败
set -euo pipefail

# 切换到项目根目录
cd "$(dirname "$0")/.."

# 可执行文件路径
BIN="build/runtime/tinyqwen"
# 模型文件路径
MODEL="model.tqwen"

# canonical prompt（"中国的首都是"）+ 16 个 greedy token
# 这是参考实现（fp32 double 累加）产出的确定性结果，作为正确性的黄金标准
GOLDEN="generated_ids: 2130 198 32 13 94305 245 46553 198 33 13 64118 55135 198 34 13 66521"

# ---------------------------------------------------------------------------
# 步骤 1/3：编译
# ---------------------------------------------------------------------------
echo "[1/3] 编译..."
# 并行编译：sysctl -n hw.ncpu 是 macOS 获取核心数，nproc 是 Linux 等价物
# >/dev/null 抑制编译输出，只在失败时由 set -e 触发退出
cmake --build build -j"$(sysctl -n hw.ncpu 2>/dev/null || nproc 2>/dev/null || echo 4)" >/dev/null

# ---------------------------------------------------------------------------
# 步骤 2/3：单元测试
# ---------------------------------------------------------------------------
echo "[2/3] 单元测试..."
# 运行测试套件，tail -1 只打印最后一行汇总信息（如 "42 tests, 0 failed"）
./build/tests/tinyqwen_tests | tail -1

# ---------------------------------------------------------------------------
# 步骤 3/3：golden token 对照
# ---------------------------------------------------------------------------
# 如果模型文件不存在，跳过此步骤（允许在没有模型的环境下跑单测）
if [[ ! -f "$MODEL" ]]; then
  # ${MODEL} 加花括号：紧邻的全角逗号是 UTF-8 多字节，bash 会把高位字节
  # 并进变量名，$MODEL，被当成未定义变量，在 set -u 下直接报错
  echo "[3/3] 跳过 golden token 对照（找不到 ${MODEL}，先 export）"
  exit 0
fi

echo "[3/3] golden token 对照..."
# 运行 tinyqwen runtime：
#   --tokens         指定 prompt 的 token ID 列表（"中国的首都是"的 tokenization）
#   --max-new-tokens 最多生成 16 个新 token
#   --max-seq-len    最大序列长度限制为 32
#   --eos -1         禁用 EOS token（强制生成满 16 个 token，便于对比）
#   2>/dev/null      抑制 stderr 日志
# grep 提取 generated_ids 行
ACTUAL="$("$BIN" --model "$MODEL" --tokens 105538,59975,100132 \
  --max-new-tokens 16 --max-seq-len 32 --eos -1 2>/dev/null | grep '^generated_ids:')"

# 逐字符串比较实际输出与 golden 参考
if [[ "$ACTUAL" == "$GOLDEN" ]]; then
  echo "PASS  生成 token 与参考实现逐位一致"
else
  # 不一致时打印详细对比信息
  echo "FAIL  生成 token 与参考不一致！"
  echo "  期望: $GOLDEN"
  echo "  实际: $ACTUAL"
  echo "  （若本次改动有意改变数值，如量化，请人工确认后更新本脚本的 GOLDEN）"
  exit 1
fi
