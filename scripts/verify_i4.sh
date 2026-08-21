#!/usr/bin/env bash
# ============================================================================
# verify_i4.sh — INT4 量化正确性门禁
# ============================================================================
# 用途：验证 INT4 量化模型的推理结果与 fp32 参考实现的偏差在可接受范围内。
#       是量化 pipeline 的质量守门员，确保量化没有引入不可接受的精度损失。
#
# 用法：
#   ./scripts/verify_i4.sh [model_i4.tqwen] [model.tqwen]
#
# 参数说明：
#   $1 — INT4 量化模型路径（默认 model_i4.tqwen）
#   $2 — fp32 参考模型路径（默认 model.tqwen）
#
# 做四件事（按顺序）：
#   1. Build（Release）              — 确保代码能编过
#   2. Unit tests                    — 跑全部单元测试（含 INT4 kernel 测试）
#   3. Token comparison (fp32 vs I4) — 同一 prompt 下对比两者生成的 token，
#                                      允许至多 MAX_DIFF=2 个 token 不同
#   4. Accuracy audit                — 调用 verify_i4_accuracy.py 做更细粒度的
#                                      精度审计（可选，脚本缺失时跳过）
#
# 容差依据：INT4 RTN/HQQ 量化会改变权重精度，导致 logits 微小偏移，
#          greedy decoding 可能在 top-1 接近的位置产生分歧。经验上 <=2 个
#          token 差异属于正常范围。
#
# 依赖环境：
#   - cmake + C++ 编译器
#   - build/runtime/tinyqwen
#   - model_i4.tqwen（由 tools/export_qwen_to_tiny_i4.py 导出）
#   - model.tqwen（可选，缺失时使用硬编码 golden）
# ============================================================================

# set -e: 任何命令失败立即退出；-u: 引用未定义变量报错；-o pipefail: 管道中任一命令失败则整体失败
set -euo pipefail

# 切换到项目根目录
cd "$(dirname "$0")/.."

# INT4 量化模型路径（第一个参数，默认 model_i4.tqwen）
MODEL_I4="${1:-model_i4.tqwen}"
# fp32 参考模型路径（第二个参数，默认 model.tqwen）
MODEL_FP32="${2:-model.tqwen}"
# 可执行文件路径
BIN="build/runtime/tinyqwen"
# canonical prompt 的 token ID 列表（"中国的首都是"）
PROMPT_TOKENS="105538,59975,100132"
# 最多生成的新 token 数
MAX_NEW=16
# 最大序列长度限制
MAX_SEQ=32

# ---------------------------------------------------------------------------
# 步骤 1/4：编译
# ---------------------------------------------------------------------------
echo "[1/4] Build..."
# 并行编译，抑制输出
cmake --build build -j"$(sysctl -n hw.ncpu 2>/dev/null || nproc 2>/dev/null || echo 4)" >/dev/null

# ---------------------------------------------------------------------------
# 步骤 2/4：单元测试
# ---------------------------------------------------------------------------
echo "[2/4] Unit tests..."
# 运行测试套件，tail -1 只打印汇总行
./build/tests/tinyqwen_tests | tail -1

# ---------------------------------------------------------------------------
# 步骤 3/4：Token comparison (fp32 vs INT4)
# ---------------------------------------------------------------------------
# 如果 INT4 模型不存在，跳过 token 对比并提示导出方法
if [[ ! -f "$MODEL_I4" ]]; then
  echo "[3/4] SKIP token comparison ($MODEL_I4 not found)"
  echo "  Hint: run python tools/export_qwen_to_tiny_i4.py --out $MODEL_I4"
  exit 0
fi

echo "[3/4] Token comparison (fp32 vs INT4)..."

# 构造 runtime 公共参数：prompt tokens + 生成长度 + 序列限制 + 禁用 EOS
COMMON_ARGS="--tokens $PROMPT_TOKENS --max-new-tokens $MAX_NEW --max-seq-len $MAX_SEQ --eos -1"

# 获取 fp32 参考结果：优先用本地模型实跑，否则使用硬编码 golden
if [[ -f "$MODEL_FP32" ]]; then
  # 实际运行 fp32 模型获取参考 token 序列
  FP32_IDS=$("$BIN" --model "$MODEL_FP32" $COMMON_ARGS 2>/dev/null | grep '^generated_ids:' | sed 's/generated_ids: //')
else
  # fp32 模型不存在时使用硬编码 golden（与 verify.sh 中的 GOLDEN 一致）
  FP32_IDS="2130 198 32 13 94305 245 46553 198 33 13 64118 55135 198 34 13 66521"
fi

# 运行 INT4 模型获取实际 token 序列
I4_IDS=$("$BIN" --model "$MODEL_I4" $COMMON_ARGS 2>/dev/null | grep '^generated_ids:' | sed 's/generated_ids: //')

# 将空格分隔的 token 序列拆分为数组
IFS=' ' read -ra FP32_ARR <<< "$FP32_IDS"
IFS=' ' read -ra I4_ARR <<< "$I4_IDS"

# 逐位置对比，统计不同的 token 数量
DIFF_COUNT=0
# 最大允许差异数
MAX_DIFF=2
for i in "${!FP32_ARR[@]}"; do
  if [[ "${FP32_ARR[$i]:-}" != "${I4_ARR[$i]:-}" ]]; then
    ((DIFF_COUNT++)) || true  # || true 防止 set -e 在计数为 0 时退出
    # 打印每个不一致位置的双方值
    echo "  pos $i: fp32=${FP32_ARR[$i]:-?} vs i4=${I4_ARR[$i]:-?}"
  fi
done

# 判断差异是否在允许范围内
if [[ $DIFF_COUNT -le $MAX_DIFF ]]; then
  echo "PASS  token diff $DIFF_COUNT/$MAX_NEW (max allowed: $MAX_DIFF)"
else
  echo "FAIL  token diff $DIFF_COUNT/$MAX_NEW (max allowed: $MAX_DIFF)"
  echo "  fp32: $FP32_IDS"
  echo "  i4:   $I4_IDS"
  exit 1
fi

# ---------------------------------------------------------------------------
# 步骤 4/4：Accuracy audit（可选的更细粒度精度审计）
# ---------------------------------------------------------------------------
echo "[4/4] Accuracy audit..."
ACCURACY_SCRIPT="tools/verify_i4_accuracy.py"
# 仅当审计脚本和 fp32 模型都存在时才执行
if [[ -f "$ACCURACY_SCRIPT" && -f "$MODEL_FP32" ]]; then
  python3 "$ACCURACY_SCRIPT" --model-i4 "$MODEL_I4" --model-fp32 "$MODEL_FP32"
else
  echo "  SKIP ($ACCURACY_SCRIPT or $MODEL_FP32 not found)"
fi

echo ""
echo "=== INT4 verification complete ==="
