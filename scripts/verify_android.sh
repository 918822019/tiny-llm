#!/usr/bin/env bash
# ============================================================================
# verify_android.sh — Android 正确性门禁
# ============================================================================
# 用途：在 Android 设备上跑 canonical prompt，验证生成 token 符合预期。
#       是 Android 优化 pipeline 的质量守门员。
#
# 用法：
#   ./scripts/verify_android.sh                      # fp32/fp16：逐位对 golden
#   MODEL=model_i4.tqwen ./scripts/verify_android.sh # 量化模型：容差对照
#
# 两种对照口径：
#   - 非量化模型（默认）：与参考实现逐位一致（fp32 double 累加，确定性结果）
#   - 量化模型（--quantized 或文件名含 _i4）：量化必然改变数值，逐位一致是错误
#     的期望。改为允许至多 MAX_TOKEN_DIFF 个 token 不同。
#
# 环境变量：
#   MODEL          — 模型文件路径（默认 model.tqwen）
#   BUILD_DIR      — 交叉编译产物目录（默认 build-android）
#   DEVICE_DIR     — 设备端工作目录（默认 /data/local/tmp/tinyqwen）
#   MAX_TOKEN_DIFF — 量化模型允许的最大 token 差异数（默认 2）
#
# 前置条件：
#   - adb 已连接设备
#   - scripts/build_android.sh 已编译（缺失时自动触发编译）
#   - 模型已导出
# ============================================================================

# set -e: 任何命令失败立即退出；-u: 引用未定义变量报错；-o pipefail: 管道中任一命令失败则整体失败
set -euo pipefail

# 切换到项目根目录
cd "$(dirname "$0")/.."

# 交叉编译产物目录
BUILD_DIR="${BUILD_DIR:-build-android}"
# 本地 binary 路径
BIN="$BUILD_DIR/runtime/tinyqwen"
# 模型文件路径
MODEL="${MODEL:-model.tqwen}"
# 设备端工作目录
DEVICE_DIR="${DEVICE_DIR:-/data/local/tmp/tinyqwen}"
# 量化模型允许的最大 token 差异数
MAX_TOKEN_DIFF="${MAX_TOKEN_DIFF:-2}"

# golden token 序列（canonical prompt "中国的首都是" + 16 greedy tokens）
GOLDEN_IDS="2130 198 32 13 94305 245 46553 198 33 13 64118 55135 198 34 13 66521"

# ---------------------------------------------------------------------------
# 判断是否为量化模型：显式 --quantized 参数 或 文件名含 _i4
# ---------------------------------------------------------------------------
QUANTIZED=0
for a in "$@"; do
  # 遍历所有参数，查找 --quantized 标志
  [[ "$a" == "--quantized" ]] && QUANTIZED=1
done
# basename 取文件名，检查是否包含 _i4 子串
[[ "$(basename "$MODEL")" == *_i4* ]] && QUANTIZED=1

# 如果 binary 不存在，自动触发编译
if [[ ! -f "$BIN" ]]; then
  echo "[verify-android] binary not found, building..."
  ./scripts/build_android.sh
fi

# 检查模型文件是否存在
if [[ ! -f "$MODEL" ]]; then
  echo "error: model file not found: ${MODEL}（先用 tools/export_qwen_to_tiny.py 导出）" >&2
  exit 1
fi

# 设备端模型路径：按文件名分开存，不同 dtype 可共存
DEVICE_MODEL="$DEVICE_DIR/models/$(basename "$MODEL")"

# ---------------------------------------------------------------------------
# 步骤 1/2：Push binary & model to device
# ---------------------------------------------------------------------------
echo "[1/2] Push binary & model to device..."

# 在设备上创建工作目录和模型子目录
adb shell "mkdir -p $DEVICE_DIR/models"
# push binary 到设备
adb push "$BIN" "$DEVICE_DIR/tinyqwen" >/dev/null
# 设置可执行权限
adb shell "chmod +x $DEVICE_DIR/tinyqwen"

# 增量传输模型：比较本地和设备端的文件大小
LOCAL_SIZE=$(stat -f%z "$MODEL" 2>/dev/null || stat -c%s "$MODEL")
REMOTE_SIZE=$(adb shell "stat -c%s $DEVICE_MODEL 2>/dev/null || echo 0" | tr -d '\r')
if [[ "$LOCAL_SIZE" != "$REMOTE_SIZE" ]]; then
  echo "  pushing $(basename "$MODEL") ($LOCAL_SIZE bytes)..."
  adb push "$MODEL" "$DEVICE_MODEL"
else
  echo "  $(basename "$MODEL") already on device, skipping push"
fi

# ---------------------------------------------------------------------------
# 步骤 2/2：Token 对照
# ---------------------------------------------------------------------------
echo "[2/2] Token 对照（$(basename "$MODEL")）..."

# 在设备上运行 tinyqwen，提取生成的 token ID 序列
# tr -d '\r' 去除 Windows 换行符，sed 去掉 "generated_ids: " 前缀
ACTUAL_IDS=$(adb shell "cd $DEVICE_DIR && ./tinyqwen \
  --model $DEVICE_MODEL \
  --tokens 105538,59975,100132 \
  --max-new-tokens 16 --max-seq-len 32 --eos -1 2>/dev/null" \
  | grep '^generated_ids:' | tr -d '\r' | sed 's/generated_ids: //')

# 检查设备端是否产出了有效结果
if [[ -z "$ACTUAL_IDS" ]]; then
  echo "FAIL  设备端没有产出 generated_ids（运行失败？）" >&2
  exit 1
fi

# ---------------------------------------------------------------------------
# 根据是否为量化模型选择不同的对照策略
# ---------------------------------------------------------------------------
if [[ "$QUANTIZED" -eq 0 ]]; then
  # ---- 非量化模型：逐位精确匹配 ----
  if [[ "$ACTUAL_IDS" == "$GOLDEN_IDS" ]]; then
    echo "PASS  生成 token 与参考实现逐位一致（Android）"
  else
    echo "FAIL  生成 token 与参考不一致！"
    echo "  期望: $GOLDEN_IDS"
    echo "  实际: $ACTUAL_IDS"
    echo "  （若本次改动有意改变数值，如量化，用 MODEL=model_i4.tqwen 或 --quantized）"
    exit 1
  fi
else
  # ---- 量化模型：容差对照 ----
  # 将空格分隔的 token 序列拆分为数组
  IFS=' ' read -ra GOLDEN_ARR <<< "$GOLDEN_IDS"
  IFS=' ' read -ra ACTUAL_ARR <<< "$ACTUAL_IDS"
  DIFF_COUNT=0
  # 逐个位置对比，统计不同的 token 数量
  for i in "${!GOLDEN_ARR[@]}"; do
    if [[ "${GOLDEN_ARR[$i]:-}" != "${ACTUAL_ARR[$i]:-}" ]]; then
      ((DIFF_COUNT++)) || true  # || true 防止 set -e 在计数为 0 时退出
      # 打印每个不一致的位置及双方值
      echo "  pos $i: golden=${GOLDEN_ARR[$i]:-?} vs 设备=${ACTUAL_ARR[$i]:-?}"
    fi
  done
  # 判断差异是否在允许范围内
  if [[ "$DIFF_COUNT" -le "$MAX_TOKEN_DIFF" ]]; then
    echo "PASS  量化容差口径：token diff $DIFF_COUNT/${#GOLDEN_ARR[@]}（上限 ${MAX_TOKEN_DIFF}）"
  else
    echo "FAIL  量化容差口径：token diff $DIFF_COUNT/${#GOLDEN_ARR[@]}（上限 ${MAX_TOKEN_DIFF}）"
    echo "  golden: $GOLDEN_IDS"
    echo "  设备:   $ACTUAL_IDS"
    exit 1
  fi
fi
