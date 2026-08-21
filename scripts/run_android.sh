#!/usr/bin/env bash
# ============================================================================
# run_android.sh — 将 tinyqwen binary + 模型 push 到 Android 设备并运行 decode
# ============================================================================
# 用途：自动化完成「push binary → push 模型 → 在设备上执行推理」的完整流程。
#       支持增量传输：仅当设备上缺少模型或大小不一致时才重新 push（模型可达 2GB）。
#
# 用法：
#   ./scripts/run_android.sh <model.tqwen> <tokens.json> [tinyqwen 的额外参数...]
#
# 示例：
#   ./scripts/run_android.sh model.tqwen prompt_tokens.json
#   ./scripts/run_android.sh model_f16.tqwen tokens.json --matvec-impl neon
#
# 前置条件：
#   - PATH 里有 adb
#   - 已连接一台 Android 设备
#   - scripts/build_android.sh 已完成交叉编译
#
# 环境变量：
#   BUILD_DIR  — 交叉编译产物目录（默认 build-android）
#   DEVICE_DIR — 设备端工作目录（默认 /data/local/tmp/tinyqwen）
#
# 设备端文件布局：
#   $DEVICE_DIR/tinyqwen          — 可执行文件
#   $DEVICE_DIR/models/<name>     — 模型文件（按文件名分开存，fp32/fp16/i4 可共存）
#   $DEVICE_DIR/tokens.json       — prompt token 文件
#   $DEVICE_DIR/profile.json      — 性能剖析输出
# ============================================================================

# set -e: 任何命令失败立即退出；-u: 引用未定义变量报错；-o pipefail: 管道中任一命令失败则整体失败
set -euo pipefail

# 切换到项目根目录
cd "$(dirname "$0")/.."

# 第一个参数为模型文件路径，第二个参数为 token 文件路径
MODEL="${1:?usage: run_android.sh <model.tqwen> <tokens.json> [tinyqwen args...]}"
TOKENS="${2:?usage: run_android.sh <model.tqwen> <tokens.json> [tinyqwen args...]}"
# shift 2 移除已消费的两个参数，剩余参数作为 tinyqwen 的额外运行参数
shift 2

# 交叉编译产物目录
BUILD_DIR="${BUILD_DIR:-build-android}"
# 设备端工作目录
DEVICE_DIR="${DEVICE_DIR:-/data/local/tmp/tinyqwen}"
# 本地 binary 路径
BIN="$BUILD_DIR/runtime/tinyqwen"
# 设备端模型路径：按文件名分开存储，不同 dtype 的模型可共存，切换时不用重传整个模型
DEVICE_MODEL="$DEVICE_DIR/models/$(basename "$MODEL")"

# 检查本地 binary 是否存在
[[ -f "$BIN" ]] || { echo "error: $BIN not found; run scripts/build_android.sh first" >&2; exit 1; }
# 检查模型文件是否存在
[[ -f "$MODEL" ]] || { echo "error: model file not found: $MODEL" >&2; exit 1; }
# 检查 token 文件是否存在
[[ -f "$TOKENS" ]] || { echo "error: tokens file not found: $TOKENS" >&2; exit 1; }

# 在设备上创建工作目录和模型子目录
adb shell "mkdir -p $DEVICE_DIR/models"
# push binary 到设备（抑制 adb push 的输出噪声）
adb push "$BIN" "$DEVICE_DIR/tinyqwen" >/dev/null
# 设置可执行权限
adb shell "chmod +x $DEVICE_DIR/tinyqwen"

# ---------------------------------------------------------------------------
# 增量传输模型：仅当设备上缺少该模型或大小不一致时才重新 push
# ---------------------------------------------------------------------------
# stat -f%z 是 macOS 语法获取文件大小，stat -c%s 是 Linux 语法
LOCAL_SIZE=$(stat -f%z "$MODEL" 2>/dev/null || stat -c%s "$MODEL")
# 读取设备端文件大小，不存在时返回 0
REMOTE_SIZE=$(adb shell "stat -c%s $DEVICE_MODEL 2>/dev/null || echo 0" | tr -d '\r')
if [[ "$LOCAL_SIZE" != "$REMOTE_SIZE" ]]; then
  # 大小不一致或文件不存在，需要重新 push
  echo "pushing $(basename "$MODEL") ($LOCAL_SIZE bytes)..."
  adb push "$MODEL" "$DEVICE_MODEL"
else
  # 大小一致，跳过传输（节省时间，模型可达 2GB）
  echo "$(basename "$MODEL") already on device ($REMOTE_SIZE bytes), skipping push"
fi

# push token 文件到设备
adb push "$TOKENS" "$DEVICE_DIR/tokens.json" >/dev/null

# ---------------------------------------------------------------------------
# 构造设备端命令行参数
# ---------------------------------------------------------------------------
# 额外参数逐个进行单引号转义后拼进 adb shell 命令串。
# 原因：整条命令要过两层 shell（本机 bash → 设备端 sh），未转义的参数中
# 如果包含空格或特殊字符会被错误拆分。
REMOTE_ARGS=""
for a in "$@"; do
  # sed 将参数中的单引号替换为 '\'' （结束当前单引号串、插入转义单引号、重新开始）
  REMOTE_ARGS="$REMOTE_ARGS '$(printf '%s' "$a" | sed "s/'/'\\\\''/g")'"
done

echo "--- run ---"
# 在设备上执行 tinyqwen：
#   cd 到工作目录
#   --model         指定模型文件路径
#   --tokens-json   指定 prompt token 文件
#   --profile-out   指定性能剖析输出路径
#   $REMOTE_ARGS    拼接的额外参数
adb shell "cd $DEVICE_DIR && ./tinyqwen \
  --model $DEVICE_MODEL \
  --tokens-json $DEVICE_DIR/tokens.json \
  --profile-out $DEVICE_DIR/profile.json$REMOTE_ARGS"
