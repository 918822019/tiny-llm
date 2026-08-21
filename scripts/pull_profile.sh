#!/usr/bin/env bash
# ============================================================================
# pull_profile.sh — 从 Android 设备拉回 profile JSON
# ============================================================================
# 用途：将 run_android.sh 运行后在设备上生成的性能剖析数据（profile.json）
#       拉回到本地，供后续分析使用。
#
# 用法：
#   ./scripts/pull_profile.sh [local_out.json]
#
# 参数说明：
#   local_out.json — 本地保存路径（默认 profile_android.json）
#
# 环境变量：
#   DEVICE_DIR — 设备端工作目录（默认 /data/local/tmp/tinyqwen）
#
# 前置条件：
#   - adb 可用且设备已连接
#   - 设备上已有 profile.json（由 run_android.sh --profile-out 生成）
# ============================================================================

# set -e: 任何命令失败立即退出；-u: 引用未定义变量报错；-o pipefail: 管道中任一命令失败则整体失败
set -euo pipefail

# 设备端工作目录（与 run_android.sh 保持一致）
DEVICE_DIR="${DEVICE_DIR:-/data/local/tmp/tinyqwen}"
# 本地输出文件路径，第一个参数可覆盖默认值
OUT="${1:-profile_android.json}"

# 使用 adb pull 将设备端的 profile.json 拉到本地指定路径
adb pull "$DEVICE_DIR/profile.json" "$OUT"

# 打印保存成功提示
echo "saved -> $OUT"
