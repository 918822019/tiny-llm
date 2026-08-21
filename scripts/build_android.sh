#!/usr/bin/env bash
# ============================================================================
# build_android.sh — 为 Android（arm64-v8a）交叉编译 tinyqwen CLI
# ============================================================================
# 用途：使用 Android NDK 将 tinyqwen 交叉编译为 arm64-v8a 架构的可执行文件，
#       供 Android 设备端运行。
#
# 用法：
#   ./scripts/build_android.sh               # NDK 自动探测（见 resolve_ndk）
#   ./scripts/build_android.sh /path/to/ndk  # 或显式指定 NDK 路径
#
# 产物：build-android/runtime/tinyqwen
#
# 环境变量：
#   ANDROID_NDK      — NDK 根目录（优先级低于命令行参数 $1）
#   ANDROID_ABI      — 目标 ABI（默认 arm64-v8a）
#   ANDROID_PLATFORM — 最低 API level（默认 android-28）
#   BUILD_DIR        — 构建输出目录（默认 build-android）
#
# NDK 探测级联顺序：
#   1. 命令行参数 $1
#   2. $ANDROID_NDK 环境变量
#   3. Android SDK 里的最新版 NDK（~/Library/Android/sdk/ndk/）
#   4. homebrew cask 安装路径（/opt/homebrew/share/android-ndk 等）
# ============================================================================

# set -e: 任何命令失败立即退出；-u: 引用未定义变量报错；-o pipefail: 管道中任一命令失败则整体失败
set -euo pipefail

# 切换到项目根目录
cd "$(dirname "$0")/.."

# ---------------------------------------------------------------------------
# resolve_ndk() — NDK 路径探测函数
# ---------------------------------------------------------------------------
# 按优先级依次尝试多个候选路径，以 toolchain cmake 文件存在为准。
# 参数：$1 = 可选的显式 NDK 路径
# 输出：找到的 NDK 根目录路径（stdout）
# 返回：0 = 找到，1 = 未找到
# 与 scripts/doctor_android.sh 的 resolve_ndk 保持同一套逻辑。
# ---------------------------------------------------------------------------
resolve_ndk() {
  local c  # 局部变量，用于存储候选路径

  # 优先级 1：命令行参数 $1 指定的路径
  if [[ -n "${1:-}" && -f "$1/build/cmake/android.toolchain.cmake" ]]; then
    echo "$1"; return 0  # 找到，输出路径并成功返回
  fi

  # 优先级 2：$ANDROID_NDK 环境变量
  if [[ -n "${ANDROID_NDK:-}" && -f "$ANDROID_NDK/build/cmake/android.toolchain.cmake" ]]; then
    echo "$ANDROID_NDK"; return 0
  fi

  # 优先级 3：Android Studio SDK 安装的 NDK（带版本号子目录）
  # sort -V 按版本号排序，tail -1 取最新版本
  if [[ -d "$HOME/Library/Android/sdk/ndk" ]]; then
    c=$(ls -1 "$HOME/Library/Android/sdk/ndk" 2>/dev/null | sort -V | tail -1)
    if [[ -n "$c" && -f "$HOME/Library/Android/sdk/ndk/$c/build/cmake/android.toolchain.cmake" ]]; then
      echo "$HOME/Library/Android/sdk/ndk/$c"; return 0
    fi
  fi

  # 优先级 4：homebrew cask 安装路径（Apple Silicon 和 Intel Mac 两种路径）
  for c in /opt/homebrew/share/android-ndk /usr/local/share/android-ndk; do
    [[ -f "$c/build/cmake/android.toolchain.cmake" ]] && { echo "$c"; return 0; }
  done

  return 1  # 所有候选都未找到
}

# 调用 resolve_ndk 探测 NDK，失败则打印安装指引并退出
if ! NDK=$(resolve_ndk "${1:-}"); then
  echo "error: 未找到 Android NDK" >&2
  echo "  安装：brew install --cask android-ndk" >&2
  echo "  或：  export ANDROID_NDK=/path/to/ndk（或作为 \$1 传入）" >&2
  exit 1
fi

# 打印实际使用的 NDK 路径
echo "[build-android] 使用 NDK：$NDK"
# 拼接 toolchain cmake 文件的完整路径
TOOLCHAIN="$NDK/build/cmake/android.toolchain.cmake"

# 目标 ABI，默认 arm64-v8a（NEON/f16 kernel 的主战场）
ABI="${ANDROID_ABI:-arm64-v8a}"
# 最低 Android API level，默认 28（Android 9.0）
PLATFORM="${ANDROID_PLATFORM:-android-28}"
# 构建输出目录
BUILD_DIR="${BUILD_DIR:-build-android}"

# 调用 cmake 配置交叉编译：
#   -S .             源码目录为当前目录
#   -B "$BUILD_DIR"  构建输出目录
#   -DCMAKE_TOOLCHAIN_FILE  指定 Android NDK 的 toolchain 文件
#   -DANDROID_ABI          目标 CPU 架构
#   -DANDROID_PLATFORM     最低 API level
#   -DCMAKE_BUILD_TYPE     Release 模式（开启优化）
cmake -S . -B "$BUILD_DIR" \
  -DCMAKE_TOOLCHAIN_FILE="$TOOLCHAIN" \
  -DANDROID_ABI="$ABI" \
  -DANDROID_PLATFORM="$PLATFORM" \
  -DCMAKE_BUILD_TYPE=Release

# 并行编译：getconf _NPROCESSORS_ONLN 获取 CPU 核心数，失败时回退到 4
cmake --build "$BUILD_DIR" -j"$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)"

# 打印产物路径
echo "built: $BUILD_DIR/runtime/tinyqwen"
