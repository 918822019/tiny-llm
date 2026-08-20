#!/usr/bin/env bash
# 为 Android（arm64-v8a）交叉编译 tinyqwen CLI。
#
#   ./scripts/build_android.sh               # NDK 自动探测（见 resolve_ndk）
#   ./scripts/build_android.sh /path/to/ndk  # 或显式指定
#
# 产物：build-android/runtime/tinyqwen
set -euo pipefail

cd "$(dirname "$0")/.."

# NDK 探测级联：$1 > $ANDROID_NDK > Android SDK 里的最新版 > homebrew cask 路径。
# 有效性以 toolchain 文件存在为准（光有目录不算）。
# 与 scripts/doctor_android.sh 的 resolve_ndk 保持同一套逻辑。
resolve_ndk() {
  local c
  if [[ -n "${1:-}" && -f "$1/build/cmake/android.toolchain.cmake" ]]; then
    echo "$1"; return 0
  fi
  if [[ -n "${ANDROID_NDK:-}" && -f "$ANDROID_NDK/build/cmake/android.toolchain.cmake" ]]; then
    echo "$ANDROID_NDK"; return 0
  fi
  # Android Studio SDK 安装的 NDK 带版本号目录，取最新（sort -V 按版本号排）。
  if [[ -d "$HOME/Library/Android/sdk/ndk" ]]; then
    c=$(ls -1 "$HOME/Library/Android/sdk/ndk" 2>/dev/null | sort -V | tail -1)
    if [[ -n "$c" && -f "$HOME/Library/Android/sdk/ndk/$c/build/cmake/android.toolchain.cmake" ]]; then
      echo "$HOME/Library/Android/sdk/ndk/$c"; return 0
    fi
  fi
  # homebrew: brew install --cask android-ndk
  for c in /opt/homebrew/share/android-ndk /usr/local/share/android-ndk; do
    [[ -f "$c/build/cmake/android.toolchain.cmake" ]] && { echo "$c"; return 0; }
  done
  return 1
}

if ! NDK=$(resolve_ndk "${1:-}"); then
  echo "error: 未找到 Android NDK" >&2
  echo "  安装：brew install --cask android-ndk" >&2
  echo "  或：  export ANDROID_NDK=/path/to/ndk（或作为 \$1 传入）" >&2
  exit 1
fi
echo "[build-android] 使用 NDK：$NDK"
TOOLCHAIN="$NDK/build/cmake/android.toolchain.cmake"

ABI="${ANDROID_ABI:-arm64-v8a}"
PLATFORM="${ANDROID_PLATFORM:-android-28}"
BUILD_DIR="${BUILD_DIR:-build-android}"

cmake -S . -B "$BUILD_DIR" \
  -DCMAKE_TOOLCHAIN_FILE="$TOOLCHAIN" \
  -DANDROID_ABI="$ABI" \
  -DANDROID_PLATFORM="$PLATFORM" \
  -DCMAKE_BUILD_TYPE=Release

cmake --build "$BUILD_DIR" -j"$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)"

echo "built: $BUILD_DIR/runtime/tinyqwen"
