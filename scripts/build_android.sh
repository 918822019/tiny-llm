#!/usr/bin/env bash
# Cross-compile the tinyqwen CLI for Android (arm64-v8a).
#
#   export ANDROID_NDK=/path/to/ndk      # or pass it as $1
#   ./scripts/build_android.sh
#
# Output: build-android/runtime/tinyqwen
set -euo pipefail

cd "$(dirname "$0")/.."

NDK="${1:-${ANDROID_NDK:-}}"
if [[ -z "$NDK" ]]; then
  echo "error: set ANDROID_NDK or pass the NDK path as \$1" >&2
  exit 1
fi
TOOLCHAIN="$NDK/build/cmake/android.toolchain.cmake"
if [[ ! -f "$TOOLCHAIN" ]]; then
  echo "error: toolchain not found: $TOOLCHAIN" >&2
  exit 1
fi

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
