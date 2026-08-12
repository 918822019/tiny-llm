# Android NDK 交叉编译

v1 使用命令行 binary，不做 App / JNI。

## 前置

- Android NDK（r25+ 测试通过即可）；
- adb，设备已开启开发者模式；
- 目标 ABI：`arm64-v8a`（INT4/NEON kernel 的主战场）。

## 编译

```bash
export ANDROID_NDK=$HOME/Library/Android/sdk/ndk/26.1.10909125   # 按实际路径

cmake -S . -B build-android \
  -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
  -DANDROID_ABI=arm64-v8a \
  -DANDROID_PLATFORM=android-28 \
  -DCMAKE_BUILD_TYPE=Release

cmake --build build-android -j
```

也可以直接用脚本：`scripts/build_android.sh`。

产物：

```text
build-android/runtime/tinyqwen
```

## 注意事项

- `ANDROID_PLATFORM` 需要 ≥ 28（C++17 运行时 + posix_memalign 行为稳定）；
- 不要引入第三方依赖；profiler 的 JSON 为手写输出，无需 JSON 库；
- 编译报 `libc++_shared.so` 相关问题时，确认 NDK toolchain 默认使用
  `ANDROID_STL=c++_static`（本项目无额外要求，保持默认即可）；
- Release 下默认 `-O3`；FP32 reference 不额外开 fast-math，保证与 PyTorch 对齐。
