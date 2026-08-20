#!/usr/bin/env bash
# Android pipeline 一键预检（doctor）：把「设备 → NDK → 模型 → 构建 → 基线」
# 整条链路的阻塞点一次列全，每项给 PASS/WARN/FAIL 和精确修复命令。
#
#   ./scripts/doctor_android.sh
#
# 退出码：0 = 全绿可跑；1 = 仅有警告（不影响跑）；2 = 存在阻塞错误。
# 设备相关检查只在设备已连接时执行（否则统一归到「设备」那一项报错）。
set -euo pipefail
cd "$(dirname "$0")/.."

ERRORS=0
WARNINGS=0
pass() { echo "  ✅ $1"; }
warn() { echo "  ⚠️  $1"; WARNINGS=$((WARNINGS + 1)); }
fail() { echo "  ❌ $1"; ERRORS=$((ERRORS + 1)); }

MODEL_PRIMARY="model_f16.tqwen"   # 当前主测精度（fp16）
MODEL="${MODEL:-$MODEL_PRIMARY}"  # 允许 MODEL=xxx.tqwen 换检查对象

echo "=== tinyqwen Android 预检 ==="

# ---- 1. adb ---------------------------------------------------------------
echo "[adb]"
SERIAL=""
if command -v adb >/dev/null 2>&1; then
  pass "adb 可用：$(adb version 2>/dev/null | head -1)"
else
  fail "adb 不存在。安装：brew install --cask android-platform-tools"
fi

# ---- 2. 设备 ---------------------------------------------------------------
echo "[设备]"
if command -v adb >/dev/null 2>&1; then
  # adb devices 输出：首行标题，之后每行 "序列号<TAB>状态"。
  DEVICES_READY=$(adb devices 2>/dev/null | awk 'NR>1 && $2=="device" {print $1}')
  DEVICES_UNAUTH=$(adb devices 2>/dev/null | awk 'NR>1 && $2=="unauthorized" {print $1}')
  DEVICES_OFFLINE=$(adb devices 2>/dev/null | awk 'NR>1 && $2=="offline" {print $1}')
  if [[ -n "$DEVICES_READY" ]]; then
    SERIAL=$(echo "$DEVICES_READY" | head -1)
    N_READY=$(echo "$DEVICES_READY" | wc -l | tr -d ' ')
    [[ "$N_READY" -gt 1 ]] && warn "检测到 $N_READY 台设备，使用第一台 $SERIAL"
    pass "设备已连接并授权：$SERIAL"
    if adb -s "$SERIAL" shell "echo ok" 2>/dev/null | grep -q ok; then
      pass "adb shell 可执行命令"
    else
      fail "adb shell 执行失败（重新拔插数据线，或 adb kill-server && adb start-server）"
      SERIAL=""
    fi
  elif [[ -n "$DEVICES_UNAUTH" ]]; then
    fail "设备未授权（unauthorized）：解锁手机屏幕，点「允许 USB 调试」弹窗（勾选一律允许）"
  elif [[ -n "$DEVICES_OFFLINE" ]]; then
    fail "设备离线（offline）：重新拔插数据线；adb kill-server && adb start-server 后重试"
  else
    fail "未检测到设备。逐项检查：① 数据线支持传输数据（非纯充电线）② 手机已开开发者选项 + USB 调试 ③ 手机已解锁"
  fi
else
  fail "跳过设备检查（无 adb）"
fi

# ---- 3. 设备能力（仅设备可用时） ---------------------------------------------
if [[ -n "$SERIAL" ]]; then
  echo "[设备能力]"
  ABI=$(adb -s "$SERIAL" shell "getprop ro.product.cpu.abi" 2>/dev/null | tr -d '\r')
  if [[ "$ABI" == "arm64-v8a" ]]; then
    pass "ABI = arm64-v8a（NEON/f16 kernel 主战场）"
  else
    fail "ABI = ${ABI:-unknown}，本项目只交叉编译 arm64-v8a"
  fi
  # 磁盘空间：模型 + KV cache + profile，留 512MB 余量。
  if [[ -f "$MODEL" ]]; then
    NEED_KB=$(( ($(stat -f%z "$MODEL" 2>/dev/null || stat -c%s "$MODEL") / 1024) + 512 * 1024 ))
    # toybox df -k：列 = Filesystem 1K-blocks Used Available Use% Mounted
    AVAIL_KB=$(adb -s "$SERIAL" shell "df -k /data/local/tmp 2>/dev/null | tail -1" \
      | tr -d '\r' | awk '{print $4}')
    if [[ "${AVAIL_KB:-}" =~ ^[0-9]+$ ]]; then
      if (( AVAIL_KB >= NEED_KB )); then
        pass "设备空间充足：可用 $((AVAIL_KB / 1024))MB ≥ 需要 $((NEED_KB / 1024))MB"
      else
        fail "设备空间不足：可用 $((AVAIL_KB / 1024))MB < 需要 $((NEED_KB / 1024))MB（清理 /data/local/tmp 或换小模型）"
      fi
    else
      warn "无法读取设备磁盘空间（df 输出异常），跳过"
    fi
  fi
  if adb -s "$SERIAL" shell "command -v taskset" 2>/dev/null | grep -q taskset; then
    pass "taskset 可用（绑核用）"
  else
    warn "设备无 taskset，bench 将无法绑大核（测速抖动会变大）"
  fi
  if adb -s "$SERIAL" shell "ls /sys/class/thermal/thermal_zone*/temp 2>/dev/null | head -1" | grep -q temp; then
    pass "thermal zone 可读（热门禁用）"
  else
    warn "无法读取 /sys/class/thermal，热门禁将跳过"
  fi
fi

# ---- 4. NDK -----------------------------------------------------------------
echo "[NDK]"
# 与 scripts/build_android.sh 同一套探测级联（两处各自内嵌，项目风格脚本自包含）。
resolve_ndk() {
  local c
  if [[ -n "${1:-}" && -f "$1/build/cmake/android.toolchain.cmake" ]]; then
    echo "$1"; return 0
  fi
  if [[ -n "${ANDROID_NDK:-}" && -f "$ANDROID_NDK/build/cmake/android.toolchain.cmake" ]]; then
    echo "$ANDROID_NDK"; return 0
  fi
  if [[ -d "$HOME/Library/Android/sdk/ndk" ]]; then
    c=$(ls -1 "$HOME/Library/Android/sdk/ndk" 2>/dev/null | sort -V | tail -1)
    if [[ -n "$c" && -f "$HOME/Library/Android/sdk/ndk/$c/build/cmake/android.toolchain.cmake" ]]; then
      echo "$HOME/Library/Android/sdk/ndk/$c"; return 0
    fi
  fi
  for c in /opt/homebrew/share/android-ndk /usr/local/share/android-ndk; do
    [[ -f "$c/build/cmake/android.toolchain.cmake" ]] && { echo "$c"; return 0; }
  done
  return 1
}
if NDK_PATH=$(resolve_ndk "${1:-}"); then
  pass "NDK：$NDK_PATH"
else
  fail "未找到 Android NDK。安装：brew install --cask android-ndk（约 1GB），或 export ANDROID_NDK=/path/to/ndk"
fi

# ---- 5. 模型 -----------------------------------------------------------------
echo "[模型]"
if [[ -f "$MODEL" ]]; then
  SIZE=$(du -h "$MODEL" | cut -f1)
  pass "主测模型：${MODEL}（${SIZE}）"
else
  fail "主测模型缺失：${MODEL}（导出：python tools/export_qwen_to_tiny.py --model <HF目录> --out ${MODEL}，fp16 需 --dtype f16）"
fi
for m in model.tqwen model_f16.tqwen model_i4.tqwen; do
  [[ "$m" == "$MODEL" ]] && continue
  [[ -f "$m" ]] && echo "      另有：${m}（$(du -h "$m" | cut -f1)）"
done

# ---- 6. 构建产物 ---------------------------------------------------------------
echo "[构建]"
if [[ -f "build-android/runtime/tinyqwen" ]]; then
  pass "Android binary 已编译：build-android/runtime/tinyqwen"
else
  warn "Android binary 未编译。执行：./scripts/build_android.sh"
fi

# ---- 7. 基线 -------------------------------------------------------------------
echo "[基线]"
if [[ -f "benchmarks/baseline_android.json" ]]; then
  pass "Android 基线已建立：$(python3 -c "import json;d=json.load(open('benchmarks/baseline_android.json'));print(d.get('label','?'), d.get('decode_median_ms','?'),'ms/tok')" 2>/dev/null || echo '?')"
else
  warn "无 Android 基线。首次成功测速后执行：MODEL=${MODEL} ./scripts/set_baseline_android.sh <label> --extra-args \"...\""
fi

# ---- 汇总 -----------------------------------------------------------------------
echo ""
if (( ERRORS == 0 && WARNINGS == 0 )); then
  echo "结论：✅ 全部就绪，可以跑 ./scripts/verify_android.sh / bench_android.sh"
  exit 0
elif (( ERRORS == 0 )); then
  echo "结论：⚠️ 无阻塞错误，$WARNINGS 个警告（可按提示补齐后再跑）"
  exit 1
else
  echo "结论：❌ $ERRORS 个阻塞错误，$WARNINGS 个警告——先按上面提示修复错误项"
  exit 2
fi
