#!/usr/bin/env bash
# ============================================================================
# doctor_android.sh — Android pipeline 一键预检（doctor）
# ============================================================================
# 用途：把「设备 → NDK → 模型 → 构建 → 基线」整条链路的阻塞点一次列全，
#       每项给 PASS/WARN/FAIL 和精确修复命令。在跑 verify_android / bench_android
#       之前先执行此脚本，快速定位环境缺失项。
#
# 用法：
#   ./scripts/doctor_android.sh
#
# 退出码：
#   0 = 全绿可跑
#   1 = 仅有警告（不影响跑，但建议补齐）
#   2 = 存在阻塞错误（必须先修复）
#
# 检查项目（按顺序）：
#   1. adb         — Android Debug Bridge 是否可用
#   2. 设备        — 是否有已连接并授权的设备
#   3. 设备能力    — ABI、磁盘空间、taskset、thermal zone（仅设备可用时）
#   4. NDK         — Android NDK 是否安装并可探测
#   5. 模型        — 主测模型文件是否存在
#   6. 构建产物    — Android binary 是否已编译
#   7. 基线        — Android 性能基线是否已建立
#
# 注意：设备相关检查只在设备已连接时执行，否则统一归到「设备」那一项报错。
# ============================================================================

# set -e: 任何命令失败立即退出；-u: 引用未定义变量报错；-o pipefail: 管道中任一命令失败则整体失败
set -euo pipefail

# 切换到项目根目录
cd "$(dirname "$0")/.."

# 错误计数器和警告计数器
ERRORS=0
WARNINGS=0

# 三个辅助函数：分别打印 PASS/WARN/FAIL 状态并更新计数器
pass() { echo "  ✅ $1"; }
warn() { echo "  ⚠️  $1"; WARNINGS=$((WARNINGS + 1)); }
fail() { echo "  ❌ $1"; ERRORS=$((ERRORS + 1)); }

# 当前主测精度为 fp16，MODEL_PRIMARY 是默认检查的模型文件名
MODEL_PRIMARY="model_f16.tqwen"
# MODEL 环境变量可覆盖默认检查对象
MODEL="${MODEL:-$MODEL_PRIMARY}"

echo "=== tinyqwen Android 预检 ==="

# ===========================================================================
# [1] adb 检查
# ===========================================================================
echo "[adb]"
# SERIAL 变量保存检测到的设备序列号，后续设备能力检查复用
SERIAL=""
if command -v adb >/dev/null 2>&1; then
  # adb 存在，打印版本信息
  pass "adb 可用：$(adb version 2>/dev/null | head -1)"
else
  # adb 不存在，提示安装方法
  fail "adb 不存在。安装：brew install --cask android-platform-tools"
fi

# ===========================================================================
# [2] 设备检查
# ===========================================================================
echo "[设备]"
if command -v adb >/dev/null 2>&1; then
  # adb devices 输出格式：首行标题，之后每行 "序列号<TAB>状态"
  # 筛选状态为 "device"（已授权且在线）的设备
  DEVICES_READY=$(adb devices 2>/dev/null | awk 'NR>1 && $2=="device" {print $1}')
  # 筛选状态为 "unauthorized"（需要用户在手机上确认授权）的设备
  DEVICES_UNAUTH=$(adb devices 2>/dev/null | awk 'NR>1 && $2=="unauthorized" {print $1}')
  # 筛选状态为 "offline"（离线/断开）的设备
  DEVICES_OFFLINE=$(adb devices 2>/dev/null | awk 'NR>1 && $2=="offline" {print $1}')

  if [[ -n "$DEVICES_READY" ]]; then
    # 取第一台已就绪设备的序列号
    SERIAL=$(echo "$DEVICES_READY" | head -1)
    # 统计已就绪设备数量
    N_READY=$(echo "$DEVICES_READY" | wc -l | tr -d ' ')
    # 多台设备时发出警告，告知使用哪一台
    [[ "$N_READY" -gt 1 ]] && warn "检测到 $N_READY 台设备，使用第一台 $SERIAL"
    pass "设备已连接并授权：$SERIAL"

    # 进一步验证 adb shell 能否实际执行命令（排除假连接）
    if adb -s "$SERIAL" shell "echo ok" 2>/dev/null | grep -q ok; then
      pass "adb shell 可执行命令"
    else
      fail "adb shell 执行失败（重新拔插数据线，或 adb kill-server && adb start-server）"
      SERIAL=""  # 清空序列号，后续设备能力检查将跳过
    fi

  elif [[ -n "$DEVICES_UNAUTH" ]]; then
    # 设备已连接但未授权 USB 调试
    fail "设备未授权（unauthorized）：解锁手机屏幕，点「允许 USB 调试」弹窗（勾选一律允许）"

  elif [[ -n "$DEVICES_OFFLINE" ]]; then
    # 设备处于离线状态
    fail "设备离线（offline）：重新拔插数据线；adb kill-server && adb start-server 后重试"

  else
    # 完全没有检测到任何设备
    fail "未检测到设备。逐项检查：① 数据线支持传输数据（非纯充电线）② 手机已开开发者选项 + USB 调试 ③ 手机已解锁"
  fi
else
  # 无 adb 时无法检查设备
  fail "跳过设备检查（无 adb）"
fi

# ===========================================================================
# [3] 设备能力检查（仅在设备可用时执行）
# ===========================================================================
if [[ -n "$SERIAL" ]]; then
  echo "[设备能力]"

  # 检查 CPU ABI：本项目只交叉编译 arm64-v8a
  ABI=$(adb -s "$SERIAL" shell "getprop ro.product.cpu.abi" 2>/dev/null | tr -d '\r')
  if [[ "$ABI" == "arm64-v8a" ]]; then
    pass "ABI = arm64-v8a（NEON/f16 kernel 主战场）"
  else
    fail "ABI = ${ABI:-unknown}，本项目只交叉编译 arm64-v8a"
  fi

  # ---- 磁盘空间检查 ----
  # 模型 + KV cache + profile 至少需要留 512MB 余量
  if [[ -f "$MODEL" ]]; then
    # 计算需要的空间：模型文件大小（KB）+ 512MB 余量
    # stat -f%z 是 macOS 语法，stat -c%s 是 Linux 语法，两者兼容
    NEED_KB=$(( ($(stat -f%z "$MODEL" 2>/dev/null || stat -c%s "$MODEL") / 1024) + 512 * 1024 ))
    # toybox df -k 输出格式：Filesystem 1K-blocks Used Available Use% Mounted
    # 取第 4 列（Available）作为可用空间
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

  # ---- taskset 检查（绑核用，减少测速抖动）----
  if adb -s "$SERIAL" shell "command -v taskset" 2>/dev/null | grep -q taskset; then
    pass "taskset 可用（绑核用）"
  else
    warn "设备无 taskset，bench 将无法绑大核（测速抖动会变大）"
  fi

  # ---- thermal zone 检查（热门禁用功能依赖）----
  if adb -s "$SERIAL" shell "ls /sys/class/thermal/thermal_zone*/temp 2>/dev/null | head -1" | grep -q temp; then
    pass "thermal zone 可读（热门禁用）"
  else
    warn "无法读取 /sys/class/thermal，热门禁将跳过"
  fi
fi

# ===========================================================================
# [4] NDK 检查
# ===========================================================================
echo "[NDK]"

# ---------------------------------------------------------------------------
# resolve_ndk() — NDK 路径探测函数（与 scripts/build_android.sh 同一套逻辑）
# ---------------------------------------------------------------------------
# 项目风格：每个脚本自包含，不 source 公共库。
# ---------------------------------------------------------------------------
resolve_ndk() {
  local c
  # 优先级 1：命令行参数
  if [[ -n "${1:-}" && -f "$1/build/cmake/android.toolchain.cmake" ]]; then
    echo "$1"; return 0
  fi
  # 优先级 2：环境变量
  if [[ -n "${ANDROID_NDK:-}" && -f "$ANDROID_NDK/build/cmake/android.toolchain.cmake" ]]; then
    echo "$ANDROID_NDK"; return 0
  fi
  # 优先级 3：Android SDK 里的最新版
  if [[ -d "$HOME/Library/Android/sdk/ndk" ]]; then
    c=$(ls -1 "$HOME/Library/Android/sdk/ndk" 2>/dev/null | sort -V | tail -1)
    if [[ -n "$c" && -f "$HOME/Library/Android/sdk/ndk/$c/build/cmake/android.toolchain.cmake" ]]; then
      echo "$HOME/Library/Android/sdk/ndk/$c"; return 0
    fi
  fi
  # 优先级 4：homebrew cask 路径
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

# ===========================================================================
# [5] 模型文件检查
# ===========================================================================
echo "[模型]"
if [[ -f "$MODEL" ]]; then
  # du -h 获取人类可读的文件大小，cut -f1 只取大小部分
  SIZE=$(du -h "$MODEL" | cut -f1)
  pass "主测模型：${MODEL}（${SIZE}）"
else
  fail "主测模型缺失：${MODEL}（导出：python tools/export_qwen_to_tiny.py --model <HF目录> --out ${MODEL}，fp16 需 --dtype f16）"
fi

# 列出其他已存在的模型文件（供参考）
for m in model.tqwen model_f16.tqwen model_i4.tqwen; do
  # 跳过当前主测模型，避免重复显示
  [[ "$m" == "$MODEL" ]] && continue
  # 如果该模型文件存在，打印其名称和大小
  [[ -f "$m" ]] && echo "      另有：${m}（$(du -h "$m" | cut -f1)）"
done

# ===========================================================================
# [6] 构建产物检查
# ===========================================================================
echo "[构建]"
if [[ -f "build-android/runtime/tinyqwen" ]]; then
  pass "Android binary 已编译：build-android/runtime/tinyqwen"
else
  warn "Android binary 未编译。执行：./scripts/build_android.sh"
fi

# ===========================================================================
# [7] 基线检查
# ===========================================================================
echo "[基线]"
if [[ -f "benchmarks/baseline_android.json" ]]; then
  # 读取基线 JSON 中的 label 和 decode_median_ms 字段
  pass "Android 基线已建立：$(python3 -c "import json;d=json.load(open('benchmarks/baseline_android.json'));print(d.get('label','?'), d.get('decode_median_ms','?'),'ms/tok')" 2>/dev/null || echo '?')"
else
  warn "无 Android 基线。首次成功测速后执行：MODEL=${MODEL} ./scripts/set_baseline_android.sh <label> --extra-args \"...\""
fi

# ===========================================================================
# 汇总结果并根据错误/警告数量决定退出码
# ===========================================================================
echo ""
if (( ERRORS == 0 && WARNINGS == 0 )); then
  # 全部通过
  echo "结论：✅ 全部就绪，可以跑 ./scripts/verify_android.sh / bench_android.sh"
  exit 0
elif (( ERRORS == 0 )); then
  # 只有警告，没有阻塞错误
  echo "结论：⚠️ 无阻塞错误，$WARNINGS 个警告（可按提示补齐后再跑）"
  exit 1
else
  # 有阻塞错误
  echo "结论：❌ $ERRORS 个阻塞错误，$WARNINGS 个警告——先按上面提示修复错误项"
  exit 2
fi
