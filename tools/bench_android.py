#!/usr/bin/env python3
"""Android 端侧标准基准测试。

在设备上跑标准负载，拉回 profile JSON，在本机算统计。
与 bench.py 共用负载定义和 summarize 逻辑，保证数字口径一致。

特性：
- 热门禁（thermal gate）：测量前等设备降温，消除热降频干扰
- 绑核（core pinning）：taskset 绑大核，减少大小核迁移抖动
- 同场 A/B：--extra-args 非空时自动启用，交错测量抗漂移
- 回归检测：自动对比 baseline，超阈值报警

用法：
    python tools/bench_android.py --label neon-android
    python tools/bench_android.py --label neon-android --runs 3 --extra-args "--matvec-impl neon"
    python tools/bench_android.py --label neon-android --pin-cores all --no-thermal-gate
    python tools/bench_android.py --label neon-android --fail-on-regression
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import bench  # noqa: E402

DEVICE_DIR = "/data/local/tmp/tinyqwen"
BUILD_DIR_DEFAULT = "build-android"

# 热门禁默认参数
THERMAL_MAX_DEFAULT = 45000  # millidegree Celsius
THERMAL_POLL_INTERVAL = 10  # seconds
THERMAL_TIMEOUT = 180  # seconds

# ---------- adb helpers ----------

def adb(*args: str, capture=True) -> str:
    cmd = ["adb"] + list(args)
    r = subprocess.run(cmd, capture_output=capture, text=True, check=True)
    return r.stdout.strip() if capture else ""


def adb_shell(cmd: str) -> str:
    return adb("shell", cmd)


def adb_getprop(prop: str) -> str:
    return adb_shell(f"getprop {prop}")


# ---------- thermal gate ----------

def read_device_thermal() -> int | None:
    """读取设备 CPU thermal zone 温度（millidegree Celsius）。失败返回 None。"""
    try:
        raw = adb_shell(
            "cat /sys/class/thermal/thermal_zone0/temp 2>/dev/null || echo -1")
        val = int(raw.strip())
        return val if val >= 0 else None
    except (ValueError, subprocess.CalledProcessError):
        return None


def wait_for_thermal_cool(max_temp_mc: int = THERMAL_MAX_DEFAULT,
                          poll_interval: int = THERMAL_POLL_INTERVAL,
                          timeout: int = THERMAL_TIMEOUT) -> None:
    """等待设备温度降到阈值以下。超时则打印警告继续。"""
    temp = read_device_thermal()
    if temp is None:
        print("  [thermal] 无法读取设备温度，跳过热门禁")
        return
    if temp <= max_temp_mc:
        return

    print(f"  [thermal] 设备温度 {temp / 1000:.1f}°C > 阈值 {max_temp_mc / 1000:.1f}°C，等待降温...")
    start = time.time()
    while temp is not None and temp > max_temp_mc:
        if time.time() - start > timeout:
            print(f"  [thermal] ⚠️ 等待超时 {timeout}s，当前 {temp / 1000:.1f}°C，继续测量（结果可能受热降频影响）")
            return
        time.sleep(poll_interval)
        temp = read_device_thermal()
        if temp is not None:
            elapsed = int(time.time() - start)
            print(f"  [thermal] {elapsed}s 已等待，当前 {temp / 1000:.1f}°C")
    print(f"  [thermal] 温度已降至阈值以下，继续测量")


# ---------- core pinning ----------

def detect_big_cores() -> list[int]:
    """检测设备大核编号（频率最高的那组 CPU）。"""
    try:
        raw = adb_shell(
            "for cpu in /sys/devices/system/cpu/cpu[0-9]*; do "
            "echo $(basename $cpu):$(cat $cpu/cpufreq/cpuinfo_max_freq 2>/dev/null || echo 0); "
            "done")
        freqs = {}
        for line in raw.strip().splitlines():
            parts = line.split(":")
            if len(parts) == 2:
                cpu_id = int(parts[0].replace("cpu", ""))
                freq = int(parts[1])
                freqs[cpu_id] = freq
        if not freqs:
            return []
        max_freq = max(freqs.values())
        return sorted(cpu_id for cpu_id, f in freqs.items() if f == max_freq)
    except (subprocess.CalledProcessError, ValueError):
        return []


def build_taskset_prefix(pin_cores: str) -> str:
    """返回 taskset 命令前缀。返回空字符串表示不绑核。"""
    if pin_cores == "all":
        return ""
    if pin_cores == "big":
        cores = detect_big_cores()
        if not cores:
            print("  [pin] 无法检测大核，跳过绑核")
            return ""
        mask = 0
        for c in cores:
            mask |= (1 << c)
        print(f"  [pin] 绑大核: cpu{','.join(map(str, cores))} (mask=0x{mask:x})")
        return f"taskset 0x{mask:x} "
    # 手动指定：逗号分隔的 CPU 编号
    try:
        cores = [int(c.strip()) for c in pin_cores.split(",")]
        mask = 0
        for c in cores:
            mask |= (1 << c)
        print(f"  [pin] 绑核: cpu{','.join(map(str, cores))} (mask=0x{mask:x})")
        return f"taskset 0x{mask:x} "
    except ValueError:
        print(f"  [pin] 无法解析 --pin-cores '{pin_cores}'，跳过绑核")
        return ""


# ---------- device env ----------

def device_env_info() -> dict:
    info = {}
    try:
        info["device_model"] = adb_getprop("ro.product.model")
    except Exception:
        info["device_model"] = "unknown"
    try:
        info["soc"] = adb_getprop("ro.board.platform")
    except Exception:
        info["soc"] = "unknown"
    try:
        info["cpu_abi"] = adb_getprop("ro.product.cpu.abi")
    except Exception:
        info["cpu_abi"] = "unknown"
    try:
        info["android_version"] = adb_getprop("ro.build.version.release")
    except Exception:
        info["android_version"] = "unknown"
    try:
        freq = adb_shell("cat /sys/devices/system/cpu/cpu7/cpufreq/cpuinfo_max_freq 2>/dev/null "
                         "|| cat /sys/devices/system/cpu/cpu3/cpufreq/cpuinfo_max_freq 2>/dev/null "
                         "|| echo unknown")
        info["big_core_max_khz"] = freq
    except Exception:
        info["big_core_max_khz"] = "unknown"
    try:
        info["governor"] = adb_shell(
            "cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo unknown")
    except Exception:
        info["governor"] = "unknown"
    try:
        info["thermal_zone0"] = adb_shell(
            "cat /sys/class/thermal/thermal_zone0/temp 2>/dev/null || echo unknown")
    except Exception:
        info["thermal_zone0"] = "unknown"
    return info


# ---------- push / run ----------

def ensure_pushed(binary: str, model: str) -> None:
    """确保 binary 和 model 已 push 到设备。模型按大小跳过重复 push。"""
    adb_shell(f"mkdir -p {DEVICE_DIR}")
    adb("push", binary, f"{DEVICE_DIR}/tinyqwen")
    adb_shell(f"chmod +x {DEVICE_DIR}/tinyqwen")

    local_size = os.path.getsize(model)
    remote_size = adb_shell(f"stat -c%s {DEVICE_DIR}/model.tqwen 2>/dev/null || echo 0")
    try:
        remote_size = int(remote_size.strip())
    except ValueError:
        remote_size = 0
    if local_size != remote_size:
        print(f"  pushing model ({local_size} bytes)...")
        adb("push", model, f"{DEVICE_DIR}/model.tqwen", capture=False)
    else:
        print(f"  model already on device ({remote_size} bytes), skipping push")


# 模块级状态：由 main() 设置，run_once_on_device 使用
_taskset_prefix = ""
_thermal_gate_enabled = True
_thermal_max = THERMAL_MAX_DEFAULT


def run_once_on_device(extra_args: list[str] | None = None) -> dict:
    """在设备上跑一次标准负载，拉回 profile JSON 并返回解析结果。"""
    if _thermal_gate_enabled:
        wait_for_thermal_cool(_thermal_max)

    tokens_csv = ",".join(map(str, bench.CANONICAL_PROMPT))
    cmd_parts = [
        f"cd {DEVICE_DIR} &&",
        f"{_taskset_prefix}./tinyqwen",
        f"--model {DEVICE_DIR}/model.tqwen",
        f"--tokens {tokens_csv}",
        f"--max-new-tokens {bench.DECODE_TOKENS}",
        f"--max-seq-len {bench.MAX_SEQ_LEN}",
        f"--eos -1",
        f"--profile-out {DEVICE_DIR}/profile.json",
    ]
    if extra_args:
        cmd_parts.extend(extra_args)
    device_cmd = " ".join(cmd_parts)
    adb_shell(device_cmd)

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
        local_profile = tf.name
    adb("pull", f"{DEVICE_DIR}/profile.json", local_profile)
    with open(local_profile) as f:
        profile = json.load(f)
    Path(local_profile).unlink(missing_ok=True)
    return profile


def measure(runs: int, extra_args: list[str] | None = None) -> tuple[float, float, dict]:
    medians, p95s, last = [], [], None
    for r in range(runs):
        prof = run_once_on_device(extra_args)
        st = bench.summarize(prof)
        medians.append(st["decode_median_ms"])
        p95s.append(st["decode_p95_ms"])
        last = st
        print(f"  run {r + 1}/{runs}: median={st['decode_median_ms']:.2f}ms "
              f"p95={st['decode_p95_ms']:.2f}ms")
    return statistics.median(medians), statistics.median(p95s), last


def measure_ab(runs: int, variant_args: list[str]) -> tuple[float, float, float, float, dict]:
    """同场 A/B：每遍先测对照（无额外参数 = ref）再测变体，交错抗热降频。"""
    ctrl_meds, ctrl_p95s, var_meds, var_p95s = [], [], [], []
    last_var = None
    for r in range(runs):
        cs = bench.summarize(run_once_on_device(None))
        vs = bench.summarize(run_once_on_device(variant_args))
        ctrl_meds.append(cs["decode_median_ms"])
        ctrl_p95s.append(cs["decode_p95_ms"])
        var_meds.append(vs["decode_median_ms"])
        var_p95s.append(vs["decode_p95_ms"])
        last_var = vs
        print(f"  run {r + 1}/{runs}: 对照={cs['decode_median_ms']:.2f}ms  "
              f"变体={vs['decode_median_ms']:.2f}ms")
    return (statistics.median(ctrl_meds), statistics.median(ctrl_p95s),
            statistics.median(var_meds), statistics.median(var_p95s), last_var)


def main() -> None:
    global _taskset_prefix, _thermal_gate_enabled, _thermal_max

    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--binary", default=None,
                   help="交叉编译产物路径（默认 build-android/runtime/tinyqwen）")
    p.add_argument("--model", default="model.tqwen")
    p.add_argument("--label", default="unlabeled")
    p.add_argument("--runs", type=int, default=1)
    p.add_argument("--json", default=None, help="结果写 JSON 文件")
    p.add_argument("--extra-args", default="",
                   help="原样传给设备端 binary 的额外 CLI 参数")
    # 热门禁
    p.add_argument("--thermal-max", type=int, default=THERMAL_MAX_DEFAULT,
                   help=f"热门禁温度阈值（millidegree Celsius，默认 {THERMAL_MAX_DEFAULT}）")
    p.add_argument("--no-thermal-gate", action="store_true",
                   help="跳过热门禁（快速迭代用）")
    # 绑核
    p.add_argument("--pin-cores", default="big",
                   help="绑核模式：big（默认，绑大核）/ all（不绑）/ 逗号分隔的 CPU 编号")
    # 回归检测
    p.add_argument("--baseline", default="benchmarks/baseline_android.json",
                   help="回归检测对比的基线文件")
    p.add_argument("--no-check-regression", action="store_true",
                   help="跳过回归检测")
    p.add_argument("--regression-threshold", type=float, default=5.0,
                   help="回归阈值百分比（默认 5）")
    p.add_argument("--fail-on-regression", action="store_true",
                   help="回归时非零退出（CI 用）")
    args = p.parse_args()

    build_dir = os.environ.get("BUILD_DIR", BUILD_DIR_DEFAULT)
    binary = args.binary or f"{build_dir}/runtime/tinyqwen"
    if not os.path.isfile(binary):
        sys.exit(f"error: binary not found: {binary}; run scripts/build_android.sh first")
    if not os.path.isfile(args.model):
        sys.exit(f"error: model not found: {args.model}")

    extra = shlex.split(args.extra_args)
    ab_mode = bool(extra)
    runs = max(1, args.runs)

    # 配置热门禁
    _thermal_gate_enabled = not args.no_thermal_gate
    _thermal_max = args.thermal_max

    print(f"[bench-android] label={args.label} runs={runs}  "
          f"负载: prompt={len(bench.CANONICAL_PROMPT)} tok, "
          f"decode={bench.DECODE_TOKENS} tok (丢弃预热 {bench.WARMUP})")
    if extra:
        print(f"[bench-android] 额外参数: {' '.join(extra)}（同场 A/B 模式）")
    print(f"[bench-android] 热门禁: {'开' if _thermal_gate_enabled else '关'}"
          f"{'（阈值 ' + str(_thermal_max // 1000) + '°C）' if _thermal_gate_enabled else ''}"
          f"  绑核: {args.pin_cores}")

    print("[bench-android] pushing binary & model...")
    ensure_pushed(binary, args.model)

    # 配置绑核
    _taskset_prefix = build_taskset_prefix(args.pin_cores)

    dev_env = device_env_info()
    print(f"[bench-android] device: {dev_env.get('device_model', '?')} "
          f"({dev_env.get('soc', '?')}) Android {dev_env.get('android_version', '?')}")

    if ab_mode:
        ctrl_med, ctrl_p95, med, p95, last = measure_ab(runs, extra)
        ab_ratio = ctrl_med / med if med > 0 else float("inf")
    else:
        med, p95, last = measure(runs, extra)
        ctrl_med = ctrl_p95 = ab_ratio = None

    host_env = bench.env_info()
    result = {
        "label": args.label,
        "platform": "android",
        "decode_median_ms": round(med, 2),
        "decode_p95_ms": round(p95, 2),
        "decode_samples": last["decode_samples"],
        "runs": runs,
        "top_ops": last["top_ops"],
        "prefill_total_ms": last["prefill_total_ms"],
        "device_env": dev_env,
        "host_env": host_env,
        "pin_cores": args.pin_cores,
        "thermal_gate": _thermal_gate_enabled,
    }
    if extra:
        result["extra_args"] = extra
    if ab_mode:
        result["ab_control_median_ms"] = round(ctrl_med, 2)
        result["ab_ratio"] = round(ab_ratio, 2)

    # 人类可读汇总
    print(f"\n  decode 延迟/token：median={med:.2f}ms  p95={p95:.2f}ms")
    if ab_mode:
        print(f"  同场 A/B：对照 {ctrl_med:.2f}ms → 变体 {med:.2f}ms = {ab_ratio:.2f}×")
    print(f"  样本量：每遍 {last['decode_samples']} 个稳态 decode token")
    print(f"  设备：{dev_env.get('device_model', '?')} / {dev_env.get('soc', '?')}")
    print(f"  commit：{host_env['git_commit']}")
    print("  耗时 top op：")
    for op in last["top_ops"][:5]:
        print(f"    {op['op']:<28} {op['total_ms']:>9.2f} ms")

    # 回归检测
    if not args.no_check_regression:
        regressed = bench.check_regression(
            med, args.baseline, args.regression_threshold, args.fail_on_regression)
        if regressed:
            result["regression_detected"] = True

    if args.json:
        Path(args.json).write_text(json.dumps(result, ensure_ascii=False, indent=2))
        print(f"\n  已写入 {args.json}")


if __name__ == "__main__":
    main()
