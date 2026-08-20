#!/usr/bin/env python3
"""Android 端侧标准基准测试。

在设备上跑标准负载，拉回 profile JSON，在本机算统计。
与 bench.py 共用负载定义和 summarize 逻辑，保证数字口径一致。

特性：
- 热门禁（thermal gate）：测量前等设备降温，消除热降频干扰
- 绑核（core pinning）：taskset 绑大核，减少大小核迁移抖动
- 同场 A/B：--extra-args 或 --control-model 非默认时自动启用，交错测量抗漂移
- 跨模型 A/B：换 dtype 是换文件而非换 flag，用 --control-model 表达
- 模型按文件名在设备上分开缓存，切 dtype 不触发整模型重传
- 资源采样：推理过程中后台采样进程 RSS（常驻/峰值内存）与各核的实时频率，
  连同 runtime 报告的 KV cache 大小一起随结果输出
- 回归检测：自动对比 baseline，超阈值报警

用法：
    python tools/bench_android.py --label neon-android
    python tools/bench_android.py --label neon-android --runs 3 --extra-args "--matvec-impl neon"
    python tools/bench_android.py --label neon-android --pin-cores all --no-thermal-gate
    python tools/bench_android.py --label neon-android --fail-on-regression
    # i4 vs fp32 跨 dtype 同场对比
    python tools/bench_android.py --label i4-android --runs 3 \
        --model model_i4.tqwen --control-model model.tqwen
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import re
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

# 结构化历史：每次测量追加一行 JSON，供 Web 控制台趋势图 / 历史表使用。
# optimization_log.md 仍是权威叙述账本；jsonl 是机器可读的补充。
HISTORY_PATH = "benchmarks/history_android.jsonl"
# 资源时序曲线：每次测量存一个 series JSON（内存/温度/频率/利用率 vs 时间），
# history 行里只放文件名引用，不把大数组塞进 jsonl。控制台 /api/series 读它画图。
SERIES_DIR = "benchmarks/series"


def append_history(result: dict, source: str = "bench", path: str = HISTORY_PATH) -> None:
    """把一次测量结果追加进历史 jsonl。

    result 用 bench_android.py main() 产出的 result dict 的字段命名；
    record_android.py 构造同构 dict 后也会调这里（source="record"）。
    """
    row = {
        "ts": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "source": source,
        "label": result.get("label"),
        "model": result.get("model"),
        "extra_args": result.get("extra_args", []),
        "median_ms": result.get("decode_median_ms"),
        "p95_ms": result.get("decode_p95_ms"),
        "ttft_ms": result.get("ttft_ms"),
        "total_ms": result.get("total_ms"),
        "prompt_tokens": result.get("prefill_tokens"),
        "generated_tokens": result.get("generated_tokens"),
        "runs": result.get("runs"),
        "device": (result.get("device_env") or {}).get("device_model"),
        "soc": (result.get("device_env") or {}).get("soc"),
        "commit": (result.get("host_env") or {}).get("git_commit"),
        "pin_cores": result.get("pin_cores"),
        "mt_threads": result.get("mt_threads"),
    }
    if result.get("ab_ratio") is not None:
        row["ab_ratio"] = result["ab_ratio"]
    # 资源采样：扁平化成标量便于趋势图。未采样时字段整体缺省（不写 null）。
    res = result.get("resources") or {}
    mem, freq = res.get("mem"), res.get("freq")
    if mem:
        row["peak_rss_mb"] = mem.get("peak_rss_mb")
        row["avg_rss_mb"] = mem.get("avg_rss_mb")
        row["kv_cache_mb"] = mem.get("kv_cache_mb")
    if freq:
        row["freq_min_khz"] = freq.get("min_khz")
        row["freq_median_khz"] = freq.get("median_khz")
    cres = result.get("ab_control_resources") or {}
    if cres.get("mem"):
        row["ab_control_peak_rss_mb"] = cres["mem"].get("peak_rss_mb")
    # 资源曲线文件引用（控制台按此画图；不存数据本体）
    if result.get("series_file"):
        row["series_file"] = result["series_file"]
    # 数据集行的 TTFT 散点 artifact（TTFT vs prompt 长度）
    if result.get("ttft_file"):
        row["ttft_file"] = result["ttft_file"]
    # 数据集负载标记（source="dataset" 的行才有；区分于 canonical 负载）
    for k in ("dataset_name", "dataset_n_prompts", "dataset_token_range"):
        if result.get(k) is not None:
            row[k] = result[k]
    # PMU 内存带宽（--pmu 才有）
    pmu = result.get("pmu") or {}
    if pmu.get("dram_bw_gbps") is not None:
        row["pmu_dram_bw_gbps"] = pmu["dram_bw_gbps"]
        row["pmu_dram_traffic_gb"] = pmu.get("dram_traffic_gb")
    if pmu.get("ipc") is not None:
        row["pmu_ipc"] = pmu["ipc"]
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


# 资源采样（内存 / 实时频率）
RESOURCE_SAMPLE_INTERVAL_S = 0.1  # 采样间隔（秒）
DEVICE_RUN_LOG = f"{DEVICE_DIR}/run_stdout.log"  # 设备端推理 stdout/stderr 合并文件
DEVICE_TRACE = f"{DEVICE_DIR}/mem_freq_trace.txt"  # 设备端资源采样 trace

# ---- PMU 内存带宽测量（--pmu）----
# 用 NDK 自带 simpleperf 的 PMU 计数器真测"推理期间搬了多少数据"。核心事件：
#   raw-bus-access-shared —— CPU 发出的总线访问（Normal/Cacheable/Shareable），
#   流式 matvec 下基本 = DRAM 行填充，每次 ≈ 一条 64B cache line。
#   bus_access × 64B ≈ DRAM 读流量（上界，含少量非 DRAM 总线事务）。
# 这台 PLK110 上 l3d-cache 系列事件不受支持（计数恒 0），bus-access 可用。
SIMPLEPERF_DEVICE = f"{DEVICE_DIR}/simpleperf"
PMU_EVENTS = [
    "raw-bus-access-shared:u",  # 总线访问（DRAM 流量代理）
    "raw-l1d-cache-refill-rd:u",  # L1D refill 读
    "raw-l2d-cache-refill-rd:u",  # L2D refill 读
    "cpu-cycles:u",
    "instructions:u",
]
PMU_LINE_BYTES = 64  # 每次总线访问按一条 cache line 计

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
    """读取设备 CPU 温度（millidegree Celsius，取 CPU 类 zone 的最大值）。失败返回 None。

    只认 type 以 "cpu" 开头的 zone（含 cpullc 簇级 / cpu-x-y 核级），原因：
    - 早期版本只读 thermal_zone0，但 zone0 在很多设备上不是 CPU，热门禁会漏判；
    - 取**所有** zone 的最大也不行：实测 OnePlus PLK110 上 cpu-hw-trip-*
      zone 是**硬件关机阈值**（恒定 95°C）、PMIC/充电 zone 也常年偏高，
      会把热门禁永久卡死。CPU 类 zone 才是"热降频"要防的对象。
    兜底：设备上没有一个 cpu* zone 可读时退回 thermal_zone0。
    """
    try:
        raw = adb_shell(
            'for z in /sys/class/thermal/thermal_zone*; do '
            'echo "$(cat $z/type 2>/dev/null) $(cat $z/temp 2>/dev/null)"; done')
        temps = []
        for line in raw.splitlines():
            parts = line.split()
            if len(parts) != 2:
                continue
            ztype, temp_str = parts
            if not ztype.startswith("cpu") or "trip" in ztype:
                continue
            try:
                v = int(temp_str)
            except ValueError:
                continue
            if 0 <= v <= 150000:
                temps.append(v)
        if temps:
            return max(temps)
        # 兜底：有些设备 zone 命名不带 cpu 前缀，退回 zone0。
        zone0 = adb_shell("cat /sys/class/thermal/thermal_zone0/temp 2>/dev/null || echo -1")
        v = int(zone0.strip())
        return v if v >= 0 else None
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


def build_taskset_prefix(pin_cores: str) -> tuple[str, int | None]:
    """返回 (taskset 命令前缀, 绑核数量)。前缀为空字符串表示不绑核；
    绑核数量为 None 表示未绑核（线程数应留给 kernel 默认）。

    返回绑核数是给 TINYQWEN_MT_THREADS 对齐用的：多线程 matvec kernel 默认
    按 hardware_concurrency()（全部核）开线程，绑到少量大核后若不对齐，
    线程会在绑定的核上过度竞争，测速失真（见 run_once_on_device 注释）。

    同时把绑定的核记进 _pinned_cores：资源采样聚合实时频率时优先看这些
    真正扛算的核，小核的低频不会掩盖大核热降频。
    """
    global _pinned_cores
    _pinned_cores = []
    if pin_cores == "all":
        return "", None
    if pin_cores == "big":
        cores = detect_big_cores()
        if not cores:
            print("  [pin] 无法检测大核，跳过绑核")
            return "", None
        mask = 0
        for c in cores:
            mask |= (1 << c)
        # 注意：Android toybox taskset 只认裸十六进制 mask（无 0x 前缀、
        # 不支持 CPU 列表），实测 "0xc0" / "6,7" 均报 bad mask。
        print(f"  [pin] 绑大核: cpu{','.join(map(str, cores))} (mask=0x{mask:x})")
        _pinned_cores = cores
        return f"taskset {mask:x} ", len(cores)
    # 手动指定：逗号分隔的 CPU 编号
    try:
        cores = [int(c.strip()) for c in pin_cores.split(",")]
        mask = 0
        for c in cores:
            mask |= (1 << c)
        print(f"  [pin] 绑核: cpu{','.join(map(str, cores))} (mask=0x{mask:x})")
        _pinned_cores = cores
        return f"taskset {mask:x} ", len(cores)
    except ValueError:
        print(f"  [pin] 无法解析 --pin-cores '{pin_cores}'，跳过绑核")
        return "", None


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

def device_model_path(local_model: str) -> str:
    """设备端模型路径：按本地文件名分开存，让 fp32/fp16/i4 在设备上共存。

    早期版本所有模型都推到 model.tqwen，切一次 dtype 就要重传整个模型
    （fp32 约 2GB）；更糟的是两个模型大小恰好相同时会静默测错模型。
    """
    return f"{DEVICE_DIR}/models/{os.path.basename(local_model)}"


def ensure_binary_pushed(binary: str) -> None:
    adb_shell(f"mkdir -p {DEVICE_DIR}/models")
    adb("push", binary, f"{DEVICE_DIR}/tinyqwen")
    adb_shell(f"chmod +x {DEVICE_DIR}/tinyqwen")


def ensure_model_pushed(local_model: str) -> str:
    """确保模型已在设备上，返回设备端路径。大小一致则跳过 push。"""
    remote = device_model_path(local_model)
    local_size = os.path.getsize(local_model)
    raw = adb_shell(f"stat -c%s {remote} 2>/dev/null || echo 0")
    try:
        remote_size = int(raw.strip())
    except ValueError:
        remote_size = 0
    name = os.path.basename(local_model)
    if local_size != remote_size:
        print(f"  pushing {name} ({local_size} bytes)...")
        adb("push", local_model, remote, capture=False)
    else:
        print(f"  {name} already on device ({remote_size} bytes), skipping push")
    return remote


def find_simpleperf_local() -> str | None:
    """在 NDK 里找 arm64 simpleperf（与 build_android.sh 的 NDK 探测同级联）。"""
    cands: list[str] = []
    if os.environ.get("ANDROID_NDK"):
        cands.append(os.environ["ANDROID_NDK"])
    home = os.path.expanduser("~")
    sdk_ndk = os.path.join(home, "Library/Android/sdk/ndk")
    if os.path.isdir(sdk_ndk):
        versions = sorted(os.listdir(sdk_ndk))
        cands += [os.path.join(sdk_ndk, v) for v in versions]
    cands += ["/opt/homebrew/share/android-ndk", "/usr/local/share/android-ndk"]
    for ndk in cands:
        sp = os.path.join(ndk, "simpleperf/bin/android/arm64/simpleperf")
        if os.path.isfile(sp):
            return sp
    return None


def ensure_simpleperf_pushed() -> str:
    """确保 simpleperf 在设备上可执行，返回设备端路径。大小一致则跳过 push。"""
    local = find_simpleperf_local()
    if not local:
        raise RuntimeError("未找到 NDK simpleperf（装 NDK 或 export ANDROID_NDK）")
    local_size = os.path.getsize(local)
    raw = adb_shell(f"stat -c%s {SIMPLEPERF_DEVICE} 2>/dev/null || echo 0")
    try:
        remote_size = int(raw.strip())
    except ValueError:
        remote_size = 0
    if local_size != remote_size:
        print(f"  pushing simpleperf ({local_size} bytes)...")
        adb("push", local, SIMPLEPERF_DEVICE, capture=False)
    adb_shell(f"chmod +x {SIMPLEPERF_DEVICE}")
    return SIMPLEPERF_DEVICE


def parse_pmu_output(text: str) -> dict:
    """解析 simpleperf stat 的计数器表。

    行形如 `   1,033,449,701  raw-bus-access-shared:u   # 540.876 M/sec`。
    tinyqwen 自己的输出（gen/generated_ids/[init]）不会匹配这个模式。
    """
    counters: dict[str, int] = {}
    pat = re.compile(r"^\s*([\d,]+)\s+([A-Za-z0-9_.:-]+)\s*#")
    for line in text.splitlines():
        m = pat.match(line)
        if m:
            counters[m.group(2)] = int(m.group(1).replace(",", ""))
    return counters


def summarize_pmu(counters: dict, wall_s: float) -> dict:
    """PMU 计数器 → 带宽/IPC 等派生量。

    bus_access 计的是总线访问次数；每次对应多少字节取决于 SoC 总线宽度
    （64B 行填充 or 32B beat），无法从设备侧确定，所以给上下界：
    lo = ×32B，hi = ×64B。真实 DRAM 流量在两者之间。
    """
    if not counters:
        return {}

    def get(short: str) -> int | None:
        for k, v in counters.items():
            if k.split(":")[0] == short:
                return v
        return None

    bus = get("raw-bus-access-shared")
    cyc = get("cpu-cycles")
    ins = get("instructions")
    out: dict = {"events": counters}
    if bus is not None:
        gb_lo = bus * (PMU_LINE_BYTES // 2) / 1e9
        gb_hi = bus * PMU_LINE_BYTES / 1e9
        out["bus_access"] = bus
        out["dram_traffic_gb_lo"] = round(gb_lo, 2)
        out["dram_traffic_gb_hi"] = round(gb_hi, 2)
        # 兼容字段：取上界（保守报"至少搬了这么多"的反面——至多）
        out["dram_traffic_gb"] = round(gb_hi, 2)
        if wall_s > 0:
            out["dram_bw_gbps_lo"] = round(gb_lo / wall_s, 2)
            out["dram_bw_gbps"] = round(gb_hi / wall_s, 2)
    if cyc and ins:
        out["ipc"] = round(ins / cyc, 2)
    for short, key in (("raw-l1d-cache-refill-rd", "l1d_refill_rd"),
                       ("raw-l2d-cache-refill-rd", "l2d_refill_rd")):
        v = get(short)
        if v is not None:
            out[key] = v
    return out


def measure_pmu_on_device(model_on_device: str,
                          extra_args: list[str] | None = None) -> tuple[dict, dict]:
    """PMU 模式跑一次标准负载，返回 (profile, pmu_summary)。

    simpleperf stat 前台裹住 tinyqwen（不后台化，否则拿不到正确 PID/计数器）；
    计数器表与 tinyqwen 输出混在同一 stdout，靠 parse_pmu_output 的严格模式分离。
    与资源采样互斥：后台采样读 /proc/<pid> 会读到 simpleperf 而非 tinyqwen。
    """
    global _last_temp_mc
    if _thermal_gate_enabled:
        wait_for_thermal_cool(_thermal_max)
    _last_temp_mc = read_device_thermal()

    simpleperf = SIMPLEPERF_DEVICE
    cpu_arg = f"--cpu {','.join(str(c) for c in _pinned_cores)} " if _pinned_cores else ""
    env_prefix = f"TINYQWEN_MT_THREADS={_mt_threads} " if _mt_threads else ""
    tokens_csv = ",".join(map(str, bench.CANONICAL_PROMPT))
    run_args = [
        f"--model {model_on_device}",
        f"--tokens {tokens_csv}",
        f"--max-new-tokens {bench.DECODE_TOKENS}",
        f"--max-seq-len {bench.MAX_SEQ_LEN}",
        "--eos -1",
        f"--profile-out {DEVICE_DIR}/profile.json",
    ]
    if extra_args:
        run_args.extend(extra_args)
    cmd = (f"cd {DEVICE_DIR} && {env_prefix}{_taskset_prefix}"
           f"{simpleperf} stat {cpu_arg}-e {','.join(PMU_EVENTS)} -- "
           f"./tinyqwen " + " ".join(run_args))
    try:
        device_out = adb_shell(cmd)
    except subprocess.CalledProcessError as e:
        out = (getattr(e, "stdout", "") or "").strip()
        err = (getattr(e, "stderr", "") or "").strip()
        print(f"  ❌ PMU 运行失败（exit={e.returncode}）", file=sys.stderr)
        if out:
            print(f"  stdout: {out[:800]}", file=sys.stderr)
        if err:
            print(f"  stderr: {err[:800]}", file=sys.stderr)
        sys.exit(1)

    counters = parse_pmu_output(device_out)
    wall_s = 0.0
    m = re.search(r"Total test time:\s*([\d.]+)\s*seconds", device_out)
    if m:
        wall_s = float(m.group(1))

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
        local_profile = tf.name
    adb("pull", f"{DEVICE_DIR}/profile.json", local_profile)
    with open(local_profile) as f:
        profile = json.load(f)
    Path(local_profile).unlink(missing_ok=True)
    return profile, summarize_pmu(counters, wall_s)


# 模块级状态：由 configure_measurement() 设置，run_once_on_device 使用
_taskset_prefix = ""
_mt_threads: int | None = None  # TINYQWEN_MT_THREADS；None = 不设置（kernel 默认）
_thermal_gate_enabled = True
_thermal_max = THERMAL_MAX_DEFAULT
_last_temp_mc: int | None = None  # 最近一次开跑前的设备温度（报告用）
_resource_sample_enabled = True  # 资源采样（进程 RSS / 各核实实时频率）
_pinned_cores: list[int] = []  # 绑定的核（资源采样按大核聚合实时频率用）
_pmu_enabled = False  # PMU 内存带宽测量（与资源采样互斥，见 measure）
# 最近一次测量的时序数据（measure/measure_ab 写入，main/record 存盘画图用）。
# 多遍测量只保留最后一遍的 series——资源曲线逐遍基本重合，存全部是冗余。
_last_series: dict = {}
_last_control_series: dict = {}  # A/B 模式下的对照组


def configure_measurement(pin_cores: str = "big", threads: int | None = None,
                          thermal_gate: bool = True,
                          thermal_max: int = THERMAL_MAX_DEFAULT) -> None:
    """配置测量全局参数：绑核 + 线程数对齐 + 热门禁。

    bench_android.py main() 与 record_android.py 都从这里走同一套配置，
    保证两条路径的测量口径一致（此前 record 直调 measure() 而从不绑核，
    与 bench 的数字没有可比性）。measure()/measure_ab() 之前调用。
    """
    global _taskset_prefix, _mt_threads, _thermal_gate_enabled, _thermal_max
    _taskset_prefix, pinned_count = build_taskset_prefix(pin_cores)
    if threads is not None:
        _mt_threads = threads if threads > 0 else None
    else:
        _mt_threads = pinned_count
    if _mt_threads:
        print(f"  [threads] TINYQWEN_MT_THREADS={_mt_threads}"
              f"{'（= 绑核数）' if threads is None else '（显式指定）'}")
    _thermal_gate_enabled = thermal_gate
    _thermal_max = thermal_max


def build_device_script(model_on_device: str, extra_args: list[str] | None,
                        sample_resources: bool,
                        override_args: list[str] | None = None) -> str:
    """构造设备端跑一次负载的 shell 脚本。

    override_args 非 None 时完全替换默认的 tinyqwen 参数（数据集批量测试用；
    env_prefix / taskset / 资源采样 wrapper 照常生效）；默认 None = canonical
    标准负载，行为不变。

    开启资源采样时把 tinyqwen 放后台，前台循环每 100ms 采样直到进程退出：
    - /proc/<pid>/status 的 VmRSS（当前常驻内存）/ VmHWM（峰值，内核记录，
      比自己取 max 可靠——采样再粗也不会漏掉峰值）；
    - 各核 cpufreq/scaling_cur_freq（运行时实时频率）。静态的
      cpuinfo_max_freq 只说明"最高能跑多快"，这里采的才是"推理时实际跑
      多快"——热降频的直接证据；
    - 全部 thermal zone 的 type + 温度（host 侧按 cpu* 且非 trip 过滤取 max，
      与 read_device_thermal 同口径）；
    - /proc/stat 各核计数（host 侧相邻样本做差算每核利用率）。
    权重是 fread 全量读进进程内存的（model_loader.cpp，非 mmap），所以 RSS
    就是"权重 + KV cache + workspace"的真实占用，不需要另外估算。
    trace 是完整时序，host 侧除了出聚合值还会存成 series JSON 画曲线。
    关闭采样时按原样前台运行，零干扰。
    """
    tokens_csv = ",".join(map(str, bench.CANONICAL_PROMPT))
    # 线程数对齐：多线程 matvec kernel（neon_mt*）读 TINYQWEN_MT_THREADS，
    # 默认按 hardware_concurrency()（全部核）开线程。绑核后必须把线程数压到
    # 绑定的核数，否则线程在少量大核上过度竞争，测速失真。不设绑核时不注入，
    # 让 kernel 走自己的默认。
    env_prefix = f"TINYQWEN_MT_THREADS={_mt_threads} " if _mt_threads else ""
    if override_args is not None:
        # 数据集批量测试等场景：整套 tinyqwen 参数由调用方给定。
        run_args = list(override_args)
    else:
        run_args = [
            f"--model {model_on_device}",
            f"--tokens {tokens_csv}",
            f"--max-new-tokens {bench.DECODE_TOKENS}",
            f"--max-seq-len {bench.MAX_SEQ_LEN}",
            "--eos -1",
            f"--profile-out {DEVICE_DIR}/profile.json",
        ]
        if extra_args:
            run_args.extend(extra_args)
    run_cmd = f"{env_prefix}{_taskset_prefix}./tinyqwen " + " ".join(run_args)

    if not sample_resources:
        return f"cd {DEVICE_DIR} && {run_cmd}"

    # 后台跑推理 + 前台采样循环。stdout/stderr 合并重定向到 run_stdout.log：
    # runtime 的 [init] kv cache 行打在 stderr，必须合并才能解析。wait 收回
    # 退出码并 exit，让 adb 把失败语义原样传回 host 侧。
    #
    # 采样循环刻意**只用 shell 内建**（read / case / 参数展开），不 fork
    # grep/cat/sed/basename：Android 上 fork 很贵，早期版本每轮 fork 几十次，
    # 采样间隔从 100ms 涨到 3s+，曲线分辨率全毁。现在每轮只 fork date/sleep。
    trace = DEVICE_TRACE
    return "\n".join([
        f"cd {DEVICE_DIR}",
        f"rm -f {DEVICE_RUN_LOG} {trace}",
        f"( {run_cmd} ) > {DEVICE_RUN_LOG} 2>&1 &",
        "PID=$!",
        "while kill -0 $PID 2>/dev/null; do",
        "  {",
        '    echo "S $(date +%s%N 2>/dev/null || echo 0)"',
        "    while read line; do",
        '      case "$line" in VmRSS:*|VmHWM:*) echo "$line";; esac',
        "    done < /proc/$PID/status 2>/dev/null",
        "    for f in /sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq; do",
        "      d=${f%/cpufreq/scaling_cur_freq}; v=''",
        '      read v < "$f" 2>/dev/null',
        '      echo "F ${d##*/} ${v:-0}"',
        "    done",
        "    for z in /sys/class/thermal/thermal_zone*; do",
        "      t=''; v=''",
        '      read t < "$z/type" 2>/dev/null',
        '      read v < "$z/temp" 2>/dev/null',
        '      echo "T ${z##*/} $t $v"',
        "    done",
        "    while read line; do",
        '      case "$line" in cpu[0-9]*) echo "U $line";; esac',
        "    done < /proc/stat 2>/dev/null",
        f"  }} >> {trace} 2>/dev/null",
        f"  sleep {RESOURCE_SAMPLE_INTERVAL_S}",
        "done",
        "wait $PID",
        "EXIT=$?",
        f"cat {DEVICE_RUN_LOG}",
        "exit $EXIT",
    ])


def pull_device_trace() -> str:
    """拉回资源采样 trace；不存在（跑得太短一次都没采到）返回空串。"""
    try:
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tf:
            local = tf.name
        adb("pull", DEVICE_TRACE, local)
        raw = Path(local).read_text()
        Path(local).unlink(missing_ok=True)
        return raw
    except Exception:
        return ""


def parse_trace_samples(raw: str) -> list[dict]:
    """把 trace 按 S 行切成样本组，原值保留。一组内各行：

        S <ns 时间戳>                     采样时刻（部分设备 date 不支持 %N，值为 0）
        VmRSS:/VmHWM:  <kB> kB            进程常驻 / 峰值内存
        F cpu0 1804800                    各核实时频率（kHz）
        T thermal_zone3 cpu-0-1-0 38000   thermal zone 名 / type / 温度（millidegree）
        U cpu0 12345 67 890 ...           /proc/stat 各核 jiffies 计数
    """
    samples: list[dict] = []
    cur: dict | None = None
    for line in raw.splitlines():
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "S":
            ts = 0
            if len(parts) >= 2:
                try:
                    ts = int(parts[1])
                except ValueError:
                    ts = 0
            cur = {"ts_ns": ts, "rss_kb": None, "hwm_kb": None,
                   "freqs": {}, "temps": {}, "stat": {}}
            samples.append(cur)
            continue
        if cur is None:
            continue
        # /proc/<pid>/status 的行形如 "VmRSS:\t 1900000 kB"（3 个 token）
        if parts[0] in ("VmRSS:", "VmHWM:") and len(parts) >= 2:
            try:
                kb = int(parts[1])
            except ValueError:
                continue
            if parts[0] == "VmRSS:":
                cur["rss_kb"] = kb
            else:
                cur["hwm_kb"] = kb
        elif parts[0] == "F" and len(parts) == 3:
            try:
                cur["freqs"][parts[1]] = int(parts[2])
            except ValueError:
                pass
        elif parts[0] == "T" and len(parts) == 4:
            try:
                cur["temps"][parts[2]] = int(parts[3])
            except ValueError:
                pass
        elif parts[0] == "U" and len(parts) >= 6:
            try:
                cur["stat"][parts[1]] = [int(x) for x in parts[2:]]
            except ValueError:
                pass
    return samples


def sample_cpu_temp_mc(temps: dict) -> int | None:
    """单个样本的 CPU 温度：cpu* 且非 trip zone 取 max（与 read_device_thermal
    同口径——hw-trip 是硬件关机阈值、PMIC zone 常年偏高，都不能算）。"""
    vals = [v for t, v in temps.items()
            if t.startswith("cpu") and "trip" not in t and 0 <= v <= 150000]
    return max(vals) if vals else None


def _core_util_pct(prev: list[int], cur: list[int]) -> float | None:
    """/proc/stat 列：user nice system idle iowait irq softirq steal ...
    利用率 = 1 - Δ(idle+iowait) / Δ(total)；计数没涨（采样太快）返回 None。"""
    if len(prev) < 4 or len(cur) < 4:
        return None
    dt = sum(cur) - sum(prev)
    if dt <= 0:
        return None
    pidle = prev[3] + (prev[4] if len(prev) > 4 else 0)
    cidle = cur[3] + (cur[4] if len(cur) > 4 else 0)
    return round((1 - (cidle - pidle) / dt) * 100, 1)


def build_series(samples: list[dict]) -> dict:
    """样本序列 → 画图用时序数据；无有效样本返回 {}。

    t_rel_s 优先用设备时间戳（date +%s%N）；不可用（全 0 / 非单调）退回
    采样间隔 × 序号（近似，仅影响 x 轴精度）。
    """
    samples = [s for s in samples if s["rss_kb"] is not None or s["freqs"]]
    if not samples:
        return {}
    ts = [s["ts_ns"] for s in samples]
    if all(ts) and all(b > a for a, b in zip(ts, ts[1:])):
        t_rel = [round((t - ts[0]) / 1e9, 3) for t in ts]
    else:
        t_rel = [round(i * RESOURCE_SAMPLE_INTERVAL_S, 3)
                 for i in range(len(samples))]
    cpus = sorted({c for s in samples for c in s["freqs"]})
    stat_cpus = sorted({c for s in samples for c in s["stat"]})
    util: dict = {c: [None] * len(samples) for c in stat_cpus}
    for i in range(1, len(samples)):
        for c in stat_cpus:
            p, q = samples[i - 1]["stat"].get(c), samples[i]["stat"].get(c)
            if p and q:
                util[c][i] = _core_util_pct(p, q)
    return {
        "samples": len(samples),
        "t_rel_s": t_rel,
        "rss_mb": [round(s["rss_kb"] / 1024, 1) if s["rss_kb"] is not None else None
                   for s in samples],
        "temp_c": [t / 1000 if (t := sample_cpu_temp_mc(s["temps"])) is not None
                   else None for s in samples],
        "freq_khz": {c: [s["freqs"].get(c) for s in samples] for c in cpus},
        "util_pct": util,
    }


def parse_resource_trace(raw: str) -> tuple[dict, dict]:
    """解析设备端采样 trace，返回 (聚合值, 时序数据)。

    聚合值 {"mem","freq","temp","util"} 进结果 JSON 与日志；
    时序数据（build_series）存 series 文件供控制台画曲线。
    无样本时两者均为 {}。
    """
    samples = parse_trace_samples(raw)
    series = build_series(samples)
    out: dict = {}

    rss_kb = [s["rss_kb"] for s in samples if s["rss_kb"] is not None]
    hwm_kb = [s["hwm_kb"] for s in samples if s["hwm_kb"] is not None]
    if rss_kb or hwm_kb:
        peak_kb = max(hwm_kb) if hwm_kb else max(rss_kb)
        out["mem"] = {
            "peak_rss_mb": round(peak_kb / 1024, 1),
            "avg_rss_mb": round(statistics.mean(rss_kb) / 1024, 1) if rss_kb else None,
            "samples": len(rss_kb),
        }

    freq_map = series.get("freq_khz") or {}
    if freq_map:
        # 优先按绑定的大核聚合（真正扛算的核）；识别不出来才退回全部核。
        big_names = sorted({f"cpu{c}" for c in _pinned_cores} & set(freq_map)) \
            if _pinned_cores else []
        pick = big_names or sorted(freq_map)
        flat = [v for name in pick for v in freq_map[name] if v is not None]
        if flat:
            out["freq"] = {
                "scope": "big" if big_names else "all",
                "min_khz": min(flat),
                "median_khz": int(statistics.median(flat)),
                "max_khz": max(flat),
            }

    temps_c = [t for t in series.get("temp_c", []) if t is not None]
    if temps_c:
        out["temp"] = {"start_c": temps_c[0], "max_c": max(temps_c),
                       "end_c": temps_c[-1]}

    util_map = series.get("util_pct") or {}
    if util_map:
        def _avg(vals: list) -> float | None:
            v = [x for x in vals if x is not None]
            return round(statistics.mean(v), 1) if v else None

        big_names_u = sorted({f"cpu{c}" for c in _pinned_cores} & set(util_map)) \
            if _pinned_cores else []
        pick_u = big_names_u or sorted(util_map)
        all_avg = _avg([u for c in sorted(util_map) for u in util_map[c]])
        big_avg = _avg([u for c in pick_u for u in util_map[c]])
        if all_avg is not None:
            out["util"] = {"all_avg_pct": all_avg, "big_avg_pct": big_avg}
    return out, series


def merge_resources(rs_list: list[dict]) -> dict:
    """合并多遍测量的资源采样：峰值/范围取极值，中位量取中位。"""
    rs = [r for r in rs_list if r]
    if not rs:
        return {}
    out: dict = {}
    mems = [r["mem"] for r in rs if r.get("mem")]
    if mems:
        avgs = [m["avg_rss_mb"] for m in mems if m.get("avg_rss_mb") is not None]
        out["mem"] = {
            "peak_rss_mb": round(max(m["peak_rss_mb"] for m in mems), 1),
            "avg_rss_mb": round(statistics.median(avgs), 1) if avgs else None,
            "samples": mems[-1]["samples"],
        }
        # 每遍都一样的结构量（KV cache / GDN state / 权重大小）取最后一个非空值
        for k in ("kv_cache_mb", "gdn_state_mb", "model_mb"):
            v = next((m[k] for m in reversed(mems) if m.get(k) is not None), None)
            if v is not None:
                out["mem"][k] = v
    freqs = [r["freq"] for r in rs if r.get("freq")]
    if freqs:
        out["freq"] = {
            "scope": freqs[-1]["scope"],
            "min_khz": min(f["min_khz"] for f in freqs),
            "median_khz": int(statistics.median(f["median_khz"] for f in freqs)),
            "max_khz": max(f["max_khz"] for f in freqs),
        }
    return out


def format_resources(res: dict) -> list[str]:
    """把资源采样格式化成人类可读的行；未采样返回空列表。"""
    lines = []
    mem = res.get("mem")
    if mem:
        s = f"peak RSS {mem['peak_rss_mb']} MB"
        if mem.get("avg_rss_mb") is not None:
            s += f"（avg {mem['avg_rss_mb']}，{mem['samples']} 个采样点）"
        parts = [s]
        if mem.get("model_mb") is not None:
            parts.append(f"权重 {mem['model_mb']} MB")
        if mem.get("kv_cache_mb") is not None:
            parts.append(f"KV cache {mem['kv_cache_mb']} MB")
        if mem.get("gdn_state_mb") is not None:
            parts.append(f"GDN state {mem['gdn_state_mb']} MB")
        lines.append("端侧内存：" + "，".join(parts))
    freq = res.get("freq")
    if freq:
        scope = "大核" if freq["scope"] == "big" else "全核"
        lines.append(f"推理期间{scope}频率：{freq['min_khz']} ~ {freq['max_khz']} kHz"
                     f"（中位 {freq['median_khz']}）")
    temp = res.get("temp")
    if temp:
        lines.append(f"设备温度：起 {temp['start_c']}°C → 峰 {temp['max_c']}°C"
                     f"（止 {temp['end_c']}°C）")
    util = res.get("util")
    if util:
        s = f"CPU 利用率：全核均值 {util['all_avg_pct']}%"
        if util.get("big_avg_pct") is not None:
            s += f"，扛算核均值 {util['big_avg_pct']}%"
        lines.append(s)
    return lines


def format_pmu(pmu: dict) -> list[str]:
    """把 PMU 结果格式化成人类可读行。"""
    if not pmu:
        return []
    lines = []
    if pmu.get("dram_bw_gbps") is not None:
        lines.append(f"DRAM 带宽：≈{pmu.get('dram_bw_gbps_lo')} ~ {pmu['dram_bw_gbps']} GB/s"
                     f"（流量 {pmu.get('dram_traffic_gb_lo')} ~ {pmu.get('dram_traffic_gb')} GB，"
                     f"bus_access {pmu.get('bus_access'):,} × 32~64B）")
    elif pmu.get("dram_traffic_gb") is not None:
        lines.append(f"DRAM 流量：≈{pmu.get('dram_traffic_gb_lo')} ~ {pmu['dram_traffic_gb']} GB"
                     f"（bus_access {pmu.get('bus_access'):,} × 32~64B）")
    if pmu.get("ipc") is not None:
        def _c(v):
            return f"{v:,}" if isinstance(v, int) else "?"

        lines.append(f"IPC：{pmu['ipc']}"
                     f"（L1D refill {_c(pmu.get('l1d_refill_rd'))}，"
                     f"L2D refill {_c(pmu.get('l2d_refill_rd'))}）")
    return lines


def save_series(series: dict, label: str) -> str | None:
    """把时序数据存成 benchmarks/series/<时间>_<label>.json，返回相对路径。

    空 series（未采样 / 采样失败）返回 None。文件名带时间戳防覆盖。
    """
    if not series or not series.get("samples"):
        return None
    Path(SERIES_DIR).mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    safe = re.sub(r"[^\w.-]", "_", label)[:40] or "unlabeled"
    path = f"{SERIES_DIR}/{ts}_{safe}.json"
    Path(path).write_text(json.dumps(series, ensure_ascii=False), encoding="utf-8")
    return path


def run_once_on_device(model_on_device: str,
                       extra_args: list[str] | None = None) -> tuple[dict, dict]:
    """在设备上跑一次标准负载，拉回 profile JSON 和资源采样。

    返回 (profile, resources)；resources 形如 {"mem": ..., "freq": ...}，
    未采样或采样失败时为 {}。
    """
    global _last_temp_mc
    if _thermal_gate_enabled:
        wait_for_thermal_cool(_thermal_max)
    _last_temp_mc = read_device_thermal()

    device_cmd = build_device_script(model_on_device, extra_args,
                                     _resource_sample_enabled)
    try:
        device_out = adb_shell(device_cmd)
    except subprocess.CalledProcessError as e:
        # adb_shell 带 capture_output，异常里带设备侧输出——打印出来再退出，
        # 而不是让用户面对一个裸 traceback 猜设备端发生了什么。
        out = (getattr(e, "stdout", "") or "").strip()
        err = (getattr(e, "stderr", "") or "").strip()
        print(f"  ❌ 设备端运行失败（exit={e.returncode}）", file=sys.stderr)
        if out:
            print(f"  设备 stdout: {out[:800]}", file=sys.stderr)
        if err:
            print(f"  设备 stderr: {err[:800]}", file=sys.stderr)
        print("  常见原因：OOM 被杀（模型过大/seq 过长）；未知 CLI 参数；"
              "模型文件损坏（删设备端 models/ 重推）；权限问题", file=sys.stderr)
        sys.exit(1)

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
        local_profile = tf.name
    adb("pull", f"{DEVICE_DIR}/profile.json", local_profile)
    with open(local_profile) as f:
        profile = json.load(f)
    Path(local_profile).unlink(missing_ok=True)

    # 资源采样：trace 拉回解析 + [init] 行里的运行期内存分配。
    # kv cache / gdn state 的 [init] 行打在 stderr，已合并进 device_out。
    resources: dict = {}
    series: dict = {}
    if _resource_sample_enabled:
        resources, series = parse_resource_trace(pull_device_trace())
        m = re.search(r"kv cache: ([0-9.]+) MB", device_out)
        if m:
            resources.setdefault("mem", {})["kv_cache_mb"] = float(m.group(1))
        m = re.search(r"gdn state: ([0-9.]+) MB", device_out)
        if m:
            resources.setdefault("mem", {})["gdn_state_mb"] = float(m.group(1))
    return profile, resources, series


def measure(model_on_device: str, runs: int,
            extra_args: list[str] | None = None) -> tuple[float, float, dict, dict]:
    """返回 (decode 中位, p95, 最后一遍的 summarize, 合并后的资源采样)。"""
    global _last_series
    medians, p95s, last, res_list = [], [], None, []
    for r in range(runs):
        prof, resources, series = run_once_on_device(model_on_device, extra_args)
        st = bench.summarize(prof)
        medians.append(st["decode_median_ms"])
        p95s.append(st["decode_p95_ms"])
        last = st
        res_list.append(resources)
        if series:
            _last_series = series
        print(f"  run {r + 1}/{runs}: median={st['decode_median_ms']:.2f}ms "
              f"p95={st['decode_p95_ms']:.2f}ms")
    return (statistics.median(medians), statistics.median(p95s), last,
            merge_resources(res_list))


def measure_ab(runs: int, variant_model: str, variant_args: list[str],
               control_model: str,
               control_args: list[str] | None = None
               ) -> tuple[float, float, float, float, dict, dict, dict]:
    """同场 A/B：每遍先测对照再测变体，交错抗热降频。

    对照与变体可以是不同模型（fp32 vs i4 这类换 dtype 的对比不是换 flag
    而是换文件），也可以是同一模型的不同 impl 参数。

    返回 (对照中位, 对照p95, 变体中位, 变体p95, 变体最后一遍 summarize,
    对照资源采样, 变体资源采样)。两侧都采样：跨 dtype 对比时内存占用差异
    （如 i4 权重 ≈ fp32 的 1/4）和时延同样重要。
    """
    global _last_series, _last_control_series
    ctrl_meds, ctrl_p95s, var_meds, var_p95s = [], [], [], []
    ctrl_res_list, var_res_list = [], []
    last_var = None
    for r in range(runs):
        cprof, cres, cseries = run_once_on_device(control_model, control_args)
        cs = bench.summarize(cprof)
        vprof, vres, vseries = run_once_on_device(variant_model, variant_args)
        vs = bench.summarize(vprof)
        ctrl_meds.append(cs["decode_median_ms"])
        ctrl_p95s.append(cs["decode_p95_ms"])
        var_meds.append(vs["decode_median_ms"])
        var_p95s.append(vs["decode_p95_ms"])
        ctrl_res_list.append(cres)
        var_res_list.append(vres)
        if cseries:
            _last_control_series = cseries
        if vseries:
            _last_series = vseries
        last_var = vs
        print(f"  run {r + 1}/{runs}: 对照={cs['decode_median_ms']:.2f}ms  "
              f"变体={vs['decode_median_ms']:.2f}ms")
    return (statistics.median(ctrl_meds), statistics.median(ctrl_p95s),
            statistics.median(var_meds), statistics.median(var_p95s), last_var,
            merge_resources(ctrl_res_list), merge_resources(var_res_list))


def main() -> None:
    global _resource_sample_enabled

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
    p.add_argument("--control-model", default=None,
                   help="同场 A/B 的对照模型（默认与 --model 相同）。指定为不同文件即可"
                        "做跨 dtype 对比，例如 --model model_i4.tqwen --control-model model.tqwen")
    p.add_argument("--control-args", default="",
                   help="对照组的额外 CLI 参数（默认无，即 ref 配置）")
    # 热门禁
    p.add_argument("--thermal-max", type=int, default=THERMAL_MAX_DEFAULT,
                   help=f"热门禁温度阈值（millidegree Celsius，默认 {THERMAL_MAX_DEFAULT}）")
    p.add_argument("--no-thermal-gate", action="store_true",
                   help="跳过热门禁（快速迭代用）")
    # 绑核
    p.add_argument("--pin-cores", default="big",
                   help="绑核模式：big（默认，绑大核）/ all（不绑）/ 逗号分隔的 CPU 编号")
    p.add_argument("--threads", type=int, default=None,
                   help="覆盖 TINYQWEN_MT_THREADS（多线程 matvec 的线程数）。"
                        "默认：绑核时 = 绑核数（防过度竞争），不绑核时不设置；"
                        "显式传 0 = 强制不设置（用 kernel 默认）")
    # 资源采样
    p.add_argument("--no-resource-sample", action="store_true",
                   help="跳过设备端资源采样（进程 RSS / 各核实实时频率，快速迭代用）")
    p.add_argument("--pmu", action="store_true",
                   help="PMU 内存带宽测量：simpleperf 计数器估 DRAM 流量/带宽 + IPC。"
                        "单遍测量；与资源采样互斥（自动关闭后者）")
    # 回归检测
    p.add_argument("--baseline", default="benchmarks/baseline_android.json",
                   help="回归检测对比的基线文件")
    p.add_argument("--no-check-regression", action="store_true",
                   help="跳过回归检测")
    p.add_argument("--regression-threshold", type=float, default=5.0,
                   help="回归阈值百分比（默认 5）")
    p.add_argument("--fail-on-regression", action="store_true",
                   help="回归时非零退出（CI 用）")
    p.add_argument("--no-history", action="store_true",
                   help=f"不往 {HISTORY_PATH} 追加本次结果（结构化历史）")
    args = p.parse_args()

    build_dir = os.environ.get("BUILD_DIR", BUILD_DIR_DEFAULT)
    binary = args.binary or f"{build_dir}/runtime/tinyqwen"
    if not os.path.isfile(binary):
        sys.exit(f"error: binary not found: {binary}; run scripts/build_android.sh first")
    if not os.path.isfile(args.model):
        sys.exit(f"error: model not found: {args.model}")
    control_model = args.control_model or args.model
    if not os.path.isfile(control_model):
        sys.exit(f"error: control model not found: {control_model}")

    extra = shlex.split(args.extra_args)
    control_extra = shlex.split(args.control_args)
    cross_model = os.path.abspath(control_model) != os.path.abspath(args.model)
    ab_mode = bool(extra) or cross_model
    runs = max(1, args.runs)

    # 配置测量参数（热门禁 + 绑核 + 线程数对齐）
    configure_measurement(args.pin_cores, args.threads,
                          thermal_gate=not args.no_thermal_gate,
                          thermal_max=args.thermal_max)

    # 配置资源采样
    _resource_sample_enabled = not args.no_resource_sample

    # PMU 模式：与资源采样互斥（simpleperf 裹住 tinyqwen 后 /proc/<pid> 读到的
    # 是 simpleperf 不是推理进程），单遍测量。
    global _pmu_enabled
    _pmu_enabled = args.pmu
    if _pmu_enabled:
        if cross_model or control_extra:
            sys.exit("error: --pmu 是单配置测量，不支持跨模型 / 对照参数")
        _resource_sample_enabled = False
        ensure_simpleperf_pushed()

    print(f"[bench-android] label={args.label} runs={runs}  "
          f"负载: prompt={len(bench.CANONICAL_PROMPT)} tok, "
          f"decode={bench.DECODE_TOKENS} tok (丢弃预热 {bench.WARMUP})")
    if extra:
        print(f"[bench-android] 额外参数: {' '.join(extra)}"
              f"{'' if _pmu_enabled else '（同场 A/B 模式）'}")
    if cross_model:
        print(f"[bench-android] 跨模型同场 A/B: 对照 {os.path.basename(control_model)} → "
              f"变体 {os.path.basename(args.model)}")
    print(f"[bench-android] 热门禁: {'开' if _thermal_gate_enabled else '关'}"
          f"{'（阈值 ' + str(_thermal_max // 1000) + '°C）' if _thermal_gate_enabled else ''}"
          f"  绑核: {args.pin_cores}  "
          f"资源采样: {'开' if _resource_sample_enabled else '关'}")

    print("[bench-android] pushing binary & model...")
    ensure_binary_pushed(binary)
    variant_on_device = ensure_model_pushed(args.model)
    control_on_device = (ensure_model_pushed(control_model) if cross_model
                         else variant_on_device)

    dev_env = device_env_info()
    print(f"[bench-android] device: {dev_env.get('device_model', '?')} "
          f"({dev_env.get('soc', '?')}) Android {dev_env.get('android_version', '?')}")

    global _last_series, _last_control_series
    _last_series = {}
    _last_control_series = {}
    pmu_summary: dict = {}
    if _pmu_enabled:
        prof, pmu_summary = measure_pmu_on_device(variant_on_device, extra)
        last = bench.summarize(prof)
        med, p95 = last["decode_median_ms"], last["decode_p95_ms"]
        var_res = {}
        ctrl_med = ctrl_p95 = ab_ratio = ctrl_res = None
    elif ab_mode:
        ctrl_med, ctrl_p95, med, p95, last, ctrl_res, var_res = measure_ab(
            runs, variant_on_device, extra, control_on_device, control_extra)
        ab_ratio = ctrl_med / med if med > 0 else float("inf")
    else:
        med, p95, last, var_res = measure(variant_on_device, runs, extra)
        ctrl_med = ctrl_p95 = ab_ratio = ctrl_res = None
    if _pmu_enabled:
        ab_mode = False  # PMU 单配置：result/print 不走 A/B 分支

    # 权重文件大小注入内存口径，方便对账：RSS ≈ 权重 + KV cache + 其余开销
    if var_res.get("mem") is not None:
        var_res["mem"]["model_mb"] = round(
            os.path.getsize(args.model) / (1024 * 1024), 1)
    if cross_model and ctrl_res and ctrl_res.get("mem") is not None:
        ctrl_res["mem"]["model_mb"] = round(
            os.path.getsize(control_model) / (1024 * 1024), 1)

    host_env = bench.env_info()
    result = {
        "label": args.label,
        "platform": "android",
        "decode_median_ms": round(med, 2),
        "decode_p95_ms": round(p95, 2),
        "decode_samples": last["decode_samples"],
        "runs": runs,
        "top_ops": last["top_ops"],
        # 负载与阶段指标：prefill/decode 长度 + TTFT + forward 总耗时
        "prefill_tokens": last["prefill_tokens"],
        "prefill_total_ms": last["prefill_total_ms"],
        "generated_tokens": last["generated_tokens"],
        "ttft_ms": round(last["ttft_ms"], 2),
        "total_ms": round(last["total_ms"], 2) if last.get("total_ms") is not None else None,
        "device_env": dev_env,
        "host_env": host_env,
        "pin_cores": args.pin_cores,
        "mt_threads": _mt_threads if _mt_threads else "kernel-default",
        "temp_at_start_mc": _last_temp_mc,
        "thermal_gate": _thermal_gate_enabled,
        "resource_sample": _resource_sample_enabled,
        "resources": var_res,
        "pmu": pmu_summary,
        "model": os.path.basename(args.model),
    }
    if extra:
        result["extra_args"] = extra
    if ab_mode:
        result["ab_control_median_ms"] = round(ctrl_med, 2)
        result["ab_ratio"] = round(ab_ratio, 2)
    if cross_model:
        result["ab_control_model"] = os.path.basename(control_model)
        if ctrl_res:
            result["ab_control_resources"] = ctrl_res
    if control_extra:
        result["ab_control_args"] = control_extra

    # 资源时序曲线落盘（控制台画图用），结果里只存文件引用
    series_file = save_series(_last_series, args.label)
    if series_file:
        result["series_file"] = series_file
    if cross_model:
        ctrl_series_file = save_series(_last_control_series, args.label + "-control")
        if ctrl_series_file:
            result["ab_control_series_file"] = ctrl_series_file

    # 人类可读汇总
    print(f"\n  TTFT：{last['ttft_ms']:.2f} ms（prefill {last['prefill_tokens']} tok）")
    print(f"  TOPT：median={med:.2f} ms/tok  p95={p95:.2f}"
          f"（decode 共 {last['generated_tokens']} tok，丢预热 {bench.WARMUP}，"
          f"稳态样本 {last['decode_samples']}）")
    if last.get("total_ms") is not None:
        print(f"  forward 总耗时：{last['total_ms']:.1f} ms")
    if ab_mode:
        print(f"  同场 A/B：对照 {ctrl_med:.2f}ms → 变体 {med:.2f}ms = {ab_ratio:.2f}×")
        if cross_model:
            print(f"    对照模型 {os.path.basename(control_model)} → "
                  f"变体模型 {os.path.basename(args.model)}")
    print(f"  设备：{dev_env.get('device_model', '?')} / {dev_env.get('soc', '?')}")
    print(f"  commit：{host_env['git_commit']}")
    print("  耗时 top op：")
    for op in last["top_ops"][:5]:
        print(f"    {op['op']:<28} {op['total_ms']:>9.2f} ms")
    for line in format_resources(var_res):
        print(f"  {line}")
    if cross_model and ctrl_res:
        for line in format_resources(ctrl_res):
            print(f"  [对照] {line}")
    if pmu_summary:
        for line in format_pmu(pmu_summary):
            print(f"  {line}")

    # 回归检测
    if not args.no_check_regression:
        regressed = bench.check_regression(
            med, args.baseline, args.regression_threshold, args.fail_on_regression)
        if regressed:
            result["regression_detected"] = True

    if args.json:
        Path(args.json).write_text(json.dumps(result, ensure_ascii=False, indent=2))
        print(f"\n  已写入 {args.json}")

    if not args.no_history:
        append_history(result, source="bench")


if __name__ == "__main__":
    main()
