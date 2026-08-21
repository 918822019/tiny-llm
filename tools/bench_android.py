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
    # 基本 Android 测速
    python tools/bench_android.py --label neon-android
    # 多遍 + 额外参数
    python tools/bench_android.py --label neon-android --runs 3 --extra-args "--matvec-impl neon"
    # 绑全部核 + 关闭热门禁（快速迭代用）
    python tools/bench_android.py --label neon-android --pin-cores all --no-thermal-gate
    # CI 模式：回归时失败退出
    python tools/bench_android.py --label neon-android --fail-on-regression
    # i4 vs fp32 跨 dtype 同场对比
    python tools/bench_android.py --label i4-android --runs 3 \\
        --model model_i4.tqwen --control-model model.tqwen

输入：
    - Android 设备通过 adb 连接
    - build-android/runtime/tinyqwen 交叉编译产物
    - .tqwen 模型文件

输出：
    - 终端打印性能汇总（TTFT、TOPT、资源采样等）
    - 追加 benchmarks/history_android.jsonl 结构化历史
    - 可选：--json 写入完整结果 JSON
    - 资源时序曲线存 benchmarks/series/ 目录
"""

# 启用延迟注解求值
from __future__ import annotations

# ---- 标准库导入 ----
import argparse       # 命令行参数解析
import datetime       # 时间戳生成（历史记录、series 文件名）
import json           # JSON 序列化/反序列化
import os             # 环境变量、文件路径操作
import re             # 正则表达式（解析设备输出、PMU 计数器）
import shlex          # shell 字符串安全分割
import statistics     # 统计函数（median、mean）
import subprocess     # 调用外部进程（adb、simpleperf）
import sys            # 系统退出、stderr
import tempfile       # 临时文件（profile JSON 中转）
import time           # 计时、sleep（热门禁轮询）
from pathlib import Path  # 路径操作

# 把 tools/ 目录加入搜索路径，以便导入同目录下的 bench 模块
sys.path.insert(0, str(Path(__file__).parent))
import bench  # noqa: E402  # 复用 CANONICAL_PROMPT、summarize、env_info、percentile 等

# ---- 常量定义 ----
# 设备端工作目录：binary、模型、profile 输出都放在这里
DEVICE_DIR = "/data/local/tmp/tinyqwen"
# Android 交叉编译产物的默认目录
BUILD_DIR_DEFAULT = "build-android"

# 结构化历史文件路径：每次测量追加一行 JSON，供 Web 控制台趋势图 / 历史表使用。
# optimization_log.md 仍是权威叙述账本；jsonl 是机器可读的补充。
HISTORY_PATH = "benchmarks/history_android.jsonl"
# 资源时序曲线目录：每次测量存一个 series JSON（内存/温度/频率/利用率 vs 时间），
# history 行里只放文件名引用，不把大数组塞进 jsonl。控制台 /api/series 读它画图。
SERIES_DIR = "benchmarks/series"


def append_history(result: dict, source: str = "bench", path: str = HISTORY_PATH) -> None:
    """把一次测量结果追加进历史 jsonl 文件。

    result 用 bench_android.py main() 产出的 result dict 的字段命名；
    record_android.py 构造同构 dict 后也会调这里（source="record"）。
    扁平化关键字段为一行 JSON，便于后续工具（Web 控制台、趋势图）直接消费。

    Args:
        result: 测量结果字典（包含 label、decode_median_ms、device_env 等字段）
        source: 数据来源标记（"bench" / "record" / "dataset"）
        path: 历史 jsonl 文件路径
    """
    # 构造一行历史记录，只保留关键标量字段
    row = {
        "ts": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),  # ISO 时间戳
        "source": source,                   # 来源标记
        "label": result.get("label"),       # 配置标签
        "model": result.get("model"),       # 模型文件名
        "extra_args": result.get("extra_args", []),  # 额外 CLI 参数
        "median_ms": result.get("decode_median_ms"),  # decode 中位延迟
        "p95_ms": result.get("decode_p95_ms"),        # decode P95 延迟
        "ttft_ms": result.get("ttft_ms"),             # TTFT
        "total_ms": result.get("total_ms"),           # forward 总耗时
        "prompt_tokens": result.get("prefill_tokens"),  # prompt token 数
        "generated_tokens": result.get("generated_tokens"),  # 生成 token 数
        "runs": result.get("runs"),         # 测量遍数
        "device": (result.get("device_env") or {}).get("device_model"),  # 设备型号
        "soc": (result.get("device_env") or {}).get("soc"),              # SoC 型号
        "commit": (result.get("host_env") or {}).get("git_commit"),      # git commit
        "pin_cores": result.get("pin_cores"),      # 绑核模式
        "mt_threads": result.get("mt_threads"),    # 多线程线程数
    }
    # A/B 模式的加速比
    if result.get("ab_ratio") is not None:
        row["ab_ratio"] = result["ab_ratio"]
    # 资源采样：扁平化成标量便于趋势图。未采样时字段整体缺省（不写 null）。
    res = result.get("resources") or {}
    mem, freq = res.get("mem"), res.get("freq")
    if mem:
        row["peak_rss_mb"] = mem.get("peak_rss_mb")     # 峰值 RSS（MB）
        row["avg_rss_mb"] = mem.get("avg_rss_mb")       # 平均 RSS（MB）
        row["kv_cache_mb"] = mem.get("kv_cache_mb")     # KV cache 大小（MB）
    if freq:
        row["freq_min_khz"] = freq.get("min_khz")        # 最低实时频率（kHz）
        row["freq_median_khz"] = freq.get("median_khz")  # 中位实时频率（kHz）
    # A/B 对照组的资源采样
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
        row["pmu_dram_bw_gbps"] = pmu["dram_bw_gbps"]          # DRAM 带宽（GB/s）
        row["pmu_dram_traffic_gb"] = pmu.get("dram_traffic_gb")  # DRAM 流量（GB）
    if pmu.get("ipc") is not None:
        row["pmu_ipc"] = pmu["ipc"]  # IPC（Instructions Per Cycle）
    # 确保目录存在并追加写入
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


# ---- 资源采样常量 ----
# 采样间隔（秒）：100ms 足够捕捉频率变化，又不至于让采样本身成为负担
RESOURCE_SAMPLE_INTERVAL_S = 0.1
# 设备端推理 stdout/stderr 合并日志文件路径
DEVICE_RUN_LOG = f"{DEVICE_DIR}/run_stdout.log"
# 设备端资源采样 trace 文件路径（内存/频率/温度/利用率时序数据）
DEVICE_TRACE = f"{DEVICE_DIR}/mem_freq_trace.txt"

# ---- PMU 内存带宽测量（--pmu）----
# 用 NDK 自带 simpleperf 的 PMU 计数器真测"推理期间搬了多少数据"。核心事件：
#   raw-bus-access-shared —— CPU 发出的总线访问（Normal/Cacheable/Shareable），
#   流式 matvec 下基本 = DRAM 行填充，每次 ≈ 一条 64B cache line。
#   bus_access × 64B ≈ DRAM 读流量（上界，含少量非 DRAM 总线事务）。
# 这台 PLK110 上 l3d-cache 系列事件不受支持（计数恒 0），bus-access 可用。
# simpleperf 在设备上的存放路径
SIMPLEPERF_DEVICE = f"{DEVICE_DIR}/simpleperf"
# PMU 硬件事件列表
PMU_EVENTS = [
    "raw-bus-access-shared:u",   # 总线访问（DRAM 流量代理），:u = 仅用户态
    "raw-l1d-cache-refill-rd:u", # L1D cache refill 读（L1 miss）
    "raw-l2d-cache-refill-rd:u", # L2D cache refill 读（L2 miss）
    "cpu-cycles:u",              # CPU 周期数
    "instructions:u",            # 指令数（与 cycles 一起算 IPC）
]
# 每次总线访问按一条 cache line（64 字节）计算 DRAM 流量
PMU_LINE_BYTES = 64

# ---- 热门禁默认参数 ----
# 温度阈值（millidegree Celsius）：45°C = 45000 millidegree
THERMAL_MAX_DEFAULT = 45000
# 温度轮询间隔（秒）
THERMAL_POLL_INTERVAL = 10
# 等待降温的最大超时（秒）
THERMAL_TIMEOUT = 180


# ---------- adb helpers（封装 adb 命令调用）----------

def adb(*args: str, capture=True) -> str:
    """执行 adb 命令并返回 stdout。

    Args:
        *args: adb 子命令及参数（如 "shell", "ls /data"）
        capture: True 捕获输出返回字符串；False 不捕获（用于 push 等大输出场景）

    Returns:
        str: adb stdout 去掉首尾空白后的内容（capture=True 时）
    """
    cmd = ["adb"] + list(args)  # 拼接完整 adb 命令
    # 同步执行，check=True 在非零退出时报错
    r = subprocess.run(cmd, capture_output=capture, text=True, check=True)
    return r.stdout.strip() if capture else ""


def adb_shell(cmd: str) -> str:
    """执行 adb shell 命令。

    Args:
        cmd: 要在设备 shell 中执行的命令字符串

    Returns:
        str: 命令输出
    """
    return adb("shell", cmd)


def adb_getprop(prop: str) -> str:
    """读取 Android 系统属性。

    Args:
        prop: 属性名（如 ro.product.model）

    Returns:
        str: 属性值
    """
    return adb_shell(f"getprop {prop}")


# ---------- thermal gate（热门禁：防止热降频干扰测速）----------

def read_device_thermal() -> int | None:
    """读取设备 CPU 温度（millidegree Celsius，取 CPU 类 zone 的最大值）。

    只认 type 以 "cpu" 开头的 zone（含 cpullc 簇级 / cpu-x-y 核级），原因：
    - 早期版本只读 thermal_zone0，但 zone0 在很多设备上不是 CPU，热门禁会漏判；
    - 取**所有** zone 的最大也不行：实测 OnePlus PLK110 上 cpu-hw-trip-*
      zone 是**硬件关机阈值**（恒定 95°C）、PMIC/充电 zone 也常年偏高，
      会把热门禁永久卡死。CPU 类 zone 才是"热降频"要防的对象。
    兜底：设备上没有一个 cpu* zone 可读时退回 thermal_zone0。

    Returns:
        int | None: CPU 温度（millidegree Celsius），读取失败返回 None
    """
    try:
        # 遍历所有 thermal zone，读取 type 和 temp
        raw = adb_shell(
            'for z in /sys/class/thermal/thermal_zone*; do '
            'echo "$(cat $z/type 2>/dev/null) $(cat $z/temp 2>/dev/null)"; done')
        temps = []  # 收集所有有效 CPU 温度
        for line in raw.splitlines():
            parts = line.split()
            if len(parts) != 2:
                continue  # 格式不对跳过
            ztype, temp_str = parts
            # 只认 cpu 开头的 zone，排除 hw-trip（硬件关机阈值）
            if not ztype.startswith("cpu") or "trip" in ztype:
                continue
            try:
                v = int(temp_str)  # 解析温度值
            except ValueError:
                continue  # 无法解析跳过
            # 合理范围检查：0~150°C（150000 millidegree）
            if 0 <= v <= 150000:
                temps.append(v)
        if temps:
            return max(temps)  # 取所有 CPU zone 的最高温度
        # 兜底：有些设备 zone 命名不带 cpu 前缀，退回 zone0
        zone0 = adb_shell("cat /sys/class/thermal/thermal_zone0/temp 2>/dev/null || echo -1")
        v = int(zone0.strip())
        return v if v >= 0 else None
    except (ValueError, subprocess.CalledProcessError):
        return None  # 任何异常都返回 None，不阻塞测速


def wait_for_thermal_cool(max_temp_mc: int = THERMAL_MAX_DEFAULT,
                          poll_interval: int = THERMAL_POLL_INTERVAL,
                          timeout: int = THERMAL_TIMEOUT) -> None:
    """等待设备温度降到阈值以下再开始测量。

    如果当前温度已低于阈值则立即返回。超时后打印警告但仍继续测量。

    Args:
        max_temp_mc: 温度阈值（millidegree Celsius）
        poll_interval: 轮询间隔（秒）
        timeout: 最大等待时间（秒）
    """
    temp = read_device_thermal()
    if temp is None:
        # 无法读取温度，跳过热门禁
        print("  [thermal] 无法读取设备温度，跳过热门禁")
        return
    if temp <= max_temp_mc:
        return  # 温度已在阈值以下，无需等待

    # 温度过高，开始轮询等待降温
    print(f"  [thermal] 设备温度 {temp / 1000:.1f}°C > 阈值 {max_temp_mc / 1000:.1f}°C，等待降温...")
    start = time.time()  # 记录等待起始时间
    while temp is not None and temp > max_temp_mc:
        if time.time() - start > timeout:
            # 超时警告，但仍然继续（不让测量被阻塞死）
            print(f"  [thermal] ⚠️ 等待超时 {timeout}s，当前 {temp / 1000:.1f}°C，继续测量（结果可能受热降频影响）")
            return
        time.sleep(poll_interval)  # 等待一个轮询周期
        temp = read_device_thermal()  # 重新读取温度
        if temp is not None:
            elapsed = int(time.time() - start)  # 已等待秒数
            print(f"  [thermal] {elapsed}s 已等待，当前 {temp / 1000:.1f}°C")
    print(f"  [thermal] 温度已降至阈值以下，继续测量")


# ---------- core pinning（绑核：减少大小核迁移抖动）----------

def detect_big_cores() -> list[int]:
    """检测设备大核编号（频率最高的那组 CPU）。

    通过读取每个 CPU 的 cpuinfo_max_freq 找出最高频率的核组。

    Returns:
        list[int]: 大核的 CPU 编号列表（升序），检测失败返回空列表
    """
    try:
        # 遍历所有 CPU，读取其最大频率
        raw = adb_shell(
            "for cpu in /sys/devices/system/cpu/cpu[0-9]*; do "
            "echo $(basename $cpu):$(cat $cpu/cpufreq/cpuinfo_max_freq 2>/dev/null || echo 0); "
            "done")
        freqs = {}  # {cpu_id: max_freq_khz}
        for line in raw.strip().splitlines():
            parts = line.split(":")
            if len(parts) == 2:
                cpu_id = int(parts[0].replace("cpu", ""))  # 从 "cpu7" 提取 7
                freq = int(parts[1])                        # 最大频率（kHz）
                freqs[cpu_id] = freq
        if not freqs:
            return []
        max_freq = max(freqs.values())  # 找到最高频率
        # 返回所有达到最高频率的 CPU 编号
        return sorted(cpu_id for cpu_id, f in freqs.items() if f == max_freq)
    except (subprocess.CalledProcessError, ValueError):
        return []


def build_taskset_prefix(pin_cores: str) -> tuple[str, int | None]:
    """构造 taskset 命令前缀用于绑核。

    返回 (taskset 命令前缀, 绑核数量)。前缀为空字符串表示不绑核；
    绑核数量为 None 表示未绑核（线程数应留给 kernel 默认）。

    返回绑核数是给 TINYQWEN_MT_THREADS 对齐用的：多线程 matvec kernel 默认
    按 hardware_concurrency()（全部核）开线程，绑到少量大核后若不对齐，
    线程会在绑定的核上过度竞争，测速失真。

    同时把绑定的核记进全局 _pinned_cores：资源采样聚合实时频率时优先看这些
    真正扛算的核，小核的低频不会掩盖大核热降频。

    Args:
        pin_cores: 绑核模式："big"（大核）/ "all"（不绑）/ 逗号分隔的 CPU 编号

    Returns:
        tuple[str, int | None]: (taskset 前缀字符串, 绑定核数量)
    """
    global _pinned_cores  # 修改模块级变量，供资源采样使用
    _pinned_cores = []
    if pin_cores == "all":
        return "", None  # 不绑核
    if pin_cores == "big":
        cores = detect_big_cores()  # 自动检测大核
        if not cores:
            print("  [pin] 无法检测大核，跳过绑核")
            return "", None
        mask = 0  # CPU affinity mask（位掩码）
        for c in cores:
            mask |= (1 << c)  # 设置对应核的位
        # 注意：Android toybox taskset 只认裸十六进制 mask（无 0x 前缀、
        # 不支持 CPU 列表），实测 "0xc0" / "6,7" 均报 bad mask。
        print(f"  [pin] 绑大核: cpu{','.join(map(str, cores))} (mask=0x{mask:x})")
        _pinned_cores = cores
        return f"taskset {mask:x} ", len(cores)
    # 手动指定：逗号分隔的 CPU 编号（如 "6,7"）
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


# ---------- device env（设备环境信息采集）----------

def device_env_info() -> dict:
    """采集 Android 设备的硬件/软件环境信息。

    Returns:
        dict: 包含 device_model、soc、cpu_abi、android_version 等字段
    """
    info = {}
    # 设备型号（如 "PLK110"）
    try:
        info["device_model"] = adb_getprop("ro.product.model")
    except Exception:
        info["device_model"] = "unknown"
    # SoC 平台（如 "snapdragon"）
    try:
        info["soc"] = adb_getprop("ro.board.platform")
    except Exception:
        info["soc"] = "unknown"
    # CPU ABI（如 "arm64-v8a"）
    try:
        info["cpu_abi"] = adb_getprop("ro.product.cpu.abi")
    except Exception:
        info["cpu_abi"] = "unknown"
    # Android 版本号
    try:
        info["android_version"] = adb_getprop("ro.build.version.release")
    except Exception:
        info["android_version"] = "unknown"
    # 大核最大频率（kHz）
    try:
        freq = adb_shell("cat /sys/devices/system/cpu/cpu7/cpufreq/cpuinfo_max_freq 2>/dev/null "
                         "|| cat /sys/devices/system/cpu/cpu3/cpufreq/cpuinfo_max_freq 2>/dev/null "
                         "|| echo unknown")
        info["big_core_max_khz"] = freq
    except Exception:
        info["big_core_max_khz"] = "unknown"
    # CPU 调频策略（governor）
    try:
        info["governor"] = adb_shell(
            "cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo unknown")
    except Exception:
        info["governor"] = "unknown"
    # thermal_zone0 温度（原始值，仅供参考）
    try:
        info["thermal_zone0"] = adb_shell(
            "cat /sys/class/thermal/thermal_zone0/temp 2>/dev/null || echo unknown")
    except Exception:
        info["thermal_zone0"] = "unknown"
    return info


# ---------- push / run（推送文件和执行命令）----------

def device_model_path(local_model: str) -> str:
    """计算模型在设备端的存储路径。

    按本地文件名分开存储，让 fp32/fp16/i4 不同量化版本的模型在设备上共存。
    早期版本所有模型都推到 model.tqwen，切一次 dtype 就要重传整个模型
    （fp32 约 2GB）；更糟的是两个模型大小恰好相同时会静默测错模型。

    Args:
        local_model: 本地模型文件路径

    Returns:
        str: 设备端模型文件的完整路径
    """
    return f"{DEVICE_DIR}/models/{os.path.basename(local_model)}"


def ensure_binary_pushed(binary: str) -> None:
    """确保 tinyqwen binary 已推送到设备并赋予执行权限。

    Args:
        binary: 本地 binary 文件路径
    """
    adb_shell(f"mkdir -p {DEVICE_DIR}/models")  # 确保设备端目录存在
    adb("push", binary, f"{DEVICE_DIR}/tinyqwen")  # 推送 binary
    adb_shell(f"chmod +x {DEVICE_DIR}/tinyqwen")   # 赋予执行权限


def ensure_model_pushed(local_model: str) -> str:
    """确保模型文件已在设备上，大小一致则跳过推送。

    Args:
        local_model: 本地模型文件路径

    Returns:
        str: 设备端模型文件路径
    """
    remote = device_model_path(local_model)       # 计算设备端目标路径
    local_size = os.path.getsize(local_model)     # 本地文件大小
    # 查询设备端文件大小（不存在则为 0）
    raw = adb_shell(f"stat -c%s {remote} 2>/dev/null || echo 0")
    try:
        remote_size = int(raw.strip())
    except ValueError:
        remote_size = 0
    name = os.path.basename(local_model)
    if local_size != remote_size:
        # 大小不一致，需要重新推送
        print(f"  pushing {name} ({local_size} bytes)...")
        adb("push", local_model, remote, capture=False)  # capture=False 显示进度
    else:
        # 大小一致，跳过推送
        print(f"  {name} already on device ({remote_size} bytes), skipping push")
    return remote


def find_simpleperf_local() -> str | None:
    """在 NDK 安装目录中查找 arm64 版本的 simpleperf 可执行文件。

    搜索顺序与 build_android.sh 的 NDK 探测同级联：
    1. ANDROID_NDK 环境变量
    2. ~/Library/Android/sdk/ndk/ 下的最新版本
    3. Homebrew 安装路径

    Returns:
        str | None: simpleperf 本地路径，未找到返回 None
    """
    cands: list[str] = []  # 候选 NDK 路径列表
    # 优先检查环境变量
    if os.environ.get("ANDROID_NDK"):
        cands.append(os.environ["ANDROID_NDK"])
    # macOS SDK 管理器安装的 NDK
    home = os.path.expanduser("~")
    sdk_ndk = os.path.join(home, "Library/Android/sdk/ndk")
    if os.path.isdir(sdk_ndk):
        versions = sorted(os.listdir(sdk_ndk))  # 按版本号排序
        cands += [os.path.join(sdk_ndk, v) for v in versions]
    # Homebrew / Linux 常见安装路径
    cands += ["/opt/homebrew/share/android-ndk", "/usr/local/share/android-ndk"]
    # 逐个检查 simpleperf 是否存在
    for ndk in cands:
        sp = os.path.join(ndk, "simpleperf/bin/android/arm64/simpleperf")
        if os.path.isfile(sp):
            return sp
    return None


def ensure_simpleperf_pushed() -> str:
    """确保 simpleperf 已在设备上可执行，大小一致则跳过推送。

    Returns:
        str: 设备端 simpleperf 路径

    Raises:
        RuntimeError: 本地找不到 NDK simpleperf
    """
    local = find_simpleperf_local()
    if not local:
        raise RuntimeError("未找到 NDK simpleperf（装 NDK 或 export ANDROID_NDK）")
    local_size = os.path.getsize(local)
    # 查询设备端已有文件大小
    raw = adb_shell(f"stat -c%s {SIMPLEPERF_DEVICE} 2>/dev/null || echo 0")
    try:
        remote_size = int(raw.strip())
    except ValueError:
        remote_size = 0
    if local_size != remote_size:
        print(f"  pushing simpleperf ({local_size} bytes)...")
        adb("push", local, SIMPLEPERF_DEVICE, capture=False)
    adb_shell(f"chmod +x {SIMPLEPERF_DEVICE}")  # 赋予执行权限
    return SIMPLEPERF_DEVICE


def parse_pmu_output(text: str) -> dict:
    """解析 simpleperf stat 输出的 PMU 计数器表。

    simpleperf 的输出行形如：
        1,033,449,701  raw-bus-access-shared:u   # 540.876 M/sec
    用正则匹配提取事件名和计数值。tinyqwen 自己的输出（gen/generated_ids/[init]）
    不会匹配这个模式，因此可以安全地从混合 stdout 中分离。

    Args:
        text: simpleperf stat 的完整 stdout 文本

    Returns:
        dict: {事件名: 计数值} 字典
    """
    counters: dict[str, int] = {}
    # 匹配 simpleperf 计数器行的正则：数字（带逗号分隔）+ 事件名 + # 注释
    pat = re.compile(r"^\s*([\d,]+)\s+([A-Za-z0-9_.:-]+)\s*#")
    for line in text.splitlines():
        m = pat.match(line)
        if m:
            # group(1) 是计数值（去逗号转 int），group(2) 是事件名
            counters[m.group(2)] = int(m.group(1).replace(",", ""))
    return counters


def summarize_pmu(counters: dict, wall_s: float) -> dict:
    """将 PMU 计数器原始值转换为带宽/IPC 等派生指标。

    bus_access 计的是总线访问次数；每次对应多少字节取决于 SoC 总线宽度
    （64B 行填充 or 32B beat），无法从设备侧确定，所以给上下界：
    lo = ×32B，hi = ×64B。真实 DRAM 流量在两者之间。

    Args:
        counters: parse_pmu_output() 返回的事件计数字典
        wall_s: wall-clock 耗时（秒），由 simpleperf 报告

    Returns:
        dict: 包含 bus_access、dram_traffic_gb、dram_bw_gbps、ipc 等字段
    """
    if not counters:
        return {}

    def get(short: str) -> int | None:
        """按事件名前缀（冒号前部分）查找计数值。"""
        for k, v in counters.items():
            if k.split(":")[0] == short:
                return v
        return None

    bus = get("raw-bus-access-shared")  # 总线访问次数
    cyc = get("cpu-cycles")             # CPU 周期数
    ins = get("instructions")           # 指令数
    out: dict = {"events": counters}    # 保留原始计数器
    if bus is not None:
        # DRAM 流量估算：下界 ×32B，上界 ×64B
        gb_lo = bus * (PMU_LINE_BYTES // 2) / 1e9  # 32B 下界（GB）
        gb_hi = bus * PMU_LINE_BYTES / 1e9          # 64B 上界（GB）
        out["bus_access"] = bus                       # 原始总线访问次数
        out["dram_traffic_gb_lo"] = round(gb_lo, 2)  # DRAM 流量下界
        out["dram_traffic_gb_hi"] = round(gb_hi, 2)  # DRAM 流量上界
        # 兼容字段：取上界（保守报"至少搬了这么多"的反面——至多）
        out["dram_traffic_gb"] = round(gb_hi, 2)
        if wall_s > 0:
            # DRAM 带宽 = 流量 / 时间
            out["dram_bw_gbps_lo"] = round(gb_lo / wall_s, 2)  # 带宽下界（GB/s）
            out["dram_bw_gbps"] = round(gb_hi / wall_s, 2)     # 带宽上界（GB/s）
    if cyc and ins:
        # IPC = 指令数 / 周期数
        out["ipc"] = round(ins / cyc, 2)
    # 提取 L1D/L2D refill 计数
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

    Args:
        model_on_device: 设备端模型文件路径
        extra_args: 额外 CLI 参数列表

    Returns:
        tuple[dict, dict]: (profiler JSON, PMU 汇总字典)
    """
    global _last_temp_mc  # 更新最近温度记录
    # PMU 模式下仍执行热门禁
    if _thermal_gate_enabled:
        wait_for_thermal_cool(_thermal_max)
    _last_temp_mc = read_device_thermal()  # 记录开跑前的温度

    simpleperf = SIMPLEPERF_DEVICE  # 设备端 simpleperf 路径
    # 如果绑定了特定核，simpleperf 也只在这些核上计数
    cpu_arg = f"--cpu {','.join(str(c) for c in _pinned_cores)} " if _pinned_cores else ""
    # 线程数环境变量前缀
    env_prefix = f"TINYQWEN_MT_THREADS={_mt_threads} " if _mt_threads else ""
    tokens_csv = ",".join(map(str, bench.CANONICAL_PROMPT))  # prompt token CSV
    # tinyqwen 运行参数
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
    # 拼接完整设备端命令：cd → env → taskset → simpleperf stat → tinyqwen
    cmd = (f"cd {DEVICE_DIR} && {env_prefix}{_taskset_prefix}"
           f"{simpleperf} stat {cpu_arg}-e {','.join(PMU_EVENTS)} -- "
           f"./tinyqwen " + " ".join(run_args))
    try:
        device_out = adb_shell(cmd)  # 在设备上执行
    except subprocess.CalledProcessError as e:
        # 设备端运行失败，打印详细错误信息后退出
        out = (getattr(e, "stdout", "") or "").strip()
        err = (getattr(e, "stderr", "") or "").strip()
        print(f"  ❌ PMU 运行失败（exit={e.returncode}）", file=sys.stderr)
        if out:
            print(f"  stdout: {out[:800]}", file=sys.stderr)
        if err:
            print(f"  stderr: {err[:800]}", file=sys.stderr)
        sys.exit(1)

    # 从混合输出中解析 PMU 计数器
    counters = parse_pmu_output(device_out)
    # 提取 wall-clock 耗时（simpleperf 在末尾打印）
    wall_s = 0.0
    m = re.search(r"Total test time:\s*([\d.]+)\s*seconds", device_out)
    if m:
        wall_s = float(m.group(1))

    # 拉回 profile JSON 到本机
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
        local_profile = tf.name
    adb("pull", f"{DEVICE_DIR}/profile.json", local_profile)
    with open(local_profile) as f:
        profile = json.load(f)
    Path(local_profile).unlink(missing_ok=True)  # 清理临时文件
    return profile, summarize_pmu(counters, wall_s)


# ---- 模块级状态变量：由 configure_measurement() 设置，run_once_on_device 使用 ----
_taskset_prefix = ""              # taskset 命令前缀（空串 = 不绑核）
_mt_threads: int | None = None    # TINYQWEN_MT_THREADS 值；None = 不设置（kernel 默认）
_thermal_gate_enabled = True      # 是否启用热门禁
_thermal_max = THERMAL_MAX_DEFAULT  # 热门禁温度阈值
_last_temp_mc: int | None = None  # 最近一次开跑前的设备温度（报告用）
_resource_sample_enabled = True   # 是否启用资源采样（进程 RSS / 各核实实时频率）
_pinned_cores: list[int] = []     # 绑定的核列表（资源采样按大核聚合实时频率用）
_pmu_enabled = False              # PMU 内存带宽测量开关（与资源采样互斥）
# 最近一次测量的时序数据（measure/measure_ab 写入，main/record 存盘画图用）。
# 多遍测量只保留最后一遍的 series——资源曲线逐遍基本重合，存全部是冗余。
_last_series: dict = {}
_last_control_series: dict = {}   # A/B 模式下的对照组时序数据


def configure_measurement(pin_cores: str = "big", threads: int | None = None,
                          thermal_gate: bool = True,
                          thermal_max: int = THERMAL_MAX_DEFAULT) -> None:
    """配置测量全局参数：绑核 + 线程数对齐 + 热门禁。

    bench_android.py main() 与 record_android.py 都从这里走同一套配置，
    保证两条路径的测量口径一致（此前 record 直调 measure() 而从不绑核，
    与 bench 的数字没有可比性）。应在 measure()/measure_ab() 之前调用。

    Args:
        pin_cores: 绑核模式（"big" / "all" / 逗号分隔 CPU 编号）
        threads: 显式指定线程数；None = 自动对齐到绑核数
        thermal_gate: 是否启用热门禁
        thermal_max: 热门禁温度阈值（millidegree Celsius）
    """
    global _taskset_prefix, _mt_threads, _thermal_gate_enabled, _thermal_max
    # 构建 taskset 前缀并获取绑核数
    _taskset_prefix, pinned_count = build_taskset_prefix(pin_cores)
    if threads is not None:
        # 显式指定线程数：>0 使用该值，<=0 不设置（kernel 默认）
        _mt_threads = threads if threads > 0 else None
    else:
        # 未显式指定：自动对齐到绑核数（防过度竞争）
        _mt_threads = pinned_count
    if _mt_threads:
        print(f"  [threads] TINYQWEN_MT_THREADS={_mt_threads}"
              f"{'（= 绑核数）' if threads is None else '（显式指定）'}")
    _thermal_gate_enabled = thermal_gate
    _thermal_max = thermal_max


def build_device_script(model_on_device: str, extra_args: list[str] | None,
                        sample_resources: bool,
                        override_args: list[str] | None = None) -> str:
    """构造设备端跑一次负载的完整 shell 脚本。

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

    Args:
        model_on_device: 设备端模型文件路径
        extra_args: 额外 CLI 参数列表
        sample_resources: 是否启用资源采样
        override_args: 覆盖默认 tinyqwen 参数（数据集测试用）

    Returns:
        str: 可在 adb shell 中执行的完整 shell 脚本
    """
    tokens_csv = ",".join(map(str, bench.CANONICAL_PROMPT))  # prompt token CSV
    # 线程数对齐：多线程 matvec kernel（neon_mt*）读 TINYQWEN_MT_THREADS，
    # 默认按 hardware_concurrency()（全部核）开线程。绑核后必须把线程数压到
    # 绑定的核数，否则线程在少量大核上过度竞争，测速失真。
    env_prefix = f"TINYQWEN_MT_THREADS={_mt_threads} " if _mt_threads else ""
    if override_args is not None:
        # 数据集批量测试等场景：整套 tinyqwen 参数由调用方给定
        run_args = list(override_args)
    else:
        # 标准负载参数
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
    # 拼接 tinyqwen 运行命令（含环境变量和 taskset 前缀）
    run_cmd = f"{env_prefix}{_taskset_prefix}./tinyqwen " + " ".join(run_args)

    if not sample_resources:
        # 不采样时直接前台运行，最简单
        return f"cd {DEVICE_DIR} && {run_cmd}"

    # ---- 资源采样模式：后台跑推理 + 前台采样循环 ----
    # stdout/stderr 合并重定向到 run_stdout.log：runtime 的 [init] kv cache
    # 行打在 stderr，必须合并才能解析。wait 收回退出码并 exit，让 adb 把
    # 失败语义原样传回 host 侧。
    #
    # 采样循环刻意**只用 shell 内建**（read / case / 参数展开），不 fork
    # grep/cat/sed/basename：Android 上 fork 很贵，早期版本每轮 fork 几十次，
    # 采样间隔从 100ms 涨到 3s+，曲线分辨率全毁。现在每轮只 fork date/sleep。
    trace = DEVICE_TRACE  # trace 文件路径
    # 返回多行 shell 脚本
    return "\n".join([
        f"cd {DEVICE_DIR}",                                    # 切换到工作目录
        f"rm -f {DEVICE_RUN_LOG} {trace}",                     # 清理旧的日志和 trace
        f"( {run_cmd} ) > {DEVICE_RUN_LOG} 2>&1 &",           # 后台运行推理，合并 stdout/stderr
        "PID=$!",                                               # 记录后台进程 PID
        "while kill -0 $PID 2>/dev/null; do",                  # 循环直到进程退出
        "  {",                                                  # 开始一组命令的输出重定向
        '    echo "S $(date +%s%N 2>/dev/null || echo 0)"',    # S 行：纳秒时间戳
        "    while read line; do",                              # 读 /proc/$PID/status
        '      case "$line" in VmRSS:*|VmHWM:*) echo "$line";; esac',  # 只取 VmRSS/VmHWM 行
        "    done < /proc/$PID/status 2>/dev/null",
        "    for f in /sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq; do",  # 遍历 CPU
        "      d=${f%/cpufreq/scaling_cur_freq}; v=''",         # 提取 CPU 名
        '      read v < "$f" 2>/dev/null',                      # 读当前频率
        '      echo "F ${d##*/} ${v:-0}"',                      # F 行：cpu名 频率
        "    done",
        "    for z in /sys/class/thermal/thermal_zone*; do",    # 遍历 thermal zone
        "      t=''; v=''",
        '      read t < "$z/type" 2>/dev/null',                 # 读 zone 类型
        '      read v < "$z/temp" 2>/dev/null',                 # 读温度
        '      echo "T ${z##*/} $t $v"',                        # T 行：zone名 类型 温度
        "    done",
        "    while read line; do",                              # 读 /proc/stat
        '      case "$line" in cpu[0-9]*) echo "U $line";; esac',  # U 行：CPU 利用率原始计数
        "    done < /proc/stat 2>/dev/null",
        f"  }} >> {trace} 2>/dev/null",                         # 追加到 trace 文件
        f"  sleep {RESOURCE_SAMPLE_INTERVAL_S}",                # 等待下一个采样周期
        "done",
        "wait $PID",                                            # 等待后台进程结束
        "EXIT=$?",                                              # 保存退出码
        f"cat {DEVICE_RUN_LOG}",                                # 输出推理日志
        "exit $EXIT",                                           # 以推理进程的退出码退出
    ])


def pull_device_trace() -> str:
    """从设备拉回资源采样 trace 文件内容。

    如果 trace 文件不存在（跑得太短一次都没采到）返回空串。

    Returns:
        str: trace 文件文本内容
    """
    try:
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tf:
            local = tf.name  # 创建本地临时文件
        adb("pull", DEVICE_TRACE, local)  # 从设备拉回
        raw = Path(local).read_text()      # 读取内容
        Path(local).unlink(missing_ok=True)  # 清理临时文件
        return raw
    except Exception:
        return ""  # 拉取失败返回空串


def parse_trace_samples(raw: str) -> list[dict]:
    """把设备端 trace 文本按 S 行切成样本组，保留原始值。

    每组内的行格式：
        S <ns 时间戳>                     采样时刻（部分设备 date 不支持 %N，值为 0）
        VmRSS:/VmHWM:  <kB> kB            进程常驻 / 峰值内存
        F cpu0 1804800                    各核实时频率（kHz）
        T thermal_zone3 cpu-0-1-0 38000   thermal zone 名 / type / 温度（millidegree）
        U cpu0 12345 67 890 ...           /proc/stat 各核 jiffies 计数

    Args:
        raw: trace 文件的原始文本

    Returns:
        list[dict]: 样本列表，每个样本包含 ts_ns、rss_kb、hwm_kb、freqs、temps、stat
    """
    samples: list[dict] = []  # 所有样本
    cur: dict | None = None   # 当前正在构建的样本
    for line in raw.splitlines():
        parts = line.split()
        if not parts:
            continue  # 空行跳过
        if parts[0] == "S":
            # S 行标志新样本的开始
            ts = 0
            if len(parts) >= 2:
                try:
                    ts = int(parts[1])  # 纳秒时间戳
                except ValueError:
                    ts = 0
            # 初始化新样本字典
            cur = {"ts_ns": ts, "rss_kb": None, "hwm_kb": None,
                   "freqs": {}, "temps": {}, "stat": {}}
            samples.append(cur)
            continue
        if cur is None:
            continue  # S 行之前的杂行忽略
        # /proc/<pid>/status 的行形如 "VmRSS:\t 1900000 kB"（3 个 token）
        if parts[0] in ("VmRSS:", "VmHWM:") and len(parts) >= 2:
            try:
                kb = int(parts[1])  # 内存值（kB）
            except ValueError:
                continue
            if parts[0] == "VmRSS:":
                cur["rss_kb"] = kb   # 当前常驻内存
            else:
                cur["hwm_kb"] = kb   # 峰值内存
        elif parts[0] == "F" and len(parts) == 3:
            # F 行：cpu名 频率
            try:
                cur["freqs"][parts[1]] = int(parts[2])  # {cpu名: 频率kHz}
            except ValueError:
                pass
        elif parts[0] == "T" and len(parts) == 4:
            # T 行：zone名 类型 温度
            try:
                cur["temps"][parts[2]] = int(parts[3])  # {zone类型: 温度millidegree}
            except ValueError:
                pass
        elif parts[0] == "U" and len(parts) >= 6:
            # U 行：/proc/stat 的 CPU 计数
            try:
                cur["stat"][parts[1]] = [int(x) for x in parts[2:]]  # {cpu名: [jiffies...]}
            except ValueError:
                pass
    return samples


def sample_cpu_temp_mc(temps: dict) -> int | None:
    """从单个样本的温度字典中提取 CPU 温度。

    只认 cpu* 且非 trip 的 zone（与 read_device_thermal 同口径），
    hw-trip 是硬件关机阈值、PMIC zone 常年偏高，都不能算。

    Args:
        temps: {zone_type: temperature_millidegree} 字典

    Returns:
        int | None: CPU 温度最大值（millidegree），无有效值返回 None
    """
    vals = [v for t, v in temps.items()
            if t.startswith("cpu") and "trip" not in t and 0 <= v <= 150000]
    return max(vals) if vals else None


def _core_util_pct(prev: list[int], cur: list[int]) -> float | None:
    """根据相邻两个 /proc/stat 样本计算单核 CPU 利用率。

    /proc/stat 列：user nice system idle iowait irq softirq steal ...
    利用率 = 1 - Δ(idle+iowait) / Δ(total)
    计数没涨（采样太快两次读到相同值）返回 None。

    Args:
        prev: 上一个样本的 jiffies 列表
        cur: 当前样本的 jiffies 列表

    Returns:
        float | None: 利用率百分比（0~100），无效返回 None
    """
    if len(prev) < 4 or len(cur) < 4:
        return None  # 数据不完整
    dt = sum(cur) - sum(prev)  # 总 jiffies 增量
    if dt <= 0:
        return None  # 计数没变化
    # idle = index 3, iowait = index 4（如果有）
    pidle = prev[3] + (prev[4] if len(prev) > 4 else 0)
    cidle = cur[3] + (cur[4] if len(cur) > 4 else 0)
    return round((1 - (cidle - pidle) / dt) * 100, 1)


def build_series(samples: list[dict]) -> dict:
    """将样本序列转换为画图用时序数据结构。

    t_rel_s 优先用设备时间戳（date +%s%N）；不可用（全 0 / 非单调）退回
    采样间隔 × 序号（近似，仅影响 x 轴精度）。

    Args:
        samples: parse_trace_samples() 返回的样本列表

    Returns:
        dict: 包含 samples、t_rel_s、rss_mb、temp_c、freq_khz、util_pct 字段；
              无有效样本返回 {}
    """
    # 过滤掉没有任何有效数据的样本
    samples = [s for s in samples if s["rss_kb"] is not None or s["freqs"]]
    if not samples:
        return {}
    # 尝试使用设备时间戳
    ts = [s["ts_ns"] for s in samples]
    if all(ts) and all(b > a for a, b in zip(ts, ts[1:])):
        # 时间戳有效且单调递增：转为相对秒数
        t_rel = [round((t - ts[0]) / 1e9, 3) for t in ts]
    else:
        # 时间戳不可用：用采样间隔 × 序号作为近似时间轴
        t_rel = [round(i * RESOURCE_SAMPLE_INTERVAL_S, 3)
                 for i in range(len(samples))]
    # 收集所有出现过的 CPU 名
    cpus = sorted({c for s in samples for c in s["freqs"]})
    stat_cpus = sorted({c for s in samples for c in s["stat"]})
    # 计算每核利用率的时序（相邻样本做差）
    util: dict = {c: [None] * len(samples) for c in stat_cpus}
    for i in range(1, len(samples)):
        for c in stat_cpus:
            p, q = samples[i - 1]["stat"].get(c), samples[i]["stat"].get(c)
            if p and q:
                util[c][i] = _core_util_pct(p, q)
    return {
        "samples": len(samples),  # 样本总数
        "t_rel_s": t_rel,         # 相对时间轴（秒）
        # RSS 内存时序（kB → MB）
        "rss_mb": [round(s["rss_kb"] / 1024, 1) if s["rss_kb"] is not None else None
                   for s in samples],
        # CPU 温度时序（millidegree → °C），使用海象运算符简化
        "temp_c": [t / 1000 if (t := sample_cpu_temp_mc(s["temps"])) is not None
                   else None for s in samples],
        # 各核频率时序
        "freq_khz": {c: [s["freqs"].get(c) for s in samples] for c in cpus},
        # 各核利用率时序
        "util_pct": util,
    }


def parse_resource_trace(raw: str) -> tuple[dict, dict]:
    """解析设备端采样 trace，返回 (聚合值, 时序数据)。

    聚合值 {"mem","freq","temp","util"} 进结果 JSON 与日志；
    时序数据（build_series）存 series 文件供控制台画曲线。
    无样本时两者均为 {}。

    Args:
        raw: trace 文件原始文本

    Returns:
        tuple[dict, dict]: (聚合统计字典, 时序数据字典)
    """
    samples = parse_trace_samples(raw)  # 解析原始 trace 为样本列表
    series = build_series(samples)       # 构建时序数据结构
    out: dict = {}  # 聚合统计结果

    # ---- 内存聚合 ----
    rss_kb = [s["rss_kb"] for s in samples if s["rss_kb"] is not None]
    hwm_kb = [s["hwm_kb"] for s in samples if s["hwm_kb"] is not None]
    if rss_kb or hwm_kb:
        peak_kb = max(hwm_kb) if hwm_kb else max(rss_kb)  # 峰值取 HWM 优先
        out["mem"] = {
            "peak_rss_mb": round(peak_kb / 1024, 1),        # 峰值 RSS（MB）
            "avg_rss_mb": round(statistics.mean(rss_kb) / 1024, 1) if rss_kb else None,  # 平均 RSS
            "samples": len(rss_kb),                           # 有效采样点数
        }

    # ---- 频率聚合 ----
    freq_map = series.get("freq_khz") or {}
    if freq_map:
        # 优先按绑定的大核聚合（真正扛算的核）；识别不出来才退回全部核
        big_names = sorted({f"cpu{c}" for c in _pinned_cores} & set(freq_map)) \
            if _pinned_cores else []
        pick = big_names or sorted(freq_map)  # 选择要聚合的核集合
        flat = [v for name in pick for v in freq_map[name] if v is not None]  # 展平为列表
        if flat:
            out["freq"] = {
                "scope": "big" if big_names else "all",  # 标记是大核还是全核
                "min_khz": min(flat),                     # 最低频率
                "median_khz": int(statistics.median(flat)),  # 中位频率
                "max_khz": max(flat),                     # 最高频率
            }

    # ---- 温度聚合 ----
    temps_c = [t for t in series.get("temp_c", []) if t is not None]
    if temps_c:
        out["temp"] = {"start_c": temps_c[0], "max_c": max(temps_c),
                       "end_c": temps_c[-1]}

    # ---- 利用率聚合 ----
    util_map = series.get("util_pct") or {}
    if util_map:
        def _avg(vals: list) -> float | None:
            """计算非 None 值的平均值。"""
            v = [x for x in vals if x is not None]
            return round(statistics.mean(v), 1) if v else None

        # 优先按绑定的大核聚合
        big_names_u = sorted({f"cpu{c}" for c in _pinned_cores} & set(util_map)) \
            if _pinned_cores else []
        pick_u = big_names_u or sorted(util_map)
        all_avg = _avg([u for c in sorted(util_map) for u in util_map[c]])  # 全核均值
        big_avg = _avg([u for c in pick_u for u in util_map[c]])            # 大核均值
        if all_avg is not None:
            out["util"] = {"all_avg_pct": all_avg, "big_avg_pct": big_avg}
    return out, series


def merge_resources(rs_list: list[dict]) -> dict:
    """合并多遍测量的资源采样结果。

    峰值/范围取极值（跨遍的最大最小），中位量取中位。

    Args:
        rs_list: 每遍的资源采样字典列表

    Returns:
        dict: 合并后的资源采样字典
    """
    rs = [r for r in rs_list if r]  # 过滤空字典
    if not rs:
        return {}
    out: dict = {}
    # ---- 合并内存 ----
    mems = [r["mem"] for r in rs if r.get("mem")]
    if mems:
        avgs = [m["avg_rss_mb"] for m in mems if m.get("avg_rss_mb") is not None]
        out["mem"] = {
            "peak_rss_mb": round(max(m["peak_rss_mb"] for m in mems), 1),  # 峰值取最大
            "avg_rss_mb": round(statistics.median(avgs), 1) if avgs else None,  # 均值取中位
            "samples": mems[-1]["samples"],  # 采样点数取最后一遍
        }
        # 每遍都一样的结构量（KV cache / GDN state / 权重大小）取最后一个非空值
        for k in ("kv_cache_mb", "gdn_state_mb", "model_mb"):
            v = next((m[k] for m in reversed(mems) if m.get(k) is not None), None)
            if v is not None:
                out["mem"][k] = v
    # ---- 合并频率 ----
    freqs = [r["freq"] for r in rs if r.get("freq")]
    if freqs:
        out["freq"] = {
            "scope": freqs[-1]["scope"],                                        # scope 取最后一遍
            "min_khz": min(f["min_khz"] for f in freqs),                       # 全局最低
            "median_khz": int(statistics.median(f["median_khz"] for f in freqs)),  # 中位的中位
            "max_khz": max(f["max_khz"] for f in freqs),                       # 全局最高
        }
    return out


def format_resources(res: dict) -> list[str]:
    """把资源采样结果格式化成人类可读的文本行。

    Args:
        res: 资源采样聚合字典

    Returns:
        list[str]: 格式化后的文本行列表；未采样返回空列表
    """
    lines = []
    # ---- 内存 ----
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
    # ---- 频率 ----
    freq = res.get("freq")
    if freq:
        scope = "大核" if freq["scope"] == "big" else "全核"
        lines.append(f"推理期间{scope}频率：{freq['min_khz']} ~ {freq['max_khz']} kHz"
                     f"（中位 {freq['median_khz']}）")
    # ---- 温度 ----
    temp = res.get("temp")
    if temp:
        lines.append(f"设备温度：起 {temp['start_c']}°C → 峰 {temp['max_c']}°C"
                     f"（止 {temp['end_c']}°C）")
    # ---- 利用率 ----
    util = res.get("util")
    if util:
        s = f"CPU 利用率：全核均值 {util['all_avg_pct']}%"
        if util.get("big_avg_pct") is not None:
            s += f"，扛算核均值 {util['big_avg_pct']}%"
        lines.append(s)
    return lines


def format_pmu(pmu: dict) -> list[str]:
    """把 PMU 测量结果格式化成人类可读的文本行。

    Args:
        pmu: summarize_pmu() 返回的 PMU 汇总字典

    Returns:
        list[str]: 格式化后的文本行列表
    """
    if not pmu:
        return []
    lines = []
    # DRAM 带宽（有上下界）
    if pmu.get("dram_bw_gbps") is not None:
        lines.append(f"DRAM 带宽：≈{pmu.get('dram_bw_gbps_lo')} ~ {pmu['dram_bw_gbps']} GB/s"
                     f"（流量 {pmu.get('dram_traffic_gb_lo')} ~ {pmu.get('dram_traffic_gb')} GB，"
                     f"bus_access {pmu.get('bus_access'):,} × 32~64B）")
    elif pmu.get("dram_traffic_gb") is not None:
        # 只有流量没有带宽（wall_s = 0 的情况）
        lines.append(f"DRAM 流量：≈{pmu.get('dram_traffic_gb_lo')} ~ {pmu.get('dram_traffic_gb')} GB"
                     f"（bus_access {pmu.get('bus_access'):,} × 32~64B）")
    # IPC 和 cache refill
    if pmu.get("ipc") is not None:
        def _c(v):
            """格式化整数计数值（加千分位逗号），None 显示 ?"""
            return f"{v:,}" if isinstance(v, int) else "?"

        lines.append(f"IPC：{pmu['ipc']}"
                     f"（L1D refill {_c(pmu.get('l1d_refill_rd'))}，"
                     f"L2D refill {_c(pmu.get('l2d_refill_rd'))}）")
    return lines


def save_series(series: dict, label: str) -> str | None:
    """把时序数据存成 benchmarks/series/<时间>_<label>.json，返回相对路径。

    空 series（未采样 / 采样失败）返回 None。文件名带时间戳防覆盖。

    Args:
        series: build_series() 返回的时序数据字典
        label: 配置标签（用于文件名）

    Returns:
        str | None: series 文件的相对路径，空数据返回 None
    """
    if not series or not series.get("samples"):
        return None
    Path(SERIES_DIR).mkdir(parents=True, exist_ok=True)  # 确保目录存在
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")  # 时间戳防覆盖
    safe = re.sub(r"[^\w.-]", "_", label)[:40] or "unlabeled"  # 文件名安全化
    path = f"{SERIES_DIR}/{ts}_{safe}.json"
    Path(path).write_text(json.dumps(series, ensure_ascii=False), encoding="utf-8")
    return path


def run_once_on_device(model_on_device: str,
                       extra_args: list[str] | None = None) -> tuple[dict, dict]:
    """在 Android 设备上跑一次标准负载，拉回 profile JSON 和资源采样。

    Args:
        model_on_device: 设备端模型文件路径
        extra_args: 额外 CLI 参数列表

    Returns:
        tuple[dict, dict, dict]: (profile JSON, 资源采样聚合, 时序数据)
    """
    global _last_temp_mc
    # 热门禁：等设备降温
    if _thermal_gate_enabled:
        wait_for_thermal_cool(_thermal_max)
    _last_temp_mc = read_device_thermal()  # 记录开跑前温度

    # 构造设备端 shell 脚本
    device_cmd = build_device_script(model_on_device, extra_args,
                                     _resource_sample_enabled)
    try:
        device_out = adb_shell(device_cmd)  # 在设备上执行
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

    # 拉回 profile JSON
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
        local_profile = tf.name
    adb("pull", f"{DEVICE_DIR}/profile.json", local_profile)
    with open(local_profile) as f:
        profile = json.load(f)
    Path(local_profile).unlink(missing_ok=True)

    # 资源采样：trace 拉回解析 + [init] 行里的运行期内存分配
    # kv cache / gdn state 的 [init] 行打在 stderr，已合并进 device_out
    resources: dict = {}
    series: dict = {}
    if _resource_sample_enabled:
        # 解析 trace 得到聚合值和时序数据
        resources, series = parse_resource_trace(pull_device_trace())
        # 从 runtime 输出中提取 KV cache 大小
        m = re.search(r"kv cache: ([0-9.]+) MB", device_out)
        if m:
            resources.setdefault("mem", {})["kv_cache_mb"] = float(m.group(1))
        # 从 runtime 输出中提取 GDN state 大小
        m = re.search(r"gdn state: ([0-9.]+) MB", device_out)
        if m:
            resources.setdefault("mem", {})["gdn_state_mb"] = float(m.group(1))
    return profile, resources, series


def measure(model_on_device: str, runs: int,
            extra_args: list[str] | None = None) -> tuple[float, float, dict, dict]:
    """执行多遍标准测速，返回汇总统计。

    Args:
        model_on_device: 设备端模型文件路径
        runs: 测量遍数
        extra_args: 额外 CLI 参数列表

    Returns:
        tuple: (decode 中位数, p95, 最后一遍 summarize 字典, 合并后的资源采样)
    """
    global _last_series
    medians, p95s, last, res_list = [], [], None, []
    for r in range(runs):
        # 在设备上跑一遍
        prof, resources, series = run_once_on_device(model_on_device, extra_args)
        st = bench.summarize(prof)  # 用 bench.py 的统一统计逻辑
        medians.append(st["decode_median_ms"])
        p95s.append(st["decode_p95_ms"])
        last = st
        res_list.append(resources)
        if series:
            _last_series = series  # 保留最后一遍的时序数据
        print(f"  run {r + 1}/{runs}: median={st['decode_median_ms']:.2f}ms "
              f"p95={st['decode_p95_ms']:.2f}ms")
    # 取各遍中位数的中位数和各遍 P95 的中位数
    return (statistics.median(medians), statistics.median(p95s), last,
            merge_resources(res_list))


def measure_ab(runs: int, variant_model: str, variant_args: list[str],
               control_model: str,
               control_args: list[str] | None = None
               ) -> tuple[float, float, float, float, dict, dict, dict]:
    """同场 A/B 测试：每遍先测对照再测变体，交错进行抗热降频漂移。

    对照与变体可以是不同模型（fp32 vs i4 这类换 dtype 的对比不是换 flag
    而是换文件），也可以是同一模型的不同 impl 参数。

    Args:
        runs: 测量遍数
        variant_model: 变体模型设备端路径
        variant_args: 变体的额外 CLI 参数
        control_model: 对照模型设备端路径
        control_args: 对照的额外 CLI 参数

    Returns:
        tuple: (对照中位, 对照p95, 变体中位, 变体p95, 变体最后一遍 summarize,
                对照资源采样, 变体资源采样)
    """
    global _last_series, _last_control_series
    ctrl_meds, ctrl_p95s, var_meds, var_p95s = [], [], [], []
    ctrl_res_list, var_res_list = [], []
    last_var = None
    for r in range(runs):
        # 每遍先测对照再测变体（交错抗漂移）
        cprof, cres, cseries = run_once_on_device(control_model, control_args)
        cs = bench.summarize(cprof)
        vprof, vres, vseries = run_once_on_device(variant_model, variant_args)
        vs = bench.summarize(vprof)
        # 收集统计量
        ctrl_meds.append(cs["decode_median_ms"])
        ctrl_p95s.append(cs["decode_p95_ms"])
        var_meds.append(vs["decode_median_ms"])
        var_p95s.append(vs["decode_p95_ms"])
        ctrl_res_list.append(cres)
        var_res_list.append(vres)
        if cseries:
            _last_control_series = cseries  # 保留对照组时序
        if vseries:
            _last_series = vseries          # 保留变体时序
        last_var = vs
        print(f"  run {r + 1}/{runs}: 对照={cs['decode_median_ms']:.2f}ms  "
              f"变体={vs['decode_median_ms']:.2f}ms")
    return (statistics.median(ctrl_meds), statistics.median(ctrl_p95s),
            statistics.median(var_meds), statistics.median(var_p95s), last_var,
            merge_resources(ctrl_res_list), merge_resources(var_res_list))


def main() -> None:
    """bench_android.py 的主入口函数。

    解析命令行参数 → 推送文件到设备 → 执行测速 → 汇总统计 → 打印结果
    → 回归检测 → 追加历史 → 可选写 JSON。
    """
    global _resource_sample_enabled

    # 创建参数解析器
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    # 交叉编译产物路径
    p.add_argument("--binary", default=None,
                   help="交叉编译产物路径（默认 build-android/runtime/tinyqwen）")
    # 模型文件
    p.add_argument("--model", default="model.tqwen")
    # 标签名
    p.add_argument("--label", default="unlabeled")
    # 测量遍数
    p.add_argument("--runs", type=int, default=1)
    # JSON 输出路径
    p.add_argument("--json", default=None, help="结果写 JSON 文件")
    # 额外 CLI 参数
    p.add_argument("--extra-args", default="",
                   help="原样传给设备端 binary 的额外 CLI 参数")
    # A/B 对照模型
    p.add_argument("--control-model", default=None,
                   help="同场 A/B 的对照模型（默认与 --model 相同）。指定为不同文件即可"
                        "做跨 dtype 对比，例如 --model model_i4.tqwen --control-model model.tqwen")
    # A/B 对照参数
    p.add_argument("--control-args", default="",
                   help="对照组的额外 CLI 参数（默认无，即 ref 配置）")
    # ---- 热门禁参数 ----
    p.add_argument("--thermal-max", type=int, default=THERMAL_MAX_DEFAULT,
                   help=f"热门禁温度阈值（millidegree Celsius，默认 {THERMAL_MAX_DEFAULT}）")
    p.add_argument("--no-thermal-gate", action="store_true",
                   help="跳过热门禁（快速迭代用）")
    # ---- 绑核参数 ----
    p.add_argument("--pin-cores", default="big",
                   help="绑核模式：big（默认，绑大核）/ all（不绑）/ 逗号分隔的 CPU 编号")
    p.add_argument("--threads", type=int, default=None,
                   help="覆盖 TINYQWEN_MT_THREADS（多线程 matvec 的线程数）。"
                        "默认：绑核时 = 绑核数（防过度竞争），不绑核时不设置；"
                        "显式传 0 = 强制不设置（用 kernel 默认）")
    # ---- 资源采样参数 ----
    p.add_argument("--no-resource-sample", action="store_true",
                   help="跳过设备端资源采样（进程 RSS / 各核实实时频率，快速迭代用）")
    p.add_argument("--pmu", action="store_true",
                   help="PMU 内存带宽测量：simpleperf 计数器估 DRAM 流量/带宽 + IPC。"
                        "单遍测量；与资源采样互斥（自动关闭后者）")
    # ---- 回归检测参数 ----
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

    # 确定 binary 路径
    build_dir = os.environ.get("BUILD_DIR", BUILD_DIR_DEFAULT)
    binary = args.binary or f"{build_dir}/runtime/tinyqwen"
    # 检查必要文件是否存在
    if not os.path.isfile(binary):
        sys.exit(f"error: binary not found: {binary}; run scripts/build_android.sh first")
    if not os.path.isfile(args.model):
        sys.exit(f"error: model not found: {args.model}")
    control_model = args.control_model or args.model
    if not os.path.isfile(control_model):
        sys.exit(f"error: control model not found: {control_model}")

    # 解析额外参数
    extra = shlex.split(args.extra_args)
    control_extra = shlex.split(args.control_args)
    # 判断是否为跨模型对比
    cross_model = os.path.abspath(control_model) != os.path.abspath(args.model)
    # 有额外参数或跨模型时自动进入 A/B 模式
    ab_mode = bool(extra) or cross_model
    runs = max(1, args.runs)

    # 配置测量参数（热门禁 + 绑核 + 线程数对齐）
    configure_measurement(args.pin_cores, args.threads,
                          thermal_gate=not args.no_thermal_gate,
                          thermal_max=args.thermal_max)

    # 配置资源采样开关
    _resource_sample_enabled = not args.no_resource_sample

    # PMU 模式：与资源采样互斥（simpleperf 裹住 tinyqwen 后 /proc/<pid> 读到的
    # 是 simpleperf 不是推理进程），单遍测量。
    global _pmu_enabled
    _pmu_enabled = args.pmu
    if _pmu_enabled:
        if cross_model or control_extra:
            sys.exit("error: --pmu 是单配置测量，不支持跨模型 / 对照参数")
        _resource_sample_enabled = False  # PMU 与资源采样互斥
        ensure_simpleperf_pushed()  # 确保 simpleperf 在设备上

    # 打印测速配置摘要
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

    # 推送 binary 和模型到设备
    print("[bench-android] pushing binary & model...")
    ensure_binary_pushed(binary)
    variant_on_device = ensure_model_pushed(args.model)
    control_on_device = (ensure_model_pushed(control_model) if cross_model
                         else variant_on_device)

    # 采集设备环境信息
    dev_env = device_env_info()
    print(f"[bench-android] device: {dev_env.get('device_model', '?')} "
          f"({dev_env.get('soc', '?')}) Android {dev_env.get('android_version', '?')}")

    # 初始化时序数据容器
    global _last_series, _last_control_series
    _last_series = {}
    _last_control_series = {}
    pmu_summary: dict = {}

    # ---- 执行测速 ----
    if _pmu_enabled:
        # PMU 模式：单遍，不走 A/B
        prof, pmu_summary = measure_pmu_on_device(variant_on_device, extra)
        last = bench.summarize(prof)
        med, p95 = last["decode_median_ms"], last["decode_p95_ms"]
        var_res = {}
        ctrl_med = ctrl_p95 = ab_ratio = ctrl_res = None
    elif ab_mode:
        # A/B 模式：交错测对照和变体
        ctrl_med, ctrl_p95, med, p95, last, ctrl_res, var_res = measure_ab(
            runs, variant_on_device, extra, control_on_device, control_extra)
        ab_ratio = ctrl_med / med if med > 0 else float("inf")  # 加速比
    else:
        # 普通模式：只测变体
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

    # 采集 host 环境信息
    host_env = bench.env_info()
    # 组装完整结果字典
    result = {
        "label": args.label,
        "platform": "android",
        "decode_median_ms": round(med, 2),
        "decode_p95_ms": round(p95, 2),
        "decode_samples": last["decode_samples"],
        "runs": runs,
        "top_ops": last["top_ops"],
        # 负载与阶段指标
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

    # ---- 人类可读汇总打印 ----
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
    # 打印资源采样信息
    for line in format_resources(var_res):
        print(f"  {line}")
    if cross_model and ctrl_res:
        for line in format_resources(ctrl_res):
            print(f"  [对照] {line}")
    # 打印 PMU 信息
    if pmu_summary:
        for line in format_pmu(pmu_summary):
            print(f"  {line}")

    # 回归检测
    if not args.no_check_regression:
        regressed = bench.check_regression(
            med, args.baseline, args.regression_threshold, args.fail_on_regression)
        if regressed:
            result["regression_detected"] = True

    # 写入 JSON 文件（如果指定了 --json）
    if args.json:
        Path(args.json).write_text(json.dumps(result, ensure_ascii=False, indent=2))
        print(f"\n  已写入 {args.json}")

    # 追加结构化历史
    if not args.no_history:
        append_history(result, source="bench")


# 脚本直接运行入口
if __name__ == "__main__":
    main()
