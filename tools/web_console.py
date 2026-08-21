#!/usr/bin/env python3
"""tinyqwen Android 端侧测试 Web 控制面板（零依赖，仅 Python 标准库）。

提供一个浏览器界面来操作 Android 测速 pipeline，与命令行工具一一对应，
只是把键盘换成按钮。所有前端资源内嵌在 tools/webui/ 目录中。

用法：
    python3 tools/web_console.py [--port 8765]
浏览器打开 http://127.0.0.1:<port>。

能力（与命令行 pipeline 一一对应，只是把键盘换成按钮）：
- 设备状态看板：scripts/doctor_android.sh 同款检查项，可视化
- 一键触发：verify / bench / record / 基线重建，实时日志流
- 历史结果与趋势：benchmarks/history_android.jsonl + optimization_log.md
- 基线管理：查看 benchmarks/baseline_android.json，一键重建

安全边界：只监听 127.0.0.1（本地单用户工具）；设备是独占资源，
同一时刻只允许一个设备任务（忙时 409）。

API 路由：
    GET  /                    → 控制台主页 HTML
    GET  /assets/*            → 静态资源（CSS/JS）
    GET  /api/status          → 设备/环境状态 JSON
    GET  /api/history         → 测量历史 JSON
    GET  /api/jobs            → 任务列表 JSON
    GET  /api/jobs/<id>       → 单个任务详情 + 日志
    GET  /api/jobs/<id>/tail  → 日志增量读取（轮询用）
    GET  /api/series?file=X   → 资源时序曲线数据
    GET  /api/ttft?file=X     → TTFT 散点数据
    POST /api/jobs            → 创建新任务
    POST /api/jobs/<id>/kill  → 终止运行中的任务
"""

# 启用延迟注解求值
from __future__ import annotations

# ---- 标准库导入 ----
import argparse       # 命令行参数解析
import json           # JSON 序列化/反序列化
import os             # 环境变量、进程操作
import re             # 正则表达式（路径匹配、版本号解析）
import shlex          # shell 引号处理（日志记录命令）
import subprocess     # 调用外部进程（adb、pipeline 脚本）
import sys            # 系统退出、stderr
import threading      # 线程锁（任务并发控制）
import time           # 时间戳、sleep
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer  # HTTP 服务器
from pathlib import Path  # 路径操作
from urllib.parse import urlparse, parse_qs  # URL 解析

# 项目根目录
REPO_ROOT = Path(__file__).resolve().parent.parent
# 把 tools/ 加入搜索路径
sys.path.insert(0, str(Path(__file__).parent))
import bench_android  # noqa: E402  # 复用设备探测、资源采样等
import visualize  # noqa: E402      # 复用优化日志解析

# ---- 文件路径常量 ----
HISTORY_PATH = REPO_ROOT / "benchmarks/history_android.jsonl"  # 结构化历史
JOBS_DIR = REPO_ROOT / "benchmarks/jobs"                       # 任务日志目录
JOB_INDEX = JOBS_DIR / "index.jsonl"                           # 任务索引
BASELINE_PATH = REPO_ROOT / "benchmarks/baseline_android.json" # Android 基线
OPT_LOG = REPO_ROOT / "docs/optimization_log.md"                # 优化日志
MODEL_GLOB = "*.tqwen"                                         # 模型文件 glob 模式

# 支持的任务类型
JOB_TYPES = ("verify", "bench", "record", "baseline", "dataset", "tokenize")


def now_iso() -> str:
    """返回当前时间的 ISO 格式字符串（带时区）。"""
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


# ---------- NDK 探测（与 scripts/build_android.sh 的 resolve_ndk 同级联）----------

def _version_key(name: str):
    """将版本号字符串拆分为可比较的列表（数字部分按数值排序）。"""
    return [int(x) if x.isdigit() else x for x in re.split(r"[.]", name)]


def resolve_ndk() -> str | None:
    """查找 Android NDK 安装路径。

    搜索顺序：ANDROID_NDK 环境变量 → ~/Library/Android/sdk/ndk/ 最新版
    → Homebrew 安装路径。

    Returns:
        str | None: NDK 路径，未找到返回 None
    """
    cands: list[str] = []  # 候选路径列表
    env = os.environ.get("ANDROID_NDK")
    if env:
        cands.append(env)
    # macOS SDK 管理器安装的 NDK
    sdk = Path.home() / "Library/Android/sdk/ndk"
    if sdk.is_dir():
        vers = sorted((p for p in sdk.iterdir() if p.is_dir()),
                      key=lambda p: _version_key(p.name))
        if vers:
            cands.append(str(vers[-1]))  # 取最新版本
    # Homebrew / Linux 常见路径
    cands += ["/opt/homebrew/share/android-ndk", "/usr/local/share/android-ndk"]
    # 验证 toolchain 文件存在
    for c in cands:
        if (Path(c) / "build/cmake/android.toolchain.cmake").is_file():
            return c
    return None


# ---------- adb（带 timeout 的安全包装：状态接口不能因设备异常挂死）----------

def adb_quiet(args: list[str], timeout: int = 15):
    """安全执行 adb 命令，不抛异常。

    Args:
        args: adb 子命令及参数列表
        timeout: 超时秒数

    Returns:
        tuple: (returncode, stdout, stderr)；adb 缺失返回 127，超时返回 124
    """
    try:
        r = subprocess.run(["adb"] + args, capture_output=True, text=True,
                           timeout=timeout)
        return r.returncode, r.stdout.strip(), r.stderr.strip()
    except FileNotFoundError:
        return 127, "", "adb not found"  # adb 未安装
    except subprocess.TimeoutExpired:
        return 124, "", "adb timeout"  # 超时


def _device_serial_state() -> tuple[str, str]:
    """获取已连接设备的序列号和状态。

    Returns:
        tuple[str, str]: (serial, state)；无设备返回 ("", "none")
    """
    rc, out, _ = adb_quiet(["devices"])
    if rc != 0:
        return "", "adb-error"
    # 跳过 "List of devices attached" 表头行
    for line in out.splitlines()[1:]:
        parts = line.split()
        if len(parts) >= 2:
            return parts[0], parts[1]  # serial, state（device/unauthorized/offline）
    return "", "none"


# ---------- /api/status（设备/环境状态采集）----------

def collect_jsonls() -> list[dict]:
    """收集 benchmarks/ 下的数据集 batch 输入清单。

    供「发起测试」dataset 类型下拉使用。附 tokenize_batch.py 写的 meta
    （条数 / token 长度区间）；history 是测量输出不是输入，排除。

    Returns:
        list[dict]: JSONL 文件信息列表
    """
    out: list[dict] = []
    bdir = REPO_ROOT / "benchmarks"
    if not bdir.is_dir():
        return out
    for p in sorted(bdir.glob("*.jsonl")):
        if p.name == "history_android.jsonl":
            continue  # 排除历史记录文件
        item: dict = {"name": p.name, "size": p.stat().st_size}
        # 尝试读取 meta 文件
        meta_p = p.with_suffix(p.suffix + ".meta.json")
        if meta_p.is_file():
            try:
                m = json.loads(meta_p.read_text(encoding="utf-8"))
                if m.get("num") is not None:
                    item["num"] = m["num"]  # 条数
                if m.get("token_len_min") is not None and m.get("token_len_max") is not None:
                    item["token_range"] = f"{m['token_len_min']}~{m['token_len_max']}"
            except (json.JSONDecodeError, OSError):
                pass
        out.append(item)
    return out


def collect_status() -> dict:
    """采集完整的设备/环境状态信息。

    检查项包括：设备连接、NDK、binary、模型文件、基线。
    每个检查项有 key/name/state(ok/warn/fail)/msg 四个字段。

    Returns:
        dict: 包含 checks、device、host_models、device_models 等字段的状态字典
    """
    checks: list[dict] = []  # 检查项列表

    def add(key: str, name: str, state: str, msg: str):
        """添加一个检查项。"""
        checks.append({"key": key, "name": name, "state": state, "msg": msg})

    binary = REPO_ROOT / "build-android/runtime/tinyqwen"  # Android binary 路径
    ndk = resolve_ndk()  # 探测 NDK

    # ---- adb / 设备连接检查 ----
    serial, dstate = _device_serial_state()
    if serial and dstate == "device":
        add("device", "设备连接", "ok", f"{serial}（已授权）")
    elif dstate == "unauthorized":
        add("device", "设备连接", "fail", "未授权：解锁手机，点「允许 USB 调试」")
    elif dstate == "offline":
        add("device", "设备连接", "fail", "设备离线：重新拔插数据线")
    elif dstate == "adb-error":
        add("device", "设备连接", "fail", "adb 不可用")
    else:
        add("device", "设备连接", "fail",
            "未检测到设备：检查数据线/开发者选项/USB 调试")

    device_info: dict = {}       # 设备详细信息
    device_models: list[dict] = []  # 设备端模型文件列表
    if serial and dstate == "device":
        # 采集设备环境信息
        try:
            device_info = bench_android.device_env_info()
        except Exception as e:  # 设备状态瞬息万变，任何异常都降级
            device_info = {"error": str(e)}
        try:
            temp = bench_android.read_device_thermal()  # CPU 温度
            device_info["temp_mc"] = temp
            device_info["big_cores"] = bench_android.detect_big_cores()  # 大核列表
        except Exception:
            pass
        # 列出设备端已推送的模型文件
        rc, out, _ = adb_quiet(["shell", f"ls -l {bench_android.DEVICE_DIR}/models 2>/dev/null"])
        if rc == 0:
            for line in out.splitlines():
                # 解析 ls -l 输出提取文件大小和名称
                m = re.match(r"-\S+\s+\d+\s+\d+\s+\d+\s+(\d+)\s+[\d-]+\s+[\d:]+\s+(\S+)", line)
                if m:
                    device_models.append({"name": m.group(2), "size": int(m.group(1))})

    # ---- NDK 检查 ----
    if ndk:
        add("ndk", "Android NDK", "ok", ndk)
    else:
        add("ndk", "Android NDK", "fail",
            "未找到：brew install --cask android-ndk 或 export ANDROID_NDK")

    # ---- binary 检查 ----
    if binary.is_file():
        mt = time.strftime("%m-%d %H:%M", time.localtime(binary.stat().st_mtime))
        add("binary", "Android binary", "ok", f"build-android（{mt} 编译）")
    else:
        add("binary", "Android binary", "warn", "未编译：./scripts/build_android.sh")

    # ---- host 模型文件检查 ----
    host_models = sorted(
        [{"name": p.name, "size": p.stat().st_size}
         for p in REPO_ROOT.glob(MODEL_GLOB)],
        key=lambda x: x["name"])
    if host_models:
        add("model", "模型文件", "ok", "、".join(m["name"] for m in host_models))
    else:
        add("model", "模型文件", "warn", "仓库根目录没有 .tqwen 模型")

    # ---- 基线检查 ----
    baseline = None
    if BASELINE_PATH.is_file():
        try:
            baseline = json.loads(BASELINE_PATH.read_text())
            add("baseline", "Android 基线", "ok",
                f"{baseline.get('label')}：{baseline.get('decode_median_ms')} ms/tok")
        except (json.JSONDecodeError, OSError):
            add("baseline", "Android 基线", "warn", "基线文件损坏")
    else:
        add("baseline", "Android 基线", "warn",
            "未建立：跑一次「重建基线」任务（type=baseline）")

    return {
        "ts": now_iso(),              # 采集时间戳
        "checks": checks,             # 检查项列表
        "device": {"serial": serial, "state": dstate, **device_info},  # 设备信息
        "host_models": host_models,   # host 端模型列表
        "device_models": device_models,  # 设备端模型列表
        "ndk": ndk,                   # NDK 路径
        "binary_built": binary.is_file(),  # binary 是否已编译
        "baseline": baseline,         # 基线数据
        "jsonls": collect_jsonls(),   # 数据集 JSONL 清单
    }


# ---------- /api/history（测量历史加载）----------

def load_history() -> dict:
    """加载测量历史：jsonl 结构化历史 + optimization_log.md 权威账本。

    两者按 (label, median) 去重，避免 record 双写重复。

    Returns:
        dict: {"history": [...], "log": [...]}
    """
    rows: list[dict] = []
    # 读取 jsonl 历史
    if HISTORY_PATH.is_file():
        for line in HISTORY_PATH.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    rows.append(obj)
            except json.JSONDecodeError:
                continue
    # 读取 optimization_log.md 的权威账本行（含 macOS/Android 全部历史）
    log_rows: list[dict] = []
    try:
        for r in visualize.parse_opt_log(str(OPT_LOG)):
            log_rows.append({
                "ts": "", "source": "log", "label": r["label"],
                "median_ms": r["median_ms"], "p95_ms": r["p95_ms"],
                "commit": r["commit"], "note": r["note"],
                "rolled_back": r["rolled_back"],
            })
    except Exception:
        pass

    # record 写日志时 label 会带「（Android）」后缀，去重键归一化掉
    def norm(s):
        """去掉 label 尾部的「（Android）」后缀用于去重比较。"""
        return re.sub(r"（Android）$", "", s or "").strip()

    # 构建已见集合
    seen = {(norm(r.get("label")), round(r.get("median_ms") or 0, 2)) for r in rows}
    # 过滤掉 jsonl 中已有的记录
    log_rows = [r for r in log_rows
                if (norm(r["label"]), round(r["median_ms"], 2)) not in seen]
    return {"history": rows, "log": log_rows}


# ---------- 任务执行器 ----------

JOBS: dict[str, dict] = {}      # 内存中的任务字典 {job_id: job_dict}
JOBS_LOCK = threading.Lock()    # 保护 JOBS 的线程锁


def _job_log_path(job_id: str) -> Path:
    """返回任务的日志文件路径。"""
    return JOBS_DIR / f"{job_id}.log"


def load_job_index() -> None:
    """启动时把已完成任务从 index.jsonl 读回内存（重启不丢历史）。"""
    if not JOB_INDEX.is_file():
        return
    for line in JOB_INDEX.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            j = json.loads(line)
            JOBS[j["id"]] = j  # 恢复到内存
        except (json.JSONDecodeError, KeyError):
            continue


def _save_job(j: dict) -> None:
    """将任务摘要追加到 index.jsonl（持久化）。"""
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    with JOB_INDEX.open("a", encoding="utf-8") as f:
        f.write(json.dumps(j, ensure_ascii=False) + "\n")


def build_command(spec: dict, job_id: str) -> tuple[list[str], dict]:
    """按任务类型构造命令行。与 CLI pipeline 逐一对应。

    Args:
        spec: 任务规格字典（包含 type、model、label 等字段）
        job_id: 任务 ID（用于结果文件命名）

    Returns:
        tuple[list[str], dict]: (命令参数列表, 环境变量字典)
    """
    t = spec["type"]  # 任务类型
    model = spec.get("model") or "model.tqwen"  # 模型文件
    env = dict(os.environ)  # 复制当前环境变量
    env["MODEL"] = model  # 注入模型名
    env["PYTHONUNBUFFERED"] = "1"  # Python 无缓冲输出（实时日志流）

    if t == "verify":
        # 验证任务：运行 verify_android.sh
        return ["./scripts/verify_android.sh"], env

    label = spec.get("label") or ""
    if t == "bench":
        # 测速任务：调用 bench_android.py
        argv = ["python3", "-u", "tools/bench_android.py",
                "--label", label, "--model", model,
                "--runs", str(spec.get("runs", 1)),
                "--json", f"benchmarks/jobs/{job_id}_result.json"]
        # 按需追加可选参数
        if spec.get("extra_args"):
            argv += ["--extra-args", str(spec["extra_args"])]
        if spec.get("control_model"):
            argv += ["--control-model", str(spec["control_model"])]
        if spec.get("control_args"):
            argv += ["--control-args", str(spec["control_args"])]
        if spec.get("pin_cores"):
            argv += ["--pin-cores", str(spec["pin_cores"])]
        if spec.get("threads"):
            argv += ["--threads", str(spec["threads"])]
        if spec.get("no_thermal_gate"):
            argv += ["--no-thermal-gate"]
        return argv, env

    if t == "record":
        # 记录任务：调用 record_android.sh
        argv = ["./scripts/record_android.sh", label]
        if spec.get("skip_verify"):
            argv.append("--skip-verify")
        if spec.get("runs"):
            argv += ["--runs", str(spec["runs"])]
        if spec.get("extra_args"):
            argv += ["--extra-args", str(spec["extra_args"])]
        if spec.get("control_model"):
            argv += ["--control-model", str(spec["control_model"])]
        if spec.get("control_args"):
            argv += ["--control-args", str(spec["control_args"])]
        if spec.get("pin_cores"):
            argv += ["--pin-cores", str(spec["pin_cores"])]
        return argv, env

    if t == "baseline":
        # 基线重建任务
        argv = ["./scripts/set_baseline_android.sh", label]
        if spec.get("extra_args"):
            argv += ["--extra-args", str(spec["extra_args"])]
        return argv, env

    if t == "tokenize":
        # 生成数据集 batch 输入（tokenize_batch.py）：dataset 测试的前置步骤，
        # 跑完 /api/status 的 jsonls 清单自动更新。
        name = str(spec.get("out_name") or "").strip()
        argv = ["python3", "-u", "tools/tokenize_batch.py",
                "--num", str(spec.get("num", 16)),
                "--min-tokens", str(spec.get("min_tokens", 8)),
                "--max-tokens", str(spec.get("max_tokens", 96)),
                "--out", f"benchmarks/{name}.jsonl"]
        return argv, env

    if t == "dataset":
        # 数据集负载：真实 prompt 批量测 TTFT/TOPT（tools/bench_dataset.py）
        argv = ["python3", "-u", "tools/bench_dataset.py",
                "--label", label, "--model", model,
                "--jsonl", f"benchmarks/{spec.get('jsonl')}",
                "--json", f"benchmarks/jobs/{job_id}_result.json"]
        if spec.get("decode_tokens"):
            argv += ["--decode-tokens", str(spec["decode_tokens"])]
        if spec.get("seq_len"):
            argv += ["--max-seq-len", str(spec["seq_len"])]
        if spec.get("extra_args"):
            argv += ["--extra-args", str(spec["extra_args"])]
        if spec.get("pin_cores"):
            argv += ["--pin-cores", str(spec["pin_cores"])]
        if spec.get("no_thermal_gate"):
            argv += ["--no-thermal-gate"]
        return argv, env

    raise ValueError(f"unknown job type: {t}")


def validate_spec(spec: dict) -> str | None:
    """验证任务规格的合法性。

    Args:
        spec: 任务规格字典

    Returns:
        str | None: 错误消息；None = 合法
    """
    # 检查任务类型
    if spec.get("type") not in JOB_TYPES:
        return f"type 必须是 {JOB_TYPES} 之一"
    # 检查模型文件存在
    model = spec.get("model") or "model.tqwen"
    if not (REPO_ROOT / model).is_file():
        return f"模型不存在：{model}"
    # bench/record/baseline/dataset 需要非空 label
    if spec["type"] not in ("verify", "tokenize") and not (spec.get("label") or "").strip():
        return "bench/record/baseline/dataset 需要非空 label"
    # runs 范围检查
    runs = spec.get("runs", 1)
    if not isinstance(runs, int) or not (1 <= runs <= 10):
        return "runs 必须是 1..10 的整数"
    # dataset 类型的 jsonl 文件名安全检查
    if spec.get("type") == "dataset":
        j = spec.get("jsonl") or ""
        if not j or "/" in j or "\\" in j or ".." in j:
            return "dataset 需要合法的 jsonl 文件名（benchmarks/ 下）"
        if not (REPO_ROOT / "benchmarks" / j).is_file():
            return f"jsonl 不存在：benchmarks/{j}（先跑 tools/tokenize_batch.py 生成）"
    # tokenize 类型参数校验
    if spec.get("type") == "tokenize":
        num = spec.get("num", 16)
        if not isinstance(num, int) or not (1 <= num <= 500):
            return "tokenize 条数必须是 1..500 的整数"
        lo, hi = spec.get("min_tokens", 8), spec.get("max_tokens", 96)
        if not isinstance(lo, int) or not isinstance(hi, int) or lo > hi or lo < 1:
            return "token 长度区间非法（需 1 <= min <= max 整数）"
        name = str(spec.get("out_name") or "").strip()
        if not name or not re.fullmatch(r"[\w.-]+", name):
            return "输出名只能含字母/数字/_/-/.（不含路径）"
    # 参数注入面：extra_args 会被 shlex 分割后进入 argv，禁止 shell 元字符
    for k in ("extra_args", "control_args"):
        v = spec.get(k) or ""
        if any(ch in v for ch in ";|&$`><\n"):
            return f"{k} 含非法字符"
    return None


def start_job(spec: dict) -> dict:
    """创建并启动一个新任务。

    Args:
        spec: 任务规格字典

    Returns:
        dict: {"id": job_id}

    Raises:
        RuntimeError: 设备忙（已有任务在运行）
        ValueError: 规格校验失败
    """
    with JOBS_LOCK:
        # 检查是否有正在运行的任务（设备独占）
        busy = [j for j in JOBS.values() if j.get("status") == "running"]
        if busy:
            raise RuntimeError(f"设备忙：任务 {busy[0]['id']} 正在运行")
        # 校验规格
        err = validate_spec(spec)
        if err:
            raise ValueError(err)
        # 生成任务 ID（时间戳格式）
        job_id = time.strftime("%Y%m%d-%H%M%S")
        # 同一秒多次提交时加后缀防撞
        suffix = 0
        while job_id in JOBS:
            suffix += 1
            job_id = time.strftime("%Y%m%d-%H%M%S") + f"-{suffix}"
        # 构造命令
        argv, env = build_command(spec, job_id)
        # 初始化任务字典
        job = {
            "id": job_id, "type": spec["type"],
            "label": spec.get("label", ""), "model": spec.get("model") or "model.tqwen",
            "argv": argv, "status": "starting", "exit": None,
            "started": now_iso(), "ended": None,
        }
        JOBS[job_id] = job
    # 确保目录存在
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    # 在后台线程中执行任务
    threading.Thread(target=_run_job, args=(job_id, argv, env), daemon=True).start()
    return {"id": job_id}


def _run_job(job_id: str, argv: list[str], env: dict) -> None:
    """在后台线程中执行任务（内部函数，由 start_job 启动）。

    Args:
        job_id: 任务 ID
        argv: 命令参数列表
        env: 环境变量字典
    """
    job = JOBS[job_id]
    job["status"] = "running"
    log_path = _job_log_path(job_id)
    try:
        # start_new_session：子进程自成进程组，kill 时 killpg 不会波及本服务
        proc = subprocess.Popen(
            argv, cwd=REPO_ROOT, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,  # 合并 stderr 到 stdout
            text=True, encoding="utf-8", errors="replace",    # 文本模式，容错编码
            start_new_session=True)                            # 独立进程组
        job["pid"] = proc.pid  # 记录 PID（用于后续 kill）
        # 实时写入日志文件
        with log_path.open("w", encoding="utf-8") as lf:
            lf.write(f"$ {' '.join(shlex.quote(a) for a in argv)}\n")  # 记录完整命令
            lf.flush()
            assert proc.stdout is not None
            for line in proc.stdout:  # 逐行读取并写入
                lf.write(line)
                lf.flush()
        rc = proc.wait()  # 等待进程结束
    except Exception as e:
        # 任务启动失败
        with log_path.open("a", encoding="utf-8") as lf:
            lf.write(f"\n[web_console] 任务启动失败：{e}\n")
        rc = -1
    # 更新任务状态
    job["exit"] = rc
    job["status"] = "done" if rc == 0 else "failed"
    job["ended"] = now_iso()
    # 持久化到 index.jsonl
    _save_job({k: job.get(k) for k in
               ("id", "type", "label", "model", "status", "exit", "started", "ended")})


def kill_job(job_id: str) -> None:
    """终止运行中的任务。

    先杀 host 端子进程组，再杀设备端 tinyqwen 孤儿进程。

    Args:
        job_id: 任务 ID

    Raises:
        KeyError: 任务不存在或已结束
    """
    job = JOBS.get(job_id)
    if not job or job.get("status") != "running":
        raise KeyError("任务不存在或已结束")
    pid = job.get("pid")
    if pid:
        try:
            # 杀整个进程组（start_new_session 创建的）
            os.killpg(os.getpgid(pid), 15)  # SIGTERM
        except (ProcessLookupError, PermissionError):
            try:
                os.kill(pid, 15)  # 退而求其次杀单个进程
            except ProcessLookupError:
                pass
    # 设备侧善后：host 端 adb 客户端被杀后，设备端进程会成孤儿继续跑
    # （占着设备、把机器烧热，污染下一次测量）。设备同一时刻只有一个
    # tinyqwen 任务（独占），按进程名精确杀即可。
    try:
        subprocess.run(["adb", "shell",
                        "pkill -x tinyqwen 2>/dev/null || pkill -f 'tinyqwen --model'"],
                       capture_output=True, timeout=10)
    except Exception:
        pass


def jobs_snapshot() -> list[dict]:
    """获取所有任务的快照（按 ID 倒序排列，最新的在前）。

    Returns:
        list[dict]: 任务摘要列表
    """
    with JOBS_LOCK:
        out = []
        for j in JOBS.values():
            # 只暴露安全字段
            out.append({k: j.get(k) for k in
                        ("id", "type", "label", "model", "status", "exit",
                         "started", "ended")})
    return sorted(out, key=lambda x: x["id"], reverse=True)


# ---------- HTTP 服务器 ----------

WEBUI_DIR = Path(__file__).resolve().parent / "webui"  # 前端资源目录
# MIME 类型映射
_MIME = {".css": "text/css; charset=utf-8", ".js": "application/javascript; charset=utf-8",
         ".html": "text/html; charset=utf-8", ".svg": "image/svg+xml", ".json": "application/json"}
_asset_cache: dict[str, bytes] = {}  # 静态资源缓存（prod 模式）
DEV_MODE = False  # 开发模式标志（禁用缓存）


def _read_asset(rel_path: str) -> bytes:
    """从 tools/webui/ 读取一个共享前端文件。

    dev 模式每次读盘（改 CSS/JS 直接刷新生效），prod 模式缓存到内存。

    Args:
        rel_path: 相对于 WEBUI_DIR 的文件路径

    Returns:
        bytes: 文件二进制内容
    """
    if not DEV_MODE and rel_path in _asset_cache:
        return _asset_cache[rel_path]  # 命中缓存
    p = WEBUI_DIR / rel_path
    data = p.read_bytes()
    if not DEV_MODE:
        _asset_cache[rel_path] = data  # 存入缓存
    return data


class Handler(BaseHTTPRequestHandler):
    """HTTP 请求处理器：路由 API 请求和静态资源。"""
    server_version = "tinyqwen-console/1.0"  # Server 头标识

    def log_message(self, fmt, *args):
        """静默默认访问日志（避免刷屏）。"""
        pass

    def _send(self, code: int, body: bytes, ctype: str, cache: str = "no-store"):
        """发送 HTTP 响应。"""
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", cache)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, obj):
        """发送 JSON 响应。"""
        self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"),
                   "application/json; charset=utf-8")

    def do_GET(self):
        """处理 GET 请求：静态资源 + API 端点。"""
        url = urlparse(self.path)
        path, query = url.path, parse_qs(url.query)
        try:
            if path == "/":
                # 控制台主页
                self._send(200, _read_asset("console/index.html"),
                           "text/html; charset=utf-8")
            elif path.startswith("/assets/"):
                # 静态资源（CSS/JS/SVG 等）
                rel = path[len("/assets/"):]
                if ".." in rel or rel.startswith("/"):
                    return self._json(400, {"error": "invalid path"})  # 防路径穿越
                try:
                    data = _read_asset(rel)
                except FileNotFoundError:
                    return self._json(404, {"error": "asset not found"})
                ext = Path(rel).suffix
                ctype = _MIME.get(ext, "application/octet-stream")
                # dev 模式不缓存，prod 模式缓存 5 分钟
                self._send(200, data, ctype, cache="no-cache" if DEV_MODE else "public, max-age=300")
            elif path == "/api/status":
                # 设备/环境状态
                self._json(200, collect_status())
            elif path == "/api/history":
                # 测量历史
                self._json(200, load_history())
            elif path == "/api/jobs":
                # 任务列表
                self._json(200, {"jobs": jobs_snapshot()})
            elif re.fullmatch(r"/api/jobs/[\w-]+", path):
                # 单个任务详情 + 完整日志
                job_id = path.rsplit("/", 1)[1]
                job = JOBS.get(job_id)
                if not job:
                    return self._json(404, {"error": "任务不存在"})
                log = ""
                lp = _job_log_path(job_id)
                if lp.is_file():
                    # 只读最后 500KB 防止超大日志撑爆内存
                    log = lp.read_text(encoding="utf-8", errors="replace")[-500_000:]
                meta = {k: job.get(k) for k in
                        ("id", "type", "label", "model", "status", "exit",
                         "started", "ended", "argv")}
                self._json(200, {"job": meta, "log": log})
            elif re.fullmatch(r"/api/jobs/[\w-]+/tail", path):
                # 日志增量读取（前端轮询用）
                job_id = path.split("/")[3]
                if job_id not in JOBS:
                    return self._json(404, {"error": "任务不存在"})
                offset = int((query.get("from") or ["0"])[0])  # 上次读到的偏移
                lp = _job_log_path(job_id)
                size = lp.stat().st_size if lp.is_file() else 0
                data = ""
                if lp.is_file() and size > offset:
                    with lp.open("r", encoding="utf-8", errors="replace") as f:
                        f.seek(offset)  # 跳到上次位置
                        data = f.read()  # 读取新增内容
                self._json(200, {
                    "from": offset, "size": size, "data": data,
                    "status": JOBS[job_id].get("status"),
                    "exit": JOBS[job_id].get("exit"),
                })
            elif path == "/api/series":
                # 资源时序曲线数据（bench_android.save_series 落盘）
                # 只认 SERIES_DIR 下的文件名，防路径穿越
                name = (query.get("file") or [""])[0]
                if not name or "/" in name or "\\" in name or ".." in name:
                    return self._json(400, {"error": "bad file name"})
                sp = REPO_ROOT / bench_android.SERIES_DIR / name
                if not sp.is_file():
                    return self._json(404, {"error": "series 不存在"})
                self._json(200, json.loads(sp.read_text(encoding="utf-8")))
            elif path == "/api/ttft":
                # 数据集行的 TTFT 散点 artifact（bench_dataset.save_ttft_scatter）
                name = (query.get("file") or [""])[0]
                if not name or "/" in name or "\\" in name or ".." in name:
                    return self._json(400, {"error": "bad file name"})
                sp = REPO_ROOT / bench_android.SERIES_DIR / name
                if not sp.is_file():
                    return self._json(404, {"error": "ttft artifact 不存在"})
                self._json(200, json.loads(sp.read_text(encoding="utf-8")))
            else:
                self._json(404, {"error": "not found"})
        except Exception as e:
            self._json(500, {"error": str(e)})

    def do_POST(self):
        """处理 POST 请求：创建任务 / 终止任务。"""
        url = urlparse(self.path)
        path = url.path
        try:
            length = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(length)) if length else {}
            if path == "/api/jobs":
                # 创建新任务
                try:
                    r = start_job(body)
                    self._json(200, r)
                except ValueError as e:
                    self._json(400, {"error": str(e)})  # 参数错误
                except RuntimeError as e:
                    self._json(409, {"error": str(e)})  # 设备忙
            elif re.fullmatch(r"/api/jobs/[\w-]+/kill", path):
                # 终止任务
                job_id = path.split("/")[3]
                try:
                    kill_job(job_id)
                    self._json(200, {"ok": True})
                except KeyError as e:
                    self._json(404, {"error": str(e)})
            else:
                self._json(404, {"error": "not found"})
        except json.JSONDecodeError:
            self._json(400, {"error": "invalid JSON"})
        except Exception as e:
            self._json(500, {"error": str(e)})


def main() -> None:
    """web_console.py 的主入口函数。

    解析参数 → 加载任务历史 → 启动 HTTP 服务器。
    """
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8765)   # 监听端口
    ap.add_argument("--host", default="127.0.0.1",
                    help="默认仅本机回环；不建议改")
    ap.add_argument("--dev", action="store_true",
                    help="开发模式:静态资源不缓存(改 CSS/JS 直接刷新生效)")
    args = ap.parse_args()

    global DEV_MODE
    DEV_MODE = args.dev  # 设置开发模式标志

    JOBS_DIR.mkdir(parents=True, exist_ok=True)  # 确保任务目录存在
    load_job_index()  # 从 index.jsonl 恢复历史任务

    # 预读 index.html（prod 模式缓存到内存）
    try:
        _read_asset("console/index.html")
    except FileNotFoundError:
        print("[web-console] 错误:找不到 tools/webui/console/index.html", file=sys.stderr)
        sys.exit(1)

    # 启动 HTTP 服务器
    try:
        server = ThreadingHTTPServer((args.host, args.port), Handler)
    except OSError as e:
        print(f"[web-console] 端口 {args.port} 已占用: {e}", file=sys.stderr)
        print(f"[web-console] 尝试 --port {args.port + 1} 或 lsof -iTCP:{args.port}", file=sys.stderr)
        sys.exit(1)
    mode = "dev(热重载)" if args.dev else "prod"
    print(f"[web-console] tinyqwen Android 测试控制台 ({mode})")
    print(f"[web-console] 打开浏览器：http://{args.host}:{args.port}")
    print(f"[web-console] Ctrl-C 退出")
    try:
        server.serve_forever()  # 阻塞主线程直到 Ctrl-C
    except KeyboardInterrupt:
        print("\n[web-console] 已退出")
        server.server_close()


# ---------------------------------------------------------------------------
# UI 已移至 tools/webui/ 共享前端目录:
#   console/index.html (骨架) + tokens.css + base.css + console/console.css
#   + util.js + chart.js + palette.js + console/console.js
# `/` 路由读 index.html,`/assets/*` 路由提供 CSS/JS 静态资源。
# ---------------------------------------------------------------------------


# 脚本直接运行入口
if __name__ == "__main__":
    main()
