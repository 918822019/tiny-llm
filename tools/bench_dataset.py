#!/usr/bin/env python3
"""tinyqwen 数据集负载测试（Android 端侧）：真实 prompt 批量测 TTFT/TOPT。

与 canonical 标准负载（3 tok prompt，tools/bench_android.py）的区别：
- 输入是数据集里长度不等的真实文本（先用 tools/tokenize_batch.py 生成 JSONL）；
- runtime 走批量模式（--batch-tokens-jsonl）：一次进程内每条 prompt 独立
  reset → prefill → decode，逐条计时；
- 测的是 **TTFT / TOPT 在一组真实 prompt 上的分布**，不是 canonical 负载
  的单点值。两者的数字**不可直接互比**（负载不同 = 尺子不同）：TOPT 口径
  相近可参考，TTFT 只有本模式能测。落 history 时 source="dataset" 标记区分。

用法：
    # 先生成 batch 输入
    python tools/tokenize_batch.py --num 16 --out benchmarks/poetry16.jsonl
    # 再跑数据集测速
    python tools/bench_dataset.py --label poetry-neon \\
        --jsonl benchmarks/poetry16.jsonl --model model_f16.tqwen \\
        --extra-args "--matvec-impl neon_mt_kv_nt --ops-impl neon"

输入：
    - tokenize_batch.py 生成的 JSONL 文件（每行 {"tokens": [...]}）
    - Android 设备通过 adb 连接
    - build-android/runtime/tinyqwen 交叉编译产物

输出：
    - 终端打印 TTFT/TOPT 分布统计（含按 prompt 长度分桶）
    - 追加 benchmarks/history_android.jsonl（source="dataset"）
    - TTFT 散点 artifact 存 benchmarks/series/
    - 可选：--json 写入完整结果 JSON
"""

# 启用延迟注解求值
from __future__ import annotations

# ---- 标准库导入 ----
import argparse       # 命令行参数解析
import datetime       # 时间戳（TTFT 散点文件名）
import json           # JSON 序列化/反序列化
import os             # 文件大小查询
import re             # 正则表达式（文件名安全化、runtime 输出解析）
import shlex          # shell 字符串安全分割
import statistics     # 统计函数（median）
import subprocess     # 调用外部进程（adb）
import sys            # 系统退出
import tempfile       # 临时文件（batch result 中转）
from pathlib import Path  # 路径操作

# 把 tools/ 目录加入搜索路径
sys.path.insert(0, str(Path(__file__).parent))
import bench  # noqa: E402  # 复用 env_info、percentile 等
import bench_android as ba  # noqa: E402  # 复用 adb helpers、资源采样、测量配置等

# 设备端 batch 输入/输出文件路径
DEVICE_BATCH_INPUT = f"{ba.DEVICE_DIR}/batch_input.jsonl"   # batch 输入 JSONL
DEVICE_BATCH_RESULT = f"{ba.DEVICE_DIR}/batch_result.json"  # batch 结果 JSON

# TTFT 按 prompt 长度分桶的边界（token 数，左闭右开）
TTFT_BUCKETS = [(8, 16), (16, 32), (32, 64), (64, 96)]


def push_jsonl(local_jsonl: str) -> None:
    """推送 batch 输入 JSONL 到设备；大小一致则跳过（与模型缓存同策略）。

    Args:
        local_jsonl: 本地 JSONL 文件路径
    """
    local_size = os.path.getsize(local_jsonl)  # 本地文件大小
    # 查询设备端已有文件大小
    raw = ba.adb_shell(f"stat -c%s {DEVICE_BATCH_INPUT} 2>/dev/null || echo 0")
    try:
        remote_size = int(raw.strip())
    except ValueError:
        remote_size = 0
    if local_size != remote_size:
        # 大小不一致，需要重新推送
        print(f"  pushing {os.path.basename(local_jsonl)} ({local_size} bytes)...")
        ba.adb("push", local_jsonl, DEVICE_BATCH_INPUT, capture=False)
    else:
        # 大小一致，跳过
        print(f"  {os.path.basename(local_jsonl)} already on device, skipping push")


def summarize_batch(b: dict) -> dict:
    """将 batch_result.json 转换为分布统计。

    C++ runtime 只出原始数据（每条 prompt 的 ttft_ms、decode_ms 列表），
    所有统计口径都在这里定义。

    Args:
        b: batch_result.json 的内容字典，包含 prompts 列表和 total_wall_ms

    Returns:
        dict: 包含 n_prompts、prompt_len_*、ttft_*、topt_*、decode_steps 等字段
    """
    prompts = b["prompts"]  # 每条 prompt 的结果列表
    ttfts = [p["ttft_ms"] for p in prompts]               # 所有 TTFT 值
    lens = [p["prompt_tokens"] for p in prompts]           # 所有 prompt 长度
    # 每条 prompt 内部 decode 延迟的中位数（先 per-prompt 聚合再跨 prompt 取中位）
    per_prompt_meds = [statistics.median(p["decode_ms"])
                       for p in prompts if p["decode_ms"]]
    # 全部 decode 步的延迟列表（用于全局 P95）
    all_decode = [v for p in prompts for v in p["decode_ms"]]

    # 按 prompt 长度分桶统计 TTFT
    buckets = []
    for lo, hi in TTFT_BUCKETS:
        # 找出长度在 [lo, hi) 范围内的 prompt 索引
        idx = [i for i, n in enumerate(lens) if lo <= n < hi]
        if not idx:
            continue  # 该桶没有样本，跳过
        bs = [ttfts[i] for i in idx]    # 该桶的 TTFT 值
        bl = [lens[i] for i in idx]     # 该桶的 prompt 长度
        buckets.append({
            "range": f"{lo}-{hi}",                                       # 桶范围标签
            "n": len(idx),                                                # 样本数
            "ttft_median_ms": round(statistics.median(bs), 2),           # TTFT 中位数
            # prefill 吞吐口径：每 token 的 prefill 耗时（按长度归一化）
            "ms_per_token": round(statistics.median(t / l for t, l in zip(bs, bl)), 3),
        })

    return {
        "n_prompts": len(prompts),                              # prompt 总数
        "prompt_len_min": min(lens),                             # 最短 prompt 长度
        "prompt_len_median": statistics.median(lens),            # prompt 长度中位数
        "prompt_len_max": max(lens),                             # 最长 prompt 长度
        "ttft_median_ms": round(statistics.median(ttfts), 2),   # TTFT 全局中位数
        "ttft_p95_ms": round(bench.percentile(sorted(ttfts), 95), 2),  # TTFT P95
        # TTFT 归一化指标：每 token 的 prefill 耗时中位数
        "ttft_ms_per_token": round(
            statistics.median(t / l for t, l in zip(ttfts, lens)), 3),
        "ttft_buckets": buckets,                                 # 分桶统计
        "topt_median_ms": round(statistics.median(per_prompt_meds), 2),  # TOPT 中位数的中位数
        "topt_p95_ms": round(bench.percentile(sorted(all_decode), 95), 2),  # TOPT 全局 P95
        "decode_steps": len(all_decode),                         # decode 总步数
        "total_wall_ms": round(b["total_wall_ms"], 1),           # wall-clock 总耗时
    }


def save_ttft_scatter(prompts: list[dict], label: str) -> str | None:
    """保存逐 prompt 的 (长度, TTFT, TOPT) 散点数据为 JSON 文件。

    batch_result.json 在设备上，host 侧拉回后默认即删——这份 artifact 是
    prefill 扩展性（TTFT vs prompt 长度）的持久记录，控制台曲线面板读它画散点。

    Args:
        prompts: batch_result.json 中的 prompts 列表
        label: 配置标签（用于文件名）

    Returns:
        str | None: 散点文件的相对路径，无数据返回 None
    """
    # 构建散点数据列表
    pts = [{"prompt_tokens": p["prompt_tokens"],
            "ttft_ms": round(p["ttft_ms"], 3),
            "topt_median_ms": round(statistics.median(p["decode_ms"]), 3)
            if p["decode_ms"] else None}
           for p in prompts]
    if not pts:
        return None
    Path(ba.SERIES_DIR).mkdir(parents=True, exist_ok=True)  # 确保目录存在
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")   # 时间戳防覆盖
    safe = re.sub(r"[^\w.-]", "_", label)[:40] or "unlabeled"  # 文件名安全化
    path = f"{ba.SERIES_DIR}/{ts}_{safe}.ttft.json"
    Path(path).write_text(json.dumps({"points": pts}, ensure_ascii=False),
                          encoding="utf-8")
    return path


def main() -> None:
    """bench_dataset.py 的主入口函数。

    解析参数 → 推送文件 → 在设备上跑批量测速 → 汇总统计 → 打印结果 → 写历史。
    """
    # 创建参数解析器
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--label", default="dataset-unlabeled")        # 标签名
    p.add_argument("--jsonl", required=True,
                   help="tokenize_batch.py 生成的 batch 输入")    # JSONL 文件路径
    p.add_argument("--model", default="model.tqwen")              # 模型文件
    p.add_argument("--binary", default=None)                      # binary 路径
    p.add_argument("--decode-tokens", type=int, default=32,
                   help="每条 prompt 的 decode 长度（默认 32，与 canonical 一致保 TOPT 可比）")
    p.add_argument("--max-seq-len", type=int, default=128,
                   help="KV 容量（默认 128 = 容纳长 prompt + decode）")
    p.add_argument("--extra-args", default="",
                   help="原样传给设备端 binary 的额外 CLI 参数（如 kernel 配方）")
    p.add_argument("--json", default=None, help="结果写 JSON 文件")
    # ---- 测量环境参数（与 bench_android 同款，默认值一致）----
    p.add_argument("--pin-cores", default="big")                  # 绑核模式
    p.add_argument("--threads", type=int, default=None)           # 线程数覆盖
    p.add_argument("--thermal-max", type=int, default=ba.THERMAL_MAX_DEFAULT)  # 热门禁阈值
    p.add_argument("--no-thermal-gate", action="store_true",
                   help="跳过热门禁（batch 跑得久，默认不建议关）")
    p.add_argument("--no-resource-sample", action="store_true")   # 跳过资源采样
    p.add_argument("--no-history", action="store_true")           # 不追加历史
    args = p.parse_args()

    # 确定 binary 路径
    build_dir = os.environ.get("BUILD_DIR", ba.BUILD_DIR_DEFAULT)
    binary = args.binary or f"{build_dir}/runtime/tinyqwen"
    # 检查必要文件
    if not os.path.isfile(binary):
        sys.exit(f"error: binary not found: {binary}; run scripts/build_android.sh first")
    if not os.path.isfile(args.model):
        sys.exit(f"error: model not found: {args.model}")
    if not os.path.isfile(args.jsonl):
        sys.exit(f"error: jsonl not found: {args.jsonl}; run tools/tokenize_batch.py first")

    # 解析额外参数
    extra = shlex.split(args.extra_args)

    # 读取数据集 meta 信息（tokenize_batch.py 写的），用于记录负载指纹
    meta = {}
    meta_path = Path(args.jsonl).with_suffix(Path(args.jsonl).suffix + ".meta.json")
    if meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass  # meta 损坏不影响主流程

    # 配置测量参数（与 bench_android 共用同一套）
    ba.configure_measurement(args.pin_cores, args.threads,
                             thermal_gate=not args.no_thermal_gate,
                             thermal_max=args.thermal_max)
    ba._resource_sample_enabled = not args.no_resource_sample

    # 打印配置摘要
    print(f"[bench-dataset] label={args.label}  jsonl={os.path.basename(args.jsonl)}"
          f"（{meta.get('num', '?')} 条，token 长度 "
          f"{meta.get('token_len_min', '?')}~{meta.get('token_len_max', '?')}，"
          f"中位 {meta.get('token_len_median', '?')}）")
    print(f"[bench-dataset] decode={args.decode_tokens} tok/prompt  "
          f"max_seq_len={args.max_seq_len}  "
          f"热门禁: {'开' if not args.no_thermal_gate else '关'}  绑核: {args.pin_cores}  "
          f"资源采样: {'开' if ba._resource_sample_enabled else '关'}")

    # 推送 binary、模型和 JSONL 到设备
    print("[bench-dataset] pushing binary & model & jsonl...")
    ba.ensure_binary_pushed(binary)
    model_on_device = ba.ensure_model_pushed(args.model)
    push_jsonl(args.jsonl)

    # 热门禁等待
    if ba._thermal_gate_enabled:
        ba.wait_for_thermal_cool(ba._thermal_max)
    temp_at_start = ba.read_device_thermal()  # 记录开跑前温度

    # 构造 override_args：完全替换默认 tinyqwen 参数（数据集批量模式）
    override_args = [
                        f"--model {model_on_device}",
                        f"--batch-tokens-jsonl {DEVICE_BATCH_INPUT}",  # batch 输入
                        f"--batch-out {DEVICE_BATCH_RESULT}",          # batch 输出
                        f"--max-new-tokens {args.decode_tokens}",      # 每条 prompt 的 decode 长度
                        f"--max-seq-len {args.max_seq_len}",           # KV 容量
                        "--eos -1",                                    # 禁用 EOS
                    ] + extra
    # 用 build_device_script 构造完整的设备端 shell 脚本（含资源采样 wrapper）
    script = ba.build_device_script(model_on_device, None,
                                    ba._resource_sample_enabled,
                                    override_args=override_args)
    try:
        device_out = ba.adb_shell(script)  # 在设备上执行
    except subprocess.CalledProcessError as e:
        out = (getattr(e, "stdout", "") or "").strip()
        err = (getattr(e, "stderr", "") or "").strip()
        print(f"  ❌ 设备端运行失败（exit={e.returncode}）", file=sys.stderr)
        if out:
            print(f"  设备 stdout: {out[:800]}", file=sys.stderr)
        if err:
            print(f"  设备 stderr: {err[:800]}", file=sys.stderr)
        sys.exit(1)

    # 拉回 batch 结果 JSON
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
        local_result = tf.name
    ba.adb("pull", DEVICE_BATCH_RESULT, local_result)
    batch = json.loads(Path(local_result).read_text())
    Path(local_result).unlink(missing_ok=True)  # 清理临时文件
    st = summarize_batch(batch)  # 计算分布统计

    # 资源采样（整场一条时序）
    resources: dict = {}
    series: dict = {}
    if ba._resource_sample_enabled:
        resources, series = ba.parse_resource_trace(ba.pull_device_trace())
        # 从 runtime 输出中提取 KV cache 大小
        m = re.search(r"kv cache: ([0-9.]+) MB", device_out)
        if m:
            resources.setdefault("mem", {})["kv_cache_mb"] = float(m.group(1))
    # 注入权重大小便于对账
    if resources.get("mem") is not None:
        resources["mem"]["model_mb"] = round(
            os.path.getsize(args.model) / (1024 * 1024), 1)

    # 采集 host 环境信息
    host_env = bench.env_info()
    # 组装结果字典
    result = {
        "label": args.label,
        "platform": "android",
        "mode": "dataset",
        # headline 字段与 canonical 结果同名，方便工具复用；负载不同不可直接比
        "decode_median_ms": st["topt_median_ms"],
        "decode_p95_ms": st["topt_p95_ms"],
        "ttft_ms": st["ttft_median_ms"],
        "total_ms": st["total_wall_ms"],
        "dataset_stats": st,                            # 完整的数据集统计
        "dataset_name": meta.get("csv") or os.path.basename(args.jsonl),  # 数据集来源
        "dataset_n_prompts": st["n_prompts"],           # prompt 数量
        "dataset_token_range": f"{st['prompt_len_min']}~{st['prompt_len_max']}",  # token 长度范围
        "decode_tokens_per_prompt": args.decode_tokens, # 每条 prompt 的 decode 长度
        "max_seq_len": args.max_seq_len,                # KV 容量
        "device_env": ba.device_env_info(),
        "host_env": host_env,
        "pin_cores": args.pin_cores,
        "mt_threads": ba._mt_threads if ba._mt_threads else "kernel-default",
        "temp_at_start_mc": temp_at_start,
        "thermal_gate": ba._thermal_gate_enabled,
        "resource_sample": ba._resource_sample_enabled,
        "resources": resources,
        "model": os.path.basename(args.model),
    }
    if extra:
        result["extra_args"] = extra
    # 保存资源时序曲线
    series_file = ba.save_series(series, args.label)
    if series_file:
        result["series_file"] = series_file
    # 保存 TTFT 散点 artifact
    ttft_file = save_ttft_scatter(batch["prompts"], args.label)
    if ttft_file:
        result["ttft_file"] = ttft_file

    # ---- 人类可读汇总打印 ----
    print(f"\n  数据集负载：{st['n_prompts']} 条 prompt，长度 "
          f"{st['prompt_len_min']}~{st['prompt_len_max']}（中位 {st['prompt_len_median']}）")
    print(f"  TTFT：median={st['ttft_median_ms']:.2f} ms  p95={st['ttft_p95_ms']:.2f}"
          f"（归一 {st['ttft_ms_per_token']:.3f} ms/tok）")
    # 打印每个分桶的 TTFT 统计
    for bk in st["ttft_buckets"]:
        print(f"    prompt {bk['range']:>5} tok：n={bk['n']:<2} "
              f"TTFT 中位 {bk['ttft_median_ms']:>8.2f} ms（{bk['ms_per_token']:.3f} ms/tok）")
    print(f"  TOPT：median={st['topt_median_ms']:.2f} ms/tok"
          f"（各 prompt 中位的中位）  全量 p95={st['topt_p95_ms']:.2f}"
          f"（{st['decode_steps']} 个 decode 步）")
    print(f"  总耗时：{st['total_wall_ms'] / 1000:.1f} s（wall clock）")
    print(f"  设备：{result['device_env'].get('device_model', '?')} / "
          f"{result['device_env'].get('soc', '?')}  commit：{host_env['git_commit']}")
    # 打印资源采样信息
    for line in ba.format_resources(resources):
        print(f"  {line}")
    if series_file:
        print(f"  资源曲线：{series_file}")

    # 追加结构化历史（source="dataset" 与 canonical 行区分）
    if not args.no_history:
        ba.append_history(result, source="dataset")

    # 写入 JSON 文件
    if args.json:
        Path(args.json).write_text(json.dumps(result, ensure_ascii=False, indent=2))
        print(f"\n  已写入 {args.json}")


# 脚本直接运行入口
if __name__ == "__main__":
    main()
