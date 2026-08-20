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
    python tools/tokenize_batch.py --num 16 --out benchmarks/poetry16.jsonl
    python tools/bench_dataset.py --label poetry-neon \
        --jsonl benchmarks/poetry16.jsonl --model model_f16.tqwen \
        --extra-args "--matvec-impl neon_mt_kv_nt --ops-impl neon"
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
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import bench  # noqa: E402
import bench_android as ba  # noqa: E402

DEVICE_BATCH_INPUT = f"{ba.DEVICE_DIR}/batch_input.jsonl"
DEVICE_BATCH_RESULT = f"{ba.DEVICE_DIR}/batch_result.json"

# TTFT 按 prompt 长度分桶（token 数，左闭右开）
TTFT_BUCKETS = [(8, 16), (16, 32), (32, 64), (64, 96)]


def push_jsonl(local_jsonl: str) -> None:
    """push batch 输入；大小一致则跳过（与模型缓存同策略）。"""
    local_size = os.path.getsize(local_jsonl)
    raw = ba.adb_shell(f"stat -c%s {DEVICE_BATCH_INPUT} 2>/dev/null || echo 0")
    try:
        remote_size = int(raw.strip())
    except ValueError:
        remote_size = 0
    if local_size != remote_size:
        print(f"  pushing {os.path.basename(local_jsonl)} ({local_size} bytes)...")
        ba.adb("push", local_jsonl, DEVICE_BATCH_INPUT, capture=False)
    else:
        print(f"  {os.path.basename(local_jsonl)} already on device, skipping push")


def summarize_batch(b: dict) -> dict:
    """batch_result.json → 分布统计。C++ 只出原始数，口径都在这里。"""
    prompts = b["prompts"]
    ttfts = [p["ttft_ms"] for p in prompts]
    lens = [p["prompt_tokens"] for p in prompts]
    per_prompt_meds = [statistics.median(p["decode_ms"])
                       for p in prompts if p["decode_ms"]]
    all_decode = [v for p in prompts for v in p["decode_ms"]]

    buckets = []
    for lo, hi in TTFT_BUCKETS:
        idx = [i for i, n in enumerate(lens) if lo <= n < hi]
        if not idx:
            continue
        bs = [ttfts[i] for i in idx]
        bl = [lens[i] for i in idx]
        buckets.append({
            "range": f"{lo}-{hi}",
            "n": len(idx),
            "ttft_median_ms": round(statistics.median(bs), 2),
            # prefill 吞吐口径：每 token 的 prefill 耗时（按长度归一）
            "ms_per_token": round(statistics.median(t / l for t, l in zip(bs, bl)), 3),
        })

    return {
        "n_prompts": len(prompts),
        "prompt_len_min": min(lens),
        "prompt_len_median": statistics.median(lens),
        "prompt_len_max": max(lens),
        "ttft_median_ms": round(statistics.median(ttfts), 2),
        "ttft_p95_ms": round(bench.percentile(sorted(ttfts), 95), 2),
        "ttft_ms_per_token": round(
            statistics.median(t / l for t, l in zip(ttfts, lens)), 3),
        "ttft_buckets": buckets,
        "topt_median_ms": round(statistics.median(per_prompt_meds), 2),
        "topt_p95_ms": round(bench.percentile(sorted(all_decode), 95), 2),
        "decode_steps": len(all_decode),
        "total_wall_ms": round(b["total_wall_ms"], 1),
    }


def save_ttft_scatter(prompts: list[dict], label: str) -> str | None:
    """逐 prompt 的 (长度, TTFT, TOPT) 散点存成 <series 目录>/<时间>_<label>.ttft.json。

    batch_result.json 在设备上，host 侧拉回后默认即删——这份 artifact 是
    prefill 扩展性（TTFT vs prompt 长度）的持久记录，控制台曲线面板读它画散点。
    """
    pts = [{"prompt_tokens": p["prompt_tokens"],
            "ttft_ms": round(p["ttft_ms"], 3),
            "topt_median_ms": round(statistics.median(p["decode_ms"]), 3)
            if p["decode_ms"] else None}
           for p in prompts]
    if not pts:
        return None
    Path(ba.SERIES_DIR).mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    safe = re.sub(r"[^\w.-]", "_", label)[:40] or "unlabeled"
    path = f"{ba.SERIES_DIR}/{ts}_{safe}.ttft.json"
    Path(path).write_text(json.dumps({"points": pts}, ensure_ascii=False),
                          encoding="utf-8")
    return path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--label", default="dataset-unlabeled")
    p.add_argument("--jsonl", required=True,
                   help="tokenize_batch.py 生成的 batch 输入")
    p.add_argument("--model", default="model.tqwen")
    p.add_argument("--binary", default=None)
    p.add_argument("--decode-tokens", type=int, default=32,
                   help="每条 prompt 的 decode 长度（默认 32，与 canonical 一致保 TOPT 可比）")
    p.add_argument("--max-seq-len", type=int, default=128,
                   help="KV 容量（默认 128 = 容纳长 prompt + decode）")
    p.add_argument("--extra-args", default="",
                   help="原样传给设备端 binary 的额外 CLI 参数（如 kernel 配方）")
    p.add_argument("--json", default=None, help="结果写 JSON 文件")
    # 测量环境（与 bench_android 同款，默认值一致）
    p.add_argument("--pin-cores", default="big")
    p.add_argument("--threads", type=int, default=None)
    p.add_argument("--thermal-max", type=int, default=ba.THERMAL_MAX_DEFAULT)
    p.add_argument("--no-thermal-gate", action="store_true",
                   help="跳过热门禁（batch 跑得久，默认不建议关）")
    p.add_argument("--no-resource-sample", action="store_true")
    p.add_argument("--no-history", action="store_true")
    args = p.parse_args()

    build_dir = os.environ.get("BUILD_DIR", ba.BUILD_DIR_DEFAULT)
    binary = args.binary or f"{build_dir}/runtime/tinyqwen"
    if not os.path.isfile(binary):
        sys.exit(f"error: binary not found: {binary}; run scripts/build_android.sh first")
    if not os.path.isfile(args.model):
        sys.exit(f"error: model not found: {args.model}")
    if not os.path.isfile(args.jsonl):
        sys.exit(f"error: jsonl not found: {args.jsonl}; run tools/tokenize_batch.py first")

    extra = shlex.split(args.extra_args)

    # 数据集 meta（tokenize_batch.py 写的），用于记录负载指纹
    meta = {}
    meta_path = Path(args.jsonl).with_suffix(Path(args.jsonl).suffix + ".meta.json")
    if meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass

    ba.configure_measurement(args.pin_cores, args.threads,
                             thermal_gate=not args.no_thermal_gate,
                             thermal_max=args.thermal_max)
    ba._resource_sample_enabled = not args.no_resource_sample

    print(f"[bench-dataset] label={args.label}  jsonl={os.path.basename(args.jsonl)}"
          f"（{meta.get('num', '?')} 条，token 长度 "
          f"{meta.get('token_len_min', '?')}~{meta.get('token_len_max', '?')}，"
          f"中位 {meta.get('token_len_median', '?')}）")
    print(f"[bench-dataset] decode={args.decode_tokens} tok/prompt  "
          f"max_seq_len={args.max_seq_len}  "
          f"热门禁: {'开' if not args.no_thermal_gate else '关'}  绑核: {args.pin_cores}  "
          f"资源采样: {'开' if ba._resource_sample_enabled else '关'}")

    print("[bench-dataset] pushing binary & model & jsonl...")
    ba.ensure_binary_pushed(binary)
    model_on_device = ba.ensure_model_pushed(args.model)
    push_jsonl(args.jsonl)

    if ba._thermal_gate_enabled:
        ba.wait_for_thermal_cool(ba._thermal_max)
    temp_at_start = ba.read_device_thermal()

    override_args = [
                        f"--model {model_on_device}",
                        f"--batch-tokens-jsonl {DEVICE_BATCH_INPUT}",
                        f"--batch-out {DEVICE_BATCH_RESULT}",
                        f"--max-new-tokens {args.decode_tokens}",
                        f"--max-seq-len {args.max_seq_len}",
                        "--eos -1",
                    ] + extra
    script = ba.build_device_script(model_on_device, None,
                                    ba._resource_sample_enabled,
                                    override_args=override_args)
    try:
        device_out = ba.adb_shell(script)
    except subprocess.CalledProcessError as e:
        out = (getattr(e, "stdout", "") or "").strip()
        err = (getattr(e, "stderr", "") or "").strip()
        print(f"  ❌ 设备端运行失败（exit={e.returncode}）", file=sys.stderr)
        if out:
            print(f"  设备 stdout: {out[:800]}", file=sys.stderr)
        if err:
            print(f"  设备 stderr: {err[:800]}", file=sys.stderr)
        sys.exit(1)

    # 拉回 batch 结果
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
        local_result = tf.name
    ba.adb("pull", DEVICE_BATCH_RESULT, local_result)
    batch = json.loads(Path(local_result).read_text())
    Path(local_result).unlink(missing_ok=True)
    st = summarize_batch(batch)

    # 资源采样（整场一条时序）
    resources: dict = {}
    series: dict = {}
    if ba._resource_sample_enabled:
        resources, series = ba.parse_resource_trace(ba.pull_device_trace())
        m = re.search(r"kv cache: ([0-9.]+) MB", device_out)
        if m:
            resources.setdefault("mem", {})["kv_cache_mb"] = float(m.group(1))
    if resources.get("mem") is not None:
        resources["mem"]["model_mb"] = round(
            os.path.getsize(args.model) / (1024 * 1024), 1)

    host_env = bench.env_info()
    result = {
        "label": args.label,
        "platform": "android",
        "mode": "dataset",
        # headline（与 canonical 结果字段同名，方便工具复用；负载不同不可直接比）
        "decode_median_ms": st["topt_median_ms"],
        "decode_p95_ms": st["topt_p95_ms"],
        "ttft_ms": st["ttft_median_ms"],
        "total_ms": st["total_wall_ms"],
        "dataset_stats": st,
        "dataset_name": meta.get("csv") or os.path.basename(args.jsonl),
        "dataset_n_prompts": st["n_prompts"],
        "dataset_token_range": f"{st['prompt_len_min']}~{st['prompt_len_max']}",
        "decode_tokens_per_prompt": args.decode_tokens,
        "max_seq_len": args.max_seq_len,
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
    series_file = ba.save_series(series, args.label)
    if series_file:
        result["series_file"] = series_file
    ttft_file = save_ttft_scatter(batch["prompts"], args.label)
    if ttft_file:
        result["ttft_file"] = ttft_file

    # ---- 人类可读汇总 ----
    print(f"\n  数据集负载：{st['n_prompts']} 条 prompt，长度 "
          f"{st['prompt_len_min']}~{st['prompt_len_max']}（中位 {st['prompt_len_median']}）")
    print(f"  TTFT：median={st['ttft_median_ms']:.2f} ms  p95={st['ttft_p95_ms']:.2f}"
          f"（归一 {st['ttft_ms_per_token']:.3f} ms/tok）")
    for bk in st["ttft_buckets"]:
        print(f"    prompt {bk['range']:>5} tok：n={bk['n']:<2} "
              f"TTFT 中位 {bk['ttft_median_ms']:>8.2f} ms（{bk['ms_per_token']:.3f} ms/tok）")
    print(f"  TOPT：median={st['topt_median_ms']:.2f} ms/tok"
          f"（各 prompt 中位的中位）  全量 p95={st['topt_p95_ms']:.2f}"
          f"（{st['decode_steps']} 个 decode 步）")
    print(f"  总耗时：{st['total_wall_ms'] / 1000:.1f} s（wall clock）")
    print(f"  设备：{result['device_env'].get('device_model', '?')} / "
          f"{result['device_env'].get('soc', '?')}  commit：{host_env['git_commit']}")
    for line in ba.format_resources(resources):
        print(f"  {line}")
    if series_file:
        print(f"  资源曲线：{series_file}")

    if not args.no_history:
        # append_history 白名单扁平化：dataset_* 字段原样透传，
        # source 标 dataset 与 canonical 行区分。
        ba.append_history(result, source="dataset")

    if args.json:
        Path(args.json).write_text(json.dumps(result, ensure_ascii=False, indent=2))
        print(f"\n  已写入 {args.json}")


if __name__ == "__main__":
    main()
