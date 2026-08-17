#!/usr/bin/env python3
"""tinyqwen vs MNN 速度对比 benchmark。

同一个 Qwen2.5-0.5B 模型，不同量化 / 引擎：
  - tinyqwen fp32     (model.tqwen, ~1.9 GB)
  - tinyqwen f16      (model_f16.tqwen, ~988 MB)
  - MNN 4-bit         (llm.mnn + llm.mnn.weight, ~265 MB 打包)

测量方式：
  - tinyqwen：使用内置 profiler，逐 token 计时（最精确）
  - MNN：外部 wall-clock 计时 + 输出 token 计数（llm_infer 不暴露逐 token 时间）

用法：
    python3 tools/bench_vs_mnn.py
    python3 tools/bench_vs_mnn.py --runs 5
    python3 tools/bench_vs_mnn.py --mnn-only
    python3 tools/bench_vs_mnn.py --tinyqwen-impl neon_mt_kv_nt
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
MNN_DIR = Path("/tmp/zip_verify/mnn_demo")

# 统一 prompt："中国的首都是"
PROMPT_TEXT = "中国的首都是"
PROMPT_TOKENS = [105538, 59975, 100132]  # tinyqwen 用 token id

# 生成长度控制
DECODE_TOKENS = 64
WARMUP_TOKENS = 4  # tinyqwen profiler 丢弃前 N 个 decode token


def run_tinyqwen(label: str, model: str, impl: str, ops_impl: str,
                 extra_args: list[str] | None = None) -> dict | None:
    """跑 tinyqwen 一次，返回 {median_ms, tok_per_sec, ...}。"""
    binary = str(PROJECT / "build" / "runtime" / "tinyqwen")
    model_path = str(PROJECT / model)
    if not os.path.isfile(model_path):
        print(f"  [skip] {label}: 模型文件不存在 {model}")
        return None

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
        profile_path = tf.name

    cmd = [
        binary,
        "--model", model_path,
        "--tokens", ",".join(map(str, PROMPT_TOKENS)),
        "--max-new-tokens", str(DECODE_TOKENS),
        "--max-seq-len", "128",
        "--eos", "-1",
        "--profile-out", profile_path,
        "--matvec-impl", impl,
        "--ops-impl", ops_impl,
    ]
    cmd += list(extra_args or [])

    t0 = time.perf_counter()
    result = subprocess.run(cmd, capture_output=True)
    wall_s = time.perf_counter() - t0

    if result.returncode != 0:
        print(f"  [error] {label}: tinyqwen 返回 {result.returncode}")
        print(f"    stderr: {result.stderr.decode()[:200]}")
        Path(profile_path).unlink(missing_ok=True)
        return None

    with open(profile_path) as f:
        profile = json.load(f)
    Path(profile_path).unlink(missing_ok=True)

    decode_ms = [t["latency_ms"] for t in profile["tokens"] if not t["is_prefill"]]
    prefill_ms = [t["latency_ms"] for t in profile["tokens"] if t["is_prefill"]]
    steady = decode_ms[WARMUP_TOKENS:]
    if not steady:
        print(f"  [error] {label}: decode token 不足")
        return None

    median_ms = statistics.median(steady)
    return {
        "label": label,
        "engine": "tinyqwen",
        "quant": "fp32" if "f16" not in model else "f16",
        "decode_median_ms": round(median_ms, 2),
        "decode_p95_ms": round(sorted(steady)[int(len(steady) * 0.95)], 2),
        "tok_per_sec": round(1000.0 / median_ms, 1),
        "prefill_ms": round(sum(prefill_ms), 2),
        "decode_tokens": len(decode_ms),
        "wall_s": round(wall_s, 2),
        "impl": impl,
    }


def run_mnn(label: str, max_tokens: int = DECODE_TOKENS) -> dict | None:
    """跑 MNN llm_infer 一次，wall-clock 计时。"""
    if not MNN_DIR.exists():
        print(f"  [skip] MNN: 目录不存在 {MNN_DIR}")
        print(f"    请先解压：unzip ~/Downloads/mnn_demo_3.6.1.2_macos_arm64.zip -d /tmp/zip_verify/")
        return None

    llm_infer = MNN_DIR / "llm_infer"
    config = MNN_DIR / "models" / "qwen25_v8_4bit" / "config.json"

    # prompt 文件
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, dir="/tmp") as pf:
        pf.write(PROMPT_TEXT)
        prompt_path = pf.name

    out_path = "/tmp/mnn_bench_out.jsonl"
    Path(out_path).unlink(missing_ok=True)

    cmd = [str(llm_infer), str(config), prompt_path, out_path]

    t0 = time.perf_counter()
    result = subprocess.run(cmd, capture_output=True, cwd=str(MNN_DIR))
    wall_s = time.perf_counter() - t0

    Path(prompt_path).unlink(missing_ok=True)

    if result.returncode != 0:
        print(f"  [error] MNN: 返回 {result.returncode}")
        print(f"    stderr: {result.stderr.decode()[:300]}")
        return None

    # 读输出，估算 token 数
    try:
        with open(out_path) as f:
            lines = f.readlines()
        resp = json.loads(lines[0])["response"]
    except Exception as e:
        print(f"  [error] MNN: 读取输出失败: {e}")
        return None

    # 粗略 token 估算：中文字符约 1-1.5 tok/char，这里用字符数 * 0.7 近似
    # 更准确的做法是用 tokenizer，但 MNN 用的是 .mtok 格式
    n_chars = len(resp)
    # 对于 Qwen2.5 中文，约 1.2-1.5 字符/token
    est_tokens = int(n_chars * 0.7)

    # 用 wall time 算整体 tok/s（包含 prefill）
    tok_per_sec = est_tokens / wall_s if wall_s > 0 else 0

    return {
        "label": label,
        "engine": "MNN 3.6.1.2",
        "quant": "int4",
        "wall_s": round(wall_s, 2),
        "output_chars": n_chars,
        "est_tokens": est_tokens,
        "tok_per_sec_overall": round(tok_per_sec, 1),
        "decode_median_ms": round(1000.0 / tok_per_sec, 2) if tok_per_sec > 0 else None,
        "threads": 4,
        "note": "wall-clock 计时，含 prefill + 模型加载；token 数为估算",
    }


def print_comparison(results: list[dict]) -> None:
    """打印对比表格。"""
    print("\n" + "=" * 72)
    print("  Qwen2.5-0.5B 推理速度对比 (macOS arm64)")
    print("=" * 72)

    # 表头
    print(f"\n  {'配置':<28} {'量化':>5} {'ms/tok':>8} {'tok/s':>7} {'wall(s)':>8}  备注")
    print(f"  {'-'*28} {'-'*5} {'-'*8} {'-'*7} {'-'*8}  {'-'*16}")

    for r in results:
        if r is None:
            continue
        quant = r.get("quant", "?")
        if "decode_median_ms" in r and r["decode_median_ms"]:
            ms_tok = f"{r['decode_median_ms']:.1f}"
        else:
            ms_tok = "N/A"

        tok_s = r.get("tok_per_sec") or r.get("tok_per_sec_overall") or 0
        tok_s_str = f"{tok_s:.1f}"

        wall = f"{r['wall_s']:.2f}" if "wall_s" in r else "N/A"

        note = r.get("impl", "") or r.get("note", "")
        if len(note) > 30:
            note = note[:30] + "…"

        print(f"  {r['label']:<28} {quant:>5} {ms_tok:>8} {tok_s_str:>7} {wall:>8}  {note}")

    print()


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", type=int, default=3, help="每个配置跑几遍取中位数")
    p.add_argument("--mnn-only", action="store_true", help="只测 MNN")
    p.add_argument("--tinyqwen-only", action="store_true", help="只测 tinyqwen")
    p.add_argument("--tinyqwen-impl", default=None,
                   help="额外测一个 tinyqwen matvec impl（如 neon_mt_kv_nt）")
    args = p.parse_args()

    runs = max(1, args.runs)
    results = []

    print(f"[bench_vs_mnn] 对比测试 | prompt=\"{PROMPT_TEXT}\" | decode={DECODE_TOKENS} tok | runs={runs}")
    print()

    # --- MNN ---
    if not args.tinyqwen_only:
        print("[MNN 3.6.1.2] int4 量化, 4 threads ...")
        mnn_results = []
        for i in range(runs):
            r = run_mnn(f"MNN int4 (run {i+1})")
            if r:
                mnn_results.append(r)
                print(f"  run {i+1}/{runs}: {r['wall_s']:.2f}s, ~{r['tok_per_sec_overall']:.1f} tok/s "
                      f"({r['output_chars']} chars → est {r['est_tokens']} tok)")

        if mnn_results:
            # 取 wall time 中位数那次
            mnn_results.sort(key=lambda x: x["wall_s"])
            median_run = mnn_results[len(mnn_results) // 2]
            median_run["label"] = "MNN 3.6.1.2 int4"
            results.append(median_run)
        print()

    # --- tinyqwen configs ---
    if not args.mnn_only:
        configs = [
            ("tinyqwen fp32 ref", "model.tqwen", "ref", "ref"),
            ("tinyqwen f16 ref", "model_f16.tqwen", "ref", "ref"),
            ("tinyqwen fp32 neon_mt_kv_nt", "model.tqwen", "neon_mt_kv_nt", "neon"),
            ("tinyqwen f16 neon_mt_kv_nt", "model_f16.tqwen", "neon_mt_kv_nt", "neon"),
        ]

        if args.tinyqwen_impl and args.tinyqwen_impl not in ("ref", "neon_mt_kv_nt"):
            configs.append((f"tinyqwen fp32 {args.tinyqwen_impl}",
                           "model.tqwen", args.tinyqwen_impl, "neon"))

        for label, model, impl, ops in configs:
            print(f"[{label}] ...")
            medians = []
            for i in range(runs):
                r = run_tinyqwen(f"{label} (run {i+1})", model, impl, ops)
                if r:
                    medians.append(r)
                    print(f"  run {i+1}/{runs}: median={r['decode_median_ms']:.2f} ms/tok, "
                          f"{r['tok_per_sec']:.1f} tok/s")
                else:
                    break

            if medians:
                # 取 median of medians
                medians.sort(key=lambda x: x["decode_median_ms"])
                best = medians[len(medians) // 2]
                best["label"] = label
                results.append(best)
            print()

    # --- 对比 ---
    print_comparison(results)

    # JSON 输出
    out_path = PROJECT / "benchmarks" / "mnn_comparison.json"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"  结果已写入: {out_path}")


if __name__ == "__main__":
    main()
