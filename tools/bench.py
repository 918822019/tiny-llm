#!/usr/bin/env python3
"""tinyqwen 的可复现基准测试（benchmark）。

为什么需要它：
    优化必须先测量。只有用**完全相同的负载**反复测，才能判断一次改动到底
    带来了多少提升。本脚本把"标准负载 + 稳定统计 + 环境记录"固化下来，
    让不同版本/优化之间的数字可以直接对比。

标准负载（canonical workload，勿随意改动——改了等于换尺子）：
    - 模型   ：Qwen2.5-0.5B fp32（.tqwen）
    - prompt ：固定的 token 序列（见 CANONICAL_PROMPT，不依赖 tokenizer）
    - decode ：生成 DECODE_TOKENS 个 token，丢弃前 WARMUP 个（预热）
    - 配置   ：batch=1、greedy、单线程

用法：
    python tools/bench.py --model model.tqwen --label fp32-baseline
    python tools/bench.py --model model.tqwen --label int8-w8a8 --json out.json

输出：人类可读的汇总 + 一行可直接粘进 docs/optimization_log.md 的 Markdown。
"""

from __future__ import annotations

import argparse
import json
import platform
import shlex
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

# ---- 标准负载（固定，保证跨版本可比）-------------------------------------
# "中国的首都是" 的 token 序列。写死 id 是为了让 bench 不依赖 tokenizer。
CANONICAL_PROMPT = [105538, 59975, 100132]
DECODE_TOKENS = 32  # 生成的 token 数（样本量）
WARMUP = 4  # 丢弃前几个 decode token（预热，避免冷启动抖动）
MAX_SEQ_LEN = 64  # 足够容纳 prompt + decode


def env_info() -> dict:
    """记录运行环境，保证结果可复现、可追溯。"""
    info = {
        "platform": platform.platform(),
        "processor": platform.processor() or "unknown",
        "python": platform.python_version(),
    }
    # 记录当前 git commit：性能数字必须能对应到具体代码版本。
    try:
        rev = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True).stdout.strip()
        info["git_commit"] = rev
    except Exception:
        info["git_commit"] = "unknown"
    return info


def percentile(data: list[float], p: float) -> float:
    """简单百分位（p 取 0..100）。"""
    if not data:
        return float("nan")
    s = sorted(data)
    k = (len(s) - 1) * p / 100.0
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    frac = k - lo
    return s[lo] * (1 - frac) + s[hi] * frac


def run_once(binary: str, model: str, extra_args: list[str] | None = None) -> dict:
    """跑一次标准负载，返回 profiler JSON。

    extra_args：原样追加给 binary 的额外 CLI 参数。测"需要开关才生效"的
    优化变体时用，例如 ["--matvec-impl", "double_2_float"]。
    """
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
        profile_path = tf.name
    cmd = [
        binary,
        "--model", model,
        "--tokens", ",".join(map(str, CANONICAL_PROMPT)),
        "--max-new-tokens", str(DECODE_TOKENS),
        "--max-seq-len", str(MAX_SEQ_LEN),
        "--eos", "-1",  # 禁用停止符，保证生成固定数量，样本量稳定
        "--profile-out", profile_path,
    ]
    cmd += list(extra_args or [])
    subprocess.run(cmd, capture_output=True, check=True)
    with open(profile_path) as f:
        profile = json.load(f)
    Path(profile_path).unlink(missing_ok=True)
    return profile


def summarize(profile: dict) -> dict:
    """从 profiler 的逐 token 记录里提取稳定的延迟统计。"""
    decode_ms = [t["latency_ms"] for t in profile["tokens"] if not t["is_prefill"]]
    prefill_ms = [t["latency_ms"] for t in profile["tokens"] if t["is_prefill"]]

    # 丢弃预热的 decode token，剩下的才是稳态性能。
    steady = decode_ms[WARMUP:]
    if not steady:
        raise RuntimeError("no steady decode tokens; increase DECODE_TOKENS")

    # 按总耗时排序的 top op，用于解释"时间花在哪"。
    op_totals = profile.get("op_totals", {})
    top_ops = sorted(op_totals.items(), key=lambda kv: -kv[1]["total_ms"])[:5]

    return {
        "decode_samples": len(steady),
        "decode_median_ms": statistics.median(steady),
        "decode_mean_ms": statistics.mean(steady),
        "decode_min_ms": min(steady),
        "decode_p95_ms": percentile(steady, 95),
        "prefill_total_ms": sum(prefill_ms),
        # 批量 prefill 下整个 prompt 只有一条记录，真实长度以 profile 的
        # prompt_tokens 为准（旧 profile 无此字段时退回记录数）。
        "prefill_tokens": profile.get("prompt_tokens") or len(prefill_ms),
        # LLM serving 标准指标口径：
        # - TTFT：首 token 延迟 = prefill 总耗时（token-by-token prefill 下
        #   profiler 的 first_token_ms 即各 prefill token 之和）；
        # - TOPT：每输出 token 延迟，主指标用稳态 decode 中位数（见上方
        #   decode_median_ms），decode_avg_ms 是含预热的全量平均，仅参考；
        # - decode 长度 = 生成 token 总数（含被丢弃的预热 token）。
        "ttft_ms": profile.get("first_token_ms", sum(prefill_ms)),
        "decode_avg_ms": profile.get("decode_avg_ms"),
        "generated_tokens": profile.get("generated_tokens", len(decode_ms)),
        "total_ms": profile.get("total_ms"),
        "top_ops": [{"op": n, "total_ms": round(s["total_ms"], 2)} for n, s in top_ops],
    }


def check_regression(measured_median: float, baseline_path: str,
                     threshold_pct: float = 5.0, fail: bool = False) -> bool:
    """对比 baseline，回退超过 threshold_pct% 时警告。

    返回 True 表示检测到回归。fail=True 时回归会 sys.exit(1)。
    baseline 不存在时静默跳过（返回 False）。
    """
    bp = Path(baseline_path)
    if not bp.exists():
        return False
    try:
        base = json.loads(bp.read_text())
        base_med = base["decode_median_ms"]
    except (json.JSONDecodeError, KeyError):
        return False

    regression_pct = (measured_median - base_med) / base_med * 100
    if regression_pct > threshold_pct:
        print(f"\n  ⚠️ 回归警告：本次 {measured_median:.2f} ms/tok vs 基线 {base_med:.2f} ms/tok"
              f"（+{regression_pct:.1f}%，阈值 {threshold_pct}%）")
        print(f"     基线来源：{baseline_path}（label={base.get('label', '?')}）")
        if fail:
            print("     --fail-on-regression 已启用，退出")
            sys.exit(1)
        return True
    return False


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--binary", default="build/runtime/tinyqwen")
    p.add_argument("--model", default="model.tqwen")
    p.add_argument("--label", default="unlabeled",
                   help="本次测量的名字，如 fp32-baseline / int8-w8a8")
    p.add_argument("--runs", type=int, default=1,
                   help="把标准负载跑几遍，取各遍中位数的中位数（抗单次波动）")
    p.add_argument("--json", default=None, help="可选：把结果也写成 JSON 文件")
    p.add_argument("--extra-args", default="",
                   help="原样传给 binary 的额外 CLI 参数（引号括起），"
                        "如 '--matvec-impl double_2_float'")
    p.add_argument("--baseline", default="benchmarks/baseline.json",
                   help="回归检测对比的基线文件")
    p.add_argument("--no-check-regression", action="store_true",
                   help="跳过回归检测")
    p.add_argument("--regression-threshold", type=float, default=5.0,
                   help="回归阈值百分比（默认 5）")
    p.add_argument("--fail-on-regression", action="store_true",
                   help="回归时非零退出（CI 用）")
    args = p.parse_args()

    extra = shlex.split(args.extra_args)
    runs = max(1, args.runs)
    print(f"[bench] label={args.label} runs={runs}  负载: prompt={len(CANONICAL_PROMPT)} tok, "
          f"decode={DECODE_TOKENS} tok (丢弃预热 {WARMUP})")
    if extra:
        print(f"[bench] 额外参数: {' '.join(extra)}")

    # 跑 runs 遍，收集每遍的稳态统计；headline 取"中位数的中位数"，更稳。
    medians, p95s, last_stats = [], [], None
    for r in range(runs):
        profile = run_once(args.binary, args.model, extra)
        stats = summarize(profile)
        medians.append(stats["decode_median_ms"])
        p95s.append(stats["decode_p95_ms"])
        last_stats = stats
        if runs > 1:
            print(f"  run {r + 1}/{runs}: median={stats['decode_median_ms']:.2f}ms "
                  f"p95={stats['decode_p95_ms']:.2f}ms")

    env = env_info()
    headline = dict(last_stats)  # top_ops / prefill 等沿用最后一遍
    headline["decode_median_ms"] = statistics.median(medians)
    headline["decode_p95_ms"] = statistics.median(p95s)
    headline["runs"] = runs
    headline["per_run_median_ms"] = [round(m, 2) for m in medians]

    result = {"label": args.label, **headline, "env": env}
    if extra:
        result["extra_args"] = extra  # 可复现性：记录本次测量启用了什么开关

    # ---- 人类可读汇总 ----
    print(f"\n  TTFT：{headline['ttft_ms']:.2f} ms"
          f"（prefill {last_stats['prefill_tokens']} tok）")
    print(f"  TOPT：median={headline['decode_median_ms']:.2f} ms/tok  "
          f"p95={headline['decode_p95_ms']:.2f}（decode 共 {last_stats['generated_tokens']} tok，"
          f"丢预热 {WARMUP}，稳态样本 {last_stats['decode_samples']}）")
    if headline.get("total_ms") is not None:
        print(f"  forward 总耗时：{headline['total_ms']:.1f} ms")
    if runs > 1:
        print(f"  各遍中位数：{headline['per_run_median_ms']}（取中位 {headline['decode_median_ms']:.2f}）")
    print(f"  环境：{env['processor']} @ {env['git_commit']}")
    print("  耗时 top op：")
    for op in last_stats["top_ops"]:
        print(f"    {op['op']:<28} {op['total_ms']:>9.2f} ms")

    # ---- 可粘进 optimization_log.md 的 Markdown 行 ----
    print("\n  [复制下面这行到 docs/optimization_log.md 的表格里]")
    print(f"  | {args.label} | {env['git_commit']} | "
          f"{headline['decode_median_ms']:.2f} | {headline['decode_p95_ms']:.2f} | "
          f"<填写：相比基线的提升> | <填写：原因> |")

    # 回归检测
    if not args.no_check_regression:
        check_regression(headline["decode_median_ms"], args.baseline,
                         args.regression_threshold, args.fail_on_regression)

    if args.json:
        Path(args.json).write_text(json.dumps(result, ensure_ascii=False, indent=2))
        print(f"\n  已写入 {args.json}")


if __name__ == "__main__":
    main()
