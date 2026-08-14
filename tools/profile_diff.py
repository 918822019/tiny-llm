#!/usr/bin/env python3
"""对比两个 profile JSON，输出 op 级性能差异。

快速定位优化/回退来源：哪些 op 变快了、哪些变慢了、按多少。

用法：
    python tools/profile_diff.py baseline.json optimized.json
    python tools/profile_diff.py --threshold 2 baseline.json optimized.json
    python tools/profile_diff.py --json diff.json baseline.json optimized.json
    python tools/profile_diff.py --category baseline.json optimized.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# op 分类规则：按名字前缀/后缀归类
CATEGORIES = {
    "matvec": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "lm_head"],
    "norm": ["layernorm", "rmsnorm"],
    "attention": ["attention", "rope", "kv_append"],
    "activation": ["swiglu", "silu"],
    "other": ["embed", "topk_argmax", "residual"],
}


def categorize_op(op_name: str) -> str:
    lower = op_name.lower()
    # 去掉 "layer_N." 前缀
    if "." in lower:
        suffix = lower.split(".", 1)[1]
    else:
        suffix = lower
    for cat, keywords in CATEGORIES.items():
        for kw in keywords:
            if kw in suffix:
                return cat
    return "other"


def load_op_totals(profile_path: str) -> dict[str, float]:
    """从 profile JSON 的 op_totals 提取 {op_name: total_ms}。"""
    with open(profile_path) as f:
        data = json.load(f)
    op_totals = data.get("op_totals", {})
    return {name: info["total_ms"] for name, info in op_totals.items()}


def compute_diff(before: dict[str, float], after: dict[str, float]) -> list[dict]:
    """计算每个 op 的差异。返回按 delta_ms 排序的列表。"""
    all_ops = set(before) | set(after)
    diffs = []
    for op in sorted(all_ops):
        ms_before = before.get(op, 0.0)
        ms_after = after.get(op, 0.0)
        delta_ms = ms_after - ms_before
        delta_pct = ((ms_after - ms_before) / ms_before * 100) if ms_before > 0 else (
            float("inf") if ms_after > 0 else 0.0)
        diffs.append({
            "op": op,
            "before_ms": round(ms_before, 3),
            "after_ms": round(ms_after, 3),
            "delta_ms": round(delta_ms, 3),
            "delta_pct": round(delta_pct, 1),
            "category": categorize_op(op),
        })
    return diffs


def print_diff(diffs: list[dict], threshold_pct: float, show_category: bool) -> None:
    total_before = sum(d["before_ms"] for d in diffs)
    total_after = sum(d["after_ms"] for d in diffs)
    speedup = total_before / total_after if total_after > 0 else float("inf")

    print(f"Profile Diff")
    print(f"  总延迟: {total_before:.2f} ms → {total_after:.2f} ms ({speedup:.2f}×)")
    print()

    # 按 delta_ms 排序
    improved = sorted([d for d in diffs if d["delta_pct"] < -threshold_pct],
                      key=lambda d: d["delta_ms"])
    regressed = sorted([d for d in diffs if d["delta_pct"] > threshold_pct],
                       key=lambda d: -d["delta_ms"])

    if improved:
        print("  Top 改善（变快）:")
        for d in improved[:10]:
            print(f"    {d['op']:<36} {d['before_ms']:>8.2f} → {d['after_ms']:>8.2f} ms  "
                  f"({d['delta_pct']:+.1f}%, {d['delta_ms']:+.2f} ms)")
    else:
        print(f"  无显著改善（阈值 {threshold_pct}%）")

    print()
    if regressed:
        print("  Top 回退（变慢）:")
        for d in regressed[:10]:
            print(f"    {d['op']:<36} {d['before_ms']:>8.2f} → {d['after_ms']:>8.2f} ms  "
                  f"({d['delta_pct']:+.1f}%, {d['delta_ms']:+.2f} ms)")
    else:
        print("  无显著回退")

    if show_category:
        print()
        print("  按分类汇总:")
        cat_before: dict[str, float] = {}
        cat_after: dict[str, float] = {}
        for d in diffs:
            cat = d["category"]
            cat_before[cat] = cat_before.get(cat, 0) + d["before_ms"]
            cat_after[cat] = cat_after.get(cat, 0) + d["after_ms"]
        for cat in ["matvec", "attention", "norm", "activation", "other"]:
            b = cat_before.get(cat, 0)
            a = cat_after.get(cat, 0)
            if b == 0 and a == 0:
                continue
            pct = ((a - b) / b * 100) if b > 0 else 0
            print(f"    {cat:<12} {b:>8.2f} → {a:>8.2f} ms  ({pct:+.1f}%)")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("before", help="基线 profile JSON")
    p.add_argument("after", help="优化后 profile JSON")
    p.add_argument("--threshold", type=float, default=1.0,
                   help="只显示变化超过 N%% 的 op（默认 1）")
    p.add_argument("--json", default=None, help="输出 JSON 文件")
    p.add_argument("--category", action="store_true", default=True,
                   help="显示按分类汇总（默认开启）")
    p.add_argument("--no-category", action="store_true",
                   help="不显示按分类汇总")
    args = p.parse_args()

    if not Path(args.before).exists():
        sys.exit(f"error: file not found: {args.before}")
    if not Path(args.after).exists():
        sys.exit(f"error: file not found: {args.after}")

    before = load_op_totals(args.before)
    after = load_op_totals(args.after)

    if not before:
        sys.exit(f"error: no op_totals in {args.before}")
    if not after:
        sys.exit(f"error: no op_totals in {args.after}")

    diffs = compute_diff(before, after)
    print_diff(diffs, args.threshold, args.category and not args.no_category)

    if args.json:
        total_before = sum(d["before_ms"] for d in diffs)
        total_after = sum(d["after_ms"] for d in diffs)
        output = {
            "total_before_ms": round(total_before, 2),
            "total_after_ms": round(total_after, 2),
            "speedup": round(total_before / total_after, 3) if total_after > 0 else None,
            "ops": diffs,
        }
        Path(args.json).write_text(json.dumps(output, ensure_ascii=False, indent=2))
        print(f"\n  已写入 {args.json}")


if __name__ == "__main__":
    main()
