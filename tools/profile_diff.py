#!/usr/bin/env python3
"""对比两个 profile JSON，输出 op 级性能差异。

快速定位优化/回退来源：哪些 op 变快了、哪些变慢了、按多少。

用法：
    python tools/profile_diff.py baseline.json optimized.json
    python tools/profile_diff.py --threshold 2 baseline.json optimized.json
    python tools/profile_diff.py --json diff.json baseline.json optimized.json
    python tools/profile_diff.py --category baseline.json optimized.json

核心功能：
    1. 加载两个 profile JSON 文件中的 op_totals（每个 op 的累计耗时）。
    2. 逐 op 计算差异（delta_ms、delta_pct）。
    3. 按变化幅度排序，分别列出 Top 改善和 Top 回退。
    4. 可选按分类（matvec/norm/attention/activation/other）汇总。
    5. 可选输出 JSON 格式的 diff 报告。

输入：两个 profile JSON 文件（由 C++ runtime --profile 选项生成）。
输出：控制台打印差异表格，可选写出 JSON diff 文件。
"""

from __future__ import annotations

# --- 标准库导入 ---
import argparse   # 命令行参数解析
import json       # JSON 读写
import sys        # 错误退出
from pathlib import Path  # 路径操作

# op 分类规则：按名字前缀/后缀归类
# 用于将细粒度的 op 名聚合到高层类别，方便宏观分析性能瓶颈
CATEGORIES = {
    "matvec": ["q_proj", "k_proj", "v_proj", "o_proj",     # 矩阵-向量乘法（线性投影）
               "gate_proj", "up_proj", "down_proj",         # MLP 投影
               "lm_head"],                                   # 语言模型头
    "norm": ["layernorm", "rmsnorm"],                        # 归一化操作
    "attention": ["attention", "rope", "kv_append"],          # 注意力相关（含 RoPE、KV cache）
    "activation": ["swiglu", "silu"],                         # 激活函数
    "other": ["embed", "topk_argmax", "residual"],            # 其他（embedding、采样、残差）
}


def categorize_op(op_name: str) -> str:
    """根据 op 名字判断其所属的功能分类。

    通过检查 op 名字中是否包含各分类的关键词来归类。
    先去掉 "layer_N." 前缀再匹配，避免层号干扰。

    Args:
        op_name: op 名字（如 "layer_0.self_attn.q_proj"）。

    Returns:
        分类字符串（"matvec"/"norm"/"attention"/"activation"/"other"）。
    """
    lower = op_name.lower()  # 统一小写便于匹配
    # 去掉 "layer_N." 前缀，只保留后半部分用于关键词匹配
    if "." in lower:
        suffix = lower.split(".", 1)[1]  # 取第一个点之后的部分
    else:
        suffix = lower  # 无点则用全名
    # 遍历所有分类的关键词列表
    for cat, keywords in CATEGORIES.items():
        for kw in keywords:
            if kw in suffix:  # 如果名字包含该关键词
                return cat    # 返回对应分类
    return "other"  # 未匹配任何关键词，归入 other


def load_op_totals(profile_path: str) -> dict[str, float]:
    """从 profile JSON 的 op_totals 提取 {op_name: total_ms}。

    profile JSON 的结构为 {"op_totals": {"op_name": {"total_ms": ..., ...}, ...}}，
    本函数只提取 op 名和对应的累计毫秒数。

    Args:
        profile_path: profile JSON 文件路径。

    Returns:
        {op_name: total_ms} 字典。
    """
    with open(profile_path) as f:
        data = json.load(f)  # 解析 JSON
    op_totals = data.get("op_totals", {})  # 获取 op_totals 子字典
    # 提取每个 op 的 total_ms 值
    return {name: info["total_ms"] for name, info in op_totals.items()}


def compute_diff(before: dict[str, float], after: dict[str, float]) -> list[dict]:
    """计算每个 op 的性能差异。返回按 delta_ms 排序的列表。

    对 before 和 after 中出现的所有 op 做并集遍历，计算：
    - delta_ms = after - before（负值表示变快，正值表示变慢）
    - delta_pct = (after - before) / before × 100%

    Args:
        before: 基线 {op_name: total_ms} 字典。
        after: 优化后 {op_name: total_ms} 字典。

    Returns:
        diff 字典列表，每个元素包含 op/before_ms/after_ms/delta_ms/delta_pct/category。
    """
    all_ops = set(before) | set(after)  # 取两个集合的并集（覆盖新增/删除的 op）
    diffs = []  # 存放差异结果
    for op in sorted(all_ops):  # 按 op 名字排序遍历
        ms_before = before.get(op, 0.0)  # 基线耗时（缺失则为 0）
        ms_after = after.get(op, 0.0)    # 优化后耗时（缺失则为 0）
        delta_ms = ms_after - ms_before  # 绝对差异（ms）
        # 计算百分比变化；before=0 时特殊处理避免除零
        delta_pct = ((ms_after - ms_before) / ms_before * 100) if ms_before > 0 else (
            float("inf") if ms_after > 0 else 0.0)  # 新增 op 为 inf，都为零则为 0
        diffs.append({
            "op": op,                              # op 名字
            "before_ms": round(ms_before, 3),      # 基线耗时（ms）
            "after_ms": round(ms_after, 3),        # 优化后耗时（ms）
            "delta_ms": round(delta_ms, 3),        # 绝对差异（ms）
            "delta_pct": round(delta_pct, 1),      # 百分比变化（%）
            "category": categorize_op(op),          # 功能分类
        })
    return diffs


def print_diff(diffs: list[dict], threshold_pct: float, show_category: bool) -> None:
    """打印性能差异报告。

    包括：
    1. 总延迟对比和加速比。
    2. Top 改善的 op（变快超过阈值的）。
    3. Top 回退的 op（变慢超过阈值的）。
    4. 可选的按分类汇总表。

    Args:
        diffs: compute_diff 返回的差异列表。
        threshold_pct: 显示阈值（%），变化低于此值的 op 不单独列出。
        show_category: 是否显示按分类的汇总。
    """
    # 计算总延迟
    total_before = sum(d["before_ms"] for d in diffs)  # 基线总耗时
    total_after = sum(d["after_ms"] for d in diffs)    # 优化后总耗时
    speedup = total_before / total_after if total_after > 0 else float("inf")  # 加速比

    print(f"Profile Diff")
    print(f"  总延迟: {total_before:.2f} ms → {total_after:.2f} ms ({speedup:.2f}×)")
    print()

    # 筛选出显著改善的 op（delta_pct < -threshold，即变快超过阈值）
    improved = sorted([d for d in diffs if d["delta_pct"] < -threshold_pct],
                      key=lambda d: d["delta_ms"])  # 按 delta_ms 升序（最大改善在前）
    # 筛选出显著回退的 op（delta_pct > threshold，即变慢超过阈值）
    regressed = sorted([d for d in diffs if d["delta_pct"] > threshold_pct],
                       key=lambda d: -d["delta_ms"])  # 按 delta_ms 降序（最大回退在前）

    if improved:
        print("  Top 改善（变快）:")
        for d in improved[:10]:  # 最多显示 10 个
            print(f"    {d['op']:<36} {d['before_ms']:>8.2f} → {d['after_ms']:>8.2f} ms  "
                  f"({d['delta_pct']:+.1f}%, {d['delta_ms']:+.2f} ms)")
    else:
        print(f"  无显著改善（阈值 {threshold_pct}%）")

    print()
    if regressed:
        print("  Top 回退（变慢）:")
        for d in regressed[:10]:  # 最多显示 10 个
            print(f"    {d['op']:<36} {d['before_ms']:>8.2f} → {d['after_ms']:>8.2f} ms  "
                  f"({d['delta_pct']:+.1f}%, {d['delta_ms']:+.2f} ms)")
    else:
        print("  无显著回退")

    # 可选：按分类汇总
    if show_category:
        print()
        print("  按分类汇总:")
        cat_before: dict[str, float] = {}  # 各分类的基线总耗时
        cat_after: dict[str, float] = {}   # 各分类的优化后总耗时
        for d in diffs:
            cat = d["category"]  # 获取该 op 的分类
            cat_before[cat] = cat_before.get(cat, 0) + d["before_ms"]  # 累加基线耗时
            cat_after[cat] = cat_after.get(cat, 0) + d["after_ms"]    # 累加优化后耗时
        # 按固定顺序打印各分类
        for cat in ["matvec", "attention", "norm", "activation", "other"]:
            b = cat_before.get(cat, 0)  # 该分类基线耗时
            a = cat_after.get(cat, 0)   # 该分类优化后耗时
            if b == 0 and a == 0:
                continue  # 两侧都为零则跳过
            pct = ((a - b) / b * 100) if b > 0 else 0  # 百分比变化
            print(f"    {cat:<12} {b:>8.2f} → {a:>8.2f} ms  ({pct:+.1f}%)")


def main() -> None:
    """主入口函数：执行两个 profile JSON 的对比分析。

    流程：
    1. 解析命令行参数。
    2. 校验输入文件存在性。
    3. 加载两个 profile 的 op_totals。
    4. 计算逐 op 差异。
    5. 打印差异报告。
    6. 可选写出 JSON diff 文件。
    """
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("before", help="基线 profile JSON")           # 基线文件
    p.add_argument("after", help="优化后 profile JSON")           # 优化后文件
    p.add_argument("--threshold", type=float, default=1.0,
                   help="只显示变化超过 N%% 的 op（默认 1）")     # 显示阈值
    p.add_argument("--json", default=None, help="输出 JSON 文件")  # JSON 输出路径
    p.add_argument("--category", action="store_true", default=True,
                   help="显示按分类汇总（默认开启）")             # 启用分类汇总
    p.add_argument("--no-category", action="store_true",
                   help="不显示按分类汇总")                       # 禁用分类汇总
    args = p.parse_args()

    # 校验输入文件存在
    if not Path(args.before).exists():
        sys.exit(f"error: file not found: {args.before}")
    if not Path(args.after).exists():
        sys.exit(f"error: file not found: {args.after}")

    # 加载两个 profile 的 op_totals
    before = load_op_totals(args.before)
    after = load_op_totals(args.after)

    # 校验数据有效性
    if not before:
        sys.exit(f"error: no op_totals in {args.before}")
    if not after:
        sys.exit(f"error: no op_totals in {args.after}")

    # 计算差异并打印报告
    diffs = compute_diff(before, after)
    print_diff(diffs, args.threshold, args.category and not args.no_category)

    # 可选：写出 JSON diff 文件
    if args.json:
        total_before = sum(d["before_ms"] for d in diffs)  # 基线总耗时
        total_after = sum(d["after_ms"] for d in diffs)    # 优化后总耗时
        output = {
            "total_before_ms": round(total_before, 2),      # 基线总耗时
            "total_after_ms": round(total_after, 2),        # 优化后总耗时
            "speedup": round(total_before / total_after, 3) if total_after > 0 else None,  # 加速比
            "ops": diffs,                                    # 逐 op 差异列表
        }
        Path(args.json).write_text(json.dumps(output, ensure_ascii=False, indent=2))  # 写出 JSON
        print(f"\n  已写入 {args.json}")


if __name__ == "__main__":
    main()  # 脚本入口点
