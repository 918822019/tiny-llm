#!/usr/bin/env python3
"""分析 MoE decode 的专家选择频率分布，为"按热度常驻"缓存策略提供判据。

背景（为什么要这个脚本）：
    SSD 卸载模式下 decode 已达 NVMe 带宽下限（AGENTS.md 坑 #29）：每 token 读
    ~1.68 GB、有效带宽 5.46 GB/s（峰值 84%），理论下限 259 ms/tok。**要再快只能
    读更少数据**，不是继续优化计算或调度。两条路：更激进量化，或按热度常驻
    高频专家。后者是否值得做，完全取决于专家选择分布的偏斜程度——如果接近
    均匀，常驻任何子集都省不了多少 I/O；如果高度偏斜，少量常驻专家就能覆盖
    大部分访问。

    本脚本不做任何猜测，直接从真实 decode 的选择序列算累积覆盖曲线。

数据来源：
    runtime/qwen_forward_token.cpp 的 TINYQWEN_DUMP_MOE 插桩，每层每 token 追加
    一条定长记录：

        normed_[hidden]        float32
        moe_ffn_acc_[hidden]   float32
        moe_gate_logits_[n_experts]  float32
        moe_topk_idx_[k]       int32
        moe_topk_w_[k]         float32

    本脚本只读 topk_idx 段，其余跳过。

用法：
    TINYQWEN_DUMP_MOE=/tmp/moe_dump.bin ./build/runtime/tinyqwen \
        --model model_qwen3_30b_moe_i4_fp16.tqwen --tokens ... \
        --max-new-tokens 64 --moe-ssd --moe-expert-cache-slots 4
    .venv/bin/python tools/analyze_moe_expert_freq.py /tmp/moe_dump.bin \
        --hidden 2048 --n-experts 128 --topk 8 --n-layers 48 --skip-prefill 3
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dump", help="TINYQWEN_DUMP_MOE 产出的二进制文件")
    p.add_argument("--hidden", type=int, required=True)
    p.add_argument("--n-experts", type=int, required=True)
    p.add_argument("--topk", type=int, required=True)
    p.add_argument("--n-layers", type=int, required=True)
    # prefill 也走逐 token 路径（MoE 批量 prefill 对混合架构/小 n 回退），
    # 其选择分布与 decode 不同且样本极少，默认排除。
    p.add_argument("--skip-prefill", type=int, default=0,
                   help="跳过前 N 个 token 的记录（= prompt 长度）")
    p.add_argument("--per-expert-mb", type=float, default=0.0,
                   help="单专家字节数（MB），用于把覆盖率换算成 I/O 节省与常驻内存")
    p.add_argument("--tokens-per-sec", type=float, default=0.0,
                   help="当前 decode ms/tok，用于估算端到端加速")
    p.add_argument("--io-share-pct", type=float, default=0.0,
                   help="expert I/O 占 token 时间的百分比，用于估算加速上限")
    return p.parse_args()


def main() -> int:
    a = parse_args()
    rec_floats = 2 * a.hidden + a.n_experts + 2 * a.topk
    rec_bytes = rec_floats * 4
    idx_off_bytes = (2 * a.hidden + a.n_experts) * 4

    with open(a.dump, "rb") as f:
        buf = f.read()
    n_records = len(buf) // rec_bytes
    if len(buf) % rec_bytes != 0:
        print(f"warn: 文件大小 {len(buf)} 不是记录大小 {rec_bytes} 的整数倍，"
              f"末尾 {len(buf) % rec_bytes} 字节被丢弃", file=sys.stderr)
    n_tokens = n_records // a.n_layers
    if n_tokens * a.n_layers != n_records:
        print(f"error: 记录数 {n_records} 不是 n_layers={a.n_layers} 的整数倍——"
              f"检查 --hidden/--n-experts/--topk/--n-layers 是否与模型一致",
              file=sys.stderr)
        return 1

    print(f"[parse] {n_records} 条记录 = {n_tokens} token × {a.n_layers} 层"
          f"（记录 {rec_bytes} B，topk_idx 偏移 {idx_off_bytes} B）")
    if a.skip_prefill:
        print(f"[parse] 跳过前 {a.skip_prefill} 个 token（prefill），"
              f"分析后 {n_tokens - a.skip_prefill} 个 decode token")

    # 每层一个 Counter：expert_id → 被选中次数
    per_layer: list[Counter[int]] = [Counter() for _ in range(a.n_layers)]
    for t in range(a.skip_prefill, n_tokens):
        for l in range(a.n_layers):
            off = (t * a.n_layers + l) * rec_bytes + idx_off_bytes
            idx = np.frombuffer(buf, dtype=np.int32, count=a.topk, offset=off)
            per_layer[l].update(int(x) for x in idx)

    n_decode = n_tokens - a.skip_prefill
    total_loads = n_decode * a.n_layers * a.topk
    print(f"[parse] decode 专家访问总数 = {total_loads}"
          f"（{n_decode} token × {a.n_layers} 层 × top-{a.topk}）")

    # ---- 偏斜度：每层被激活过的专家数 / 分布集中度 ----
    print("\n=== 每层专家激活情况（偏斜程度）===")
    print(f"{'层':>4} {'激活专家数':>10} {'/%d' % a.n_experts:>7} "
          f"{'top1 占比':>10} {'top8 累积':>10}")
    activated = []
    for l in range(a.n_layers):
        c = per_layer[l]
        act = len(c)
        activated.append(act)
        ordered = sorted(c.values(), reverse=True)
        tot = sum(ordered)
        top1 = ordered[0] / tot if ordered else 0.0
        top8 = sum(ordered[:8]) / tot if ordered else 0.0
        print(f"{l:>4} {act:>10} {act / a.n_experts:>6.0%} "
              f"{top1:>9.1%} {top8:>9.1%}")
    print(f"\n平均每层激活 {np.mean(activated):.1f} / {a.n_experts} 个专家"
          f"（{np.mean(activated) / a.n_experts:.0%}）")
    print(f"（均匀分布下 {n_decode} token × top-{a.topk} 最多激活 "
          f"{min(a.n_experts, n_decode * a.topk)} 个；激活数远小于它 = 高度偏斜）")

    # ---- 累积覆盖曲线：常驻 top-M 专家能覆盖多少访问 ----
    print("\n=== 累积覆盖曲线（按热度常驻的判据）===")
    print(f"{'常驻/层':>8} {'常驻总数':>9} {'覆盖率':>8} ", end="")
    if a.per_expert_mb > 0:
        print(f"{'常驻内存':>10} {'省 I/O':>9} {'省 ms/tok':>10}", end="")
    print()

    # 每层各自按热度排序，取前 M 个（各层热点不同，不能全局排序）
    per_layer_sorted = [sorted(c.values(), reverse=True) for c in per_layer]
    for m in (1, 2, 4, 8, 16, 24, 32, 48, 64):
        if m > a.n_experts:
            break
        covered = sum(sum(s[:m]) for s in per_layer_sorted)
        cov = covered / total_loads
        line = f"{m:>8} {m * a.n_layers:>9} {cov:>7.1%} "
        if a.per_expert_mb > 0:
            mem_mb = m * a.n_layers * a.per_expert_mb
            saved_io_mb = cov * a.per_expert_mb * total_loads / n_decode
            line += f"{mem_mb:>9.0f}M {saved_io_mb:>8.0f}M "
            if a.tokens_per_sec > 0 and a.io_share_pct > 0:
                # 省下的 I/O 占原 I/O 的比例 × 原 I/O 占比 = 端到端可省时间比例
                io_frac = a.io_share_pct / 100.0
                saving = a.tokens_per_sec * io_frac * cov
                line += f"{saving:>9.1f}ms"
        print(line)

    print("\n判据：覆盖率随 M 增长越快，说明分布越偏斜、常驻收益越大。")
    print("      但注意 AGENTS.md 坑 #20——常驻工作集过大会让 expert_ffn 计算")
    print("      本身变慢（实测 slots=4096 比 slots=4 慢 1.8×），收益不是单调的。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
