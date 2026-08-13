#!/usr/bin/env python3
"""pipeline 第③④步：稳定测速 + 自动写入优化日志。

用法：
    python tools/record_optimization.py --label fp32-float [--runs 3]

做三件事：
    1. 把 canonical 负载跑 runs 遍，取"中位数的中位数"（复用 tools/bench.py）；
    2. 读 benchmarks/baseline.json，自动算 vs 基线加速比；
    3. 往 docs/optimization_log.md 追加：汇总表一行 + 详细小节
       （"归因 / 为什么"留 TODO，由人填——这是最需要人判断的部分）。
"""

from __future__ import annotations

import argparse
import datetime
import json
import shlex
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import bench  # noqa: E402  复用 run_once / summarize / env_info

TABLE_MARKER = "<!-- 新的优化按时间顺序往上表追加行"
DETAIL_MARKER = "<!-- 模板：复制下面这段"


def measure(binary: str, model: str, runs: int,
            extra_args: list[str] | None = None) -> tuple[float, float, dict]:
    medians, p95s, last = [], [], None
    for r in range(runs):
        prof = bench.run_once(binary, model, extra_args)
        st = bench.summarize(prof)
        medians.append(st["decode_median_ms"])
        p95s.append(st["decode_p95_ms"])
        last = st
        print(f"  run {r + 1}/{runs}: median={st['decode_median_ms']:.2f}ms "
              f"p95={st['decode_p95_ms']:.2f}ms")
    return statistics.median(medians), statistics.median(p95s), last


def build_table_row(label, commit, med, p95, vs_base_str, note) -> str:
    return (f"| {label} | {commit} | {med:.2f} | {p95:.2f} | "
            f"{vs_base_str} | — | {note} |")


def build_detail(label, commit, med, p95, runs, last, vs_prev_str, base_note,
                 extra_suffix="") -> str:
    today = datetime.date.today().isoformat()
    top_ops = "、".join(f"`{o['op']}`" for o in last["top_ops"][:3])
    return f"""### {label}（{today}）

- **优化栈**：<基线 + 本次优化，如 fp32-baseline + XXX>
- **是什么**：<本次改了哪个 kernel / 数据结构 / 调度，一两句话>
- **假设**：<为什么预期会快：带宽 / 计算 / 并行 / 指令 哪一类>
- **结果**：decode 中位 **{med:.2f} ms/token**（{runs} 遍取中位，每遍 {last['decode_samples']} 样本），p95 {p95:.2f}
- **vs 上一配置**：{vs_prev_str}
- **基线参照**：{base_note}
- **验证**：scripts/verify.sh（31 单测 + golden token 对照）
- **瓶颈转移**：top op = {top_ops}，下一刀砍哪：<填>
- **意外 / 教训**：<填——往往最值钱>
- **复现**：`./scripts/bench.sh {label}{extra_suffix}`

---

"""


def insert_before(text: str, marker: str, block: str) -> str:
    idx = text.find(marker)
    if idx == -1:
        raise RuntimeError(f"log 里找不到标记：{marker!r}，无法定位插入点")
    return text[:idx] + block + text[idx:]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True, help="本次优化的名字")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--binary", default="build/runtime/tinyqwen")
    ap.add_argument("--model", default="model.tqwen")
    ap.add_argument("--baseline", default="benchmarks/baseline.json")
    ap.add_argument("--log", default="docs/optimization_log.md")
    ap.add_argument("--extra-args", default="",
                    help="原样传给 runtime 的额外 CLI 参数（引号括起），"
                         "如 '--matvec-impl double_2_float'")
    args = ap.parse_args()

    extra = shlex.split(args.extra_args)
    extra_suffix = (" " + args.extra_args.strip()) if extra else ""

    print(f"[record] label={args.label} runs={args.runs}"
          + (f"  额外参数: {' '.join(extra)}" if extra else ""))
    med, p95, last = measure(args.binary, args.model, args.runs, extra)
    env = bench.env_info()
    commit = env["git_commit"]

    # 读基线算加速比
    base_path = Path(args.baseline)
    if base_path.exists():
        base = json.loads(base_path.read_text())
        vs_base = base["decode_median_ms"] / med
        vs_base_str = f"{vs_base:.2f}×"
        base_note = (f"{base['label']}（{base['decode_median_ms']:.2f} ms/tok @ {base.get('commit', '?')}），"
                     f"本次 vs 基线 = {vs_base:.2f}×")
        vs_prev_str = f"<填：vs 上一配置>（vs 基线 {vs_base:.2f}×）"
    else:
        vs_base_str = "基线"
        base_note = f"未找到 {args.baseline}，本次作为基线；请先跑 scripts/set_baseline.sh"
        vs_prev_str = "—"

    # 写日志：表格行 + 详细小节
    log_path = Path(args.log)
    text = log_path.read_text()
    row = build_table_row(args.label, commit, med, p95, vs_base_str,
                          "<填：一句话归因>") + "\n"
    detail = build_detail(args.label, commit, med, p95, args.runs, last,
                          vs_prev_str, base_note, extra_suffix)
    text = insert_before(text, TABLE_MARKER, row)
    text = insert_before(text, DETAIL_MARKER, detail)
    log_path.write_text(text)

    print(f"\n[record] 已写入 {args.log}")
    print(f"  decode 中位 {med:.2f} ms/token（p95 {p95:.2f}）@ {commit}")
    print(f"  {base_note}")
    print("  ⚠️ 请手动补全日志里的 <填...>：优化栈/是什么/假设/归因/教训")


if __name__ == "__main__":
    main()
