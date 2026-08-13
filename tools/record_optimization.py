#!/usr/bin/env python3
"""pipeline 第③④步：稳定测速 + 自动写入优化日志。

用法：
    python tools/record_optimization.py --label fp32-float [--runs 3]

做四件事：
    1. 把 canonical 负载跑 runs 遍，取"中位数的中位数"（复用 tools/bench.py）；
       带 --extra-args 时自动升级为**同场 A/B**：每遍先测对照（同 binary、
       无额外参数，当前即 ref）再测变体，交错进行抗慢漂移——
       vs 历史基线会被机器漂移掩盖，同场 A/B 才是真贡献（见
       docs/optimization_log.md 的 fp32-float / double_2_float 教训）；
    2. 读 benchmarks/baseline.json，自动算 vs 基线加速比（参考用）；
    3. 参照中位偏离基线 >5% 时打印基线漂移警告；
    4. 往 docs/optimization_log.md 追加：汇总表一行 + 详细小节
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


def measure_ab(binary: str, model: str, runs: int,
               variant_args: list[str]) -> tuple[float, float, float, float, dict]:
    """同场 A/B：每遍先测对照（无额外参数）再测变体，交错进行抗慢漂移。

    返回 (对照中位, 对照p95, 变体中位, 变体p95, 最后一遍变体统计)。
    对照语义 = "同 binary、无额外参数"，当前即默认 ref 实现。
    """
    ctrl_meds, ctrl_p95s, var_meds, var_p95s = [], [], [], []
    last_var = None
    for r in range(runs):
        cs = bench.summarize(bench.run_once(binary, model))
        vs = bench.summarize(bench.run_once(binary, model, variant_args))
        ctrl_meds.append(cs["decode_median_ms"])
        ctrl_p95s.append(cs["decode_p95_ms"])
        var_meds.append(vs["decode_median_ms"])
        var_p95s.append(vs["decode_p95_ms"])
        last_var = vs
        print(f"  run {r + 1}/{runs}: 对照={cs['decode_median_ms']:.2f}ms  "
              f"变体={vs['decode_median_ms']:.2f}ms")
    return (statistics.median(ctrl_meds), statistics.median(ctrl_p95s),
            statistics.median(var_meds), statistics.median(var_p95s), last_var)


def build_table_row(label, commit, med, p95, vs_base_str, note,
                    vs_prev_col="—") -> str:
    return (f"| {label} | {commit} | {med:.2f} | {p95:.2f} | "
            f"{vs_base_str} | {vs_prev_col} | {note} |")


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
- **验证**：scripts/verify.sh（单测 + golden token 对照）
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
    # 日志"复现"行要能直接执行：额外参数走 bench.sh 的 --extra-args 通道。
    extra_suffix = f' --extra-args "{args.extra_args.strip()}"' if extra else ""

    ab_mode = bool(extra)
    print(f"[record] label={args.label} runs={args.runs}"
          + (f"  额外参数: {' '.join(extra)}（同场 A/B 模式）" if ab_mode else ""))

    if ab_mode:
        # A/B：对照 = 同 binary 无额外参数（当前即 ref）。
        ctrl_med, ctrl_p95, med, p95, last = measure_ab(
            args.binary, args.model, args.runs, extra)
        ab_ratio = ctrl_med / med
        vs_prev_str = (f"**{ab_ratio:.2f}×（同场 A/B）**——对照（无额外参数，当前即 ref）"
                       f"中位 {ctrl_med:.2f}（p95 {ctrl_p95:.2f}）→ 变体 {med:.2f}，"
                       f"同 binary 同场交错测量")
        vs_prev_col = f"{ab_ratio:.2f}×（同场）"
    else:
        med, p95, last = measure(args.binary, args.model, args.runs)
        vs_prev_str = None  # 下面与基线信息合并
        vs_prev_col = "—"

    env = bench.env_info()
    commit = env["git_commit"]

    # 读基线算加速比（vs 基线是"用户视角"参考；跨 commit 数字，会被机器漂移影响）
    base_path = Path(args.baseline)
    if base_path.exists():
        base = json.loads(base_path.read_text())
        vs_base = base["decode_median_ms"] / med
        vs_base_str = f"{vs_base:.2f}×"
        base_note = (f"{base['label']}（{base['decode_median_ms']:.2f} ms/tok @ {base.get('commit', '?')}），"
                     f"本次 vs 基线 = {vs_base:.2f}×")
        if vs_prev_str is None:
            vs_prev_str = f"<填：vs 上一配置>（vs 基线 {vs_base:.2f}×）"
        else:
            vs_prev_str += f"。（vs 基线 {vs_base:.2f}×）"
        # 基线漂移警告：用"参照中位"判断——A/B 模式用对照（它才是与基线
        # 同配置的量），否则用本次结果。>5% 说明机器状态或基线已过期。
        reference_med = ctrl_med if ab_mode else med
        drift = (reference_med - base["decode_median_ms"]) / base["decode_median_ms"]
        if abs(drift) > 0.05:
            print(f"  ⚠️ 基线漂移警告：参照中位 {reference_med:.2f} vs 基线 "
                  f"{base['decode_median_ms']:.2f}（偏离 {drift * 100:+.1f}%，>5%）"
                  f"——考虑重跑 scripts/set_baseline.sh 重建基线")
    else:
        vs_base_str = "基线"
        base_note = f"未找到 {args.baseline}，本次作为基线；请先跑 scripts/set_baseline.sh"
        if vs_prev_str is None:
            vs_prev_str = "—"

    # 写日志：表格行 + 详细小节
    log_path = Path(args.log)
    text = log_path.read_text()
    row = build_table_row(args.label, commit, med, p95, vs_base_str,
                          "<填：一句话归因>", vs_prev_col) + "\n"
    detail = build_detail(args.label, commit, med, p95, args.runs, last,
                          vs_prev_str, base_note, extra_suffix)
    text = insert_before(text, TABLE_MARKER, row)
    text = insert_before(text, DETAIL_MARKER, detail)
    log_path.write_text(text)

    print(f"\n[record] 已写入 {args.log}")
    print(f"  decode 中位 {med:.2f} ms/token（p95 {p95:.2f}）@ {commit}")
    if ab_mode:
        print(f"  同场 A/B：对照 {ctrl_med:.2f} → 变体 {med:.2f} = {ab_ratio:.2f}×")
    print(f"  {base_note}")
    print("  ⚠️ 请手动补全日志里的 <填...>：优化栈/是什么/假设/归因/教训")


if __name__ == "__main__":
    main()
