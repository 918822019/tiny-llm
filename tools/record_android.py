#!/usr/bin/env python3
"""Android 端侧优化记录：稳定测速 + 自动写入 optimization_log.md。

与 record_optimization.py 对等——区别仅在于测量跑在 Android 设备上。
日志写入同一个文件、用同样的标记和格式，便于全局对比。

用法：
    python tools/record_android.py --label neon-android --runs 3
    python tools/record_android.py --label neon-android --runs 3 --extra-args "--matvec-impl neon"
"""

from __future__ import annotations

import argparse
import datetime
import json
import shlex
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import bench  # noqa: E402
import bench_android  # noqa: E402
import record_optimization as rec  # noqa: E402


def build_detail_android(label, commit, med, p95, runs, last, vs_prev_str,
                         base_note, dev_env, extra_suffix="", model_prefix="",
                         verify_note="scripts/verify_android.sh（golden token 逐位对照）") -> str:
    today = datetime.date.today().isoformat()
    top_ops = "、".join(f"`{o['op']}`" for o in last["top_ops"][:3])
    device_str = f"{dev_env.get('device_model', '?')} / {dev_env.get('soc', '?')} / Android {dev_env.get('android_version', '?')}"
    return f"""### {label}（{today}，Android）

- **设备**：{device_str}
- **优化栈**：<基线 + 本次优化，如 android-baseline + XXX>
- **是什么**：<本次改了哪个 kernel / 数据结构 / 调度，一两句话>
- **假设**：<为什么预期会快：带宽 / 计算 / 并行 / 指令 哪一类>
- **结果**：decode 中位 **{med:.2f} ms/token**（{runs} 遍取中位，每遍 {last['decode_samples']} 样本），p95 {p95:.2f}
- **vs 上一配置**：{vs_prev_str}
- **基线参照**：{base_note}
- **验证**：{verify_note}
- **瓶颈转移**：top op = {top_ops}，下一刀砍哪：<填>
- **意外 / 教训**：<填——往往最值钱>
- **复现**：`{model_prefix}./scripts/bench_android.sh {label}{extra_suffix}`

---

"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--label", required=True, help="本次优化的名字")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--binary", default=None)
    ap.add_argument("--model", default="model.tqwen")
    ap.add_argument("--baseline", default="benchmarks/baseline_android.json")
    ap.add_argument("--log", default="docs/optimization_log.md")
    ap.add_argument("--extra-args", default="",
                    help="原样传给设备端 binary 的额外 CLI 参数")
    ap.add_argument("--control-model", default=None,
                    help="同场 A/B 的对照模型（默认与 --model 相同）。指定为不同文件即可"
                         "做跨 dtype 对比，例如 --model model_i4.tqwen --control-model model.tqwen")
    args = ap.parse_args()

    import os
    extra = shlex.split(args.extra_args)
    control_model = args.control_model or args.model
    cross_model = os.path.abspath(control_model) != os.path.abspath(args.model)
    ab_mode = bool(extra) or cross_model
    extra_suffix = f' --extra-args "{args.extra_args.strip()}"' if extra else ""
    if cross_model:
        extra_suffix += f" --control-model {control_model}"

    build_dir = os.environ.get("BUILD_DIR", bench_android.BUILD_DIR_DEFAULT)
    binary = args.binary or f"{build_dir}/runtime/tinyqwen"
    if not os.path.isfile(binary):
        sys.exit(f"error: binary not found: {binary}; run scripts/build_android.sh first")
    if not os.path.isfile(args.model):
        sys.exit(f"error: model not found: {args.model}")
    if not os.path.isfile(control_model):
        sys.exit(f"error: control model not found: {control_model}")

    print(f"[record-android] label={args.label} runs={args.runs}"
          + (f"  额外参数: {' '.join(extra)}（同场 A/B 模式）" if extra else "")
          + (f"  跨模型 A/B: {os.path.basename(control_model)} → "
             f"{os.path.basename(args.model)}" if cross_model else ""))

    print("[record-android] pushing binary & model...")
    bench_android.ensure_binary_pushed(binary)
    variant_on_device = bench_android.ensure_model_pushed(args.model)
    control_on_device = (bench_android.ensure_model_pushed(control_model) if cross_model
                         else variant_on_device)

    dev_env = bench_android.device_env_info()
    print(f"[record-android] device: {dev_env.get('device_model', '?')} "
          f"({dev_env.get('soc', '?')}) Android {dev_env.get('android_version', '?')}")

    if ab_mode:
        ctrl_med, ctrl_p95, med, p95, last = bench_android.measure_ab(
            args.runs, variant_on_device, extra, control_on_device)
        ab_ratio = ctrl_med / med if med > 0 else float("inf")
        control_desc = (f"对照模型 {os.path.basename(control_model)} " if cross_model
                        else "对照（无额外参数，当前即 ref）")
        vs_prev_str = (f"**{ab_ratio:.2f}×（同场 A/B）**——{control_desc}"
                       f"中位 {ctrl_med:.2f}（p95 {ctrl_p95:.2f}）→ 变体 {med:.2f}，"
                       f"同 binary 同场交错测量")
        vs_prev_col = f"{ab_ratio:.2f}×（同场）"
    else:
        med, p95, last = bench_android.measure(variant_on_device, args.runs,
                                               extra if extra else None)
        vs_prev_str = None
        vs_prev_col = "—"

    host_env = bench.env_info()
    commit = host_env["git_commit"]

    # 读 Android 基线算加速比
    base_path = Path(args.baseline)
    if base_path.exists():
        base = json.loads(base_path.read_text())
        vs_base = base["decode_median_ms"] / med
        vs_base_str = f"{vs_base:.2f}×"
        base_note = (f"{base['label']}（{base['decode_median_ms']:.2f} ms/tok @ "
                     f"{base.get('device_model', base.get('commit', '?'))}），"
                     f"本次 vs 基线 = {vs_base:.2f}×")
        if vs_prev_str is None:
            vs_prev_str = f"<填：vs 上一配置>（vs 基线 {vs_base:.2f}×）"
        else:
            vs_prev_str += f"。（vs 基线 {vs_base:.2f}×）"

        # 基线漂移警告
        if ab_mode:
            reference_med = ctrl_med
        else:
            reference_med = med
        drift = (reference_med - base["decode_median_ms"]) / base["decode_median_ms"]
        if abs(drift) > 0.05:
            print(f"  ⚠️ 基线漂移警告：参照中位 {reference_med:.2f} vs 基线 "
                  f"{base['decode_median_ms']:.2f}（偏离 {drift * 100:+.1f}%，>5%）"
                  f"——考虑重跑 scripts/set_baseline_android.sh 重建基线")
    else:
        vs_base_str = "基线"
        base_note = f"未找到 {args.baseline}，请先跑 scripts/set_baseline_android.sh"
        if vs_prev_str is None:
            vs_prev_str = "—"

    # 写日志
    log_path = Path(args.log)
    text = log_path.read_text()
    label_with_device = f"{args.label}（Android）"
    row = rec.build_table_row(label_with_device, commit, med, p95, vs_base_str,
                              "<填：一句话归因>", vs_prev_col)
    model_prefix = ("" if os.path.basename(args.model) == "model.tqwen"
                    else f"MODEL={args.model} ")
    verify_note = ("scripts/verify_android.sh（量化容差口径，token diff 上限 2）"
                   if "_i4" in os.path.basename(args.model)
                   else "scripts/verify_android.sh（golden token 逐位对照）")
    detail = build_detail_android(args.label, commit, med, p95, args.runs, last,
                                  vs_prev_str, base_note, dev_env, extra_suffix,
                                  model_prefix, verify_note)
    text = rec.insert_table_row(text, rec.TABLE_MARKER, row)
    text = rec.insert_before(text, rec.DETAIL_MARKER, detail)
    log_path.write_text(text)

    print(f"\n[record-android] 已写入 {args.log}")
    print(f"  decode 中位 {med:.2f} ms/token（p95 {p95:.2f}）@ {commit}")
    if ab_mode:
        print(f"  同场 A/B：对照 {ctrl_med:.2f} → 变体 {med:.2f} = {ab_ratio:.2f}×")
    print(f"  设备：{dev_env.get('device_model', '?')} / {dev_env.get('soc', '?')}")
    print(f"  {base_note}")
    print("  ⚠️ 请手动补全日志里的 <填...>：优化栈/是什么/假设/归因/教训")


if __name__ == "__main__":
    main()
