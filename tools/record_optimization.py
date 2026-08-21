#!/usr/bin/env python3
"""pipeline 第③④步：稳定测速 + 自动写入优化日志。

本脚本是本地（macOS/Linux）优化记录工具，与 record_android.py 对等。
做四件事：
    1. 把 canonical 负载跑 runs 遍，取"中位数的中位数"（复用 tools/bench.py）；
       带 --extra-args 时自动升级为**同场 A/B**：每遍先测对照再测变体，
       交错进行抗慢漂移——vs 历史基线会被机器漂移掩盖，同场 A/B 才是真
       贡献（见 docs/optimization_log.md 的 fp32-float / double_2_float 教训）。
       对照默认 = 同模型文件、无额外参数（即 ref）；跨配置对照（如 f16
       模型 vs fp32 最佳栈）用 --control-model / --control-args 显式指定；
    2. 读 benchmarks/baseline.json，自动算 vs 基线加速比（参考用）；
    3. 参照中位偏离基线 >5% 时打印基线漂移警告；
    4. 往 docs/optimization_log.md 追加：汇总表一行 + 详细小节
       （"归因 / 为什么"留 TODO，由人填——这是最需要人判断的部分）。

用法：
    # 基本记录
    python tools/record_optimization.py --label fp32-float [--runs 3]
    # 带额外参数（自动进入同场 A/B 模式）
    python tools/record_optimization.py --label neon-test --extra-args "--matvec-impl neon_mt_kv_nt"
    # 跨配置 A/B
    python tools/record_optimization.py --label f16-vs-fp32 \\
        --model model_f16.tqwen --control-model model.tqwen

输入：
    - build/runtime/tinyqwen（本地编译的 runtime）
    - .tqwen 模型文件
    - benchmarks/baseline.json（基线文件，可选）

输出：
    - docs/optimization_log.md 追加表格行和详细小节
    - 终端打印结果摘要
"""

# 启用延迟注解求值
from __future__ import annotations

# ---- 标准库导入 ----
import argparse       # 命令行参数解析
import datetime       # 日期格式化（详细小节标题）
import json           # JSON 读取（基线文件）
import shlex          # shell 字符串安全分割
import statistics     # 统计函数（median）
import sys            # 系统退出
from pathlib import Path  # 路径操作

# 把 tools/ 目录加入搜索路径，以便导入 bench 模块
sys.path.insert(0, str(Path(__file__).parent))
import bench  # noqa: E402  # 复用 run_once / summarize / env_info

# ---- Markdown 标记常量 ----
# 汇总表中新增行的插入位置标记（新行插在标记之前）
TABLE_MARKER = "<!-- 新的优化按时间顺序往上表追加行"
# 详细小节的插入位置标记（新内容插在标记之前）
DETAIL_MARKER = "<!-- 模板：复制下面这段"


def measure(binary: str, model: str, runs: int,
            extra_args: list[str] | None = None) -> tuple[float, float, dict]:
    """执行多遍标准测速，返回汇总统计。

    Args:
        binary: tinyqwen 可执行文件路径
        model: 模型文件路径
        runs: 测量遍数
        extra_args: 额外 CLI 参数列表

    Returns:
        tuple: (decode 中位数, p95, 最后一遍 summarize 字典)
    """
    medians, p95s, last = [], [], None
    for r in range(runs):
        # 跑一遍标准负载
        prof = bench.run_once(binary, model, extra_args)
        # 提取统计量
        st = bench.summarize(prof)
        medians.append(st["decode_median_ms"])
        p95s.append(st["decode_p95_ms"])
        last = st  # 保留最后一遍的完整统计
        print(f"  run {r + 1}/{runs}: median={st['decode_median_ms']:.2f}ms "
              f"p95={st['decode_p95_ms']:.2f}ms")
    # 取各遍中位数的中位数和各遍 P95 的中位数
    return statistics.median(medians), statistics.median(p95s), last


def measure_ab(binary: str, model: str, runs: int, variant_args: list[str],
               control_model: str | None = None,
               control_args: list[str] | None = None) -> tuple[float, float, float, float, dict]:
    """同场 A/B 测试：每遍先测对照再测变体，交错进行抗慢漂移。

    返回 (对照中位, 对照p95, 变体中位, 变体p95, 最后一遍变体统计)。
    对照语义 = "同 binary + control_model + control_args"；两者缺省时
    退化为旧语义（同模型文件、无额外参数，即默认 ref 实现）。
    跨配置对照（如 fp32 最佳栈 vs f16 模型）用显式参数指定。

    Args:
        binary: tinyqwen 可执行文件路径
        model: 变体模型文件路径
        runs: 测量遍数
        variant_args: 变体的额外 CLI 参数
        control_model: 对照模型文件路径（None = 与变体相同）
        control_args: 对照的额外 CLI 参数（None = 无）

    Returns:
        tuple: (对照中位, 对照p95, 变体中位, 变体p95, 变体最后一遍 summarize)
    """
    ctrl_model = control_model or model   # 对照模型缺省 = 变体模型
    ctrl_args = control_args or []        # 对照参数缺省 = 空
    ctrl_meds, ctrl_p95s, var_meds, var_p95s = [], [], [], []
    last_var = None
    for r in range(runs):
        # 每遍先测对照再测变体（交错抗漂移）
        cs = bench.summarize(bench.run_once(binary, ctrl_model, ctrl_args))
        vs = bench.summarize(bench.run_once(binary, model, variant_args))
        ctrl_meds.append(cs["decode_median_ms"])
        ctrl_p95s.append(cs["decode_p95_ms"])
        var_meds.append(vs["decode_median_ms"])
        var_p95s.append(vs["decode_p95_ms"])
        last_var = vs  # 保留最后一遍变体统计
        print(f"  run {r + 1}/{runs}: 对照={cs['decode_median_ms']:.2f}ms  "
              f"变体={vs['decode_median_ms']:.2f}ms")
    return (statistics.median(ctrl_meds), statistics.median(ctrl_p95s),
            statistics.median(var_meds), statistics.median(var_p95s), last_var)


def build_table_row(label, commit, med, p95, vs_base_str, note,
                    vs_prev_col="—") -> str:
    """构建 optimization_log.md 汇总表的 Markdown 表格行。

    Args:
        label: 配置标签名
        commit: git commit hash
        med: decode 中位延迟（ms）
        p95: decode P95 延迟（ms）
        vs_base_str: vs 基线加速比字符串
        note: 归因说明
        vs_prev_col: vs 上一配置列的内容

    Returns:
        str: Markdown 表格行字符串
    """
    return (f"| {label} | {commit} | {med:.2f} | {p95:.2f} | "
            f"{vs_base_str} | {vs_prev_col} | {note} |")


def build_detail(label, commit, med, p95, runs, last, vs_prev_str, base_note,
                 extra_suffix="", model_prefix="") -> str:
    """构建优化记录的详细小节 Markdown 文本。

    其中 <填...> 占位符由人工后续补全（优化栈/假设/归因/教训等）。

    Args:
        label: 配置标签名
        commit: git commit hash
        med: decode 中位延迟（ms）
        p95: decode P95 延迟（ms）
        runs: 测量遍数
        last: 最后一遍的 summarize 字典
        vs_prev_str: vs 上一配置的对比描述
        base_note: 基线参照说明
        extra_suffix: 复现命令中的额外参数后缀
        model_prefix: 复现命令中的 MODEL 环境变量前缀

    Returns:
        str: Markdown 格式的详细小节文本
    """
    today = datetime.date.today().isoformat()  # 今日日期
    # 耗时 top 3 算子
    top_ops = "、".join(f"`{o['op']}`" for o in last["top_ops"][:3])
    return f"""### {label}（{today}）

- **优化栈**：<基线 + 本次优化，如 fp32-baseline + XXX>
- **是什么**：<本次改了哪个 kernel / 数据结构 / 调度，一两句话>
- **假设**：<为什么预期会快：带宽 / 计算 / 并行 / 指令 哪一类>
- **结果**：TTFT **{last['ttft_ms']:.2f} ms**（prefill {last['prefill_tokens']} tok）；TOPT 中位 **{med:.2f} ms/token**（{runs} 遍取中位；decode 共 {last['generated_tokens']} tok，丢预热，稳态样本 {last['decode_samples']}），p95 {p95:.2f}；forward 总耗时 {last.get('total_ms') or 0:.1f} ms
- **vs 上一配置**：{vs_prev_str}
- **基线参照**：{base_note}
- **验证**：scripts/verify.sh（单测 + golden token 对照）
- **瓶颈转移**：top op = {top_ops}，下一刀砍哪：<填>
- **意外 / 教训**：<填——往往最值钱>
- **复现**：`{model_prefix}./scripts/bench.sh {label}{extra_suffix}`

---

"""


def insert_before(text: str, marker: str, block: str) -> str:
    """在文本中找到标记字符串，将新内容块插入到标记之前。

    Args:
        text: 原始文本
        marker: 定位标记字符串
        block: 要插入的内容块

    Returns:
        str: 插入后的文本

    Raises:
        RuntimeError: 找不到标记时抛出
    """
    idx = text.find(marker)
    if idx == -1:
        raise RuntimeError(f"log 里找不到标记：{marker!r}，无法定位插入点")
    return text[:idx] + block + text[idx:]


def insert_table_row(text: str, marker: str, row: str) -> str:
    """表格行专用插入：倒退掉标记前的空行，让新行紧贴表格末行。

    Markdown 表格中间出现空行就会断表（后半截渲染成无表头文本）——
    模板里标记注释前恰好带空行，普通 insert_before 会把行插到空行后面，
    第一次插入就把表劈成两半（实际踩过）。

    Args:
        text: 原始日志文本
        marker: 表格标记字符串
        row: 要插入的表格行

    Returns:
        str: 插入后的文本

    Raises:
        RuntimeError: 找不到标记时抛出
    """
    idx = text.find(marker)
    if idx == -1:
        raise RuntimeError(f"log 里找不到标记：{marker!r}，无法定位插入点")
    # rstrip("\n") 去掉标记前的空行，确保新行紧贴表格
    prefix = text[:idx].rstrip("\n")
    return prefix + "\n" + row + "\n" + text[idx:]


def main() -> None:
    """record_optimization.py 的主入口函数。

    解析参数 → 执行测速（普通或 A/B）→ 读基线 → 写日志。
    """
    # 创建参数解析器
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True, help="本次优化的名字")
    ap.add_argument("--runs", type=int, default=3)                # 测量遍数
    ap.add_argument("--binary", default="build/runtime/tinyqwen") # binary 路径
    ap.add_argument("--model", default="model.tqwen")             # 模型文件
    ap.add_argument("--baseline", default="benchmarks/baseline.json")  # 基线文件
    ap.add_argument("--log", default="docs/optimization_log.md")  # 日志文件
    ap.add_argument("--extra-args", default="",
                    help="原样传给 runtime 的额外 CLI 参数（引号括起），"
                         "如 '--matvec-impl double_2_float'")
    ap.add_argument("--control-model", default=None,
                    help="A/B 对照的模型文件（缺省 = 与 --model 相同）。"
                         "跨文件对照用，如 f16 变体 vs fp32 最佳栈")
    ap.add_argument("--control-args", default="",
                    help="A/B 对照的额外参数（引号括起；缺省 = 无，即默认 ref）")
    args = ap.parse_args()

    # 解析额外参数
    extra = shlex.split(args.extra_args)
    control_extra = shlex.split(args.control_args)
    control_model = args.control_model or args.model
    # 日志"复现"行要能直接执行：额外参数走 bench.sh 的 --extra-args 通道；
    # 非默认模型文件走 bench.sh 认的 MODEL 环境变量。
    extra_suffix = f' --extra-args "{args.extra_args.strip()}"' if extra else ""
    model_prefix = "" if args.model == "model.tqwen" else f"MODEL={args.model} "

    # A/B 模式判断：变体带额外参数，或显式指定了对照配置
    ab_mode = bool(extra) or bool(control_extra) or control_model != args.model
    print(f"[record] label={args.label} runs={args.runs}"
          + (f"  额外参数: {' '.join(extra)}（同场 A/B 模式）" if ab_mode else ""))

    # ---- 执行测速 ----
    if ab_mode:
        # A/B 模式
        ctrl_med, ctrl_p95, med, p95, last = measure_ab(
            args.binary, args.model, args.runs, extra, control_model, control_extra)
        ab_ratio = ctrl_med / med  # 加速比
        # 对照描述
        if control_model == args.model and not control_extra:
            ctrl_desc = "无额外参数，当前即 ref"
        else:
            ctrl_desc = f"{control_model}" + (
                f" + '{args.control_args.strip()}'" if control_extra else "")
        vs_prev_str = (f"**{ab_ratio:.2f}×（同场 A/B）**——对照（{ctrl_desc}）"
                       f"中位 {ctrl_med:.2f}（p95 {ctrl_p95:.2f}）→ 变体 {med:.2f}，"
                       f"同 binary 同场交错测量")
        vs_prev_col = f"{ab_ratio:.2f}×（同场）"
    else:
        # 普通模式
        med, p95, last = measure(args.binary, args.model, args.runs)
        vs_prev_str = None  # 下面与基线信息合并
        vs_prev_col = "—"

    # 采集环境信息
    env = bench.env_info()
    commit = env["git_commit"]

    # ---- 读基线算加速比 ----
    base_path = Path(args.baseline)
    if base_path.exists():
        base = json.loads(base_path.read_text())
        vs_base = base["decode_median_ms"] / med  # 加速比
        vs_base_str = f"{vs_base:.2f}×"
        base_note = (f"{base['label']}（{base['decode_median_ms']:.2f} ms/tok @ {base.get('commit', '?')}），"
                     f"本次 vs 基线 = {vs_base:.2f}×")
        if vs_prev_str is None:
            vs_prev_str = f"<填：vs 上一配置>（vs 基线 {vs_base:.2f}×）"
        else:
            vs_prev_str += f"。（vs 基线 {vs_base:.2f}×）"
        # 基线漂移警告
        default_control = control_model == args.model and not control_extra
        if ab_mode and not default_control:
            print("  [info] 跨配置对照，跳过基线漂移检查")
        else:
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

    # ---- 写日志 ----
    log_path = Path(args.log)
    text = log_path.read_text()  # 读取现有日志
    # 构建表格行
    row = build_table_row(args.label, commit, med, p95, vs_base_str,
                          "<填：一句话归因>", vs_prev_col)
    # 构建详细小节
    detail = build_detail(args.label, commit, med, p95, args.runs, last,
                          vs_prev_str, base_note, extra_suffix, model_prefix)
    text = insert_table_row(text, TABLE_MARKER, row)   # 表格行：防断表
    text = insert_before(text, DETAIL_MARKER, detail)   # 详细小节
    log_path.write_text(text)  # 写回

    # ---- 打印结果摘要 ----
    print(f"\n[record] 已写入 {args.log}")
    print(f"  decode 中位 {med:.2f} ms/token（p95 {p95:.2f}）@ {commit}")
    if ab_mode:
        print(f"  同场 A/B：对照 {ctrl_med:.2f} → 变体 {med:.2f} = {ab_ratio:.2f}×")
    print(f"  {base_note}")
    print("  ⚠️ 请手动补全日志里的 <填...>：优化栈/是什么/假设/归因/教训")


# 脚本直接运行入口
if __name__ == "__main__":
    main()
