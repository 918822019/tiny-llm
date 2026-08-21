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
    # 基本用法：指定模型和标签跑一次标准测速
    python tools/bench.py --model model.tqwen --label fp32-baseline
    # 输出 JSON 结果文件
    python tools/bench.py --model model.tqwen --label int8-w8a8 --json out.json
    # 多遍测量取中位数的中位数，抗单次波动
    python tools/bench.py --model model.tqwen --label neon --runs 5
    # 传递额外参数给 runtime（如切换 matvec 实现）
    python tools/bench.py --model model.tqwen --label test --extra-args "--matvec-impl double_2_float"

输入：
    - --model：.tqwen 格式的模型文件路径
    - --binary：tinyqwen C++ runtime 可执行文件路径（默认 build/runtime/tinyqwen）
    - --label：本次测量的名字标签，用于区分不同配置
    - --extra-args：原样传给 binary 的额外 CLI 参数

输出：
    - 终端打印人类可读的性能汇总（TTFT、TOPT、top op 等）
    - 一行可直接粘进 docs/optimization_log.md 的 Markdown 表格行
    - 可选：--json 指定路径写入完整结果 JSON

关键设计：
    - 固定 prompt token id（不依赖 tokenizer），保证跨版本可比
    - 丢弃前 WARMUP 个 decode token（预热），避免冷启动抖动影响稳态统计
    - 取稳态 decode 中位数作为主指标（比平均值更抗异常值）
    - 记录 git commit + 运行环境，性能数字必须能对应到具体代码版本
    - 支持回归检测：与基线对比，超阈值时警告或失败退出（CI 用）
"""

# 启用延迟注解求值，允许在类型提示中使用新语法（如 list[float]、X | Y）
from __future__ import annotations

# ---- 标准库导入 ----
import argparse       # 命令行参数解析
import json           # JSON 序列化/反序列化（profile 输出、结果存储）
import platform       # 获取操作系统/CPU/Python 版本等环境信息
import shlex          # shell 字符串分割（安全解析 --extra-args）
import statistics     # 统计函数（median、mean）
import subprocess     # 调用外部进程（tinyqwen binary、git）
import sys            # 系统退出、stderr 输出
import tempfile       # 创建临时文件存放 profile JSON
from pathlib import Path  # 路径操作（文件读写、删除）

# ---- 标准负载常量（固定，保证跨版本可比）-------------------------------------
# "中国的首都是" 的 token 序列。写死 id 是为了让 bench 不依赖 tokenizer，
# 这样即使 tokenizer 配置变了，测速负载也完全不变。
CANONICAL_PROMPT = [105538, 59975, 100132]
# 生成的 token 数量（样本量）。32 个足够算出稳定的中位数，又不会太慢。
DECODE_TOKENS = 32
# 丢弃前几个 decode token（预热）。推理引擎刚启动时可能有缓存未命中、
# JIT 编译等一次性开销，这些不属于稳态性能，必须剔除。
WARMUP = 4
# KV cache 最大序列长度。需足够容纳 prompt + decode（3+32=35），
# 设 64 留余量，避免因 seq_len 不足导致截断。
MAX_SEQ_LEN = 64


def env_info() -> dict:
    """收集当前运行环境信息，保证测速结果可复现、可追溯。

    返回包含以下字段的字典：
        - platform：操作系统标识（如 macOS-14.0-arm64）
        - processor：CPU 型号（如 arm）
        - python：Python 版本号
        - git_commit：当前 git HEAD 的短 hash（unknown 如果不在 git 仓库中）

    Returns:
        dict: 环境信息字典
    """
    # 构建基础环境信息字典
    info = {
        "platform": platform.platform(),      # 操作系统平台标识字符串
        "processor": platform.processor() or "unknown",  # CPU 处理器型号，空则标记 unknown
        "python": platform.python_version(),  # Python 解释器版本号
    }
    # 记录当前 git commit：性能数字必须能对应到具体代码版本，
    # 否则无法知道某次提速是哪次提交带来的。
    try:
        # 调用 git rev-parse 获取当前 HEAD 的短 hash（7 位）
        rev = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,   # 捕获 stdout/stderr，不打印到终端
            text=True,             # 以文本模式返回（而非 bytes）
            check=True             # 非零退出码时抛 CalledProcessError
        ).stdout.strip()           # 去掉末尾换行符
        info["git_commit"] = rev   # 存入环境信息
    except Exception:
        # git 不可用或不在仓库内时静默降级为 "unknown"
        info["git_commit"] = "unknown"
    return info


def percentile(data: list[float], p: float) -> float:
    """计算简单百分位数（线性插值法）。

    对排序后的数据做线性插值，p=50 等价于中位数，p=95 即 P95。
    不依赖 numpy，纯 Python 实现保持零外部依赖。

    Args:
        data: 数值列表（不必预先排序）
        p: 百分位，取值范围 0..100

    Returns:
        float: 百分位数值；空列表返回 NaN
    """
    if not data:
        return float("nan")         # 空数据无法计算，返回 NaN
    s = sorted(data)                # 升序排列副本
    k = (len(s) - 1) * p / 100.0   # 计算连续索引位置（可能是小数）
    lo = int(k)                     # 下界索引（整数部分）
    hi = min(lo + 1, len(s) - 1)   # 上界索引（不超过最后一个元素）
    frac = k - lo                   # 小数部分，用于线性插值权重
    # 在 s[lo] 和 s[hi] 之间做线性插值
    return s[lo] * (1 - frac) + s[hi] * frac


def run_once(binary: str, model: str, extra_args: list[str] | None = None) -> dict:
    """执行一次标准负载测速，返回 profiler 输出的 JSON 字典。

    构造 tinyqwen CLI 命令并运行，通过 --profile-out 将逐 token 计时
    数据写入临时 JSON 文件，读入后删除临时文件。

    Args:
        binary: tinyqwen 可执行文件路径
        model: .tqwen 模型文件路径
        extra_args: 原样追加给 binary 的额外 CLI 参数列表。
                    测"需要开关才生效"的优化变体时使用，
                    例如 ["--matvec-impl", "double_2_float"]。

    Returns:
        dict: profiler JSON 内容，包含 tokens（逐 token 记录）和 op_totals 等字段
    """
    # 创建临时文件存放 profile 输出；delete=False 因为需要在子进程结束后读取
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
        profile_path = tf.name      # 获取临时文件的绝对路径
    # 构造 tinyqwen CLI 命令行参数列表
    cmd = [
        binary,                     # 可执行文件路径
        "--model", model,           # 模型文件
        "--tokens", ",".join(map(str, CANONICAL_PROMPT)),  # 固定 prompt token id，逗号分隔
        "--max-new-tokens", str(DECODE_TOKENS),  # 生成 token 数 = 32
        "--max-seq-len", str(MAX_SEQ_LEN),       # KV cache 容量上限 = 64
        "--eos", "-1",              # 禁用 EOS 停止符，保证生成固定数量的 token，样本量稳定
        "--profile-out", profile_path,  # profile 输出到临时文件
    ]
    # 追加额外参数（如果有），用于测试不同的 kernel 实现等
    cmd += list(extra_args or [])
    # 同步执行 tinyqwen，capture_output 捕获所有输出，check=True 在非零退出时报错
    subprocess.run(cmd, capture_output=True, check=True)
    # 读取 profiler 输出的 JSON
    with open(profile_path) as f:
        profile = json.load(f)
    # 清理临时文件
    Path(profile_path).unlink(missing_ok=True)
    return profile


def summarize(profile: dict) -> dict:
    """从 profiler 的逐 token 记录里提取稳定的延迟统计量。

    核心逻辑：分离 prefill/decode → 丢弃预热 → 算中位数/均值/P95。
    同时提取 top op（耗时最多的算子），用于分析瓶颈在哪。

    Args:
        profile: run_once() 返回的 profiler JSON 字典

    Returns:
        dict: 包含以下键的统计字典：
            - decode_samples: 稳态 decode 样本数
            - decode_median_ms: 稳态 decode 中位延迟（ms），主指标
            - decode_mean_ms: 稳态 decode 平均延迟
            - decode_min_ms: 稳态 decode 最小延迟
            - decode_p95_ms: 稳态 decode P95 延迟
            - prefill_total_ms: prefill 总耗时
            - prefill_tokens: prompt token 数
            - ttft_ms: Time To First Token（首 token 延迟）
            - decode_avg_ms: 含预热的全量 decode 平均（仅参考）
            - generated_tokens: 生成的 token 总数（含预热）
            - total_ms: forward 总耗时
            - top_ops: 耗时前 5 的算子列表
    """
    # 从 profile 中提取 decode 阶段（非 prefill）的逐 token 延迟
    decode_ms = [t["latency_ms"] for t in profile["tokens"] if not t["is_prefill"]]
    # 提取 prefill 阶段的逐 token 延迟
    prefill_ms = [t["latency_ms"] for t in profile["tokens"] if t["is_prefill"]]

    # 丢弃前 WARMUP 个 decode token（预热期），剩下的才是稳态性能。
    # 预热期可能因缓存未填充、频率爬升等原因偏慢，混入统计会拉高方差。
    steady = decode_ms[WARMUP:]
    if not steady:
        # 如果丢弃预热后没有剩余样本，说明 DECODE_TOKENS 设太小了
        raise RuntimeError("no steady decode tokens; increase DECODE_TOKENS")

    # 按总耗时排序的 top op，用于解释"时间花在哪"。
    # op_totals 是 profiler 按算子名聚合的 {op_name: {calls, total_ms}} 字典。
    op_totals = profile.get("op_totals", {})
    # 按 total_ms 降序取前 5 个最耗时的算子
    top_ops = sorted(op_totals.items(), key=lambda kv: -kv[1]["total_ms"])[:5]

    return {
        "decode_samples": len(steady),                      # 稳态样本数量
        "decode_median_ms": statistics.median(steady),      # 中位数：主指标，比均值更抗异常值
        "decode_mean_ms": statistics.mean(steady),          # 均值：辅助参考
        "decode_min_ms": min(steady),                       # 最小值：理论最优下限
        "decode_p95_ms": percentile(steady, 95),            # P95：尾部延迟指标
        "prefill_total_ms": sum(prefill_ms),                # prefill 总耗时（所有 prompt token 之和）
        # 批量 prefill 下整个 prompt 只有一条记录，真实长度以 profile 的
        # prompt_tokens 为准（旧 profile 无此字段时退回记录数）。
        "prefill_tokens": profile.get("prompt_tokens") or len(prefill_ms),
        # LLM serving 标准指标口径说明：
        # - TTFT（Time To First Token）：首 token 延迟 = prefill 总耗时；
        #   token-by-token prefill 下 profiler 的 first_token_ms 即各 prefill token 之和
        # - TOPT（Time Per Output Token）：每输出 token 延迟，
        #   主指标用稳态 decode 中位数（decode_median_ms），
        #   decode_avg_ms 是含预热的全量平均，仅作参考
        # - decode 长度 = 生成 token 总数（含被丢弃的预热 token）
        "ttft_ms": profile.get("first_token_ms", sum(prefill_ms)),  # TTFT
        "decode_avg_ms": profile.get("decode_avg_ms"),      # 全量 decode 平均（含预热）
        "generated_tokens": profile.get("generated_tokens", len(decode_ms)),  # 生成 token 总数
        "total_ms": profile.get("total_ms"),                # forward 总耗时
        # top op 列表：取前 5 个最耗时算子，保留名称和毫秒数（保留两位小数）
        "top_ops": [{"op": n, "total_ms": round(s["total_ms"], 2)} for n, s in top_ops],
    }


def check_regression(measured_median: float, baseline_path: str,
                     threshold_pct: float = 5.0, fail: bool = False) -> bool:
    """对比基线文件，检测性能是否回退超过阈值。

    读取基线 JSON 中的 decode_median_ms，与本次测量值对比。
    回退百分比 = (本次 - 基线) / 基线 × 100%。
    超过 threshold_pct 时打印警告；fail=True 时还会 sys.exit(1)。

    Args:
        measured_median: 本次测量的稳态 decode 中位延迟（ms）
        baseline_path: 基线 JSON 文件路径
        threshold_pct: 回归阈值百分比（默认 5%）
        fail: True 时回归会导致进程以 exit code 1 退出（CI 门禁用）

    Returns:
        bool: True 表示检测到回归，False 表示正常或基线不存在
    """
    bp = Path(baseline_path)
    # 基线文件不存在时静默跳过（首次运行还没建基线）
    if not bp.exists():
        return False
    try:
        # 读取基线 JSON 并提取 decode_median_ms
        base = json.loads(bp.read_text())
        base_med = base["decode_median_ms"]
    except (json.JSONDecodeError, KeyError):
        # 基线文件格式不对或缺少关键字段，静默跳过
        return False

    # 计算回退百分比：正值 = 变慢（回归），负值 = 变快（改善）
    regression_pct = (measured_median - base_med) / base_med * 100
    # 超过阈值则报告回归
    if regression_pct > threshold_pct:
        print(f"\n  ⚠️ 回归警告：本次 {measured_median:.2f} ms/tok vs 基线 {base_med:.2f} ms/tok"
              f"（+{regression_pct:.1f}%，阈值 {threshold_pct}%）")
        print(f"     基线来源：{baseline_path}（label={base.get('label', '?')}）")
        if fail:
            # CI 模式下回归直接失败退出
            print("     --fail-on-regression 已启用，退出")
            sys.exit(1)
        return True
    return False


def main() -> None:
    """bench.py 的主入口函数。

    解析命令行参数 → 执行 N 遍标准负载 → 汇总统计 → 打印结果 → 回归检测 → 可选写 JSON。
    """
    # 创建参数解析器，description 使用模块级 docstring（RawDescriptionHelpFormatter 保留格式）
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    # tinyqwen C++ runtime 可执行文件路径
    p.add_argument("--binary", default="build/runtime/tinyqwen")
    # .tqwen 模型文件路径
    p.add_argument("--model", default="model.tqwen")
    # 本次测量的标签名，用于区分不同配置（如 fp32-baseline / int8-w8a8）
    p.add_argument("--label", default="unlabeled",
                   help="本次测量的名字，如 fp32-baseline / int8-w8a8")
    # 重复测量次数，取各遍中位数的中位数（抗单次波动）
    p.add_argument("--runs", type=int, default=1,
                   help="把标准负载跑几遍，取各遍中位数的中位数（抗单次波动）")
    # 可选的 JSON 输出文件路径
    p.add_argument("--json", default=None, help="可选：把结果也写成 JSON 文件")
    # 原样传给 binary 的额外 CLI 参数（用引号括起）
    p.add_argument("--extra-args", default="",
                   help="原样传给 binary 的额外 CLI 参数（引号括起），"
                        "如 '--matvec-impl double_2_float'")
    # 回归检测对比的基线文件路径
    p.add_argument("--baseline", default="benchmarks/baseline.json",
                   help="回归检测对比的基线文件")
    # 跳过回归检测的标志
    p.add_argument("--no-check-regression", action="store_true",
                   help="跳过回归检测")
    # 回归阈值百分比
    p.add_argument("--regression-threshold", type=float, default=5.0,
                   help="回归阈值百分比（默认 5）")
    # 回归时非零退出（CI 门禁用）
    p.add_argument("--fail-on-regression", action="store_true",
                   help="回归时非零退出（CI 用）")
    # 解析命令行参数
    args = p.parse_args()

    # 用 shlex 安全地将 extra_args 字符串拆分为参数列表（处理引号/转义）
    extra = shlex.split(args.extra_args)
    # 确保至少跑 1 遍
    runs = max(1, args.runs)
    # 打印测速配置摘要
    print(f"[bench] label={args.label} runs={runs}  负载: prompt={len(CANONICAL_PROMPT)} tok, "
          f"decode={DECODE_TOKENS} tok (丢弃预热 {WARMUP})")
    if extra:
        # 有额外参数时也打印出来，方便确认
        print(f"[bench] 额外参数: {' '.join(extra)}")

    # 跑 runs 遍，收集每遍的稳态统计；headline 取"中位数的中位数"，更稳。
    # medians/p95s 分别收集每遍的中位数和 P95，最后再取各自的中位数。
    medians, p95s, last_stats = [], [], None
    for r in range(runs):
        # 执行一次标准负载，获取 profiler 数据
        profile = run_once(args.binary, args.model, extra)
        # 从 profiler 数据中提取统计量
        stats = summarize(profile)
        # 收集本遍的中位数和 P95
        medians.append(stats["decode_median_ms"])
        p95s.append(stats["decode_p95_ms"])
        # 保留最后一遍的完整统计（top_ops / prefill 等沿用）
        last_stats = stats
        if runs > 1:
            # 多遍时逐遍打印中间结果，方便观察稳定性
            print(f"  run {r + 1}/{runs}: median={stats['decode_median_ms']:.2f}ms "
                  f"p95={stats['decode_p95_ms']:.2f}ms")

    # 收集环境信息（OS、CPU、git commit）
    env = env_info()
    # headline 复制最后一遍的统计，然后用所有遍的中位数的中位数覆盖关键字段
    headline = dict(last_stats)  # top_ops / prefill 等沿用最后一遍
    headline["decode_median_ms"] = statistics.median(medians)  # 中位数的中位数：更稳健
    headline["decode_p95_ms"] = statistics.median(p95s)        # P95 也取中位数
    headline["runs"] = runs                                     # 记录跑了多少遍
    headline["per_run_median_ms"] = [round(m, 2) for m in medians]  # 每遍的中位数列表

    # 组装最终结果字典
    result = {"label": args.label, **headline, "env": env}
    if extra:
        result["extra_args"] = extra  # 可复现性：记录本次测量启用了什么开关

    # ---- 人类可读汇总打印 ----
    # TTFT（首 token 延迟）
    print(f"\n  TTFT：{headline['ttft_ms']:.2f} ms"
          f"（prefill {last_stats['prefill_tokens']} tok）")
    # TOPT（每输出 token 延迟）：中位数 + P95 + 样本信息
    print(f"  TOPT：median={headline['decode_median_ms']:.2f} ms/tok  "
          f"p95={headline['decode_p95_ms']:.2f}（decode 共 {last_stats['generated_tokens']} tok，"
          f"丢预热 {WARMUP}，稳态样本 {last_stats['decode_samples']}）")
    # forward 总耗时（如果有）
    if headline.get("total_ms") is not None:
        print(f"  forward 总耗时：{headline['total_ms']:.1f} ms")
    # 多遍时打印每遍中位数和最终取值
    if runs > 1:
        print(f"  各遍中位数：{headline['per_run_median_ms']}（取中位 {headline['decode_median_ms']:.2f}）")
    # 环境信息：CPU 型号 + git commit
    print(f"  环境：{env['processor']} @ {env['git_commit']}")
    # 耗时 top 算子列表
    print("  耗时 top op：")
    for op in last_stats["top_ops"]:
        print(f"    {op['op']:<28} {op['total_ms']:>9.2f} ms")

    # ---- 生成可粘进 optimization_log.md 的 Markdown 表格行 ----
    print("\n  [复制下面这行到 docs/optimization_log.md 的表格里]")
    print(f"  | {args.label} | {env['git_commit']} | "
          f"{headline['decode_median_ms']:.2f} | {headline['decode_p95_ms']:.2f} | "
          f"<填写：相比基线的提升> | <填写：原因> |")

    # 回归检测：与基线对比，超阈值时打印警告
    if not args.no_check_regression:
        check_regression(headline["decode_median_ms"], args.baseline,
                         args.regression_threshold, args.fail_on_regression)

    # 如果指定了 --json，将完整结果写入 JSON 文件
    if args.json:
        Path(args.json).write_text(json.dumps(result, ensure_ascii=False, indent=2))
        print(f"\n  已写入 {args.json}")


# 脚本直接运行入口
if __name__ == "__main__":
    main()
