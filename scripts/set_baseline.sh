#!/usr/bin/env bash
# ============================================================================
# set_baseline.sh — 建立 / 更新性能基线（baseline）
# ============================================================================
# 用途：稳定跑 3 遍基准测试，将结果写入 benchmarks/baseline.json。
#       之后 record.sh 会自动读取此基线，计算"vs 基线加速比"。
#       通常在完成一个稳定版本后执行一次，作为后续优化的参照点。
#
# 用法：
#   ./scripts/set_baseline.sh <label>
#
# 示例：
#   ./scripts/set_baseline.sh fp32-baseline
#
# 参数说明：
#   label — 基线标签名，记录在 baseline.json 的 label 字段中
#
# 产物：
#   benchmarks/baseline.json — 精简后的基线数据（label、中位耗时、P95、commit）
#   benchmarks/.baseline_full.json — 临时文件（完整测速数据，脚本结束后删除）
#
# 依赖环境：
#   - python3 + tools/bench.py
#   - 已导出的模型和已编译的 runtime
# ============================================================================

# set -e: 任何命令失败立即退出；-u: 引用未定义变量报错；-o pipefail: 管道中任一命令失败则整体失败
set -euo pipefail

# 切换到项目根目录
cd "$(dirname "$0")/.."

# 第一个参数为 label，缺失时打印用法并退出
LABEL="${1:?usage: set_baseline.sh <label>}"

# 确保 benchmarks 目录存在
mkdir -p benchmarks

# 调用 bench.py 跑 3 遍取中位，输出完整的 JSON 结果到临时文件
python3 tools/bench.py --label "$LABEL" --runs 3 --json benchmarks/.baseline_full.json

# ---------------------------------------------------------------------------
# 用内联 Python 脚本从完整结果中提取关键字段，生成精简的 baseline.json
# ---------------------------------------------------------------------------
python3 - "$LABEL" <<'PYEOF'
import json, sys
# 从命令行获取 label 参数
label = sys.argv[1]
# 读取完整测速结果
d = json.load(open("benchmarks/.baseline_full.json"))
# 构造精简的基线对象：只保留关键指标
out = {
    "label": label,                                          # 基线标签
    "decode_median_ms": round(d["decode_median_ms"], 2),     # decode 阶段中位耗时（ms/token）
    "decode_p95_ms": round(d["decode_p95_ms"], 2),           # decode P95 耗时（衡量尾部延迟）
    "commit": d["env"]["git_commit"],                        # 对应的 git commit hash
    "note": "fp32 参考基线；换基线时重跑 scripts/set_baseline.sh",  # 备注
}
# 写入精简基线文件（ensure_ascii=False 允许中文）
json.dump(out, open("benchmarks/baseline.json", "w"), ensure_ascii=False, indent=2)
print("[baseline] benchmarks/baseline.json =", out)
PYEOF

# 清理临时文件
rm -f benchmarks/.baseline_full.json
