#!/usr/bin/env bash
# ============================================================================
# bench_kernels.sh — 算子微基准一键扫测（算子优化的标准入口）
# ============================================================================
# 用法:
#   ./scripts/bench_kernels.sh                 # 全族扫测 + 落 CSV + 对比基线
#   ./scripts/bench_kernels.sh --family vq2    # 透传任意 bench_kernels 参数
#   REBUILD=1 ./scripts/bench_kernels.sh       # 强制重新构建
# 环境变量:
#   BUILD_DIR  构建目录（默认 build）
#   JOBS       并行编译数（默认 8）
#   TINYQWEN_MT_THREADS  多线程内核的线程数（透传给二进制）
#
# 产物:
#   benchmarks/results/kernels_<时间戳>.csv   本次结果
#   与基线（kernels_baseline.csv）逐行对比，标出 ±10% 以上的耗时变化。
# ============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

BUILD_DIR="${BUILD_DIR:-build}"
JOBS="${JOBS:-8}"
BIN="$BUILD_DIR/benchmarks/bench_kernels"
RESULTS_DIR="benchmarks/results"
BASELINE="$RESULTS_DIR/kernels_baseline.csv"

# ---- 1. 构建 ----
if [[ ! -x "$BIN" || "${REBUILD:-0}" == "1" ]]; then
    echo "[build] cmake -B $BUILD_DIR -DCMAKE_BUILD_TYPE=Release"
    cmake -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release > /dev/null
fi
cmake --build "$BUILD_DIR" --target bench_kernels -j "$JOBS"

# ---- 2. 扫测 ----
mkdir -p "$RESULTS_DIR"
TS="$(date +%Y%m%d_%H%M%S)"
CSV="$RESULTS_DIR/kernels_${TS}.csv"
echo ""
echo "[bench] $BIN $* --csv $CSV"
"$BIN" "$@" --csv "$CSV"

# ---- 3. 与基线对比（按 family,shape,impl 对齐，标 ±10% 以上变化） ----
if [[ -f "$BASELINE" && "$CSV" != "$BASELINE" ]]; then
    echo ""
    echo "[diff] vs 基线 ${BASELINE}（|Δ|>10% 的 ms/call）"
    awk -F, '
        NR==FNR { if (FNR>1) base[$1","$2","$5]=$8; next }   # 基线: ms_per_call
        FNR>1 {
            key=$1","$2","$5
            if (key in base && base[key] > 0) {
                d=($8-base[key])/base[key]*100
                if (d>10 || d<-10)
                    printf "  %-22s %-8s %-12s %9.4f -> %9.4f ms  (%+.1f%%)\n",
                           $1,$2,$5,base[key],$8,d
            }
        }' "$BASELINE" "$CSV"
    echo "[done] 本次结果: $CSV"
else
    echo "[done] 无基线可对比；本次结果即为基线: ${CSV}"
fi
