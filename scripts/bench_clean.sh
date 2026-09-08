#!/usr/bin/env bash
# =============================================================================
# bench_clean.sh —— 干净环境下的可信 decode 测速（重启后专用）
# =============================================================================
# 为什么单独有这个脚本：AGENTS.md 坑 #7 和 #20 都是实测踩到的教训——
#   坑 #7 ：agent 自己会抢 CPU（曾见 load 5.67、两个 opencode 进程各占 125%/79%），
#           端到端计时被污染到 CPU arm 离散度 37×，一度得出完全相反的优化结论并误回退。
#           污染是加性的，所以取 min 比取中位数稳。
#   坑 #20：机器一旦被推进重度换页（实测 swap used 11.3 GB / 12 GB、pageouts 132 万），
#           此后所有计时都不可信。swap 是写操作、消耗 SSD 寿命，宁可 fail-fast。
#
# 所以本脚本做三件别的 bench 脚本都不做的事：
#   1. 测速前门禁：load / swap / 可用内存不合格就拒绝跑（"不许用 swap"是硬约束）
#   2. 取 min 而非中位数，并打印离散度 max/min —— 离散度 > 1.5 的行不可用于归因
#   3. 测速后复查：这一轮若自己把机器搞脏了（swap 涨了），标注结果不可用
#
# 用法：
#   ./scripts/bench_clean.sh                                     # 默认模型 + 默认配方
#   MODEL=model_qwen35_f16.tqwen ./scripts/bench_clean.sh
#   EXTRA_ARGS="--matvec-impl neon_mt_kv_nt --ops-impl neon" ./scripts/bench_clean.sh
#   FORCE=1 ./scripts/bench_clean.sh                             # 跳过门禁（结果不可信）
#
# 可选：设 BYTES_PER_TOKEN=<MB> 会额外算出有效带宽与"离带宽墙多远"。
#   本机带宽墙见 benchmarks/machine_ceiling/m4_ceiling.md（DRAM 整机墙 ≈114 GB/s 峰值、
#   持续 74.6 GB/s）。注意 docs/optimization_log.md 里的 "~199 GB/s" 出自更高带宽的
#   Mac（M Pro 系），不是本机 —— 引用时必须区分机器。
# =============================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

BUILD_DIR="${BUILD_DIR:-build}"
BIN="./${BUILD_DIR}/runtime/tinyqwen"
MODEL="${MODEL:-model_qwen35_f16.tqwen}"
PY="${PY:-.venv/bin/python}"

# 坑 #1：--matvec-impl 默认 ref 是标量兜底，测出来不是真速度。必须显式给优化配方，
# 否则数字会被放大十几倍。各模型的 recipe 见 model*.yaml。
EXTRA_ARGS="${EXTRA_ARGS:---matvec-impl neon_mt_kv_nt --ops-impl neon}"

# prompt ≥32 才走 Qwen3.5 批量 prefill GEMM 路径（坑 #5）；decode 32 与 bench.sh 一致
PROMPT_TOKENS="${PROMPT_TOKENS:-32}"
DECODE_TOKENS="${DECODE_TOKENS:-32}"
WARMUP="${WARMUP:-4}"          # bench.sh 约定：丢预热 4
RUNS="${RUNS:-7}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-256}"

# 门禁阈值
MAX_LOAD="${MAX_LOAD:-2.0}"          # 1 分钟 load 上限
MAX_SWAP_MB="${MAX_SWAP_MB:-0}"      # 不许用 swap
DISPERSION_LIMIT="${DISPERSION_LIMIT:-1.5}"   # 坑 #7：> 1.5 的行不可用于归因
FORCE="${FORCE:-0}"

[[ -x "$BIN" ]] || { echo "error: $BIN 不存在，先 cmake --build build -j" >&2; exit 1; }
[[ -f "$MODEL" ]] || { echo "error: 找不到模型 $MODEL" >&2; exit 1; }
[[ -x "$PY" ]] || { echo "error: 找不到 $PY（仓库根 .venv，见 AGENTS.md「Python 环境」）" >&2; exit 1; }

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

# ---------------------------------------------------------------------------
# vm_stat 取值。字段索引是个真实的坑：各行列数不同（"Pages free:" 数值在 $3、
# "Pages wired down:" 在 $4、"Pages occupied by compressor:" 在 $5、"Pageouts:" 在 $2），
# 手拼时极易取错并且不报错（取到空值 → 算出 0）。一律用 $NF 取最后一个字段，
# 对"带冒号"和"不带冒号"两种 vm_stat 格式都成立。
# ---------------------------------------------------------------------------
PAGE_SIZE="$(sysctl -n hw.pagesize)"

vm_pages() {   # vm_pages <行匹配正则> → 该行数值（页数），取不到则输出 0
    vm_stat | awk -v re="$1" '$0 ~ re { v=$NF; gsub(/\./,"",v); print v+0; exit }
               END { if (NR==0) print 0 }'
}
vm_mb() {      # 同上，换算成 MB
    echo $(( $(vm_pages "$1") * PAGE_SIZE / 1048576 ))
}

swap_used_mb() {
    sysctl vm.swapusage | sed -n 's/.*used = \([0-9.]*\)M.*/\1/p'
}
load1() {
    sysctl -n vm.loadavg | awk '{ gsub(/\[/,"",$2); print $2 }'
}

# 可用内存口径与 runtime/main.cpp 的 available_memory_bytes() 一致：
# free + inactive + purgeable。故意不取 speculative/compressor —— 坑 #26 说宁可
# 保守 fail-fast 也不赌 macOS 能回收（朴素"物理-wired"口径实测乐观 2.39×）。
avail_mb() {
    echo $(( $(vm_mb 'Pages free') + $(vm_mb 'Pages inactive') + $(vm_mb 'Pages purgeable') ))
}

mem_report() {   # 打印完整归因表；$1 = 标题
    echo "--- $1 ---"
    local free active inactive spec wired purge fb anon comp_p comp_l
    free=$(vm_mb 'Pages free');            active=$(vm_mb 'Pages active')
    inactive=$(vm_mb 'Pages inactive');    spec=$(vm_mb 'Pages speculative')
    wired=$(vm_mb 'Pages wired');          purge=$(vm_mb 'Pages purgeable')
    fb=$(vm_mb 'File-backed pages');        anon=$(vm_mb 'Anonymous pages')
    comp_p=$(vm_mb 'occupied by compressor'); comp_l=$(vm_mb 'stored in compressor')
    printf "  free %d / active %d / inactive %d / speculative %d MB\n" \
        "$free" "$active" "$inactive" "$spec"
    printf "  wired %d / compressor %d MB（逻辑量 %d MB，比值 %.1f:1 —— 远大于 2~3:1\n" \
        "$wired" "$comp_p" "$comp_l" "$(awk -v a="$comp_l" -v b="$comp_p" \
        'BEGIN{ if(b>0) printf "%.1f", a/b; else print 0 }')"
    echo "    说明里面绝大部分是全零页，不是真实数据，别被这个数字吓到）"
    printf "  File-backed %d MB（文件缓存，可回收）/ Anonymous %d MB（进程私有，须换页）\n" \
        "$fb" "$anon"
    printf "  可用(free+inactive+purgeable) %d MB = %.1f%% of 物理 %d MB\n" \
        "$(avail_mb)" \
        "$(awk -v a="$(avail_mb)" -v p="$(( $(sysctl -n hw.memsize) / 1048576 ))" \
           'BEGIN{ printf "%.1f", a*100.0/p }')" \
        "$(( $(sysctl -n hw.memsize) / 1048576 ))"
    printf "  swap used %s MB / load1 %s / Pageouts %s / Swapouts %s\n" \
        "$(swap_used_mb)" "$(load1)" "$(vm_pages 'Pageouts')" "$(vm_pages 'Swapouts')"
    printf "  macOS 自报 free: %s（与上面口径不同，它把 speculative/compressor 也算可回收）\n" \
        "$(memory_pressure 2>/dev/null | awk '/free percentage/{print $NF}' || echo '?')"
    echo
}

# ---------------------------------------------------------------------------
# 门禁
# ---------------------------------------------------------------------------
gate() {   # gate <阶段名>；返回 0 = 通过
    local swap load ok=1
    swap="$(swap_used_mb)"; load="$(load1)"
    echo "=== 机器状态门禁（$1）==="
    printf "  load1      %-8s 阈值 %-8s %s\n" "$load" "$MAX_LOAD" \
        "$(awk -v a="$load" -v b="$MAX_LOAD" 'BEGIN{print (a<=b)?"PASS":"FAIL"}')"
    printf "  swap used  %-8s 阈值 %-8s %s\n" "$swap" "$MAX_SWAP_MB" \
        "$(awk -v a="$swap" -v b="$MAX_SWAP_MB" 'BEGIN{print (a<=b)?"PASS":"FAIL"}')"
    printf "  可用内存   %-8s （budget = 可用 - 10%%，供 runtime 预检参考）\n" "$(avail_mb)"
    awk -v a="$load" -v b="$MAX_LOAD" 'BEGIN{ exit !(a<=b) }' || ok=0
    awk -v a="$swap" -v b="$MAX_SWAP_MB" 'BEGIN{ exit !(a<=b) }' || ok=0
    echo
    return $(( 1 - ok ))
}

# ---------------------------------------------------------------------------
# 测速
# ---------------------------------------------------------------------------
# 确定性伪随机 token 序列（不需要 tokenizer），与 bench_metal_prefill.sh 同一 seed
TOKENS="$("$PY" - "$PROMPT_TOKENS" <<'PY'
import sys, random
random.seed(1234)
print(",".join(str(random.randrange(0, 151936)) for _ in range(int(sys.argv[1]))))
PY
)"

# 时间取自 profiler JSON 的 decode_avg_ms，不是 wall clock ——
# 排除模型加载 / 进程启动 / prefill 的噪声（同 bench_metal_prefill.sh 的做法）。
one_run() {
    local out="$1"
    # EXTRA_ARGS 故意不加引号：需要按空格拆成多个参数
    # shellcheck disable=SC2086
    "$BIN" --model "$MODEL" --tokens "$TOKENS" \
        --max-new-tokens "$DECODE_TOKENS" --max-seq-len "$MAX_SEQ_LEN" \
        --eos -1 --profile-out "$out" $EXTRA_ARGS >/dev/null 2>&1
    "$PY" -c "import json,sys; print(json.load(open(sys.argv[1]))['decode_avg_ms'])" "$out"
}

stats() {   # stats <样本...> → "min median spread"
    "$PY" -c "
import statistics, sys
v = [float(x) for x in sys.argv[1:]]
print('%.4f %.4f %.2f' % (min(v), statistics.median(v), max(v)/min(v)))
" "$@"
}

# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
mem_report "内存归因（测速前）"

if ! gate "测速前"; then
    if [[ "$FORCE" != "1" ]]; then
        cat >&2 <<EOF
error: 机器不干净，拒绝测速。
  - load 高 → 关掉占 CPU 的进程（注意 agent 自己也会抢，坑 #7）
  - swap 非 0 → 重启，或等 macOS 收缩；swap 是写操作、消耗 SSD 寿命（坑 #20/#26）
确要在脏机器上跑（结果不可信）：FORCE=1 $0
EOF
        exit 1
    fi
    echo "warn: FORCE=1，跳过门禁 —— 下面的数字不可用于归因" >&2
    echo
fi

echo "=== 配置 ==="
printf "  模型 %s\n  配方 %s\n  prompt %s tok / decode %s tok / 预热 %s / 采样 %s\n\n" \
    "$MODEL" "$EXTRA_ARGS" "$PROMPT_TOKENS" "$DECODE_TOKENS" "$WARMUP" "$RUNS"

echo "=== 预热（丢弃）==="
for _ in $(seq "$WARMUP"); do
    one_run "$TMP/warm.json" >/dev/null || true
done

echo "=== 采样 ==="
samples=()
for i in $(seq "$RUNS"); do
    s="$(one_run "$TMP/run.json")"
    samples+=("$s")
    printf "  run %d: %s ms/tok\n" "$i" "$s"
done
echo

read -r MIN MED SPREAD <<< "$(stats "${samples[@]}")"

echo "=== 结果 ==="
printf "  min    %s ms/tok  ← 用这个（污染是加性的，取 min 比取中位数稳，坑 #7）\n" "$MIN"
printf "  median %s ms/tok\n" "$MED"
printf "  离散度 %s（max/min）%s\n" "$SPREAD" \
    "$(awk -v s="$SPREAD" -v l="$DISPERSION_LIMIT" \
       'BEGIN{ print (s<=l) ? "可用" : "超过 " l " —— 不可用于归因" }')"

if [[ -n "${BYTES_PER_TOKEN:-}" ]]; then
    echo
    echo "=== 带宽归因（BYTES_PER_TOKEN=$BYTES_PER_TOKEN MB）==="
    awk -v mb="$BYTES_PER_TOKEN" -v ms="$MIN" 'BEGIN {
        bw = (mb/1024.0) / (ms/1000.0)
        printf "  有效带宽 %.1f GB/s\n", bw
        printf "  占峰值墙(114 GB/s) %.0f%% / 占持续墙(74.6 GB/s) %.0f%%\n", bw*100/114, bw*100/74.6
        if (bw > 114) print "  !! 超过整机墙 —— 分母算错了（坑 #7：别拿文件大小当每 token 读量，稀疏查表的 embed 不整读）"
    }'
fi

echo
mem_report "内存归因（测速后）"

SWAP_AFTER="$(swap_used_mb)"
echo "=== 复查 ==="
if awk -v a="$SWAP_AFTER" -v b="$MAX_SWAP_MB" 'BEGIN{ exit !(a>b) }'; then
    echo "  FAIL: 测速期间 swap 涨到 ${SWAP_AFTER} MB —— 本轮结果不可信（坑 #20）"
elif awk -v s="$SPREAD" -v l="$DISPERSION_LIMIT" 'BEGIN{ exit !(s>l) }'; then
    echo "  FAIL: 离散度 $SPREAD 超阈值 —— 本轮结果不可用于归因（坑 #7）"
else
    echo "  PASS: swap 未增长、离散度在阈值内 —— 本轮数字可用于归因"
fi

# 便于粘进 docs/optimization_log.md 的一段
echo
echo "=== 记录用摘要 ==="
cat <<EOF
- 模型: $MODEL
- 配方: $EXTRA_ARGS
- decode: $MIN ms/tok（min of $RUNS，丢预热 $WARMUP；median $MED，离散度 $SPREAD）
- 机器: load1 $(load1) / swap ${SWAP_AFTER} MB / 可用 $(avail_mb) MB
EOF
