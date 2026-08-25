#!/usr/bin/env bash
# ============================================================================
# add_model.sh — 一条命令接入新模型（导出 → 校验 → 冒烟验证）
# ============================================================================
# 用途：把一个 HF 格式的新模型接入 tinyqwen，全流程自动化：
#   1. 确保 runtime 二进制是最新的（增量编译）
#   2. 导出 .tqwen（i4 量化 / f16 两条路径，带计时）
#   3. 文件头校验（magic / 版本 / 尺寸一致性）
#   4. 冒烟生成：固定 prompt 跑 N 个 token，解码打印，肉眼可验
#
# 用法：
#   ./scripts/add_model.sh <HF 模型目录> [选项]
#
# 选项：
#   --dtype i4|f16|vq2  导出精度（默认 i4；vq2=朴素 2-bit 块 VQ，验证链路用）
#   --vq2-iters N       vq2 k-means 迭代（默认 8）；--embed-i4 导出后把 embed 压 INT4
#   --method hqq|rtn    i4 量化算法（默认 hqq；rtn 快好几倍，快速试跑用）
#   --group-size N      i4 量化组大小（默认 64，须为 32 倍数）
#   --workers N         i4 并行量化进程数（默认 0=自动）
#   --prompt "..."      冒烟 prompt（默认量子计算三句话）
#   --smoke-tokens N    冒烟生成 token 数（默认 32）
#
# 示例：
#   ./scripts/add_model.sh models/Qwen3.5-4B                     # i4 HQQ（默认）
#   ./scripts/add_model.sh models/Qwen3.5-4B --method rtn        # 快速试跑
#   ./scripts/add_model.sh models/Qwen3.5-4B --dtype f16         # f16 满精度
#
# 完成后：建议为新模型建 model_*.yaml 注册表（参照 model_qwen35_f16.yaml），
# 记录架构参数 / recipe / 实测数字，tools/model_registry.py 会读取。
# ============================================================================

# 任何命令失败立即退出；未定义变量报错；管道任一失败即失败
set -euo pipefail

# 切到项目根目录
cd "$(dirname "$0")/.."

# ---------------------------------------------------------------------------
# 参数解析
# ---------------------------------------------------------------------------
MODEL_DIR=""                                  # HF 模型目录（位置参数）
DTYPE="i4"                                    # 导出精度
METHOD="hqq"                                  # i4 量化算法
GROUP_SIZE="64"                               # i4 量化组大小
WORKERS="0"                                   # 并行度（0=自动）
VQ2_ITERS="8"                                 # vq2 k-means 迭代次数
EMBED_I4="0"                                  # vq2 导出后是否把 embed 压成 INT4
PROMPT="请用三句话介绍一下量子计算的基本原理。"  # 冒烟 prompt
SMOKE_TOKENS="32"                             # 冒烟生成长度

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dtype)        DTYPE="$2"; shift 2 ;;
        --method)       METHOD="$2"; shift 2 ;;
        --group-size)   GROUP_SIZE="$2"; shift 2 ;;
        --workers)      WORKERS="$2"; shift 2 ;;
        --vq2-iters)    VQ2_ITERS="$2"; shift 2 ;;
        --embed-i4)     EMBED_I4="1"; shift ;;
        --prompt)       PROMPT="$2"; shift 2 ;;
        --smoke-tokens) SMOKE_TOKENS="$2"; shift 2 ;;
        -*)             echo "未知参数: $1" >&2; exit 1 ;;
        *)              MODEL_DIR="$1"; shift ;;
    esac
done

# 参数校验
if [[ -z "$MODEL_DIR" ]]; then
    echo "用法: $0 <HF 模型目录> [--dtype i4|f16] [--method hqq|rtn] [--group-size N] [--workers N]" >&2
    exit 1
fi
if [[ ! -f "$MODEL_DIR/config.json" ]]; then
    echo "错误: $MODEL_DIR/config.json 不存在（不是 HF 模型目录？）" >&2
    exit 1
fi

# Python 解释器：优先仓库 .venv（transformers/torch 齐全）
PY=".venv/bin/python"
[[ -x "$PY" ]] || PY="python3"

# 输出文件名：model_<模型名小写>_<精度>.tqwen
# （tr -d '\n' 必须先于 tr -c：否则管道尾部换行会被补集规则转成 '_'）
NAME="$(basename "$MODEL_DIR" | tr 'A-Z' 'a-z' | tr -d '\n' | tr -c 'a-z0-9-._' '_')"
OUT="model_${NAME}_${DTYPE}.tqwen"
BIN="build/runtime/tinyqwen"

echo "=========================================="
echo "接入新模型: $MODEL_DIR"
echo "  精度=$DTYPE  量化=$METHOD  group=$GROUP_SIZE  workers=$WORKERS"
echo "  输出=$OUT"
echo "=========================================="

# ---------------------------------------------------------------------------
# 步骤 1/4：确保二进制最新（增量编译，已有则秒过）
# ---------------------------------------------------------------------------
echo "[1/4] 编译 runtime..."
if [[ ! -d build ]]; then
    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release >/dev/null
fi
cmake --build build -j >/dev/null
echo "      OK: $BIN"

# ---------------------------------------------------------------------------
# 步骤 2/4：导出 .tqwen（带计时）
# ---------------------------------------------------------------------------
echo "[2/4] 导出 $DTYPE 模型..."
case "$DTYPE" in
    i4)
        time "$PY" tools/export_qwen_to_tiny_i4.py \
            --model "$MODEL_DIR" --out "$OUT" \
            --method "$METHOD" --group-size "$GROUP_SIZE" --workers "$WORKERS"
        ;;
    f16)
        time "$PY" tools/export_qwen_to_tiny.py \
            --model "$MODEL_DIR" --out "$OUT" --dtype f16
        ;;
    vq2)
        # 朴素块 VQ（k-means，无旋转）：2-bit 链路/格式验证用；精度受限
        time "$PY" tools/export_qwen_to_tiny_vq2.py \
            --model "$MODEL_DIR" --out "$OUT" --kmeans-iters "$VQ2_ITERS"
        if [[ "$EMBED_I4" == "1" ]]; then
            echo "      压缩 embed 为紧凑 INT4..."
            "$PY" tools/quantize_embed_i4.py --in "$OUT" --out "$OUT.ei4" \
                --group-size "$GROUP_SIZE"
            mv "$OUT.ei4" "$OUT"
        fi
        ;;
    *)
        echo "错误: 不支持的 dtype: $DTYPE（可选 i4 / f16 / vq2）" >&2
        exit 1
        ;;
esac

# ---------------------------------------------------------------------------
# 步骤 3/4：文件头校验（magic / 版本 / 尺寸自洽）
# ---------------------------------------------------------------------------
echo "[3/4] 校验文件头..."
OUT="$OUT" "$PY" - <<'EOF'
import os, struct, sys
p = os.environ["OUT"]                       # 输出文件路径（环境变量传入）
h = open(p, "rb").read(192)                 # .tqwen 头部固定 192 字节
assert h[:8] == b"TINYQWEN", "magic 错误（不是 TINYQWEN 文件）"
ver, dt = struct.unpack_from("<II", h, 8)   # 版本号 + dtype
assert ver >= 1, f"未知版本 {ver}"
total = struct.unpack_from("<Q", h, 88)[0]  # 头部记录的总字节数
sz = os.path.getsize(p)                     # 实际文件大小
assert total == sz, f"尺寸不一致: 头部 {total} != 实际 {sz}"
n_layers, hidden, inter = struct.unpack_from("<III", h, 16)
vocab = struct.unpack_from("<I", h, 40)[0]
print(f"      OK: v{ver} dtype={dt} layers={n_layers} hidden={hidden} "
      f"inter={inter} vocab={vocab} size={sz / 2**20:.0f}MB")
EOF

# ---------------------------------------------------------------------------
# 步骤 4/4：冒烟生成（tokenize → 生成 → 解码）
# ---------------------------------------------------------------------------
echo "[4/4] 冒烟生成（$SMOKE_TOKENS tokens）..."
TOK_JSON="/tmp/add_model_prompt_$$.json"
"$PY" tools/tokenize_prompt.py --model "$MODEL_DIR" \
    --prompt "$PROMPT" --chat --out "$TOK_JSON" >/dev/null

# 按精度选推荐内核配方（与 model_*.yaml 的 recipe 一致）
case "$DTYPE" in
    i4)  IMPL_ARGS=(--matvec-impl sdot2_mt --ops-impl neon) ;;
    f16) IMPL_ARGS=(--matvec-impl neon_mt_kv_nt --ops-impl neon) ;;
esac

# 生成（取最后一行 generated_ids）
GEN="$("$BIN" --model "$OUT" --tokens-json "$TOK_JSON" \
    --max-new-tokens "$SMOKE_TOKENS" "${IMPL_ARGS[@]}" 2>&1 | tail -1)"
rm -f "$TOK_JSON"

# 解码 + 基础合理性检查
MODEL_DIR="$MODEL_DIR" GEN="$GEN" "$PY" - <<'EOF'
import os, sys
from transformers import AutoTokenizer
ids = [int(x) for x in os.environ["GEN"].split(":")[1].split()]
tok = AutoTokenizer.from_pretrained(os.environ["MODEL_DIR"],
                                    trust_remote_code=True)
text = tok.decode(ids, skip_special_tokens=True)
print("      prompt 解码（前 300 字）:")
for line in text[:300].splitlines():
    print("        " + line)
# 合理性兜底：token 多样性过低说明生成坏了（重复退化）
assert len(ids) > 0, "没有生成任何 token"
assert len(set(ids)) >= max(3, len(ids) // 8), \
    "token 多样性过低，疑似量化/内核错误——请人工检查上面的输出"
print("      冒烟 PASS")
EOF

echo ""
echo "=========================================="
echo "完成：$OUT"
echo "后续建议："
echo "  1. 建注册表 model_${NAME}_${DTYPE}.yaml（参照 model_qwen35_f16.yaml）"
echo "  2. 正式测速：MODEL=$OUT ./scripts/bench.sh <label> \\"
echo "       --extra-args \"${IMPL_ARGS[*]}\""
echo "=========================================="
