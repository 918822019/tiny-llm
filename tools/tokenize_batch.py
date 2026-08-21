#!/usr/bin/env python3
"""把数据集批量 tokenize 成 tinyqwen 批量模式的输入（JSONL）。

数据来源默认 ModelScope 中文诗词数据集（datasets/chinese-poetry-collection，
单列 text1，每行一首诗）。整段 text1 作为 prompt——续写任务口径，
add_special_tokens=False、不走 chat template（与 run_poetry_demo.py 一致）。

选择是**确定性**的：按 csv 行序过滤 token 长度区间后取前 N 条，不 shuffle——
两次运行产出完全相同的输入，测量才可复现。

输出：
- <out>           JSONL，每行 {"tokens": [...], "text": "..."}，供 tinyqwen --batch-tokens-jsonl
- <out>.meta.json 条数与 token 长度分布（bench_dataset.py 读取记录进结果）

用法：
    # 基本用法：取 16 条诗词
    python tools/tokenize_batch.py --num 16 --out benchmarks/poetry16.jsonl
    # 自定义数据集和长度范围
    python tools/tokenize_batch.py --csv datasets/.../train.csv --num 64 \\
        --min-tokens 8 --max-tokens 96 --out /tmp/poetry64.jsonl

输入：
    - CSV 数据集文件（含 text1 列）
    - 本地 HF tokenizer 目录

输出：
    - JSONL 文件（每行一个 tokenized prompt）
    - .meta.json 元数据文件（条数、token 长度统计）
"""

# 启用延迟注解求值
from __future__ import annotations

# ---- 标准库导入 ----
import argparse       # 命令行参数解析
import csv            # CSV 文件读取
import json           # JSON 序列化
import statistics     # 统计函数（median）
import sys            # 系统退出
from pathlib import Path  # 路径操作

# ---- 默认常量 ----
DEFAULT_CSV = "datasets/chinese-poetry-collection/test.csv"   # 默认数据集路径
DEFAULT_TOKENIZER = "models/Qwen2.5-0.5B"                    # 默认 tokenizer 目录


def load_texts(csv_path: str) -> list[str]:
    """从 CSV 数据集加载所有非空文本。

    Args:
        csv_path: CSV 文件路径（需含 text1 列）

    Returns:
        list[str]: 非空文本列表（保持原始行序）
    """
    texts = []
    # utf-8-sig 去掉 BOM（Windows 导出的 CSV 常带）
    with open(csv_path, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)  # 按字典方式读取
        for row in reader:
            text = (row.get("text1") or "").strip()  # 取 text1 列并去空白
            if text:  # 过滤空行
                texts.append(text)
    return texts


def main() -> None:
    """tokenize_batch.py 的主入口函数。

    加载数据集 → 过滤长度区间 → tokenize → 写 JSONL + meta。
    """
    # 创建参数解析器
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default=DEFAULT_CSV, help=f"数据集 csv（默认 {DEFAULT_CSV}）")
    ap.add_argument("--tokenizer", default=DEFAULT_TOKENIZER,
                    help=f"本地 HF 模型目录（默认 {DEFAULT_TOKENIZER}，离线可用）")
    ap.add_argument("--out", required=True, help="输出 JSONL 路径")
    ap.add_argument("--num", type=int, default=16, help="取多少条（默认 16）")
    ap.add_argument("--min-tokens", type=int, default=8, help="token 数下限（默认 8）")
    ap.add_argument("--max-tokens", type=int, default=96,
                    help="token 数上限（默认 96；注意 prompt+decode 要装进 --max-seq-len）")
    args = ap.parse_args()

    # ---- 输入校验 ----
    if not Path(args.csv).is_file():
        sys.exit(f"error: 数据集不存在：{args.csv}")
    if not Path(args.tokenizer).is_dir():
        sys.exit(f"error: tokenizer 目录不存在：{args.tokenizer}（先把 HF 模型下到本地）")
    if args.num <= 0:
        sys.exit("error: --num 必须 > 0")
    if args.min_tokens > args.max_tokens:
        sys.exit("error: --min-tokens 不能大于 --max-tokens")

    # 延迟导入 transformers
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer)  # 加载 tokenizer

    # 加载数据集
    texts = load_texts(args.csv)
    print(f"[tokenize-batch] 数据集 {args.csv}：{len(texts)} 条原始文本")

    # ---- 按行序过滤 token 长度区间 ----
    picked: list[tuple[str, list[int]]] = []  # [(原文, token_ids), ...]
    for text in texts:
        ids = tok(text, add_special_tokens=False)["input_ids"]  # tokenize
        if args.min_tokens <= len(ids) <= args.max_tokens:
            picked.append((text, ids))  # 符合长度区间的加入
        if len(picked) >= args.num:
            break  # 达到数量上限停止
    if not picked:
        sys.exit("error: 过滤后没有符合长度区间的文本，放宽 --min/max-tokens")
    if len(picked) < args.num:
        print(f"  ⚠️ 只凑到 {len(picked)}/{args.num} 条符合 [{args.min_tokens}, "
              f"{args.max_tokens}] 的文本", file=sys.stderr)

    # ---- 写入 JSONL ----
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)  # 确保父目录存在
    with out_path.open("w", encoding="utf-8") as f:
        for text, ids in picked:
            # text 留在行里只为人类可读；C++ 侧只消费 tokens 字段
            f.write(json.dumps({"tokens": ids, "text": text}, ensure_ascii=False) + "\n")

    # ---- 计算并写入 meta ----
    lens = [len(ids) for _, ids in picked]  # 所有选中条目的 token 长度
    meta = {
        "csv": args.csv,                          # 数据来源
        "tokenizer": args.tokenizer,               # 使用的 tokenizer
        "num": len(picked),                        # 实际条数
        "min_tokens": args.min_tokens,             # 长度下限
        "max_tokens": args.max_tokens,             # 长度上限
        "token_len_min": min(lens),                # 最短 token 长度
        "token_len_median": statistics.median(lens),  # 中位 token 长度
        "token_len_max": max(lens),                # 最长 token 长度
    }
    meta_path = out_path.with_suffix(out_path.suffix + ".meta.json")
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    # 打印摘要
    print(f"[tokenize-batch] 已写入 {out_path}（{len(picked)} 条）")
    print(f"  token 长度：min={meta['token_len_min']} median={meta['token_len_median']} "
          f"max={meta['token_len_max']}")
    print(f"  meta：{meta_path}")


# 脚本直接运行入口
if __name__ == "__main__":
    main()
