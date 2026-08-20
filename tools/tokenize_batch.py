#!/usr/bin/env python3
"""把数据集批量 tokenize 成 tinyqwen 批量模式的输入（JSONL）。

数据来源默认 ModelScope 中文诗词数据集（datasets/chinese-poetry-collection，
单列 text1，每行一首诗）。整段 text1 作为 prompt——续写任务口径，
add_special_tokens=False、不走 chat template（与 run_poetry_demo.py 一致）。

选择是**确定性**的：按 csv 行序过滤 token 长度区间后取前 N 条，不 shuffle——
两次运行产出完全相同的输入，测量才可复现。

输出：
- <out>         JSONL，每行 {"tokens": [...]}，供 tinyqwen --batch-tokens-jsonl
- <out>.meta.json  条数与 token 长度分布（bench_dataset.py 读取记录进结果）

用法：
    python tools/tokenize_batch.py --num 16 --out benchmarks/poetry16.jsonl
    python tools/tokenize_batch.py --csv datasets/.../train.csv --num 64 \
        --min-tokens 8 --max-tokens 96 --out /tmp/poetry64.jsonl
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

DEFAULT_CSV = "datasets/chinese-poetry-collection/test.csv"
DEFAULT_TOKENIZER = "models/Qwen2.5-0.5B"


def load_texts(csv_path: str) -> list[str]:
    texts = []
    with open(csv_path, encoding="utf-8-sig") as f:  # utf-8-sig 去掉 BOM
        reader = csv.DictReader(f)
        for row in reader:
            text = (row.get("text1") or "").strip()
            if text:
                texts.append(text)
    return texts


def main() -> None:
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

    if not Path(args.csv).is_file():
        sys.exit(f"error: 数据集不存在：{args.csv}")
    if not Path(args.tokenizer).is_dir():
        sys.exit(f"error: tokenizer 目录不存在：{args.tokenizer}（先把 HF 模型下到本地）")
    if args.num <= 0:
        sys.exit("error: --num 必须 > 0")
    if args.min_tokens > args.max_tokens:
        sys.exit("error: --min-tokens 不能大于 --max-tokens")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer)

    texts = load_texts(args.csv)
    print(f"[tokenize-batch] 数据集 {args.csv}：{len(texts)} 条原始文本")

    picked: list[tuple[str, list[int]]] = []
    for text in texts:
        ids = tok(text, add_special_tokens=False)["input_ids"]
        if args.min_tokens <= len(ids) <= args.max_tokens:
            picked.append((text, ids))
        if len(picked) >= args.num:
            break
    if not picked:
        sys.exit("error: 过滤后没有符合长度区间的文本，放宽 --min/max-tokens")
    if len(picked) < args.num:
        print(f"  ⚠️ 只凑到 {len(picked)}/{args.num} 条符合 [{args.min_tokens}, "
              f"{args.max_tokens}] 的文本", file=sys.stderr)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for text, ids in picked:
            # text 留在行里只为人类可读；C++ 侧只消费 tokens 字段。
            f.write(json.dumps({"tokens": ids, "text": text}, ensure_ascii=False) + "\n")

    lens = [len(ids) for _, ids in picked]
    meta = {
        "csv": args.csv,
        "tokenizer": args.tokenizer,
        "num": len(picked),
        "min_tokens": args.min_tokens,
        "max_tokens": args.max_tokens,
        "token_len_min": min(lens),
        "token_len_median": statistics.median(lens),
        "token_len_max": max(lens),
    }
    meta_path = out_path.with_suffix(out_path.suffix + ".meta.json")
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[tokenize-batch] 已写入 {out_path}（{len(picked)} 条）")
    print(f"  token 长度：min={meta['token_len_min']} median={meta['token_len_median']} "
          f"max={meta['token_len_max']}")
    print(f"  meta：{meta_path}")


if __name__ == "__main__":
    main()
