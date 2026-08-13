#!/usr/bin/env python3
"""用 ModelScope 中文诗词数据集驱动 tinyqwen 做续写演示。

数据集 modelscope/chinese-poetry-collection：单列 text1，每行一首诗。
玩法：取每首诗的前两句作 prompt，让 runtime 续写，再和原诗后两句对照。

用法：
    python tools/run_poetry_demo.py [--num 3] [--max-new-tokens 24] [--csv datasets/chinese-poetry-collection/test.csv]
"""

from __future__ import annotations

import argparse
import csv
import subprocess
from pathlib import Path

MODEL_DIR = "models/Qwen2.5-0.5B"
BIN = "build/runtime/tinyqwen"
MODEL_TQWEN = "model.tqwen"


def load_poems(csv_path: str, num: int) -> list[str]:
    poems = []
    with open(csv_path, encoding="utf-8-sig") as f:  # utf-8-sig 去掉 BOM
        reader = csv.DictReader(f)
        for row in reader:
            text = (row.get("text1") or "").strip()
            if len(text) >= 10:  # 过滤过短的
                poems.append(text)
            if len(poems) >= num:
                break
    return poems


def split_prompt(poem: str) -> tuple[str, str]:
    """按第一个句号拆成 prompt（前两句）和参考（后两句）。"""
    if "。" in poem:
        head, _, tail = poem.partition("。")
        return head + "。", tail
    return poem, ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="datasets/chinese-poetry-collection/test.csv")
    ap.add_argument("--num", type=int, default=3, help="取多少首诗")
    ap.add_argument("--max-new-tokens", type=int, default=24)
    ap.add_argument("--max-seq-len", type=int, default=96)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)

    poems = load_poems(args.csv, args.num)
    print(f"从 {args.csv} 取了 {len(poems)} 首诗\n" + "=" * 60)

    for idx, poem in enumerate(poems):
        prompt, reference = split_prompt(poem)
        ids = tok(prompt, add_special_tokens=False)["input_ids"]
        tokens_csv = ",".join(map(str, ids))

        # 调 C++ runtime 生成续写 token
        cmd = [
            BIN, "--model", MODEL_TQWEN,
            "--tokens", tokens_csv,
            "--max-new-tokens", str(args.max_new_tokens),
            "--max-seq-len", str(args.max_seq_len),
            "--eos", "-1",
        ]
        r = subprocess.run(cmd, capture_output=True, text=True, check=True)

        gen_ids = []
        for line in r.stdout.splitlines():
            if line.startswith("generated_ids:"):
                gen_ids = [int(t) for t in line.split()[1:]]
        gen_text = tok.decode(gen_ids, skip_special_tokens=True)

        print(f"\n【{idx + 1}】prompt（前两句）:\n  {prompt}")
        print(f"模型续写:\n  {gen_text}")
        if reference:
            print(f"原诗后两句:\n  {reference}")
        print("-" * 60)


if __name__ == "__main__":
    main()
