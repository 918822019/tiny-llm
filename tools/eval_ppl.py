#!/usr/bin/env python3
"""tiny-llm 困惑度（PPL）评测编排。

把数据集准备与 PPL 计算解耦：
  - PPL 计算走 tiny-llm 的 `--ppl` 模式（批量 prefill + 全位置 lm_head + 交叉熵，
    一次进程加载模型、逐块 reset + forward_ppl）。
  - 数据准备支持两种来源，互不依赖：
      a) **预分词 token**（首选，最稳）：--tokens-npy / --tokens-json，
         可在任何有 tokenizer 的机器上先分好词再拿来评测；
      b) **现场 tokenize**：--text/--wikitext + --tokenizer <HF 模型目录>
         （需要本机装有 transformers 与对应模型/数据）。

产出：NLL、PPL、计分 token 数；可选 --json 落盘，便于多模型对比。

用法示例：
    # 用预分词 token 评测（推荐）
    python tools/eval_ppl.py --model model_qwen25_vq2.tqwen \
        --tokens-npy benchmarks/wikitext_test_tokens.npy --chunk-len 1024 \
        --matvec-impl neon --label vq2-rot

    # 现场 tokenize（需要 transformers + 模型 + 数据）
    python tools/eval_ppl.py --model model.tqwen --wikitext --split test \
        --tokenizer models/Qwen2.5-0.5B --chunk-len 1024

    # 只准备分块（不跑），供别处使用
    python tools/eval_ppl.py --tokens-json toks.json --chunk-len 512 --prepare-only \
        --chunks-jsonl /tmp/chunks.jsonl

注意：
  - `--ppl` 的 forward_prefill 只支持 Qwen2.x（Qwen2.5）；Qwen3.5 需另走其批量路径。
  - 分块按不重叠窗口独立计分（标准做法），块间有轻微边界效应。
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


def load_tokens(args) -> list[int]:
    """按输入来源加载/生成长串 token 序列。"""
    if args.tokens_npy:
        import numpy as np
        arr = np.load(args.tokens_npy)
        return [int(x) for x in arr.reshape(-1)]
    if args.tokens_json:
        data = json.loads(Path(args.tokens_json).read_text())
        flat: list[int] = []
        for item in data:
            if isinstance(item, list):
                flat.extend(int(x) for x in item)
            else:
                flat.append(int(item))
        return flat
    if args.text:
        return tokenize_texts([Path(args.text).read_text(errors="ignore")], args.tokenizer)
    if args.wikitext:
        return tokenize_texts(load_wikitext(args.split), args.tokenizer)
    raise SystemExit("error: 需指定 --tokens-npy / --tokens-json / --text / --wikitext 之一")


def tokenize_texts(texts: list[str], tokenizer_dir: str) -> list[int]:
    try:
        from transformers import AutoTokenizer
    except Exception as e:  # noqa: BLE001
        raise SystemExit(f"error: 现场 tokenize 需要 transformers（{e}）。"
                         f"可改用 --tokens-npy/--tokens-json 预分词输入。")
    if not tokenizer_dir:
        raise SystemExit("error: --text/--wikitext 需要 --tokenizer <HF 模型目录>")
    tok = AutoTokenizer.from_pretrained(tokenizer_dir, trust_remote_code=True, use_fast=True)
    ids: list[int] = []
    for t in texts:
        if not t.strip():
            continue
        ids.extend(tok(t, add_special_tokens=False)["input_ids"])
    return ids


def load_wikitext(split: str) -> list[str]:
    try:
        from datasets import load_dataset
    except Exception as e:  # noqa: BLE001
        raise SystemExit(f"error: --wikitext 需要 datasets 库（{e}）。"
                         f"可先 `pip install datasets`，或改用 --tokens-npy/--tokens-json。")
    name = "wikitext" if split != "raw" else "wikitext"
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split=split)
    return [row["text"] for row in ds if row.get("text", "").strip()]


def chunk_tokens(tokens: list[int], chunk_len: int) -> list[list[int]]:
    """切不重叠窗口；过短的尾巴并入最后一块（仍 < 2 则丢弃）。"""
    chunks = [tokens[i:i + chunk_len] for i in range(0, len(tokens), chunk_len)]
    chunks = [c for c in chunks if len(c) >= 2]
    return chunks


def write_chunks_jsonl(chunks: list[list[int]], path: str) -> None:
    with open(path, "w") as f:
        for c in chunks:
            f.write(json.dumps({"tokens": c}) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", help=".tqwen 模型文件")
    ap.add_argument("--binary", default="build/runtime/tinyqwen")
    # token 来源
    ap.add_argument("--tokens-npy")
    ap.add_argument("--tokens-json")
    ap.add_argument("--text")
    ap.add_argument("--wikitext", action="store_true")
    ap.add_argument("--split", default="test", help="wikitext split: test/validation/train")
    ap.add_argument("--tokenizer", help="HF 模型目录（现场 tokenize 用）")
    # 分块与运行
    ap.add_argument("--chunk-len", type=int, default=1024)
    ap.add_argument("--max-seq-len", type=int, default=0, help="默认 = chunk-len")
    ap.add_argument("--matvec-impl", default="")
    ap.add_argument("--label", default="")
    ap.add_argument("--chunks-jsonl", default="", help="分块文件路径（默认临时文件）")
    ap.add_argument("--prepare-only", action="store_true", help="只准备分块，不运行")
    ap.add_argument("--json", default="", help="把结果写入 JSON")
    args = ap.parse_args()

    chunks_path = args.chunks_jsonl or os.path.join(tempfile.gettempdir(), "ppl_chunks.jsonl")

    # 若已给定 chunks 文件且 prepare-only 之外无需重新生成，可直接复用
    if args.chunks_jsonl and os.path.exists(args.chunks_jsonl) and not (
            args.tokens_npy or args.tokens_json or args.text or args.wikitext):
        n_chunks = sum(1 for _ in open(args.chunks_jsonl))
        print(f"[prep] 复用已有分块 {args.chunks_jsonl}（{n_chunks} 块）")
    else:
        tokens = load_tokens(args)
        chunks = chunk_tokens(tokens, args.chunk_len)
        if not chunks:
            raise SystemExit("error: 无可用分块（token 太少？）")
        write_chunks_jsonl(chunks, chunks_path)
        print(f"[prep] {len(tokens)} tokens → {len(chunks)} 块 × ≤{args.chunk_len} → {chunks_path}")
        if args.prepare_only:
            return

    if not args.model:
        raise SystemExit("error: 运行需要 --model")
    max_seq = args.max_seq_len or args.chunk_len

    cmd = [args.binary, "--model", args.model, "--ppl", "--ppl-jsonl", chunks_path,
           "--max-seq-len", str(max_seq)]
    if args.matvec_impl:
        cmd += ["--matvec-impl", args.matvec_impl]
    print(f"[run] {' '.join(cmd)}")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    sys.stderr.write(proc.stderr)
    out = {}
    for line in proc.stdout.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[0] in ("ppl_nll", "ppl", "ppl_tokens"):
            out[parts[0]] = float(parts[1]) if parts[0] != "ppl_tokens" else int(float(parts[1]))
    if proc.returncode != 0 or "ppl" not in out:
        raise SystemExit(f"error: tiny-llm --ppl 失败（rc={proc.returncode}）")

    label = args.label or Path(args.model).stem
    print("\n════ PPL 结果 ════")
    print(f"  model : {args.model}  ({label})")
    print(f"  NLL   : {out.get('ppl_nll', float('nan')):.6f}")
    print(f"  PPL   : {out.get('ppl', float('nan')):.4f}")
    print(f"  tokens: {out.get('ppl_tokens', 0)}")

    if args.json:
        rec = {"label": label, "model": args.model, "chunk_len": args.chunk_len,
               "matvec_impl": args.matvec_impl, **out}
        with open(args.json, "a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"  [json] 追加到 {args.json}")


if __name__ == "__main__":
    main()
