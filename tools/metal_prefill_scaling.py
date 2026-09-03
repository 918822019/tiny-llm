#!/usr/bin/env python3
"""探针：Qwen3-0.6B MPS prefill 随 seq 的扩展性 —— 算力瓶颈还是开销/带宽瓶颈？"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM


def log(m: str) -> None:
    print(m, flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/Qwen3-0.6B")
    ap.add_argument("--seqs", default="16,32,64,128,256,512")
    ap.add_argument("--reps", type=int, default=8)
    args = ap.parse_args()

    if not torch.backends.mps.is_available():
        log("MPS 不可用")
        return 1

    vocab = 151936
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16).to("mps").eval()

    proj_per_tok = 2 * 0.44e9
    lmhead_per_tok = 2 * 1024 * vocab
    attn_per_seq2 = 28 * 4 * 128 * 16

    log("Qwen3-0.6B MPS prefill 扩展性 (fp16)")
    log(f"{'seq':>5} {'ms':>9} {'ms/tok':>8} {'GFLOPs':>9} {'impl TFLOP/s':>13}")
    for seq in [int(s) for s in args.seqs.split(",")]:
        ids = torch.randint(0, vocab, (1, seq), dtype=torch.long, device="mps")
        for _ in range(3):
            with torch.no_grad():
                model(input_ids=ids, use_cache=False)
        torch.mps.synchronize()
        ts = []
        for _ in range(args.reps):
            t = time.time()
            with torch.no_grad():
                model(input_ids=ids, use_cache=False)
            torch.mps.synchronize()
            ts.append((time.time() - t) * 1000)
        med = float(np.median(ts))
        flops = (proj_per_tok + lmhead_per_tok) * seq + attn_per_seq2 * seq * seq
        impl = flops / (med / 1000) / 1e12
        log(f"{seq:>5} {med:>9.2f} {med / seq:>8.3f} {flops / 1e9:>9.1f} {impl:>13.2f}")
    del model
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
