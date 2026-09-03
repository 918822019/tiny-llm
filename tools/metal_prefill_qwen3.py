#!/usr/bin/env python3
"""Qwen3-0.6B prefill 在 Apple GPU(Metal/MPS) vs CPU 的测速 —— Metal 路可行性 + 基线。"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM


def log(m: str) -> None:
    print(m, flush=True)


def prefill_ms(model, ids, reps: int, warmup: int = 3):
    is_mps = model.device.type == "mps"
    for _ in range(warmup):
        with torch.no_grad():
            model(input_ids=ids, use_cache=False)
    if is_mps:
        torch.mps.synchronize()
    ts = []
    for _ in range(reps):
        t = time.time()
        with torch.no_grad():
            model(input_ids=ids, use_cache=False)
        if is_mps:
            torch.mps.synchronize()
        ts.append((time.time() - t) * 1000)
    return float(np.median(ts)), ts


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/Qwen3-0.6B")
    ap.add_argument("--seq", type=int, default=128)
    ap.add_argument("--reps", type=int, default=10)
    args = ap.parse_args()

    if not torch.backends.mps.is_available():
        log("MPS 不可用，无法测 Apple GPU")
        return 1

    vocab = 151936
    ids = torch.randint(0, vocab, (1, args.seq), dtype=torch.long)
    log(f"Qwen3-0.6B prefill 测速: seq={args.seq} reps={args.reps}")

    results = {}
    for dev, dt in [("cpu", torch.float32), ("mps", torch.float16)]:
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dt).to(dev).eval()
        x = ids.to(dev)
        med, ts = prefill_ms(model, x, args.reps)
        results[dev] = med
        runs = ", ".join(f"{t:.0f}" for t in ts)
        log(f"  {dev.upper():4s} ({str(dt).replace('torch.','')}): median {med:7.2f} ms "
            f"= {med / args.seq:.3f} ms/tok   runs=[{runs}]")
        del model

    if "cpu" in results and "mps" in results:
        log(f"  => Apple GPU(Metal) / CPU 加速比: {results['cpu'] / results['mps']:.2f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
