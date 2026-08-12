#!/usr/bin/env python3
"""Generate a small random-weight .tqwen file for smoke-testing the C++ runtime.

The config is tiny (hidden=16, 2 layers) so the whole loop — loader, forward,
KV cache, profiler — runs in milliseconds with random weights. This does NOT
produce meaningful text; it validates plumbing and the binary format.

Usage:
    python tools/make_fake_model.py --out /tmp/fake.tqwen [--seed 0]
"""

from __future__ import annotations

import argparse

import numpy as np

from export_qwen_to_tiny import write_tqwen


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="fake.tqwen")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tied", type=int, default=1)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)

    cfg = {
        "n_layers": 2,
        "hidden_size": 16,
        "intermediate_size": 32,
        "n_heads": 4,
        "n_kv_heads": 2,
        "head_dim": 4,
        "vocab_size": 64,
        "max_seq_len": 32,
        "tied": args.tied,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
    }
    H, I = cfg["hidden_size"], cfg["intermediate_size"]
    qd = cfg["n_heads"] * cfg["head_dim"]
    kvd = cfg["n_kv_heads"] * cfg["head_dim"]
    V = cfg["vocab_size"]

    def w(shape):
        return (rng.standard_normal(shape) * 0.1).astype(np.float32)

    tensors = {"model.embed_tokens.weight": w((V, H))}
    for i in range(cfg["n_layers"]):
        pfx = f"model.layers.{i}."
        tensors[pfx + "input_layernorm.weight"] = np.ones(H, dtype=np.float32)
        tensors[pfx + "self_attn.q_proj.weight"] = w((qd, H))
        tensors[pfx + "self_attn.k_proj.weight"] = w((kvd, H))
        tensors[pfx + "self_attn.v_proj.weight"] = w((kvd, H))
        # Qwen2/2.5 attention has q/k/v biases (attention_bias=True).
        tensors[pfx + "self_attn.q_proj.bias"] = w((qd,))
        tensors[pfx + "self_attn.k_proj.bias"] = w((kvd,))
        tensors[pfx + "self_attn.v_proj.bias"] = w((kvd,))
        tensors[pfx + "self_attn.o_proj.weight"] = w((H, qd))
        tensors[pfx + "post_attention_layernorm.weight"] = np.ones(H, dtype=np.float32)
        tensors[pfx + "mlp.gate_proj.weight"] = w((I, H))
        tensors[pfx + "mlp.up_proj.weight"] = w((I, H))
        tensors[pfx + "mlp.down_proj.weight"] = w((H, I))
    tensors["model.norm.weight"] = np.ones(H, dtype=np.float32)
    if not args.tied:
        tensors["lm_head.weight"] = w((V, H))

    total = write_tqwen(args.out, cfg, tensors)
    print(f"wrote {args.out}: {total} bytes, {len(tensors)} tensors")


if __name__ == "__main__":
    main()
