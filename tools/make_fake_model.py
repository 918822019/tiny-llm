#!/usr/bin/env python3
"""生成随机权重的小 .tqwen 文件，用于 C++ runtime 的冒烟测试。

配置刻意很小（hidden=16，2 层），让 loader、forward、KV cache、profiler
整条链路在毫秒级跑完。权重是随机的，生成结果没有语义，只用于验证
管道和二进制格式。

用法:
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

    # 与真实 Qwen2.5 同构但极小的配置（GQA：4 个 q head 共享 2 个 kv head）。
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
        # RMSNorm 权重取 1（中性缩放）：让激活保持在良态范围内，
        # 对齐测试检验的是数学实现，而不是溢出行为。
        tensors[pfx + "input_layernorm.weight"] = np.ones(H, dtype=np.float32)
        tensors[pfx + "self_attn.q_proj.weight"] = w((qd, H))
        tensors[pfx + "self_attn.k_proj.weight"] = w((kvd, H))
        tensors[pfx + "self_attn.v_proj.weight"] = w((kvd, H))
        # Qwen2/2.5 的 attention 带 q/k/v bias（attention_bias=True）。
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
