#!/usr/bin/env python3
"""dump PyTorch/HF 参考激活值，用于 tinyqwen C++ runtime 的数值对齐。

对固定的 token 序列（单次前向、eager attention、fp32、cpu）dump：
    embed_out                     [seq, hidden]
    layer_0_attn_norm             input_layernorm 输出              [seq, hidden]
    layer_0_q / k / v             q/k/v_proj 输出（RoPE 之前）      [seq, n_heads*head_dim] 等
    layer_0_attn_out              self_attn 输出（o_proj 之后）     [seq, hidden]
    layer_0_post_attn_residual    post_attention_layernorm 的输入   [seq, hidden]
    layer_0_ffn_norm              post_attention_layernorm 输出     [seq, hidden]
    layer_0_gate / layer_0_up     mlp gate/up 投影输出              [seq, inter]
    layer_0_ffn_out               mlp down_proj 输出                [seq, hidden]
    layer_0_output                decoder layer 0 输出              [seq, hidden]
    final_norm                    model.norm 输出                   [seq, hidden]
    logits                        [seq, vocab]
    topk_values / topk_indices    最后一个位置的 top-k              [k]

旁边的 <out>.meta.json 记录复现本次运行所需的全部信息
（tokens、dtype、attention 实现等）。对齐流程见 docs/pytorch_alignment.md。

用法:
    python tools/dump_qwen_reference.py \
        --model Qwen/Qwen2.5-0.5B \
        --tokens-json prompt_tokens.json \
        --out ref.npz --topk 10
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, help="HF model dir or repo id")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--tokens-json", help="JSON produced by tools/tokenize_prompt.py")
    src.add_argument("--tokens", help="comma separated token ids, e.g. 1,2,3")
    p.add_argument("--out", default="ref.npz")
    p.add_argument("--topk", type=int, default=10)
    p.add_argument("--layer", type=int, default=0,
                   help="which layer to dump in detail (default 0)")
    return p.parse_args()


def load_tokens(args: argparse.Namespace) -> list[int]:
    if args.tokens:
        return [int(t) for t in args.tokens.split(",") if t.strip()]
    data = json.loads(Path(args.tokens_json).read_text())
    return [int(t) for t in data["tokens"]]


def main() -> None:
    args = parse_args()
    tokens = load_tokens(args)
    print(f"tokens ({len(tokens)}): {tokens}")

    import torch
    from transformers import AutoModelForCausalLM

    # 确定性参考实现：fp32、cpu、eager attention、eval 模式。
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float32,
        attn_implementation="eager",
    ).eval()

    L = args.layer
    lm = model.model
    captured: dict[str, np.ndarray] = {}

    def save(name: str, tensor: torch.Tensor) -> None:
        captured[name] = tensor.detach().float().cpu().numpy()

    hooks = []

    # reg：捕获子模块的输出。HF 的 attention / decoder layer 返回元组，
    # 这里统一解包取 hidden-state 张量。
    def reg(module: torch.nn.Module, name: str) -> None:
        hooks.append(module.register_forward_hook(
            lambda _m, _i, out, n=name: save(n, out[0] if isinstance(out, tuple) else out)))

    # reg_pre：捕获子模块的输入——用于取 post-attention 残差状态，
    # 它恰好就是 post_attention_layernorm 的输入。
    def reg_pre(module: torch.nn.Module, name: str) -> None:
        hooks.append(module.register_forward_pre_hook(
            lambda _m, inp, n=name: save(n, inp[0])))

    # hook 的位置与 docs/qwen_forward.md 的 op 一一对应；dump 的 key 名
    # 是 docs/pytorch_alignment.md 使用的契约。
    # 注意：q/k/v dump 的是 RoPE 之前的值（投影输出），与 C++ 侧
    # q/k/v 投影刚结束时的 buffer 状态对应。
    layer0 = lm.layers[L]
    save("embed_out", lm.embed_tokens(torch.tensor(tokens)))
    reg(layer0.input_layernorm, f"layer_{L}_attn_norm")
    reg(layer0.self_attn.q_proj, f"layer_{L}_q")
    reg(layer0.self_attn.k_proj, f"layer_{L}_k")
    reg(layer0.self_attn.v_proj, f"layer_{L}_v")
    reg(layer0.self_attn, f"layer_{L}_attn_out")
    reg_pre(layer0.post_attention_layernorm, f"layer_{L}_post_attn_residual")
    reg(layer0.post_attention_layernorm, f"layer_{L}_ffn_norm")
    reg(layer0.mlp.gate_proj, f"layer_{L}_gate")
    reg(layer0.mlp.up_proj, f"layer_{L}_up")
    reg(layer0.mlp.down_proj, f"layer_{L}_ffn_out")
    reg(layer0, f"layer_{L}_output")
    reg(lm.norm, "final_norm")

    input_ids = torch.tensor([tokens], dtype=torch.long)
    with torch.no_grad():
        # 显式 position_ids 保证与 C++ runtime 可比：C++ 侧
        # pos == KV cache 长度（0 起始，调度完全一致）。
        position_ids = torch.arange(len(tokens), dtype=torch.long).unsqueeze(0)
        out = model(input_ids=input_ids, position_ids=position_ids, use_cache=False)

    for h in hooks:
        h.remove()

    logits = out.logits[0]  # [seq, vocab]
    save("logits", logits)
    top = torch.topk(logits[-1], args.topk)
    captured["topk_values"] = top.values.numpy()
    captured["topk_indices"] = top.indices.numpy()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, **captured)

    meta = {
        "model": args.model,
        "tokens": tokens,
        "n_tokens": len(tokens),
        "dtype": "float32",
        "device": "cpu",
        "attn_implementation": "eager",
        "layer_dumped": L,
        "topk": args.topk,
        "note": "single forward pass with explicit position_ids = arange; "
                "compare C++ token-by-token outputs at matching positions",
    }
    meta_path = out_path.with_suffix(out_path.suffix + ".meta.json")
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n")

    print(f"saved {len(captured)} tensors -> {out_path}")
    print(f"meta -> {meta_path}")
    print("last-position top-k:",
          list(zip(captured['topk_indices'].tolist(),
                   np.round(captured['topk_values'], 4).tolist())))


if __name__ == "__main__":
    main()
