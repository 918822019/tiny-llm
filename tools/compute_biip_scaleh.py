#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""compute_biip_scaleh.py — 复算 BiIP 旋转的 scaleH，供 QAT 桥接导出器使用。

`tools/export_qat_vq2_to_tiny.py` 需要 QAT 上游用过的旋转参数。其中：

  - **sign 是确定的**：`seed = layer_idx * 100 + SUBLAYER_NAMES.index(sublayer)`，
    由 `torch.randint(0, 2, (in_features,), generator=manual_seed(seed))` 生成，
    可直接复现，无需数据。
  - **scaleH 依赖校准数据**，公式（kronq/scaleh.py）：

        diagH  = mean_over_tokens(x²)          # 未旋转模型的子层输入二阶矩
        diagW2 = (W ** 2).sum(dim=0)           # 原始 HF 权重的列能量
        scale  = (diagH / diagW2) ** 0.25
        scale /= scale.mean()                  # 归一化到均值 1
        若 diagH.max() / diagH.median() > threshold 则中和（scale = None）

本脚本用同一套校准 token 复算 diagH 与 scaleH，落盘 npz。相比在导出器里
按最小二乘反解 scaleH（会在退化列上产生 1e3~1e4 量级的离群估计），复算得到的是
上游真正用过的值。

用法:
    python tools/compute_biip_scaleh.py \
        --model models/Qwen3-0.6B \
        --calib <kronq>/.calib_cache/wikitext_n128_s512_seed42.pt \
        --out vq_qat_qwen3_06b/scaleh_qwen3_06b.npz
"""

import argparse
import time

import numpy as np
import torch

# 子层顺序即 sign 的 seed 编号来源，必须与 kronq/configs.py 的 SUBLAYER_NAMES 一致
SUBLAYER_NAMES = ["q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj"]


def sublayer_of(layer, name):
    """取 HF Qwen 层里的某个线性子层。"""
    if name in ("q_proj", "k_proj", "v_proj", "o_proj"):
        return getattr(layer.self_attn, name)
    return getattr(layer.mlp, name)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="HF 模型目录（未旋转的原始权重）")
    p.add_argument("--calib", required=True, help="校准 token 张量 .pt（[n_samples, seq_len]）")
    p.add_argument("--out", required=True, help="输出 npz")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--threshold", type=float, default=6500.0,
                   help="diagH 跨度阈值，超过则该子层中和（与 kronq 默认一致）")
    p.add_argument("--device", default="cpu", help="cpu / mps")
    args = p.parse_args()

    from transformers import AutoModelForCausalLM

    calib = torch.load(args.calib, map_location="cpu", weights_only=True)
    if calib.dim() != 2:
        raise SystemExit(f"error: 校准张量应为 [n_samples, seq_len]，实际 {tuple(calib.shape)}")
    print(f"校准数据: {tuple(calib.shape)}  id 范围 [{int(calib.min())}, {int(calib.max())}]")

    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32)
    model.eval().to(args.device)
    vocab = int(model.config.vocab_size)
    if int(calib.max()) >= vocab:
        raise SystemExit(f"error: 校准 token id 超出词表（{int(calib.max())} >= {vocab}），"
                         f"校准缓存与该模型的 tokenizer 不匹配")
    layers = model.model.layers
    print(f"模型: {len(layers)} 层 × {len(SUBLAYER_NAMES)} 子层，device={args.device}")

    # 一次前向收集所有子层输入的平方和（流式累加，不驻留激活）
    sums: dict[tuple[int, str], torch.Tensor] = {}
    counts: dict[tuple[int, str], int] = {}
    handles = []

    def make_hook(key):
        def hook(module, inp):
            x = inp[0].detach()
            if x.dim() == 3:
                x = x.reshape(-1, x.shape[-1])
            sq = (x.float() ** 2).sum(dim=0).cpu()
            if key in sums:
                sums[key] += sq
                counts[key] += x.shape[0]
            else:
                sums[key] = sq
                counts[key] = x.shape[0]
        return hook

    for li in range(len(layers)):
        for sl in SUBLAYER_NAMES:
            handles.append(sublayer_of(layers[li], sl).register_forward_pre_hook(
                make_hook((li, sl))))

    t0 = time.time()
    n_batches = (calib.shape[0] + args.batch_size - 1) // args.batch_size
    with torch.no_grad():
        for bi, s in enumerate(range(0, calib.shape[0], args.batch_size)):
            model(calib[s:s + args.batch_size].to(args.device), use_cache=False)
            print(f"  batch {bi + 1}/{n_batches}  {time.time() - t0:.0f}s", end="\r", flush=True)
    for h in handles:
        h.remove()
    print(f"\ndiagH 收集完成 {time.time() - t0:.0f}s")

    out: dict[str, np.ndarray] = {}
    spans, n_neutral = [], 0
    for li in range(len(layers)):
        for sl in SUBLAYER_NAMES:
            diag_h = (sums[(li, sl)] / counts[(li, sl)]).double()
            span = float(diag_h.max() / diag_h.median())
            spans.append(span)
            key = f"{li}.{sl}"
            out[key + ".span"] = np.array([span])
            if span > args.threshold:
                n_neutral += 1
                continue  # 中和：不写 scale，导出器据此按 scale=None 处理
            w = sublayer_of(layers[li], sl).weight.data.double().cpu()
            diag_w2 = (w ** 2).sum(dim=0).clamp(min=1e-8)
            scale = (diag_h.clamp(min=1e-8) / diag_w2).pow(0.25)
            scale = scale / scale.mean()
            out[key + ".scale"] = scale.float().numpy()

    sp = np.array(spans)
    print(f"diagH 跨度: min={sp.min():.1f} median={np.median(sp):.1f} max={sp.max():.1f}")
    print(f"中和子层数（span > {args.threshold:g}）: {n_neutral} / {len(spans)}")
    np.savez(args.out, **out)
    print(f"写出 {args.out}（{len(out)} 个数组）")


if __name__ == "__main__":
    main()
