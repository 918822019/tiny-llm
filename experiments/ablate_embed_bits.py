#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ablate_embed_bits.py — embed 位宽消融：把一个 VQ2 .tqwen 的 embed 换成
N-bit fake-quant 后的值（仍以 f16 存盘），只回答"精度还有多少余量"。

背景：Qwen3-0.6B 是 tied embedding，embed 同时充当 lm_head。上游 QAT 把
embed 钉在 4-bit（行内分组 minmax，group_size=64）——那是为了与 PTQ 对照口径
一致而定的默认值，不是测出来的最优点。embed 占部署包 44%，降位宽是唯一还没被
验证过的体积杠杆。

为什么 fake-quant 而不是真的换成 2/3-bit 存储：runtime 只有 i4 一种 embed
量化 dtype，新增 i3/i2 要动内核。而这里要回答的是**精度问题**——把 N-bit
量化误差注入后仍以 f16 存盘，PPL 就已经能反映位宽的代价，不需要新内核。
所以本脚本产出的文件体积不变，只能用来读 PPL，不能用来读体积或速度。

量化方案与上游一致：逐行分组、非对称 minmax，
    scale = (max - min) / (2^bits - 1)，zero = round(-min/scale)，
    反量化 = (clip(round(w/scale + zero)) - zero) * scale

用法:
    python experiments/ablate_embed_bits.py --in model_qwen3_06b_vq2_pack.tqwen \\
        --bits 2 --out /tmp/ablate_e2.tqwen [--group-size 64]
"""

import argparse
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
from export_qwen_to_tiny import (  # noqa: E402
    HEADER_FMT, ENTRY_FMT, DTYPE_F16, DTYPE_F32,
)

EMBED_NAME = "model.embed_tokens.weight"


def fake_quant_rowgroup(w: np.ndarray, bits: int, gs: int) -> np.ndarray:
    """逐行分组非对称 minmax 的 fake-quant（量化再反量化，形状不变）。"""
    rows, cols = w.shape
    if cols % gs:
        raise SystemExit(f"error: cols {cols} 不能被 group_size {gs} 整除")
    qmax = float((1 << bits) - 1)
    blocks = w.astype(np.float32).reshape(rows, cols // gs, gs)
    vmin = blocks.min(axis=2, keepdims=True)
    vmax = blocks.max(axis=2, keepdims=True)
    scale = (vmax - vmin) / qmax
    degenerate = scale < 1e-12          # 常量组：直接保留原值，避免除零
    safe = np.where(degenerate, 1.0, scale)
    zero = np.clip(np.round(-vmin / safe), 0.0, qmax)
    q = np.clip(np.round(blocks / safe + zero), 0.0, qmax)
    deq = (q - zero) * safe
    return np.where(degenerate, blocks, deq).reshape(rows, cols)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in", dest="src", required=True, help="输入 .tqwen（embed 须为 f16/f32）")
    p.add_argument("--out", required=True, help="输出 .tqwen")
    p.add_argument("--bits", type=int, required=True, choices=[2, 3, 4, 5, 6, 8])
    p.add_argument("--group-size", type=int, default=64)
    args = p.parse_args()

    src = Path(args.src)
    raw = bytearray(src.read_bytes())
    hdr = struct.unpack(HEADER_FMT, bytes(raw[: struct.calcsize(HEADER_FMT)]))
    n_tensors, table_off = hdr[15], hdr[16]
    esz = struct.calcsize(ENTRY_FMT)

    target = None
    for i in range(n_tensors):
        off = table_off + i * esz
        name_b, dt, nd, s0, s1, s2, s3, doff, nbytes = struct.unpack(
            ENTRY_FMT, bytes(raw[off:off + esz]))
        if name_b.rstrip(b"\x00").decode() == EMBED_NAME:
            target = (dt, (s0, s1), doff, nbytes)
            break
    if target is None:
        raise SystemExit(f"error: 找不到 {EMBED_NAME}")
    dt, (rows, cols), doff, nbytes = target
    if dt not in (DTYPE_F16, DTYPE_F32):
        raise SystemExit(f"error: embed dtype={dt} 非 f16/f32，本消融需要未量化存储的 embed")

    np_dt = np.float16 if dt == DTYPE_F16 else np.float32
    w = np.frombuffer(bytes(raw[doff:doff + nbytes]), dtype=np_dt).reshape(rows, cols)
    deq = fake_quant_rowgroup(w, args.bits, args.group_size)
    err = float(np.abs(deq - w.astype(np.float32)).max())
    rel = float(np.linalg.norm(deq - w.astype(np.float32)) / np.linalg.norm(w.astype(np.float32)))
    raw[doff:doff + nbytes] = np.ascontiguousarray(deq.astype(np_dt)).tobytes()
    Path(args.out).write_bytes(bytes(raw))
    print(f"embed {rows}x{cols} fake-quant {args.bits}-bit gs={args.group_size}: "
          f"max_abs_err={err:.6f} rel_err={rel:.4f}")
    print(f"写出 {args.out}（体积不变，仅供读 PPL）")


if __name__ == "__main__":
    main()
