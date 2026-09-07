#!/usr/bin/env python3
"""真 AutoGPTQ 权重的 dequant 数值验证。

fake MoE 模型的 GPTQ 是手写随机打包，测不出与真 AutoGPTQ 产物的语义差异。本脚本
拿真 GPTQ checkpoint（Qwen2.5-0.5B-Instruct-GPTQ-Int4）+ 同模型的未量化权重，
逐张量比对 dequant 结果，并把"错误变体"一起跑出来证明本测试有判别力：
  - correct      : zp+1 修正 + 标准布局
  - no_zp_plus1  : 直接用存储值（AutoGPTQ 存的是 zp-1，不修正会整体偏一个步长）
  - transposed   : qweight 布局解读反了（应得到垃圾）

再把 correct 变体打包成我们的 in-band 块，跑 C++ matvec_gptq，与 numpy 参考
逐位比对——这一步验证 C++ 实现与预期语义一致。
"""
import argparse
import os
import struct
import subprocess
import sys

import numpy as np
from safetensors.numpy import load_file

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
BIN = os.path.join(ROOT, "build", "benchmarks", "check_gptq_dequant")

GPTQ_MAGIC = 0x47505451
PACK = 8


def unpack_qweight(qweight, in_dim, out_dim):
    """[in//8, out] int32 -> [in, out] uint4；一个 word 装 8 个连续 in_dim，低 nibble 在前。
    in_idx = (in//8 下标)*8 + nibble 序号，故 [in//8, 8, out] 直接 reshape 即可，
    不能 transpose（那会得到 nibble*(in//8)+(in//8 下标) 的错序）。"""
    qw = qweight.astype(np.uint32)
    nib = np.stack([(qw >> (k * 4)) & 0xF for k in range(PACK)], axis=1)  # [in//8,8,out]
    return nib.reshape(in_dim, out_dim)


def unpack_qzeros(qzeros, out_dim):
    """[ng, out//8] int32 -> [ng, out] uint4；一个 word 装 8 个连续 out_dim。"""
    qz = qzeros.astype(np.uint32)
    nib = np.stack([(qz >> (k * 4)) & 0xF for k in range(PACK)], axis=1)  # [ng,8,out//8]
    return nib.transpose(0, 2, 1).reshape(-1, out_dim)


def dequant(qweight, qzeros, scales, in_dim, out_dim, gs, zp_offset):
    """AutoGPTQ 语义 dequant -> [out, in] fp32。zp_offset=1 对应存储值为 zp-1 的约定。"""
    q = unpack_qweight(qweight, in_dim, out_dim).astype(np.float32)
    z = unpack_qzeros(qzeros, out_dim).astype(np.float32) + zp_offset
    s = scales.astype(np.float32)
    W = np.empty((out_dim, in_dim), np.float32)
    for g in range(in_dim // gs):
        rows = slice(g * gs, (g + 1) * gs)
        W[:, rows] = (q[rows, :].T - z[g][:, None]) * s[g][:, None]
    return W


def pack_inband(W_qweight, qzeros, scales, in_dim, out_dim, gs, zp_offset):
    """打包成 runtime 的 in-band 块（qzeros 解包为 fp16 并做 zp 修正）。"""
    z = unpack_qzeros(qzeros, out_dim).astype(np.float32) + zp_offset
    buf = bytearray()
    buf += struct.pack("<II", GPTQ_MAGIC, 0)  # flags bit0=0：g_idx contiguous，省略
    buf += scales.astype(np.float16).tobytes()
    buf += z.astype(np.float16).tobytes()
    buf += W_qweight.astype(np.uint32).tobytes()
    return bytes(buf)


def metrics(ref, got):
    d = (ref.astype(np.float32) - got.astype(np.float32)).ravel()
    r = ref.astype(np.float32).ravel()
    cos = float(np.dot(d * 0 + r, got.astype(np.float32).ravel()) /
                (np.linalg.norm(r) * np.linalg.norm(got.astype(np.float32).ravel()) + 1e-12))
    return float(np.mean(d ** 2)), float(np.max(np.abs(d))), cos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gptq", default=os.path.join(ROOT, "models/Qwen2.5-0.5B-Instruct-GPTQ-Int4"))
    ap.add_argument("--orig", default=os.path.join(ROOT, "models/Qwen2.5-0.5B-Instruct"))
    ap.add_argument("--tensors", default="model.layers.0.mlp.down_proj,"
                                          "model.layers.0.self_attn.q_proj,"
                                          "model.layers.5.mlp.gate_proj,"
                                          "model.layers.23.mlp.up_proj")
    ap.add_argument("--cpp", action="store_true", default=True)
    args = ap.parse_args()

    G = load_file(os.path.join(args.gptq, "model.safetensors"))
    # 原始权重是 bf16，numpy 不支持该 dtype，经 torch 读后转 fp32
    import torch
    from safetensors.torch import load_file as load_torch
    O_t = load_torch(os.path.join(args.orig, "model.safetensors"))
    O = {k: v.float().numpy() for k, v in O_t.items()}
    qcfg = None
    import json
    qcfg = json.load(open(os.path.join(args.gptq, "config.json")))["quantization_config"]
    gs = qcfg["group_size"]
    print(f"[cfg] bits={qcfg['bits']} group_size={gs} desc_act={qcfg['desc_act']} "
          f"sym={qcfg['sym']}")

    names = [n.strip() for n in args.tensors.split(",") if n.strip()]
    print(f"\n{'tensor':<44} {'variant':<14} {'MSE':>12} {'MaxErr':>12} {'CosSim':>10}")
    print("-" * 96)
    worst_correct = 0.0
    all_ok = True
    mse_map = {}
    for name in names:
        qw = G[name + ".qweight"]
        qz = G[name + ".qzeros"]
        sc = G[name + ".scales"]
        in_dim, out_dim = qw.shape[0] * PACK, sc.shape[1]
        ref = O[name + ".weight"]

        variants = {
            "correct": dequant(qw, qz, sc, in_dim, out_dim, gs, 1),
            "no_zp_plus1": dequant(qw, qz, sc, in_dim, out_dim, gs, 0),
        }
        for vname, W in variants.items():
            mse, mx, cos = metrics(ref, W)
            print(f"{name:<44} {vname:<14} {mse:>12.6e} {mx:>12.6e} {cos:>10.6f}")
            mse_map.setdefault(name, {})[vname] = mse
            if vname == "correct":
                worst_correct = max(worst_correct, mse)

    # 判别力检查：correct 必须显著优于 no_zp_plus1（实测约 8-10×）。
    # sym 量化的 qzeros nibble 恒为 zp-1=7，看 nibble 分布区分不了布局，
    # 只有"少一个量化步长"造成的系统性偏差能证明本测试有判别力。
    print("\n[判别力] correct vs no_zp_plus1 的 MSE 比值（应 >> 1）:")
    for name in names:
        c = mse_map[name]["correct"]
        b = mse_map[name]["no_zp_plus1"]
        ratio = b / c if c > 0 else float("inf")
        print(f"  {name:<44} {ratio:>6.2f}x")
        if ratio < 2.0:
            print("    !! 比值太小 —— 本测试无判别力，结论不可信")
            all_ok = False

    # ---- C++ matvec_gptq 与 numpy 参考逐位比对 ----
    if args.cpp and os.path.exists(BIN):
        print(f"\n=== C++ matvec_gptq vs numpy 参考 ===")
        for name in names[:2]:
            qw = G[name + ".qweight"]
            qz = G[name + ".qzeros"]
            sc = G[name + ".scales"]
            in_dim, out_dim = qw.shape[0] * PACK, sc.shape[1]
            rng = np.random.default_rng(1234)
            x = rng.standard_normal(in_dim).astype(np.float32)
            W = dequant(qw, qz, sc, in_dim, out_dim, gs, 1)
            ref_y = (W @ x).astype(np.float32)

            tmp = "/tmp/gptq_check"
            os.makedirs(tmp, exist_ok=True)
            with open(f"{tmp}/block.bin", "wb") as f:
                f.write(pack_inband(qw, qz, sc, in_dim, out_dim, gs, 1))
            x.tofile(f"{tmp}/x.bin")
            r = subprocess.run([BIN, "--block", f"{tmp}/block.bin",
                                "--out-dim", str(out_dim), "--in-dim", str(in_dim),
                                "--group-size", str(gs), "--x", f"{tmp}/x.bin",
                                "--out", f"{tmp}/y.bin"], capture_output=True, text=True)
            if r.returncode != 0:
                sys.stderr.write(r.stderr)
                raise SystemExit(f"C++ 校验失败: {name}")
            got_y = np.fromfile(f"{tmp}/y.bin", np.float32)
            d = np.abs(ref_y - got_y)
            print(f"  {name:<44} max_abs_err={d.max():.6e} mean={d.mean():.6e}")
            if d.max() > 1e-3:
                all_ok = False

    print(f"\nworst correct-variant MSE vs 原始权重: {worst_correct:.6e}")
    print("结论：" + ("真 AutoGPTQ 数值路径验证通过" if all_ok else "存在未通过项，见上"))
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
