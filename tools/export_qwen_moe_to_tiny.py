#!/usr/bin/env python3
"""把真 MoE GPTQ checkpoint 导出成 .tqwen（v3）。

目标是 HF 的 GPTQ-Int4 MoE 模型（如 Qwen3-30B-A3B-GPTQ-Int4）：
  model_type=qwen3_moe，dense attention（全层 full attention，无 GDN），
  无共享专家，路由专家是 AutoGPTQ 打包的 int4。

内存策略：safetensors 用 safe_open 惰性按张量读，导出器逐个张量转换后立即写盘，
全程不把整个 checkpoint 或整个输出驻留内存。16.9GB 输入 / ~17GB 输出都能在
几 GB 内存内跑完。

GPTQ 张量经 export_qwen_to_tiny.repack_gptq_from_hf 重打包成 runtime 的
in-band 块（qzeros int4 解包 + zp+1 修正），该转换已对真 AutoGPTQ 数据验证过
（tools/validate_gptq_numeric.py）。
"""
from __future__ import annotations

import argparse
import json
import os
import struct
import sys

import numpy as np
from safetensors import safe_open

from export_qwen_to_tiny import (
    ALIGN, DTYPE_F16, DTYPE_F32, HEADER_FMT, MAGIC, align_up, pack_v2_ext,
    quant_rtn_to_gptq_inband, repack_gptq_from_hf,
)

FORMAT_VERSION = 3
ENTRY_FMT = "<64sII4QQQ"
DTYPE_GPTQ4 = 5
MAX_NAME = 64
MODEL_QWEN3_MOE = 3


def pack_v3_ext(ext: dict) -> bytes:
    """打包 TinyHeaderV3Ext（32B），字段顺序与 runtime/tiny_format.h 一致。"""
    return struct.pack(
        "<8I",
        ext.get("n_routed_experts", 0),
        ext.get("num_experts_per_tok", 0),
        ext.get("moe_intermediate_size", 0),
        ext.get("shared_expert_intermediate_size", 0),
        ext.get("n_shared_experts", 0),
        ext.get("moe_topk_norm", 1),
        ext.get("gptq_group_size", 0),
        0,
    )


def plan_names(cfg: dict, max_layers: int = 0) -> list[str]:
    """按 runtime 期望的名字与顺序列出全部张量名。

    顺序决定文件内布局，也决定稀疏加载时哪些 tensor 会被判定为可卸载
    （名字含 ".mlp.experts."）。路由专家排在每层最后，便于卸载区连续。

    max_layers > 0 时只导出前 N 层（冒烟测试用，避免直接写 17GB）。
    """
    n_layers = cfg["num_hidden_layers"]
    if max_layers > 0:
        n_layers = min(n_layers, max_layers)
    n_exp = cfg["num_experts"]
    names = ["model.embed_tokens.weight"]
    for i in range(n_layers):
        p = f"model.layers.{i}."
        names.append(p + "input_layernorm.weight")
        names.append(p + "post_attention_layernorm.weight")
        for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
            names.append(p + f"self_attn.{proj}.weight")
        names.append(p + "self_attn.q_norm.weight")
        names.append(p + "self_attn.k_norm.weight")
        names.append(p + "mlp.gate.weight")
        for e in range(n_exp):
            pe = p + f"mlp.experts.{e}."
            for proj in ("gate_proj", "up_proj", "down_proj"):
                names.append(pe + f"{proj}.weight")
    names.append("model.norm.weight")
    if not cfg.get("tie_word_embeddings", False):
        names.append("lm_head.weight")
    return names


def src_names_for(name: str, has_gptq: set[str]) -> tuple[str, list[str]]:
    """把一个 runtime 张量名映射到 checkpoint 里的源张量名。

    GPTQ 量化过的投影在 checkpoint 里是四个分量（qweight/qzeros/scales/g_idx），
    其余是单个张量。返回 (kind, src_names)：kind 为 "gptq" / "raw"。
    """
    base = name[: -len(".weight")] if name.endswith(".weight") else name
    if base in has_gptq:
        return "gptq", [f"{base}.{s}" for s in ("qweight", "qzeros", "scales", "g_idx")]
    return "raw", [name]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF GPTQ MoE 模型目录")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-seq-len", type=int, default=0, help="0 = 用 config 的值")
    ap.add_argument("--max-layers", type=int, default=0,
                    help="只导出前 N 层（冒烟测试用；0 = 全部）")
    ap.add_argument("--quant-lm-head", action="store_true",
                    help="把 lm_head 用 RTN 量化成 GPTQ in-band（省 ~1022 MB resident）；"
                         "误差须用 tools/validate_lm_head_quant.py 实测")
    args = ap.parse_args()

    cfg = json.load(open(os.path.join(args.model, "config.json")))
    qcfg = cfg.get("quantization_config") or json.load(
        open(os.path.join(args.model, "quantize_config.json")))
    gs = int(qcfg["group_size"])
    if int(qcfg.get("bits", 4)) != 4:
        sys.exit(f"error: only 4-bit GPTQ supported, got bits={qcfg.get('bits')}")
    if qcfg.get("desc_act", False):
        sys.exit("error: desc_act (act-order) GPTQ not supported yet")

    st_path = os.path.join(args.model, "model.safetensors")
    if not os.path.exists(st_path):
        sys.exit(f"error: {st_path} not found")

    with safe_open(st_path, framework="numpy") as f:
        ck_keys = set(f.keys())
        has_gptq = {k[: -len(".qweight")] for k in ck_keys if k.endswith(".qweight")}

        names = plan_names(cfg, args.max_layers)
        n_layers_out = cfg["num_hidden_layers"]
        if args.max_layers > 0:
            n_layers_out = min(int(n_layers_out), args.max_layers)
        # 预检：所有源张量都必须存在，缺一个就说明名字映射错了，提前报错而不是
        # 导出到一半失败（17GB 文件写一半很难收拾）
        missing = []
        for n in names:
            kind, srcs = src_names_for(n, has_gptq)
            for s in srcs:
                if s not in ck_keys:
                    missing.append(s)
        if missing:
            sys.exit(f"error: {len(missing)} source tensors missing, first 5: {missing[:5]}")

        # pass 1：算每个张量的 nbytes 与 offset（需要先拿到形状）
        entries = []
        table_end = 192 + len(names) * 120
        data_off = align_up(table_end, ALIGN)
        off = data_off
        for n in names:
            kind, srcs = src_names_for(n, has_gptq)
            if kind == "gptq":
                qw_shape = f.get_slice(srcs[0]).get_shape()
                shape = [int(qw_shape[1]), int(qw_shape[0]) * 8]
                nbytes = None  # 真正字节数在 repack 后才知道
                dtype_code = DTYPE_GPTQ4
            elif args.quant_lm_head and n == "lm_head.weight":
                # lm_head 走 RTN → GPTQ in-band：nbytes 要量化后才知道，
                # 与 GPTQ 专家同样走"先算块再定布局"的两遍流程
                sh = f.get_slice(n).get_shape()
                shape = [int(d) for d in sh]
                nbytes = None
                dtype_code = DTYPE_GPTQ4
            else:
                sh = f.get_slice(n).get_shape()
                shape = [int(d) for d in sh]
                # **保留源 dtype，不升 fp32**。源 checkpoint 的 lm_head / embed_tokens
                # / layernorm 都是 fp16，硬编码升 fp32 会让每个张量白白翻倍
                # （lm_head 594→1187 MB）且零收益——实测 fp16 相对 RMS 误差
                # 0.0000%、argmax 一致率 100%、CosSim 1.0（就是原值）。
                src_dtype = f.get_slice(n).get_dtype()
                elem = 2 if src_dtype == "F16" else 4
                dtype_code = DTYPE_F16 if src_dtype == "F16" else DTYPE_F32
                nbytes = int(np.prod(shape)) * elem
            entries.append([n, kind, shape, nbytes, None, None, dtype_code])
            if nbytes is not None:
                off = align_up(off + nbytes, ALIGN)

        sys.stderr.write(f"[moe-export] {len(names)} tensors, planning...\n")

        # GPTQ 张量的 nbytes 依赖 repack 结果，必须先算一遍（不写盘）
        for e in entries:
            if e[3] is None:
                n = e[0]
                if n == "lm_head.weight":
                    block = quant_rtn_to_gptq_inband(f.get_tensor(n), gs)
                    sys.stderr.write(
                        f"[moe-export] lm_head RTN→GPTQ: {e[3+1] and ''}"
                        f"{len(block)/1048576:.1f} MB\n")
                else:
                    _, srcs = src_names_for(n, has_gptq)
                    block = repack_gptq_from_hf(
                        f.get_tensor(srcs[0]), f.get_tensor(srcs[1]),
                        f.get_tensor(srcs[2]), f.get_tensor(srcs[3]), gs)
                e[3] = len(block)
                e[4] = block
        sys.stderr.write("[moe-export] repacked all GPTQ tensors, computing layout...\n")

        off = data_off
        for e in entries:
            e[5] = off   # entry = [name, kind, shape, nbytes, block, offset, dtype_code]
            off = align_up(off + e[3], ALIGN)
        total = off

        ext_v2 = dict(
            model_type=MODEL_QWEN3_MOE,
            linear_num_qk_heads=0, linear_num_v_heads=0,
            linear_qk_head_dim=0, linear_v_head_dim=0,
            linear_conv_kernel_dim=0,
            full_attention_interval=1,   # <= 1 → 全层 full attention
            partial_rotary_factor=1.0,
            eos_token_id=int(cfg.get("eos_token_id", 151645)),
            quant_group_size=0,
        )
        ext_v3 = dict(
            n_routed_experts=int(cfg["num_experts"]),
            num_experts_per_tok=int(cfg["num_experts_per_tok"]),
            moe_intermediate_size=int(cfg["moe_intermediate_size"]),
            shared_expert_intermediate_size=0,
            n_shared_experts=0,
            moe_topk_norm=1 if cfg.get("norm_topk_prob", True) else 0,
            gptq_group_size=gs,
        )
        reserved = pack_v2_ext(ext_v2) + pack_v3_ext(ext_v3)
        assert len(reserved) == 96, len(reserved)

        max_seq = args.max_seq_len or int(cfg.get("max_position_embeddings", 4096))
        n_heads = int(cfg["num_attention_heads"])
        n_kv = int(cfg["num_key_value_heads"])
        head_dim = int(cfg.get("head_dim", cfg["hidden_size"] // n_heads))
        hdr = struct.pack(
            HEADER_FMT,
            MAGIC, FORMAT_VERSION, DTYPE_GPTQ4,
            int(n_layers_out), int(cfg["hidden_size"]),
            int(cfg.get("intermediate_size", 0)), n_heads, n_kv, head_dim,
            int(cfg["vocab_size"]), max_seq,
            1 if cfg.get("tie_word_embeddings", False) else 0, 0,
            float(cfg["rms_norm_eps"]), float(cfg["rope_theta"]),
            len(names), 192, data_off, total, reserved,
        )

        sys.stderr.write(f"[moe-export] writing {args.out}: {total/1e9:.2f} GB...\n")
        with open(args.out, "wb") as out_f:
            out_f.write(hdr)
            for e in entries:
                n, kind, shape, nbytes, block, offset = e[:6]
                # dtype 取自 pass 1 算好的值（保留源 dtype；量化 lm_head 标 GPTQ4）。
                # 标错会让 runtime 按错误的 dtype 解析，算错且不报错
                # （AGENTS.md 坑 #19 同类失效模式）。
                dtype_code = e[6]
                entry = struct.pack(
                    ENTRY_FMT, n.encode().ljust(MAX_NAME, b"\x00")[:MAX_NAME],
                    dtype_code, len(shape),
                    shape[0] if len(shape) > 0 else 0,
                    shape[1] if len(shape) > 1 else 0,
                    shape[2] if len(shape) > 2 else 0,
                    shape[3] if len(shape) > 3 else 0,
                    offset, nbytes)
                out_f.write(entry)
            out_f.seek(data_off)
            for e in entries:
                n, kind, shape, nbytes, block, offset = e[:6]
                out_f.seek(offset)
                if block is not None:
                    out_f.write(block)
                else:
                    # 保留源 dtype：fp16 源不升 fp32（升了只是白白翻倍，实测无损）
                    arr = f.get_tensor(n)
                    target = np.float16 if arr.dtype == np.float16 else np.float32
                    out_f.write(np.ascontiguousarray(arr, target).tobytes())
            # 补齐到 total。最后一个 tensor 末尾可能已恰好对齐到 total（无需填充），
            # 此时无条件 seek(total-1) 写零会覆盖它的末字节，损坏最后一个 tensor。
            out_f.seek(0, os.SEEK_END)
            if out_f.tell() < total:
                out_f.seek(total - 1)
                out_f.write(b"\x00")

    sys.stderr.write(f"[moe-export] done: {args.out} {total/1e9:.2f} GB, "
                     f"{len(names)} tensors\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
