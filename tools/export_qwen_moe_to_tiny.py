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
import torch
from safetensors import safe_open

from export_qwen_to_tiny import (
    ALIGN, DTYPE_F16, DTYPE_F32, HEADER_FMT, MAGIC, align_up, pack_v2_ext,
    quant_rtn_to_gptq_inband, repack_gptq_from_hf, _squeeze_conv,
)

FORMAT_VERSION = 3
ENTRY_FMT = "<64sII4QQQ"
DTYPE_GPTQ4 = 5
MAX_NAME = 64
MODEL_QWEN35_MOE = 2   # Qwen3.5 MoE：GDN 混合 + 共享专家
MODEL_QWEN3_MOE = 3    # Qwen3-MoE：全层 full attention、无共享专家


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
    # GDN + full attention 混合：interval>1 时只有 (i+1)%interval==0 的层是
    # full attention，其余是 Gated DeltaNet（linear_attn）。
    interval = int(cfg.get("full_attention_interval", 0) or 0)
    has_shared = int(cfg.get("shared_expert_intermediate_size", 0) or 0) > 0
    names = ["model.embed_tokens.weight"]
    for i in range(n_layers):
        p = f"model.layers.{i}."
        names.append(p + "input_layernorm.weight")
        names.append(p + "post_attention_layernorm.weight")
        is_linear = interval > 1 and ((i + 1) % interval != 0)
        if is_linear:
            g = p + "linear_attn."
            for proj in ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj"):
                names.append(g + f"{proj}.weight")
            names.append(g + "conv1d.weight")
            names.append(g + "A_log")
            names.append(g + "dt_bias")
            names.append(g + "norm.weight")
        else:
            for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
                names.append(p + f"self_attn.{proj}.weight")
            names.append(p + "self_attn.q_norm.weight")
            names.append(p + "self_attn.k_norm.weight")
        names.append(p + "mlp.gate.weight")
        if has_shared:
            # runtime 用复数 shared_experts，checkpoint 是单数 shared_expert
            for proj in ("gate_proj", "up_proj", "down_proj"):
                names.append(p + f"mlp.shared_experts.{proj}.weight")
        for e in range(n_exp):
            pe = p + f"mlp.experts.{e}."
            for proj in ("gate_proj", "up_proj", "down_proj"):
                names.append(pe + f"{proj}.weight")
    names.append("model.norm.weight")
    if not cfg.get("tie_word_embeddings", False):
        names.append("lm_head.weight")
    return names


def src_names_for(name: str, has_gptq: set[str],
                  ckpt_prefix: str = "model.") -> tuple[str, list[str]]:
    """把一个 runtime 张量名映射到 checkpoint 里的源张量名。

    GPTQ 量化过的投影在 checkpoint 里是四个分量（qweight/qzeros/scales/g_idx），
    其余是单个张量。返回 (kind, src_names)：kind 为 "gptq" / "raw"。

    两处名字差异必须映射，否则找不到源张量：
      ① 多模态模型（Qwen3.5-MoE）文本部分前缀是 model.language_model.，
         runtime 名用 model.
      ② 共享专家 checkpoint 是单数 mlp.shared_expert.，runtime 用复数
         mlp.shared_experts.
    """
    src = name
    if ckpt_prefix != "model." and src.startswith("model."):
        src = ckpt_prefix + src[len("model."):]
    src = src.replace("mlp.shared_experts.", "mlp.shared_expert.")
    base = src[: -len(".weight")] if src.endswith(".weight") else src
    if base in has_gptq:
        return "gptq", [f"{base}.{s}" for s in ("qweight", "qzeros", "scales", "g_idx")]
    return "raw", [src]


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
    # 多模态模型（Qwen3.5-MoE）把文本配置嵌在 text_config 里，且张量前缀是
    # model.language_model.；纯文本 MoE（Qwen3-MoE）直接在顶层、前缀 model.。
    if "text_config" in cfg:
        text_cfg = dict(cfg["text_config"])
        ckpt_prefix = "model.language_model."
        text_cfg.setdefault("tie_word_embeddings", cfg.get("tie_word_embeddings", False))
        if cfg.get("quantization_config"):
            text_cfg["quantization_config"] = cfg["quantization_config"]
        # Qwen3.5 的 config 键名与 runtime 期望的不同，需归一化：
        #   rope_parameters(dict) → rope_theta
        #   linear_key_head_dim   → linear_qk_head_dim
        #   linear_num_key_heads  → linear_num_qk_heads
        #   linear_value_head_dim → linear_v_head_dim
        #   linear_num_value_heads→ linear_num_v_heads
        rp = text_cfg.get("rope_parameters")
        if isinstance(rp, dict) and "rope_theta" not in text_cfg:
            text_cfg["rope_theta"] = float(rp.get("rope_theta", rp.get("theta", 10000.0)))
        for src_k, dst_k in (("linear_key_head_dim", "linear_qk_head_dim"),
                             ("linear_num_key_heads", "linear_num_qk_heads"),
                             ("linear_value_head_dim", "linear_v_head_dim"),
                             ("linear_num_value_heads", "linear_num_v_heads")):
            if src_k in text_cfg:
                text_cfg.setdefault(dst_k, text_cfg[src_k])
        is_multimodal = True
    else:
        text_cfg = cfg
        ckpt_prefix = "model."
        is_multimodal = False
    cfg = text_cfg
    qcfg = cfg.get("quantization_config") or json.load(
        open(os.path.join(args.model, "quantize_config.json")))
    gs = int(qcfg["group_size"])
    if int(qcfg.get("bits", 4)) != 4:
        sys.exit(f"error: only 4-bit GPTQ supported, got bits={qcfg.get('bits')}")
    if qcfg.get("desc_act", False):
        sys.exit("error: desc_act (act-order) GPTQ not supported yet")

    # 支持单文件与分片 safetensors。大模型（如 Qwen3.5-35B-A3B）是多个分片，
    # 张量名 → 分片文件的映射在 model.safetensors.index.json 的 weight_map 里。
    st_path = os.path.join(args.model, "model.safetensors")
    idx_path = os.path.join(args.model, "model.safetensors.index.json")
    shard_map = {}
    handles = {}
    single = None
    if os.path.exists(idx_path):
        with open(idx_path) as fh:
            shard_map = json.load(fh).get("weight_map", {})
        for sf in sorted(set(shard_map.values())):
            sp = os.path.join(args.model, sf)
            if not os.path.exists(sp):
                sys.exit(f"error: 分片 {sp} 不存在")
            handles[sf] = safe_open(sp, framework="torch")
    elif os.path.exists(st_path):
        single = safe_open(st_path, framework="torch")
    else:
        sys.exit(f"error: 找不到 {st_path} 或 {idx_path}")

    def _handle_for(name):
        """取该张量所在的 safe_open 句柄（分片感知）。

        传入的可能是 runtime 名（model.xxx / mlp.shared_experts.xxx），也可能
        已经是 checkpoint 名（src_names_for 的产出）。_map_name 对两种都安全。
        """
        if single is not None:
            return single
        src = _map_name(name)
        sf = shard_map.get(src)
        if sf is None:
            raise KeyError(f"张量 {name}（映射后 {src}）不在 weight_map 里")
        return handles[sf]

    def _map_name(name):
        """runtime 名 → checkpoint 名。已经是 checkpoint 名则原样返回，
        避免双重映射（src_names_for 产出的名字已经是 checkpoint 名）。"""
        src = name
        if ckpt_prefix != "model." and src.startswith("model.") \
                and not src.startswith(ckpt_prefix):
            src = ckpt_prefix + src[len("model."):]
        return src.replace("mlp.shared_experts.", "mlp.shared_expert.")

    def get_tensor(name):
        """取张量并转成 numpy。

        浮点张量（bf16/fp16）转 fp32：numpy 不认识 bf16，所以用 torch framework
        读出来后 .float().numpy()。
        **整数张量（qweight/qzeros/g_idx 是 int32）必须保持原 dtype**——转 float
        会把整数值变成浮点，后续 .astype(np.uint32) 就是把浮点位模式当整数读，
        得到垃圾（AGENTS.md 坑 #19）。
        """
        t = _handle_for(name).get_tensor(_map_name(name))
        if t.dtype in (torch.bfloat16, torch.float16, torch.float32):
            return t.float().numpy()
        return t.numpy()

    def get_shape(name):
        return [int(d) for d in _handle_for(name).get_slice(_map_name(name)).get_shape()]

    def get_dtype(name):
        return _handle_for(name).get_slice(_map_name(name)).get_dtype()

    ck_keys = set(single.keys()) if single is not None else set(shard_map.keys())
    has_gptq = {k[: -len(".qweight")] for k in ck_keys if k.endswith(".qweight")}

    names = plan_names(cfg, args.max_layers)
    n_layers_out = cfg["num_hidden_layers"]
    if args.max_layers > 0:
        n_layers_out = min(int(n_layers_out), args.max_layers)
    # 预检：所有源张量都必须存在，缺一个就说明名字映射错了，提前报错而不是
    # 导出到一半失败（17GB 文件写一半很难收拾）
    missing = []
    for n in names:
        kind, srcs = src_names_for(n, has_gptq, ckpt_prefix)
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
        kind, srcs = src_names_for(n, has_gptq, ckpt_prefix)
        if kind == "gptq":
            qw_shape = get_shape(srcs[0])
            shape = [int(qw_shape[1]), int(qw_shape[0]) * 8]
            nbytes = None  # 真正字节数在 repack 后才知道
            dtype_code = DTYPE_GPTQ4
        elif args.quant_lm_head and n == "lm_head.weight":
            # lm_head 走 RTN → GPTQ in-band：nbytes 要量化后才知道，
            # 与 GPTQ 专家同样走"先算块再定布局"的两遍流程
            sh = get_shape(n)
            shape = [int(d) for d in sh]
            nbytes = None
            dtype_code = DTYPE_GPTQ4
        else:
            sh = get_shape(n)
            shape = [int(d) for d in sh]
            # depthwise conv1d 权重 HF 存 3D [out,1,kernel]，runtime 期望 2D
            # [out,kernel]。规划阶段就要 squeeze，否则 tensor 表记录 3D shape，
            # runtime 校验 ndim 会失败（写盘时的 squeeze 只改数据不改 shape）。
            if n.endswith("linear_attn.conv1d.weight") and len(shape) == 3:
                shape = [shape[0], shape[2]]
            # **保留源 dtype，不升 fp32**。源 checkpoint 的 lm_head / embed_tokens
            # / layernorm 都是 fp16，硬编码升 fp32 会让每个张量白白翻倍
            # （lm_head 594→1187 MB）且零收益——实测 fp16 相对 RMS 误差
            # 0.0000%、argmax 一致率 100%、CosSim 1.0（就是原值）。
            src_dtype = get_dtype(n)
            # bf16 与 fp16 都是 2 字节，都按 F16 写。源 checkpoint 是 bf16，
            # 若按 fp32 写会让每个张量翻倍（embed 1017→1940 MB）且零收益。
            # bf16→fp16 对权重安全：bf16 8 位尾数、fp16 10 位尾数，权重值域
            # 远小于 fp16 上限 ±65504，不会溢出。
            is_half = src_dtype in ("F16", "BF16")
            elem = 2 if is_half else 4
            dtype_code = DTYPE_F16 if is_half else DTYPE_F32
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
                block = quant_rtn_to_gptq_inband(get_tensor(n), gs)
                sys.stderr.write(
                    f"[moe-export] lm_head RTN→GPTQ: {e[3+1] and ''}"
                    f"{len(block)/1048576:.1f} MB\n")
            else:
                _, srcs = src_names_for(n, has_gptq, ckpt_prefix)
                block = repack_gptq_from_hf(
                    get_tensor(srcs[0]), get_tensor(srcs[1]),
                    get_tensor(srcs[2]), get_tensor(srcs[3]), gs)
            e[3] = len(block)
            e[4] = block
    sys.stderr.write("[moe-export] repacked all GPTQ tensors, computing layout...\n")

    off = data_off
    for e in entries:
        e[5] = off   # entry = [name, kind, shape, nbytes, block, offset, dtype_code]
        off = align_up(off + e[3], ALIGN)
    total = off

    # Qwen3.5-MoE（多模态）= ModelType 2，有 GDN 混合 + 共享专家；
    # Qwen3-MoE（纯文本）= ModelType 3，全层 full attention、无共享专家。
    shared_inter = int(cfg.get("shared_expert_intermediate_size", 0) or 0)
    interval = int(cfg.get("full_attention_interval", 0) or 0)
    ext_v2 = dict(
        model_type=MODEL_QWEN35_MOE if is_multimodal else MODEL_QWEN3_MOE,
        linear_num_qk_heads=int(cfg.get("linear_num_qk_heads", 0) or 0),
        linear_num_v_heads=int(cfg.get("linear_num_v_heads", 0) or 0),
        linear_qk_head_dim=int(cfg.get("linear_qk_head_dim", 0) or 0),
        linear_v_head_dim=int(cfg.get("linear_v_head_dim", 0) or 0),
        linear_conv_kernel_dim=int(cfg.get("linear_conv_kernel_dim", 0) or 0),
        # interval<=1 → 全层 full attention；>1 → GDN + full 混合
        full_attention_interval=interval if interval > 0 else 1,
        partial_rotary_factor=float(cfg.get("partial_rotary_factor", 1.0) or 1.0),
        eos_token_id=int(cfg.get("eos_token_id", 151645)),
        quant_group_size=0,
    )
    ext_v3 = dict(
        n_routed_experts=int(cfg["num_experts"]),
        num_experts_per_tok=int(cfg["num_experts_per_tok"]),
        moe_intermediate_size=int(cfg["moe_intermediate_size"]),
        shared_expert_intermediate_size=shared_inter,
        n_shared_experts=1 if shared_inter > 0 else 0,
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
        int(cfg.get("intermediate_size", 0) or cfg["moe_intermediate_size"]),
        n_heads, n_kv, head_dim,
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
                arr = get_tensor(n)
                # depthwise conv1d 权重 HF 存 3D [out,1,kernel]，runtime 期望 2D
                # [out,kernel]，需 squeeze（与 export_qwen_to_tiny 同款处理）。
                if n.endswith("linear_attn.conv1d.weight"):
                    arr = _squeeze_conv(arr)
                # 用规划阶段定的 dtype_code，不能用 arr.dtype——get_tensor 已把
                # bf16 转成 fp32 numpy，arr.dtype 恒为 float32，会误判成 fp32。
                target = np.float16 if e[6] == DTYPE_F16 else np.float32
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
