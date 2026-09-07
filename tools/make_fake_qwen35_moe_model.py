#!/usr/bin/env python3
# make_fake_qwen35_moe_model.py — 生成一个极小的 Qwen3.5-MoE fake 模型
#
# 用途：为 MoE SSD 卸载机制（ExpertStore）提供端到端可跑的 .tqwen 测试模型。
# 不依赖 HF（transformers 可能无 Qwen3.5 MoE）；直接按 create() 期望的 tensor
# 名/形状手写随机权重，GPTQ-打包路由专家（+ attention/router/shared，全 GPTQ），
# 写 v3 .tqwen 文件。embed/norm/q_norm/k_norm 留 fp32（与 i4 文件 mixed-dtype 同款）。
#
# 正确性锚点：C++ resident 模式 vs SSD 模式逐位一致（align_fake_qwen35_moe_model.py）。
# 无需 numpy/HF 外部参考——两种模式用同一份 GPTQ 权重，差异只在专家访问路径。
#
# 布局契约见 runtime/tiny_format.h（GptqBlockOffsets / TinyHeaderV3Ext）。
import argparse
import struct
import sys
import numpy as np

from export_qwen_to_tiny import (
    MAGIC, FORMAT_VERSION, ALIGN, HEADER_FMT, ENTRY_FMT, align_up, pack_v2_ext,
)

MODEL_QWEN35_MOE = 2
DTYPE_F32 = 0
DTYPE_GPTQ4 = 5
GPTQ_MAGIC = 0x47505451  # 'GPTQ'
GPTQ_PACK = 8
GPTQ_HEADER = 8

# ---- fake 拓扑：复用 dense Qwen3.5 的 attention 配置（GDN+full 3:1）+ MoE FFN ----
FAKE_CFG = dict(
    hidden_size=16, intermediate_size=16, num_hidden_layers=4,
    num_attention_heads=2, num_key_value_heads=1, head_dim=16, vocab_size=64,
    max_position_embeddings=64, rms_norm_eps=1e-6,
    linear_num_key_heads=2, linear_num_value_heads=2, linear_key_head_dim=8,
    linear_value_head_dim=8, linear_conv_kernel_dim=4, full_attention_interval=4,
    tie_word_embeddings=False, attention_bias=False, rope_theta=10000.0,
    partial_rotary_factor=0.25, eos_token_id=63,
    # MoE
    n_routed_experts=8, num_experts_per_tok=2, moe_intermediate_size=16,
    shared_expert_intermediate_size=16, n_shared_experts=1, gptq_group_size=8,
)


def _zero_centered(name):
    """与 dense fake 一致：这些 norm 权重导出时 +1 折叠。"""
    if name == "model.norm.weight":
        return True
    return any(name.endswith(s) for s in (
        "input_layernorm.weight", "post_attention_layernorm.weight",
        "self_attn.q_norm.weight", "self_attn.k_norm.weight"))


def pack_gptq(W, group_size, has_g_idx=False):
    """RTN 量化 + AutoGPTQ 列主序 int32 打包，in-band 布局（与 matvec_gptq_ref 一致）。
    W: [out, in] fp32。返回 bytes。"""
    out, in_dim = W.shape
    assert in_dim % group_size == 0 and in_dim % GPTQ_PACK == 0
    n_groups = in_dim // group_size
    # 每 (g, o) 一组 scale/zero（列方向分组）
    scales = np.zeros((n_groups, out), np.float16)
    qzeros = np.zeros((n_groups, out), np.float16)
    q = np.zeros((out, in_dim), np.int32)  # 量化值 [out, in]
    for g in range(n_groups):
        blk = W[:, g * group_size:(g + 1) * group_size]  # [out, gs]
        vmin = blk.min(axis=1)
        vmax = blk.max(axis=1)
        scale = np.where(vmax > vmin, (vmax - vmin) / 15.0, 0.0).astype(np.float32)
        zero = np.where(scale > 0, -vmin / np.where(scale > 0, scale, 1.0), 0.0)
        sh = scale.astype(np.float16)
        zh = zero.astype(np.float16)
        scales[g] = sh
        qzeros[g] = zh
        s = sh.astype(np.float32)
        z = zh.astype(np.float32)
        s_col = s[:, None]  # [out, 1] 广播
        z_col = z[:, None]
        qq = np.rint(blk / np.where(s_col > 0, s_col, 1.0) + z_col).astype(np.int32)
        qq = np.clip(qq, 0, 15)
        q[:, g * group_size:(g + 1) * group_size] = qq
    # qweight: [(in/8), out] u32，word = Σ_k q[o, c8*8+k] << (k*4)
    c8_count = in_dim // GPTQ_PACK
    qweight = np.zeros((c8_count, out), np.uint32)
    for c8 in range(c8_count):
        for k in range(GPTQ_PACK):
            qweight[c8] |= (q[:, c8 * GPTQ_PACK + k].astype(np.uint32) & 0xF) << (k * 4)
    # 序列化 in-band
    flags = 1 if has_g_idx else 0
    buf = bytearray()
    buf += struct.pack("<II", GPTQ_MAGIC, flags)
    buf += scales.tobytes()
    buf += qzeros.tobytes()
    if has_g_idx:
        gidx = np.arange(in_dim, dtype=np.uint32) // group_size
        buf += gidx.tobytes()
    buf += qweight.tobytes()
    return bytes(buf), q.astype(np.float32)  # 返回量化值供参考


def pack_v3_ext(ext):
    """打包 TinyHeaderV3Ext (32B)。字段顺序与 tiny_format.h 一致。"""
    return struct.pack(
        "<8I",
        ext.get("n_routed_experts", 0),
        ext.get("num_experts_per_tok", 0),
        ext.get("moe_intermediate_size", 0),
        ext.get("shared_expert_intermediate_size", 0),
        ext.get("n_shared_experts", 0),
        ext.get("moe_topk_norm", 1),
        ext.get("gptq_group_size", 0),
        0,  # pad
    )


def write_tqwen_moe(path, header_cfg, tensors, ext_v2, ext_v3):
    """写 v3 .tqwen：per-tensor (dtype, bytes)。tensors: name -> (dtype_code, bytes, shape)."""
    names = list(tensors.keys())
    # pass 1: 算 offset
    entries = []
    table_end = 192 + len(names) * 120
    data_off = align_up(table_end, ALIGN)
    off = data_off
    for n in names:
        dtype_code, data, shape = tensors[n]
        nbytes = len(data)
        entries.append((n, dtype_code, shape, off, nbytes))
        off = align_up(off + nbytes, ALIGN)
    total = off
    # header
    v2ext_bytes = pack_v2_ext(ext_v2)
    v3ext_bytes = pack_v3_ext(ext_v3)
    reserved = v2ext_bytes + v3ext_bytes  # 64 + 32 = 96
    assert len(reserved) == 96
    dtype_master = DTYPE_GPTQ4  # 主 dtype = GPTQ（embed/norm 是 per-tensor kF32）
    hdr = struct.pack(
        HEADER_FMT,
        MAGIC, FORMAT_VERSION, dtype_master,
        header_cfg["num_hidden_layers"], header_cfg["hidden_size"],
        header_cfg["intermediate_size"], header_cfg["num_attention_heads"],
        header_cfg["num_key_value_heads"], header_cfg["head_dim"],
        header_cfg["vocab_size"], header_cfg["max_position_embeddings"],
        0 if not header_cfg["tie_word_embeddings"] else 1, 0,
        header_cfg["rms_norm_eps"], header_cfg["rope_theta"],
        len(names), 192, data_off, total, reserved,
    )
    with open(path, "wb") as f:
        f.write(hdr)
        for n, dt, shape, o, nb in entries:
            shp = list(shape) + [0] * (4 - len(shape))
            f.write(struct.pack(ENTRY_FMT, n.encode().ljust(64, b"\x00")[:64],
                                dt, len(shape), *shp, o, nb))
        # pad to data_off
        f.write(b"\x00" * (data_off - f.tell()))
        for n in names:
            _, data, _ = tensors[n]
            f.write(data)
            pad = align_up(len(data), ALIGN) - len(data)
            f.write(b"\x00" * pad)
    return total


def build(seed=42):
    rng = np.random.default_rng(seed)
    C = FAKE_CFG
    H = C["hidden_size"]
    nL = C["num_hidden_layers"]
    vocab = C["vocab_size"]
    qd = C["num_attention_heads"] * C["head_dim"]
    kvd = C["num_key_value_heads"] * C["head_dim"]
    gdn_qk = C["linear_num_key_heads"] * C["linear_key_head_dim"]
    gdn_v = C["linear_num_value_heads"] * C["linear_value_head_dim"]
    gdn_conv = 2 * gdn_qk + gdn_v
    moe_inter = C["moe_intermediate_size"]
    shared_inter = C["shared_expert_intermediate_size"]
    n_exp = C["n_routed_experts"]
    gs = C["gptq_group_size"]
    interval = C["full_attention_interval"]

    def W(*shape):
        return (rng.standard_normal(shape) * 0.1).astype(np.float32)

    tensors = {}  # name -> (dtype_code, bytes, shape)
    # 装填函数：2D 投影 GPTQ 打包；1D norm/embed 留 fp32
    def add_quant(name, arr):
        # arr: [out, in] fp32 -> GPTQ in-band bytes
        b, _ = pack_gptq(arr, gs, has_g_idx=False)
        tensors[name] = (DTYPE_GPTQ4, b, arr.shape)

    def add_f32(name, arr):
        tensors[name] = (DTYPE_F32, arr.astype(np.float32).tobytes(), arr.shape)

    # embed (fp32, untied)
    add_f32("model.embed_tokens.weight", W(vocab, H))
    # final norm (zero-centered +1)
    add_f32("model.norm.weight", (np.ones(H, np.float32) * 0.1))
    # lm_head (GPTQ, untied)
    add_quant("lm_head.weight", W(vocab, H))

    for i in range(nL):
        p = f"model.layers.{i}."
        # norms (fp32, zero-centered +1)
        add_f32(p + "input_layernorm.weight", np.ones(H, np.float32) * 0.1)
        add_f32(p + "post_attention_layernorm.weight", np.ones(H, np.float32) * 0.1)
        # MoE FFN
        add_quant(p + "mlp.gate.weight", W(n_exp, H))  # router
        add_quant(p + "mlp.shared_experts.gate_proj.weight", W(shared_inter, H))
        add_quant(p + "mlp.shared_experts.up_proj.weight", W(shared_inter, H))
        add_quant(p + "mlp.shared_experts.down_proj.weight", W(H, shared_inter))
        for e in range(n_exp):
            pe = p + f"mlp.experts.{e}."
            add_quant(pe + "gate_proj.weight", W(moe_inter, H))
            add_quant(pe + "up_proj.weight", W(moe_inter, H))
            add_quant(pe + "down_proj.weight", W(H, moe_inter))
        # attention：linear (GDN) or full
        is_linear = (i + 1) % interval != 0
        if is_linear:
            g = p + "linear_attn."
            add_quant(g + "in_proj_qkv.weight", W(gdn_conv, H))
            add_quant(g + "in_proj_z.weight", W(gdn_v, H))
            add_quant(g + "in_proj_b.weight", W(C["linear_num_value_heads"], H))
            add_quant(g + "in_proj_a.weight", W(C["linear_num_value_heads"], H))
            add_quant(g + "out_proj.weight", W(H, gdn_v))
            add_f32(g + "conv1d.weight", W(gdn_conv, C["linear_conv_kernel_dim"]))
            add_f32(g + "A_log", np.log(rng.uniform(0.5, 2.0, C["linear_num_value_heads"])).astype(np.float32))
            add_f32(g + "dt_bias", np.zeros(C["linear_num_value_heads"], np.float32))
            add_f32(g + "norm.weight", np.ones(C["linear_value_head_dim"], np.float32))
        else:
            s = p + "self_attn."
            add_quant(s + "q_proj.weight", W(2 * qd, H))
            add_quant(s + "k_proj.weight", W(kvd, H))
            add_quant(s + "v_proj.weight", W(kvd, H))
            add_quant(s + "o_proj.weight", W(H, qd))
            add_f32(s + "q_norm.weight", np.ones(C["head_dim"], np.float32) * 0.1)
            add_f32(s + "k_norm.weight", np.ones(C["head_dim"], np.float32) * 0.1)

    # ext
    ext_v2 = dict(model_type=MODEL_QWEN35_MOE,
                  linear_num_qk_heads=C["linear_num_key_heads"],
                  linear_num_v_heads=C["linear_num_value_heads"],
                  linear_qk_head_dim=C["linear_key_head_dim"],
                  linear_v_head_dim=C["linear_value_head_dim"],
                  linear_conv_kernel_dim=C["linear_conv_kernel_dim"],
                  full_attention_interval=C["full_attention_interval"],
                  partial_rotary_factor=C["partial_rotary_factor"],
                  eos_token_id=C["eos_token_id"], quant_group_size=0)
    ext_v3 = dict(n_routed_experts=n_exp, num_experts_per_tok=C["num_experts_per_tok"],
                  moe_intermediate_size=moe_inter,
                  shared_expert_intermediate_size=shared_inter,
                  n_shared_experts=C["n_shared_experts"], moe_topk_norm=1,
                  gptq_group_size=gs)
    return FAKE_CFG, tensors, ext_v2, ext_v3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    cfg, tensors, ext2, ext3 = build()
    total = write_tqwen_moe(args.out, cfg, tensors, ext2, ext3)
    sys.stderr.write(
        f"[fake-moe] wrote {args.out}: {total} bytes, {len(tensors)} tensors, "
        f"experts={cfg['n_routed_experts']} per_tok={cfg['num_experts_per_tok']} "
        f"gptq_group={cfg['gptq_group_size']}\n")


if __name__ == "__main__":
    main()
