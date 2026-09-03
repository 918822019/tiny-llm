#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""export_qat_vq2_to_tiny.py — 把 KronQ VQ-QAT 训练产物桥接成 .tqwen 的 VQ2 格式。

输入是 QAT 训练器落盘的**旋转域稠密 fp32 权重** `wq_<layer>_<sublayer>.pt`
（每个大线性子层一个文件），外加对应的 HF 原始权重目录。产出可直接被
tinyqwen runtime 加载的 2-bit VQ2 模型。

与免校准导出器（export_qwen_to_tiny_vq2.py）的关键区别：

    - **不做 k-means**。QAT 权重本身已经是 VQ 量化后的稠密还原值：每 d=4 个
      连续权重只取 K≤256 个唯一向量之一。所以码本 + 索引可**逐位无损**反提取
      （np.unique），不引入任何新的量化误差。
    - **需要 BiIP 旋转参数**。QAT 在旋转域训练，权重与 HF 原始基不对齐
      （实测相关性 ≈ 0）。旋转约定（见 kernels/biip/biip_rotate.cpp）为
          W_rot = blockHadamard((W * scaleH) ⊙ sign)   沿输入维

旋转参数的三个来源，按可靠性排序：

    1. **sign：确定性生成（默认）**。上游用
       `seed = layer_idx * 100 + SUBLAYER_NAMES.index(sublayer)` 播种
       `torch.randint(0, 2, (in_features,))`，无需数据即可精确复现。
       已对 28 层 × 7 子层交叉验证：与下面的最小二乘反解一致率 99.97%。
    2. **scaleH：复算（推荐）**。`--scaleh-npz` 传入
       `tools/compute_biip_scaleh.py` 的产物（用同一份校准 token 复算
       `(diagH/diagW²)^0.25`，含 span 阈值中和判定）。
    3. **最小二乘反解（兜底）**。没有上面两者时，用归一化块 Hadamard
       `Hn = H/sqrt(B)` 的正交对称性（Hn·Hn = I）把权重转回原基：
           U = W_rot · Hn  ≈  W · diag(scaleH ⊙ sign)
       再逐输入列做标量回归。sign 由此得到的很可靠，但 scaleH 会在退化列上
       产生 1e3~1e4 量级的离群估计（真 scaleH 是四次根，不可能这么大）。

旋转基的可证伪校验（本导出器会打印）：block size 在 B=256 处 R² 尖峰
（0.81~0.85），相邻 B=128/512 骤降到 ~0.41 —— 恰好等于 runtime
`find_hadamard_block_size()` 对 1024/2048/3072 的取值。

用法:
    python tools/export_qat_vq2_to_tiny.py \
        --model models/Qwen3-0.6B \
        --qat-dir vq_qat_qwen3_06b/qat_qwen3_06b_vq_seed42 \
        --scaleh-npz vq_qat_qwen3_06b/scaleh_qwen3_06b.npz \
        --out model_qwen3_06b_vq2_qat.tqwen
"""

import argparse
import struct
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from export_qwen_to_tiny import (  # noqa: E402
    MAGIC, FORMAT_VERSION, DTYPE_F32, DTYPE_F16, DTYPE_I4, DTYPE_VQ2, MAX_NAME,
    HEADER_FMT, ENTRY_FMT,
    align_up, find_local_dir, load_config, plan_tensors,
    print_table_summary, pack_v2_ext, MODEL_QWEN35,
)
from export_qwen_to_tiny_i4 import LazyTensors, should_quantize  # noqa: E402
from export_qwen_to_tiny_vq2 import (  # noqa: E402
    VQ2_K, VQ2_BLOCK, VQ2_CODEBOOK_BYTES, vq2_tensor_nbytes,
)

# QAT 落盘文件名里的子层标识 -> .tqwen tensor 名字片段。
# 顺序即 sign 的 seed 编号来源，必须与 kronq/configs.py 的 SUBLAYER_NAMES 一致。
QAT_SUBLAYERS = {
    "q_proj": "self_attn.q_proj",
    "k_proj": "self_attn.k_proj",
    "v_proj": "self_attn.v_proj",
    "o_proj": "self_attn.o_proj",
    "gate_proj": "mlp.gate_proj",
    "up_proj": "mlp.up_proj",
    "down_proj": "mlp.down_proj",
}
SUBLAYER_NAMES = list(QAT_SUBLAYERS)
# Hadamard 块大小候选，与 runtime/qwen_model.cpp 的 find_hadamard_block_size 一致
HADAMARD_BLOCKS = (256, 128, 64, 32, 16, 8, 4, 2)


def deterministic_sign(dim: int, seed: int) -> np.ndarray:
    """复现上游的 ±1 符号向量（kronq/hadamard.py 的 generate_sign_vector）。"""
    import torch
    gen = torch.Generator()
    gen.manual_seed(seed)
    return (torch.randint(0, 2, (dim,), generator=gen).float() * 2 - 1).numpy()


def find_hadamard_block_size(dim: int) -> int:
    """取能整除 dim 的最大候选块大小。与 C++ 侧逐字对应。"""
    for bs in HADAMARD_BLOCKS:
        if dim % bs == 0:
            return bs
    return 1


def normalized_hadamard(n: int) -> np.ndarray:
    """Sylvester 构造的归一化 Hadamard 矩阵 H/sqrt(n)：正交且对称，故自逆。"""
    h = np.array([[1.0]])
    while h.shape[0] < n:
        h = np.block([[h, h], [h, -h]])
    return h / np.sqrt(n)


def unrotate(w_rot: np.ndarray, hn: np.ndarray) -> np.ndarray:
    """沿输入维（列）施加分块 Hadamard，把旋转域权重变回原基方向。"""
    rows, cols = w_rot.shape
    block = hn.shape[0]
    return (w_rot.reshape(rows, cols // block, block) @ hn).reshape(rows, cols)


def recover_rotation(w_ref: np.ndarray, w_qat: np.ndarray, hn: np.ndarray):
    """从 (HF 原始权重, QAT 旋转域权重) 反解 BiIP 旋转参数。

    返回 (sign, scale, r2)：
      - sign  — ±1 向量 [in_features]（fp32）
      - scale — 反解出的 scaleH [in_features]（fp32），= 1/|D'|
      - r2    — 只用 sign（scale 视作 1）时对 U 的解释度，用于判定旋转基是否正确
    """
    u = unrotate(w_qat, hn)
    # D'_jj 使 U·D' 最贴近 W —— 这正是"部署后等效权重误差最小"的目标
    num = (u * w_ref).sum(0)
    dp = num / np.maximum((u * u).sum(0), 1e-30)
    sign = np.where(num >= 0.0, 1.0, -1.0).astype(np.float32)
    scale = (1.0 / np.maximum(np.abs(dp), 1e-12)).astype(np.float32)
    r2 = float(1.0 - ((u - w_ref * sign) ** 2).sum() / (u * u).sum())
    return sign, scale, r2


def extract_vq2(w_qat: np.ndarray) -> bytes:
    """从 QAT 稠密权重逐位无损反提取 [码本 [K,d] fp16][uint8 索引]。

    QAT 权重每 d=4 维块只取 K≤256 个唯一向量之一，且码本值本身精确落在 fp16
    上（实测往返误差 0），所以这一步不引入任何新误差。
    """
    rows, cols = w_qat.shape
    blocks = np.ascontiguousarray(w_qat).reshape(-1, VQ2_BLOCK)
    uniq, inv = np.unique(blocks, axis=0, return_inverse=True)
    if uniq.shape[0] > VQ2_K:
        raise ValueError(f"唯一块数 {uniq.shape[0]} > K={VQ2_K}，不是 VQ{VQ2_BLOCK}/{VQ2_K} 产物")
    codebook = np.zeros((VQ2_K, VQ2_BLOCK), dtype=np.float16)
    codebook[: uniq.shape[0]] = uniq.astype(np.float16)
    # fp16 码本必须能逐位还原原权重，否则说明产物不是 fp16 码本量化的
    if not np.array_equal(codebook.astype(np.float32)[inv].reshape(rows, cols),
                          w_qat.astype(np.float32)):
        raise ValueError("fp16 码本无法逐位还原 QAT 权重")
    indices = inv.astype(np.uint8).reshape(rows, cols // VQ2_BLOCK)
    return (np.ascontiguousarray(codebook).tobytes() +
            np.ascontiguousarray(indices).tobytes())


def pack_vq2_blob(codebook: np.ndarray, indices: np.ndarray,
                  rows: int, cols: int) -> bytes:
    """把上游 pack 的 (码本, 索引) 打包成 [码本 [K,d] fp16][uint8 索引]。

    pack 里的码本条目数是实际用到的数量（实测 251~256），需补零到 K=256
    才符合 .tqwen 的定长码本布局。
    """
    if codebook.ndim != 2 or codebook.shape[1] != VQ2_BLOCK:
        raise ValueError(f"码本形状 {codebook.shape} 应为 [K, {VQ2_BLOCK}]")
    if codebook.shape[0] > VQ2_K:
        raise ValueError(f"码本条目 {codebook.shape[0]} > K={VQ2_K}")
    if indices.shape != (rows, cols // VQ2_BLOCK):
        raise ValueError(f"索引形状 {indices.shape} != {(rows, cols // VQ2_BLOCK)}")
    if int(indices.max()) >= codebook.shape[0]:
        raise ValueError(f"索引越界: max={int(indices.max())} >= K_used={codebook.shape[0]}")
    padded = np.zeros((VQ2_K, VQ2_BLOCK), dtype=np.float16)
    padded[: codebook.shape[0]] = codebook.astype(np.float16)
    return (np.ascontiguousarray(padded).tobytes() +
            np.ascontiguousarray(indices.astype(np.uint8)).tobytes())


def transcode_embed_i4(pack_embed: dict, rows: int, cols: int) -> tuple[bytes, int]:
    """把上游 embed_int4.pt 转码成 tinyqwen 的 i4 组布局。

    这是**格式转码，不是重新量化**：两侧的量化语义完全一致（行内分组、
    非对称 minmax、反量化 (q - zero) * scale、低 nibble 在前），所以 4-bit
    码字与 zero 逐位照搬，唯一损失是 scale fp32 → fp16（.tqwen 格式要求）。

    为什么不能重新量化：上游是 tied embedding，这张表同时充当 lm_head，
    在已经量化过的格点上再做一次 minmax 取整会引入纯损失。

    目标布局（runtime/tiny_format.h）：每组
        [scale_fp16 2B][zero_fp16 2B][packed_uint4 group_size/2 B]

    返回 (bytes, group_size)。
    """
    q_packed = pack_embed["q_packed"].numpy()
    scale = pack_embed["scale"].numpy()
    zero = pack_embed["zero"].numpy()
    gs = int(pack_embed["group_size"])
    if int(pack_embed["bits"]) != 4:
        raise ValueError(f"只支持 4-bit embed，实际 {int(pack_embed['bits'])}")
    if tuple(pack_embed["shape"]) != (rows, cols):
        raise ValueError(f"embed 形状 {tuple(pack_embed['shape'])} != {(rows, cols)}")
    if cols % gs:
        raise ValueError(f"cols {cols} 不能被 group_size {gs} 整除")
    groups = cols // gs
    if q_packed.shape != (rows, cols // 2) or scale.shape != (rows, groups):
        raise ValueError(f"q_packed{q_packed.shape} / scale{scale.shape} 形状与 "
                         f"rows={rows} cols={cols} gs={gs} 不符")
    if int(zero.max()) > 15:
        raise ValueError(f"zero 超出 4-bit 范围: max={int(zero.max())}")

    nib_per_group = gs // 2                       # 每组打包后的字节数
    group_total = 4 + nib_per_group               # scale(2) + zero(2) + 码字
    out = np.empty((rows, groups, group_total), dtype=np.uint8)
    out[:, :, 0:2] = scale.astype(np.float16).view(np.uint8).reshape(rows, groups, 2)
    out[:, :, 2:4] = (zero.astype(np.float16)
                      .view(np.uint8).reshape(rows, groups, 2))
    out[:, :, 4:] = q_packed.reshape(rows, groups, nib_per_group)
    return np.ascontiguousarray(out).tobytes(), gs


def probe_block_size(w_ref: np.ndarray, w_qat: np.ndarray) -> list[tuple[int, float]]:
    """扫描候选块大小，返回 [(B, R2_sign)]，用于证伪旋转基假设。"""
    out = []
    for b in sorted(HADAMARD_BLOCKS):
        if w_qat.shape[1] % b:
            continue
        _, _, r2 = recover_rotation(w_ref, w_qat, normalized_hadamard(b))
        out.append((b, r2))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="HF 模型目录（提供 embed；反解模式还用作参照）")
    p.add_argument("--pack", default="",
                   help="上游 deploy pack 目录（cb_/idx_/rot_/extra_fp16.pt）——首选："
                        "旋转参数精确自带，无需反解或复算")
    p.add_argument("--qat-dir", default="",
                   help="QAT 稠密权重目录（wq_<layer>_<sublayer>.pt）；与 --pack 二选一")
    p.add_argument("--out", required=True, help="输出 .tqwen 路径")
    p.add_argument("--scaleh-npz", default="",
                   help="tools/compute_biip_scaleh.py 复算的 scaleH（仅 --qat-dir 模式用）。"
                        "缺某子层的 key 表示上游中和了它。不给则退回最小二乘反解")
    p.add_argument("--max-seq-len", type=int, default=0, help="override max_seq_len")
    args = p.parse_args()

    if bool(args.pack) == bool(args.qat_dir):
        sys.exit("error: --pack 与 --qat-dir 恰须指定其一")
    from_pack = bool(args.pack)
    src_dir = Path(args.pack or args.qat_dir)
    if not src_dir.is_dir():
        sys.exit(f"error: 源目录不存在: {src_dir}")
    import torch  # 仅为读 .pt，放在参数校验之后

    def load_pt(name: str):
        return torch.load(src_dir / name, map_location="cpu", weights_only=True)

    model_dir = find_local_dir(args.model)
    tcfg, model_type = load_config(model_dir)
    if model_type == MODEL_QWEN35:
        sys.exit("error: 本桥接只支持 Qwen2.x / Qwen3 稠密架构")
    base_names, src_of, transform_of = plan_tensors(tcfg, model_type)
    n_layers = int(tcfg["num_hidden_layers"])

    st = LazyTensors(model_dir)

    def load_ref(tiny_name: str) -> np.ndarray:
        arr = st.get(src_of[tiny_name])
        transform = transform_of[tiny_name]
        if transform is not None:
            arr = transform(arr)
        return np.asarray(arr, dtype=np.float32)

    def qat_path(layer: int, sub: str) -> Path:
        return src_dir / f"wq_{layer}_{sub}.pt"

    def load_qat(layer: int, sub: str) -> np.ndarray:
        return load_pt(f"wq_{layer}_{sub}.pt").float().numpy()

    # ---- 第一遍：确定旋转参数 ----
    t0 = time.time()
    signs: dict[str, np.ndarray] = {}
    scales: dict[str, np.ndarray] = {}
    n_neutral = 0

    if from_pack:
        print(f"[1/2] 读 deploy pack 的精确旋转参数（{n_layers} 层 × "
              f"{len(QAT_SUBLAYERS)} 子层）...")
        for layer in range(n_layers):
            for sub, frag in QAT_SUBLAYERS.items():
                tiny = f"model.layers.{layer}.{frag}.weight"
                rot = load_pt(f"rot_{layer}_{sub}.pt")
                sign = rot["sign_vec"].float().numpy()
                in_f = len(sign)
                if not np.all(np.abs(sign) == 1.0):
                    sys.exit(f"error: rot_{layer}_{sub} 的 sign_vec 不是 ±1")
                expect_block = find_hadamard_block_size(in_f)
                if int(rot["block_size"]) != expect_block:
                    sys.exit(f"error: rot_{layer}_{sub} block_size={int(rot['block_size'])} "
                             f"!= runtime 推导值 {expect_block}")
                signs[tiny] = sign.astype(np.float32)
                if rot["scale"] is None:
                    n_neutral += 1  # 上游中和了该子层
                else:
                    scales[tiny] = rot["scale"].float().numpy().astype(np.float32)
        print(f"      block_size 全部与 runtime find_hadamard_block_size 一致 OK；"
              f"scaleH：{len(scales)} 个写入，{n_neutral} 个中和")
    else:
        scaleh = np.load(args.scaleh_npz) if args.scaleh_npz else None
        src = "复算 npz" if scaleh is not None else "最小二乘反解（有离群风险）"
        print(f"[1/2] BiIP 旋转参数（{n_layers} 层 × {len(QAT_SUBLAYERS)} 子层）"
              f"：sign=确定性生成，scaleH={src}")
        r2s: list[float] = []
        agree: list[float] = []
        hn_cache: dict[int, np.ndarray] = {}
        probe_printed = False
        for layer in range(n_layers):
            for sub, frag in QAT_SUBLAYERS.items():
                tiny = f"model.layers.{layer}.{frag}.weight"
                path = qat_path(layer, sub)
                if not path.exists():
                    sys.exit(f"error: 缺少 QAT 权重 {path}")
                w_qat = load_qat(layer, sub)
                w_ref = load_ref(tiny).astype(np.float64)
                if w_qat.shape != w_ref.shape:
                    sys.exit(f"error: {tiny} 形状不符 QAT{w_qat.shape} vs HF{w_ref.shape}")
                block = find_hadamard_block_size(w_qat.shape[1])
                if not probe_printed:
                    probe = probe_block_size(w_ref, w_qat.astype(np.float64))
                    best = max(probe, key=lambda kv: kv[1])
                    print("      块大小探测 R2_sign: " +
                          " ".join(f"B={b}:{r:+.3f}" for b, r in probe))
                    if best[0] != block:
                        sys.exit(f"error: R2 最优块大小 {best[0]} != runtime 推导值 {block}，"
                                 f"旋转基假设不成立，请核对上游配方")
                    print(f"      最优 B={best[0]} 与 runtime find_hadamard_block_size 一致 OK")
                    probe_printed = True
                if block not in hn_cache:
                    hn_cache[block] = normalized_hadamard(block)

                in_f = w_qat.shape[1]
                sign = deterministic_sign(in_f, layer * 100 + SUBLAYER_NAMES.index(sub))
                signs[tiny] = sign.astype(np.float32)
                # 最小二乘反解：既交叉校验旋转基/符号，又在没有复算 scaleH 时兜底
                est_sign, est_scale, r2 = recover_rotation(w_ref, w_qat.astype(np.float64),
                                                          hn_cache[block])
                agree.append(float((sign == est_sign).mean()))
                r2s.append(r2)
                if scaleh is not None:
                    key = f"{layer}.{sub}.scale"
                    if key in scaleh:
                        s = scaleh[key]
                        if s.shape != (in_f,):
                            sys.exit(f"error: {key} 形状 {s.shape} != ({in_f},)")
                        scales[tiny] = s.astype(np.float32)
                    else:
                        n_neutral += 1  # 上游中和了该子层
                else:
                    scales[tiny] = est_scale
                del w_qat, w_ref
            print(f"      layer {layer + 1}/{n_layers} done", end="\r", flush=True)
        r2_arr, ag_arr = np.array(r2s), np.array(agree)
        print(f"\n      交叉校验 R2_sign: min={r2_arr.min():.4f} "
              f"median={np.median(r2_arr):.4f} max={r2_arr.max():.4f}；"
              f"确定性 sign vs 反解 sign 一致率={ag_arr.mean():.4f}")
        if scaleh is not None:
            print(f"      scaleH：{len(scales)} 个子层写入，{n_neutral} 个中和（不写 rot_scale）")
        else:
            print("      warning: 未提供 --scaleh-npz，rot_scale 用反解值，"
                  "退化列上会有离群估计")

    # ---- 组装 tensor 清单：基础权重 + 每个量化子层后紧跟 rot_sign ----
    names: list[str] = []
    for name in base_names:
        names.append(name)
        if name in signs:
            prefix = name[: -len(".weight")]
            names.append(prefix + ".rot_sign")
            if name in scales:  # 中和的子层不写 rot_scale，runtime 按 scale=nullptr 处理
                names.append(prefix + ".rot_scale")

    max_seq_len = args.max_seq_len or tcfg.get("max_position_embeddings", 32768)
    tqwen_cfg = {
        "n_layers": n_layers,
        "hidden_size": tcfg["hidden_size"],
        "intermediate_size": tcfg["intermediate_size"],
        "n_heads": tcfg["num_attention_heads"],
        "n_kv_heads": tcfg["num_key_value_heads"],
        "head_dim": tcfg["head_dim"],
        "vocab_size": tcfg["vocab_size"],
        "max_seq_len": max_seq_len,
        "tied": int(tcfg.get("tie_word_embeddings", True)),
        "rms_norm_eps": tcfg["rms_norm_eps"],
        "rope_theta": tcfg["rope_theta"],
    }
    ext = {"model_type": model_type,
           "eos_token_id": tcfg.get("eos_token_id", 151645)}

    # ---- 定布局。量化子层的 shape 取自 HF config，无需再读一遍权重 ----
    hidden = int(tcfg["hidden_size"])
    inter = int(tcfg["intermediate_size"])
    qd = int(tcfg["num_attention_heads"]) * int(tcfg["head_dim"])
    kvd = int(tcfg["num_key_value_heads"]) * int(tcfg["head_dim"])
    quant_shape = {"self_attn.q_proj": (qd, hidden), "self_attn.k_proj": (kvd, hidden),
                   "self_attn.v_proj": (kvd, hidden), "self_attn.o_proj": (hidden, qd),
                   "mlp.gate_proj": (inter, hidden), "mlp.up_proj": (inter, hidden),
                   "mlp.down_proj": (hidden, inter)}

    def shape_of(name: str) -> tuple[int, ...]:
        if name in signs:
            frag = name.split(".", 3)[3][: -len(".weight")]
            return quant_shape[frag]
        if name.endswith(".rot_sign") or name.endswith(".rot_scale"):
            return (len(signs[name.rsplit(".", 1)[0] + ".weight"]),)
        arr = load_ref(name)
        return tuple(arr.shape)

    # pack 模式：embed 用上游已量化的 INT4 表（逐位转码）。上游是 tied embedding，
    # 这张表同时充当 lm_head，必须与 QAT 学生模型一致，不能拿 HF 原值或再量化一次。
    embed_name = "model.embed_tokens.weight"
    embed_i4_blob = None
    if from_pack:
        embed_i4_blob, embed_gs = transcode_embed_i4(
            load_pt("embed_int4.pt"), int(tqwen_cfg["vocab_size"]), hidden)
        ext["quant_group_size"] = embed_gs
        print(f"      embed：上游 INT4 逐位转码（group_size={embed_gs}），"
              f"{len(embed_i4_blob) / (1024 * 1024):.1f} MB")

    entries = []  # (name, shape, offset, nbytes, dtype_code)
    table_end = struct.calcsize(HEADER_FMT) + len(names) * struct.calcsize(ENTRY_FMT)
    data_offset = align_up(table_end)
    offset = data_offset
    print(f"[2/2] 写盘 {len(names)} 个 tensor（VQ2 K={VQ2_K} d={VQ2_BLOCK}）...")
    for name in names:
        shape = shape_of(name)
        assert len(name) <= MAX_NAME, f"tensor name too long: {name}"
        if name in signs:
            nbytes, dtype_code = vq2_tensor_nbytes(shape[0], shape[1]), DTYPE_VQ2
        elif name.endswith(".rot_sign") or name.endswith(".rot_scale"):
            # 旋转参数存 f16（与上游 export_tiny_vq2.py 一致，runtime 侧自动转 fp32）：
            # sign 是 ±1，f16 精确；scale 是平滑的四次根量，f16 相对精度 ~5e-4 足够
            nbytes, dtype_code = shape[0] * 2, DTYPE_F16
        elif name == embed_name and embed_i4_blob is not None:
            nbytes, dtype_code = len(embed_i4_blob), DTYPE_I4
        elif "embed_tokens" in name:
            # 无上游 INT4 表时退回 f16（体积大头，但优于 f32）
            nbytes, dtype_code = shape[0] * shape[1] * 2, DTYPE_F16
        else:
            n = 1
            for d in shape:
                n *= d
            nbytes, dtype_code = n * 4, DTYPE_F32
        entries.append((name, list(shape), offset, nbytes, dtype_code))
        offset = align_up(offset + nbytes)
    total_bytes = offset

    reserved = pack_v2_ext(ext)
    reserved = reserved + b"\x00" * (96 - len(reserved))
    header = struct.pack(
        HEADER_FMT,
        MAGIC, FORMAT_VERSION, DTYPE_VQ2,
        n_layers, hidden, inter,
        int(tqwen_cfg["n_heads"]), int(tqwen_cfg["n_kv_heads"]), int(tqwen_cfg["head_dim"]),
        int(tqwen_cfg["vocab_size"]), int(max_seq_len), int(tqwen_cfg["tied"]),
        0,
        float(tqwen_cfg["rms_norm_eps"]), float(tqwen_cfg["rope_theta"]),
        len(names), struct.calcsize(HEADER_FMT), data_offset, total_bytes,
        reserved,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as out:
        out.write(header)
        for name, shape, off, nbytes, dtype_code in entries:
            out.write(struct.pack(ENTRY_FMT, name.encode("ascii"), dtype_code, len(shape),
                                  *shape, *[0] * (4 - len(shape)), off, nbytes))
        out.write(b"\x00" * (data_offset - out.tell()))
        for name, shape, off, nbytes, dtype_code in entries:
            assert out.tell() == off, f"offset drift at {name}"
            if dtype_code == DTYPE_VQ2:
                frag = name.split(".", 3)[3]
                layer = int(name.split(".", 3)[2])
                sub = next(s for s, f in QAT_SUBLAYERS.items()
                           if f == frag[: -len(".weight")])
                if from_pack:
                    blob = pack_vq2_blob(
                        load_pt(f"cb_{layer}_{sub}.pt").float().numpy(),
                        load_pt(f"idx_{layer}_{sub}.pt").numpy(),
                        shape[0], shape[1])
                else:
                    blob = extract_vq2(load_qat(layer, sub))
            elif name.endswith(".rot_sign"):
                blob = np.ascontiguousarray(
                    signs[name.rsplit(".", 1)[0] + ".weight"].astype(np.float16)).tobytes()
            elif name.endswith(".rot_scale"):
                blob = np.ascontiguousarray(
                    scales[name.rsplit(".", 1)[0] + ".weight"].astype(np.float16)).tobytes()
            elif dtype_code == DTYPE_I4:
                blob = embed_i4_blob
            elif dtype_code == DTYPE_F16:
                blob = np.ascontiguousarray(load_ref(name).astype(np.float16)).tobytes()
            else:
                blob = np.ascontiguousarray(load_ref(name).astype(np.float32)).tobytes()
            assert len(blob) == nbytes, f"{name}: blob {len(blob)} != nbytes {nbytes}"
            out.write(blob)
            out.write(b"\x00" * (align_up(off + nbytes) - (off + nbytes)))
        assert out.tell() == total_bytes, f"final size {out.tell()} != {total_bytes}"

    mb = total_bytes / (1024 * 1024)
    print(f"\nDone: {args.out} ({mb:.1f} MB) in {time.time() - t0:.1f}s")
    print_table_summary(args.out)


if __name__ == "__main__":
    main()
