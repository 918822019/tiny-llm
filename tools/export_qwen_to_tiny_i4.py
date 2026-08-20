#!/usr/bin/env python3
"""将 Qwen2.5（HF）权重导出为 INT4 量化的 .tqwen 文件。

非对称 uint4 [0,15]，per-group(128) scale+zero_point（fp16），interleaved packing。
小向量（norm/bias/embed）保持 fp32。RTN (round-to-nearest) 量化。

用法:
    python tools/export_qwen_to_tiny_i4.py \
        --model Qwen/Qwen2.5-0.5B \
        --out model_i4.tqwen \
        --group-size 128
"""

from __future__ import annotations

import argparse
import numpy as np
import os
import struct
import sys
from pathlib import Path

# 复用现有 exporter 的基础设施
sys.path.insert(0, str(Path(__file__).parent))
from export_qwen_to_tiny import (  # noqa: E402
    MAGIC, FORMAT_VERSION, DTYPE_F32, MAX_NAME,
    HEADER_FMT, ENTRY_FMT,
    align_up, find_local_dir, load_config, plan_tensors,
    print_table_summary,
    MODEL_QWEN35,
)

DTYPE_I4 = 3
GROUP_HEADER_BYTES = 4  # scale(fp16, 2B) + zero(fp16, 2B)

# v2 ext 格式：增加 quant_group_size（从 ext_reserved 的前 4 字节）
# 7 个 u32 + float + 2 个 u32 + u32(quant_group_size) + 20 字节 reserved
EXT_I4_FMT = "<7If2II20s"  # 64 字节
assert struct.calcsize(EXT_I4_FMT) == 64


def pack_v2_ext_i4(ext: dict, group_size: int) -> bytes:
    g = lambda k: ext.get(k, 0)  # noqa: E731
    return struct.pack(
        EXT_I4_FMT,
        int(g("model_type")),
        int(g("linear_num_qk_heads")),
        int(g("linear_num_v_heads")),
        int(g("linear_qk_head_dim")),
        int(g("linear_v_head_dim")),
        int(g("linear_conv_kernel_dim")),
        int(g("full_attention_interval")),
        float(g("partial_rotary_factor")),
        int(g("eos_token_id")),
        0,  # pad
        int(group_size),
        b"\x00" * 20,
    )


def i4_row_bytes(in_dim: int, group_size: int) -> int:
    n_groups = (in_dim + group_size - 1) // group_size
    return n_groups * (GROUP_HEADER_BYTES + group_size // 2)


def should_quantize(name: str) -> bool:
    """大投影矩阵走 INT4；小向量/embed 保持 fp32。"""
    skip_patterns = [
        "embed_tokens", "layernorm", "norm.weight", "norm_weight",
        "bias", "conv1d", "A_log", "dt_bias",
    ]
    for pat in skip_patterns:
        if pat in name:
            return False
    return True


def quantize_i4_group(data: np.ndarray) -> tuple[np.float16, np.float16, np.ndarray]:
    """对一组 float32 数据做非对称 RTN 量化。

    Returns: (scale_fp16, zero_fp16, packed_uint4_bytes)
    """
    vmin = data.min()
    vmax = data.max()

    if vmax == vmin:
        scale = np.float16(0.0)
        zero = np.float16(0.0)
        quantized = np.zeros(len(data), dtype=np.uint8)
    else:
        scale_f32 = (vmax - vmin) / 15.0
        zero_f32 = -vmin / scale_f32  # uint4=0 maps to vmin

        scale = np.float16(scale_f32)
        zero = np.float16(zero_f32)

        # 用 rounded fp16 值做量化（与 C++ 反量化对齐）
        s = float(scale)
        z = float(zero)
        if s == 0:
            quantized = np.zeros(len(data), dtype=np.uint8)
        else:
            q = np.round(data / s + z).astype(np.int32)
            quantized = np.clip(q, 0, 15).astype(np.uint8)

    return scale, zero, quantized


def quantize_tensor_i4(tensor: np.ndarray, group_size: int) -> bytes:
    """将 [out_dim, in_dim] 的 fp32 tensor 量化为 interleaved INT4 格式。"""
    assert tensor.ndim == 2
    out_dim, in_dim = tensor.shape
    row_bytes = i4_row_bytes(in_dim, group_size)
    result = bytearray(out_dim * row_bytes)

    groups_per_row = (in_dim + group_size - 1) // group_size
    group_total = GROUP_HEADER_BYTES + group_size // 2

    for o in range(out_dim):
        row_offset = o * row_bytes
        for g in range(groups_per_row):
            start = g * group_size
            end = min(start + group_size, in_dim)
            group_data = tensor[o, start:end].astype(np.float32)

            scale, zero, quantized = quantize_i4_group(group_data)

            group_offset = row_offset + g * group_total
            # 写 scale + zero（fp16 little-endian）
            result[group_offset:group_offset + 2] = scale.tobytes()
            result[group_offset + 2:group_offset + 4] = zero.tobytes()

            # pack nibbles: low nibble first
            data_offset = group_offset + GROUP_HEADER_BYTES
            n_elems = len(quantized)
            for i in range(0, n_elems, 2):
                lo = quantized[i]
                hi = quantized[i + 1] if i + 1 < n_elems else 0
                result[data_offset + i // 2] = int(hi << 4) | int(lo)

    return bytes(result)


def quantize_tensor_i4_hqq(tensor: np.ndarray, group_size: int) -> bytes:
    """HQQ（半二次优化，MSE 最优、免校准）INT4 量化，输出与 RTN 完全相同的
    interleaved 格式：每组 [scale_fp16 | zero_fp16 | packed_uint4]，低 nibble 在前。

    hqq 库的反量化语义 (q - zero) × scale、分组方向（沿 in_dim、组序 row-major）
    与本仓库 C++ kernel 完全一致——探包验证过逐位重组 == hq.dequantize()——
    因此只需把 unpack 出的 q / scale / zero 直接 repack。
    """
    import torch  # 延迟导入：只有 method=hqq 才需要
    from hqq.core.quantize import BaseQuantizeConfig, HQQLinear, Quantizer

    out_dim, in_dim = tensor.shape
    if in_dim % group_size != 0:
        raise ValueError(
            f"HQQ 要求 in_dim % group_size == 0（实际 {in_dim} % {group_size}）；"
            f"该 tensor 可退回 --method rtn")

    linear = torch.nn.Linear(in_dim, out_dim, bias=False)
    linear.weight.data = torch.from_numpy(np.ascontiguousarray(tensor))
    cfg = BaseQuantizeConfig(nbits=4, group_size=group_size,
                             quant_zero=False, quant_scale=False)
    hq = HQQLinear(linear, cfg, compute_dtype=torch.float32, device="cpu")

    # unpack 到整数 q：[n_groups_total, group_size]，组序 = row * groups_per_row + g
    q = Quantizer.unpack["4bit_u8"](hq.W_q, dtype=torch.float32)
    q_np = q.round().clamp(0, 15).to(torch.uint8).numpy()
    scale = hq.meta["scale"].float().flatten().numpy().astype(np.float16)
    zero = hq.meta["zero"].float().flatten().numpy().astype(np.float16)

    row_bytes = i4_row_bytes(in_dim, group_size)
    result = bytearray(out_dim * row_bytes)
    groups_per_row = in_dim // group_size
    group_total = GROUP_HEADER_BYTES + group_size // 2
    for o in range(out_dim):
        row_offset = o * row_bytes
        for g in range(groups_per_row):
            gi = o * groups_per_row + g
            group_offset = row_offset + g * group_total
            result[group_offset:group_offset + 2] = scale[gi].tobytes()
            result[group_offset + 2:group_offset + 4] = zero[gi].tobytes()
            qv = q_np[gi]
            data_offset = group_offset + GROUP_HEADER_BYTES
            for i in range(0, group_size, 2):
                result[data_offset + i // 2] = int(qv[i + 1]) << 4 | int(qv[i])

    return bytes(result)


def write_tqwen_i4(out_path: str | Path, cfg: dict, tensors: dict,
                   group_size: int, version: int = 2, ext: dict | None = None,
                   method: str = "hqq") -> int:
    """写一个 INT4 混合 dtype 的 .tqwen 文件。

    大矩阵: INT4 量化（method=hqq 用 HQQ 优化量化，rtn 用 min-max 舍入）;
    小向量: fp32。
    """
    names = list(tensors.keys())

    table_end = struct.calcsize(HEADER_FMT) + len(names) * struct.calcsize(ENTRY_FMT)
    data_offset = align_up(table_end)

    # 第一遍：分类 + 计算 layout
    entries = []  # (name, shape, offset, nbytes, dtype_code, packed_data)
    offset = data_offset

    print(f"Quantizing {len(names)} tensors (group_size={group_size})...")
    for name in names:
        arr = np.asarray(tensors[name], dtype=np.float32)
        shape = list(arr.shape)
        assert 1 <= len(shape) <= 4, f"{name}: ndim {len(shape)} not in [1,4]"
        assert len(name) <= MAX_NAME, f"tensor name too long: {name}"

        if arr.ndim == 2 and should_quantize(name):
            # INT4 量化
            if method == "hqq":
                packed = quantize_tensor_i4_hqq(arr, group_size)
            else:
                packed = quantize_tensor_i4(arr, group_size)
            nbytes = len(packed)
            dtype_code = DTYPE_I4
            print(f"  [I4] {name:<50} {str(shape):<18} -> {nbytes:>10} bytes")
        else:
            # fp32 保持
            packed = arr.tobytes()
            nbytes = len(packed)
            dtype_code = DTYPE_F32
            print(f"  [F32] {name:<49} {str(shape):<18} -> {nbytes:>10} bytes")

        entries.append((name, shape, offset, nbytes, dtype_code, packed))
        offset = align_up(offset + nbytes)

    total_bytes = offset

    # header reserved 区
    reserved = b"\x00" * 96
    if version >= 2:
        ext_bytes = pack_v2_ext_i4(ext or {}, group_size)
        reserved = ext_bytes + b"\x00" * (96 - len(ext_bytes))

    header = struct.pack(
        HEADER_FMT,
        MAGIC,
        version,
        DTYPE_I4,  # 主 dtype = INT4
        int(cfg["n_layers"]),
        int(cfg["hidden_size"]),
        int(cfg["intermediate_size"]),
        int(cfg["n_heads"]),
        int(cfg["n_kv_heads"]),
        int(cfg["head_dim"]),
        int(cfg["vocab_size"]),
        int(cfg["max_seq_len"]),
        int(cfg["tied"]),
        0,  # reserved_u32
        float(cfg["rms_norm_eps"]),
        float(cfg["rope_theta"]),
        len(names),
        struct.calcsize(HEADER_FMT),
        data_offset,
        total_bytes,
        reserved,
    )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as out:
        out.write(header)
        for name, shape, off, nbytes, dtype_code, _ in entries:
            out.write(struct.pack(
                ENTRY_FMT,
                name.encode("ascii"),
                dtype_code,
                len(shape),
                *shape,
                *[0] * (4 - len(shape)),
                off,
                nbytes,
            ))
        # 补零到 data_offset
        out.write(b"\x00" * (data_offset - out.tell()))

        for name, shape, off, nbytes, dtype_code, packed in entries:
            assert out.tell() == off, f"alignment bug at {name}: {out.tell()} != {off}"
            out.write(packed)
            out.write(b"\x00" * (align_up(off + nbytes) - (off + nbytes)))
        assert out.tell() == total_bytes

    size = os.path.getsize(out_path)
    assert size == total_bytes, f"file size {size} != expected {total_bytes}"
    return total_bytes


# ---- 懒加载 safetensors -----------------------------------------------

class LazyTensors:
    """按需从 safetensors 文件加载并转换 tensor。"""

    def __init__(self, model_dir: Path):
        try:
            from safetensors import safe_open
        except ImportError:
            sys.exit("error: pip install safetensors")
        self._handles = []
        self._index = {}
        for st_file in sorted(model_dir.glob("*.safetensors")):
            # 用 torch 框架读：真实 Qwen 权重是 bf16，numpy 不认识 bf16；
            # get() 里统一转成 fp32（与 export_qwen_to_tiny.py 同款做法）。
            h = safe_open(str(st_file), framework="torch")
            for key in h.keys():
                self._index[key] = h
            self._handles.append(h)

    def get(self, key: str) -> np.ndarray:
        if key not in self._index:
            sys.exit(f"error: tensor '{key}' not found in safetensors")
        return self._index[key].get_tensor(key).float().numpy()


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="HF model dir")
    p.add_argument("--out", required=True, help="output .tqwen path")
    p.add_argument("--method", choices=["hqq", "rtn"], default="hqq",
                   help="量化算法：hqq（MSE 最优迭代，默认）/ rtn（min-max 舍入，回归对照）")
    p.add_argument("--group-size", type=int, default=64,
                   help="quantization group size（须 ≥32 且为 32 的倍数；默认 64）")
    p.add_argument("--max-seq-len", type=int, default=0, help="override max_seq_len")
    p.add_argument("--no-lm-head-i4", action="store_true",
                   help="tied 模型默认把 lm_head 也量化导出（embed 的独立副本，"
                        "流量 544→~70MB/token）；此开关关闭，退回 fp32 lm_head（A/B 对照用）")
    args = p.parse_args()

    if args.group_size < 32 or args.group_size % 32 != 0:
        sys.exit(f"error: --group-size 须 ≥32 且为 32 的倍数（NEON kernel 约束），"
                 f"实际 {args.group_size}")

    model_dir = find_local_dir(args.model)
    tcfg, model_type = load_config(model_dir)
    tiny_names, src_of, transform_of = plan_tensors(tcfg, model_type)

    # 构造 tinyqwen 配置
    max_seq_len = args.max_seq_len or tcfg.get("max_position_embeddings", 32768)
    tied = int(tcfg.get("tie_word_embeddings", True))
    tqwen_cfg = {
        "n_layers": tcfg["num_hidden_layers"],
        "hidden_size": tcfg["hidden_size"],
        "intermediate_size": tcfg["intermediate_size"],
        "n_heads": tcfg["num_attention_heads"],
        "n_kv_heads": tcfg["num_key_value_heads"],
        "head_dim": tcfg["head_dim"],
        "vocab_size": tcfg["vocab_size"],
        "max_seq_len": max_seq_len,
        "tied": tied,
        "rms_norm_eps": tcfg["rms_norm_eps"],
        "rope_theta": tcfg["rope_theta"],
    }

    # v2 ext
    ext = {"model_type": model_type}
    if model_type == MODEL_QWEN35:
        # Qwen3.5 的 HF config 字段名与 runtime 内部不同：
        #   linear_num_key_heads   → linear_num_qk_heads
        #   linear_num_value_heads → linear_num_v_heads
        #   linear_key_head_dim    → linear_qk_head_dim
        #   linear_value_head_dim  → linear_v_head_dim
        # partial_rotary_factor / rope_theta 在 rope_parameters 子树里。
        # 与 export_qwen_to_tiny.py 的 load_config 同一套映射。
        rp = tcfg.get("rope_parameters") or {}
        ext.update({
            "linear_num_qk_heads": tcfg.get("linear_num_key_heads", 0),
            "linear_num_v_heads": tcfg.get("linear_num_value_heads", 0),
            "linear_qk_head_dim": tcfg.get("linear_key_head_dim", 0),
            "linear_v_head_dim": tcfg.get("linear_value_head_dim", 0),
            "linear_conv_kernel_dim": tcfg.get("linear_conv_kernel_dim", 4),
            "full_attention_interval": tcfg.get("full_attention_interval", 4),
            "partial_rotary_factor": rp.get("partial_rotary_factor", tcfg.get("partial_rotary_factor", 0.25)),
            "eos_token_id": tcfg.get("eos_token_id", 248044),
        })

    # 加载权重
    st = LazyTensors(model_dir)
    tensors = {}
    for tiny_name in tiny_names:
        hf_name = src_of[tiny_name]
        arr = st.get(hf_name).astype(np.float32)
        transform = transform_of[tiny_name]
        if transform is not None:
            arr = transform(arr)
        tensors[tiny_name] = arr

    # tied 模型的 lm_head：默认额外导出一份量化副本（与 embed 同权重）。
    # embed_tokens 本身保持 fp32 供 token lookup；lm_head.weight 是 2D 且不在
    # should_quantize 的 skip 列表里，写入循环会自动量化它。runtime 侧 tied
    # 时优先绑定 lm_head.weight（见 qwen_model.cpp），把每 token 544MB 的
    # fp32 流量换成 ~70MB i4。文件 +~70MB。
    if tied and not args.no_lm_head_i4:
        tensors["lm_head.weight"] = tensors["model.embed_tokens.weight"]
        print("[lm_head] tied 模型：追加 lm_head.weight 量化副本（embed 保持 fp32 lookup）")

    # 写入
    total = write_tqwen_i4(args.out, tqwen_cfg, tensors, args.group_size,
                           version=FORMAT_VERSION, ext=ext, method=args.method)

    mb = total / (1024 * 1024)
    print(f"\nDone: {args.out} ({mb:.1f} MB)")
    print_table_summary(args.out)


if __name__ == "__main__":
    main()
