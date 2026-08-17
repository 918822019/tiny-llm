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
import os
import struct
import sys
from pathlib import Path

import numpy as np

# 复用现有 exporter 的基础设施
sys.path.insert(0, str(Path(__file__).parent))
from export_qwen_to_tiny import (  # noqa: E402
    MAGIC, FORMAT_VERSION, ALIGN, DTYPE_F32, MAX_NAME,
    HEADER_FMT, ENTRY_FMT,
    align_up, find_local_dir, load_config, plan_tensors,
    detect_model_type, text_cfg_of, print_table_summary,
    MODEL_QWEN2, MODEL_QWEN35,
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


def write_tqwen_i4(out_path: str | Path, cfg: dict, tensors: dict,
                   group_size: int, version: int = 2, ext: dict | None = None) -> int:
    """写一个 INT4 混合 dtype 的 .tqwen 文件。

    大矩阵: INT4 量化; 小向量: fp32。
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
            h = safe_open(str(st_file), framework="numpy")
            for key in h.keys():
                self._index[key] = h
            self._handles.append(h)

    def get(self, key: str) -> np.ndarray:
        if key not in self._index:
            sys.exit(f"error: tensor '{key}' not found in safetensors")
        return self._index[key].get_tensor(key)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="HF model dir")
    p.add_argument("--out", required=True, help="output .tqwen path")
    p.add_argument("--group-size", type=int, default=128, help="quantization group size")
    p.add_argument("--max-seq-len", type=int, default=0, help="override max_seq_len")
    args = p.parse_args()

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
        ext.update({
            "linear_num_qk_heads": tcfg.get("linear_num_qk_heads", 0),
            "linear_num_v_heads": tcfg.get("linear_num_v_heads", 0),
            "linear_qk_head_dim": tcfg.get("linear_qk_head_dim", 0),
            "linear_v_head_dim": tcfg.get("linear_v_head_dim", 0),
            "linear_conv_kernel_dim": tcfg.get("linear_conv_kernel_dim", 4),
            "full_attention_interval": tcfg.get("full_attention_interval", 4),
            "partial_rotary_factor": tcfg.get("partial_rotary_factor", 0.25),
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

    # 写入
    total = write_tqwen_i4(args.out, tqwen_cfg, tensors, args.group_size,
                           version=FORMAT_VERSION, ext=ext)

    mb = total / (1024 * 1024)
    print(f"\nDone: {args.out} ({mb:.1f} MB)")
    print_table_summary(args.out)


if __name__ == "__main__":
    main()
