#!/usr/bin/env python3
"""INT4 量化精度审计：逐层比较 fp32 与 I4 权重的量化误差。

读取 .tqwen 文件，将 I4 packed 权重反量化到 fp32，与原始 fp32 权重对比，
汇报每层 MSE / max-abs-error / cosine-similarity。

用法:
    python tools/verify_i4_accuracy.py \
        --model-i4 model_i4.tqwen \
        --model-fp32 model.tqwen
"""

from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

import numpy as np


# ---- tqwen 格式常量 ----
MAGIC = b"TQWN"
DTYPE_F32 = 0
DTYPE_I4 = 2


def read_header(f):
    """读取 tqwen v2 header，返回 (n_tensors, dtype, ext_info)。"""
    magic = f.read(4)
    assert magic == MAGIC, f"bad magic: {magic}"
    version = struct.unpack("<I", f.read(4))[0]
    assert version == 2, f"unsupported version: {version}"
    n_tensors, dtype = struct.unpack("<II", f.read(8))
    # ext header (v2): model_type(4) + n_layers(4) + hidden(4) + ... + quant_group_size(4)
    ext_data = f.read(64)
    # quant_group_size is at offset 44 in ext (after model_type, n_layers, hidden, intermediate,
    # n_heads, n_kv_heads, head_dim, vocab, max_seq, rope_base, rope_scaling) = 11*4=44
    quant_group_size = struct.unpack_from("<I", ext_data, 44)[0]
    return n_tensors, dtype, quant_group_size


def read_tensor_entries(f, n_tensors):
    """读取 tensor table，返回 [(name, dtype, dims, offset, nbytes)]。"""
    entries = []
    for _ in range(n_tensors):
        name_len = struct.unpack("<I", f.read(4))[0]
        name = f.read(name_len).decode("utf-8")
        tdtype, ndim = struct.unpack("<II", f.read(8))
        dims = list(struct.unpack(f"<{ndim}I", f.read(4 * ndim)))
        offset, nbytes = struct.unpack("<QQ", f.read(16))
        entries.append((name, tdtype, dims, offset, nbytes))
    return entries


def dequant_i4_tensor(data: bytes, dims: list[int], group_size: int) -> np.ndarray:
    """将 I4 packed tensor 反量化为 fp32 [out_dim, in_dim]。"""
    out_dim, in_dim = dims[0], dims[1]
    n_groups = (in_dim + group_size - 1) // group_size
    data_bytes_per_group = group_size // 2
    group_total = 4 + data_bytes_per_group

    result = np.zeros((out_dim, in_dim), dtype=np.float32)
    buf = np.frombuffer(data, dtype=np.uint8)

    for row in range(out_dim):
        row_offset = row * n_groups * group_total
        for g in range(n_groups):
            g_offset = row_offset + g * group_total
            # scale and zero as fp16
            scale_bytes = buf[g_offset:g_offset + 2].tobytes()
            zero_bytes = buf[g_offset + 2:g_offset + 4].tobytes()
            scale = float(np.frombuffer(scale_bytes, dtype=np.float16)[0])
            zero = float(np.frombuffer(zero_bytes, dtype=np.float16)[0])

            col_base = g * group_size
            for b in range(data_bytes_per_group):
                byte_val = int(buf[g_offset + 4 + b])
                lo = byte_val & 0x0F
                hi = (byte_val >> 4) & 0x0F
                idx0 = col_base + 2 * b
                idx1 = col_base + 2 * b + 1
                if idx0 < in_dim:
                    result[row, idx0] = (lo - zero) * scale
                if idx1 < in_dim:
                    result[row, idx1] = (hi - zero) * scale

    return result


def load_tensors(path: Path):
    """加载 tqwen 文件中所有 tensor，返回 {name: (dtype, dims, ndarray)}。"""
    tensors = {}
    with open(path, "rb") as f:
        n_tensors, file_dtype, group_size = read_header(f)
        entries = read_tensor_entries(f, n_tensors)
        data_base = f.tell()

        for name, tdtype, dims, offset, nbytes in entries:
            f.seek(data_base + offset)
            raw = f.read(nbytes)
            if tdtype == DTYPE_F32:
                arr = np.frombuffer(raw, dtype=np.float32).reshape(dims)
            elif tdtype == DTYPE_I4:
                arr = dequant_i4_tensor(raw, dims, group_size if group_size > 0 else 128)
            else:
                arr = None
            tensors[name] = (tdtype, dims, arr)

    return tensors, group_size


def compare_tensors(fp32_tensors, i4_tensors):
    """逐 tensor 比较，打印精度指标。"""
    print(f"\n{'Tensor':<50} {'MSE':>12} {'MaxErr':>12} {'CosSim':>10}")
    print("-" * 88)

    total_mse = 0.0
    count = 0

    for name in sorted(i4_tensors.keys()):
        i4_dtype, i4_dims, i4_arr = i4_tensors[name]
        if i4_dtype != DTYPE_I4:
            continue
        if name not in fp32_tensors:
            print(f"  {name:<48} {'MISSING in fp32':>36}")
            continue

        fp32_dtype, fp32_dims, fp32_arr = fp32_tensors[name]
        if fp32_arr is None:
            continue

        diff = fp32_arr.flatten() - i4_arr.flatten()
        mse = float(np.mean(diff ** 2))
        max_err = float(np.max(np.abs(diff)))

        # cosine similarity
        dot = float(np.dot(fp32_arr.flatten(), i4_arr.flatten()))
        norm_a = float(np.linalg.norm(fp32_arr.flatten()))
        norm_b = float(np.linalg.norm(i4_arr.flatten()))
        cos_sim = dot / (norm_a * norm_b + 1e-12)

        total_mse += mse
        count += 1

        print(f"  {name:<48} {mse:>12.6e} {max_err:>12.6e} {cos_sim:>10.6f}")

    if count > 0:
        print("-" * 88)
        avg_mse = total_mse / count
        print(f"  {'AVERAGE':<48} {avg_mse:>12.6e}")
        print(f"\n  共 {count} 个量化 tensor")
        if avg_mse > 0.01:
            print("  WARNING: 平均 MSE > 0.01，量化质量偏低")
            return False
    return True


def main():
    parser = argparse.ArgumentParser(description="INT4 量化精度审计")
    parser.add_argument("--model-i4", required=True, help="INT4 .tqwen 文件路径")
    parser.add_argument("--model-fp32", required=True, help="fp32 .tqwen 文件路径")
    parser.add_argument("--prompt-tokens", default="", help="(unused, 保持接口兼容)")
    args = parser.parse_args()

    i4_path = Path(args.model_i4)
    fp32_path = Path(args.model_fp32)

    if not i4_path.exists():
        print(f"ERROR: {i4_path} not found")
        sys.exit(1)
    if not fp32_path.exists():
        print(f"WARNING: {fp32_path} not found, 跳过精度对比")
        sys.exit(0)

    print(f"加载 fp32 模型: {fp32_path}")
    fp32_tensors, _ = load_tensors(fp32_path)
    print(f"加载 I4 模型:   {i4_path}")
    i4_tensors, group_size = load_tensors(i4_path)
    print(f"量化组大小: {group_size}")

    ok = compare_tensors(fp32_tensors, i4_tensors)
    if not ok:
        sys.exit(1)
    print("\n  精度审计 PASS")


if __name__ == "__main__":
    main()
