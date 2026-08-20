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
import numpy as np
import struct
import sys
from pathlib import Path

# ---- tqwen 格式常量 ----
MAGIC = b"TINYQWEN"  # 8 字节魔数（runtime/tiny_format.h kMagic）
DTYPE_F32 = 0
DTYPE_I4 = 3  # runtime/tiny_format.h: kI4 = 3（kI8 = 2）

# 与 exporter 一致：header 192B，entry 120B
HEADER_FMT = "<8s12Iff4Q96s"
ENTRY_FMT = "<64sII4QQQ"


def parse_header_bytes(header: bytes):
    """解析 192B header，返回 (tensor_count, dtype, quant_group_size)。

    v1/v2 共用同一 192B 布局；v1 的 reserved[96] 全 0（无量化信息），
    v2 把前 64B 解读为 TinyHeaderV2Ext（含 quant_group_size）。
    """
    magic = header[:8]
    assert magic == MAGIC, f"bad magic: {magic}"
    version = struct.unpack_from("<I", header, 8)[0]
    assert version in (1, 2), f"unsupported version: {version}"
    dtype = struct.unpack_from("<I", header, 12)[0]
    # tensor_count / tensor_table_offset 是 4×u64 的前两个（offset 64 / 72）
    tensor_count = struct.unpack_from("<Q", header, 64)[0]
    quant_group_size = 0
    if version >= 2:
        # v2 ext 复用 reserved[96] 的前 64B（起始 offset 96）；quant_group_size
        # 在 ext 内 offset 40（EXT_I4_FMT = "<7If2II20s"：7×u32+f32+2×u32+u32+20s）。
        quant_group_size = struct.unpack_from("<I", header, 96 + 40)[0]
    return tensor_count, dtype, quant_group_size


def read_header(f):
    """读取 tqwen v2 header（192B），返回 (n_tensors, dtype, quant_group_size)。"""
    header = f.read(struct.calcsize(HEADER_FMT))
    result = parse_header_bytes(header)
    f.seek(0)  # 交回调用方按 tensor_table_offset 定位
    return result


def read_tensor_entries(f, n_tensors):
    """读取 tensor 表，返回 [(name, dtype, dims, offset, nbytes)]。offset 为绝对偏移。"""
    entries = []
    entry_size = struct.calcsize(ENTRY_FMT)
    for _ in range(n_tensors):
        raw = f.read(entry_size)
        name_b, tdtype, ndim, d0, d1, d2, d3, offset, nbytes = struct.unpack(
            ENTRY_FMT, raw)
        name = name_b.split(b"\x00", 1)[0].decode("ascii")
        dims = [d0, d1, d2, d3][:ndim]
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
        header = f.read(struct.calcsize(HEADER_FMT))
        n_tensors, file_dtype, group_size = parse_header_bytes(header)
        # tensor 表位置由 header 的 tensor_table_offset 给出（=192）
        table_offset = struct.unpack_from("<Q", header, 72)[0]
        f.seek(table_offset)
        entries = read_tensor_entries(f, n_tensors)

        for name, tdtype, dims, offset, nbytes in entries:
            f.seek(offset)  # entry 里的 offset 是文件绝对偏移
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
        ref_name = name
        if name not in fp32_tensors:
            # tied 模型：i4 文件带独立的 lm_head.weight 量化副本，
            # fp32 参照是共享的 embed_tokens。
            if name == "lm_head.weight" and "model.embed_tokens.weight" in fp32_tensors:
                ref_name = "model.embed_tokens.weight"
            else:
                print(f"  {name:<48} {'MISSING in fp32':>36}")
                continue

        fp32_dtype, fp32_dims, fp32_arr = fp32_tensors[ref_name]
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

        disp = name if ref_name == name else f"{name} (vs embed)"
        print(f"  {disp:<48} {mse:>12.6e} {max_err:>12.6e} {cos_sim:>10.6f}")

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
