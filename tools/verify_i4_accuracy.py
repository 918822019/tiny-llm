#!/usr/bin/env python3
"""INT4 量化精度审计：逐层比较 fp32 与 I4 权重的量化误差。

读取 .tqwen 文件，将 I4 packed 权重反量化到 fp32，与原始 fp32 权重对比，
汇报每层 MSE / max-abs-error / cosine-similarity。

用法:
    python tools/verify_i4_accuracy.py \\
        --model-i4 model_i4.tqwen \\
        --model-fp32 model.tqwen

核心设计思路：
    1. 分别加载 fp32 和 INT4 两个 .tqwen 文件的所有 tensor。
    2. 对 INT4 tensor 执行反量化（unpack nibble → (q - zero) × scale）。
    3. 逐 tensor 计算三个精度指标：
       - MSE（均方误差）：衡量整体量化噪声水平。
       - MaxAbsError（最大绝对误差）：衡量最坏情况的偏差。
       - CosineSimilarity（余弦相似度）：衡量方向一致性（接近 1.0 为佳）。
    4. 汇总平均 MSE，超过阈值 0.01 时发出警告。

输入：两个 .tqwen 文件路径（fp32 参考 + INT4 量化版）。
输出：控制台打印逐 tensor 精度表格和审计结论。
"""

from __future__ import annotations

# --- 标准库导入 ---
import argparse   # 命令行参数解析
import numpy as np  # 数值计算、反量化
import struct     # 二进制解包（header/entry）
import sys        # 错误退出
from pathlib import Path  # 路径操作

# ---- tqwen 格式常量 ----
MAGIC = b"TINYQWEN"  # 8 字节魔数（runtime/tiny_format.h kMagic）
DTYPE_F32 = 0        # float32 dtype 编号
DTYPE_I4 = 3         # INT4 dtype 编号（runtime/tiny_format.h: kI4 = 3，kI8 = 2）

# 与 exporter 一致：header 192B，entry 120B
HEADER_FMT = "<8s12Iff4Q96s"  # header 格式字符串
ENTRY_FMT = "<64sII4QQQ"      # tensor entry 格式字符串


def parse_header_bytes(header: bytes):
    """解析 192B header，返回 (tensor_count, dtype, quant_group_size)。

    v1/v2 共用同一 192B 布局；v1 的 reserved[96] 全 0（无量化信息），
    v2 把前 64B 解读为 TinyHeaderV2Ext（含 quant_group_size）。

    header 布局关键偏移：
    - [0:8]    magic (8B)
    - [8:12]   version (u32)
    - [12:16]  dtype (u32)
    - [64:72]  tensor_count (u64)
    - [72:80]  tensor_table_offset (u64)
    - [96:192] reserved (96B)，v2 时前 64B 是 TinyHeaderV2Ext

    Args:
        header: 192 字节的 header 数据。

    Returns:
        (tensor_count, dtype, quant_group_size) 三元组。
    """
    magic = header[:8]  # 提取魔数
    assert magic == MAGIC, f"bad magic: {magic}"  # 校验魔数
    version = struct.unpack_from("<I", header, 8)[0]  # 解析版本号（offset 8）
    assert version in (1, 2), f"unsupported version: {version}"  # 校验版本
    dtype = struct.unpack_from("<I", header, 12)[0]  # 解析默认 dtype（offset 12）
    # tensor_count / tensor_table_offset 是 4×u64 的前两个（offset 64 / 72）
    tensor_count = struct.unpack_from("<Q", header, 64)[0]  # tensor 总数
    quant_group_size = 0  # 默认为 0（v1 无量化信息）
    if version >= 2:
        # v2 ext 复用 reserved[96] 的前 64B（起始 offset 96）；quant_group_size
        # 在 ext 内 offset 40（EXT_I4_FMT = "<7If2II20s"：7×u32+f32+2×u32+u32+20s）。
        # 即 header[96+40] = header[136] 处的 u32
        quant_group_size = struct.unpack_from("<I", header, 96 + 40)[0]
    return tensor_count, dtype, quant_group_size


def read_header(f):
    """读取 tqwen v2 header（192B），返回 (n_tensors, dtype, quant_group_size)。

    读取后重置文件指针到开头，交由调用方按 tensor_table_offset 定位。

    Args:
        f: 已打开的文件对象。

    Returns:
        (n_tensors, dtype, quant_group_size) 三元组。
    """
    header = f.read(struct.calcsize(HEADER_FMT))  # 读取 192 字节 header
    result = parse_header_bytes(header)  # 解析 header
    f.seek(0)  # 交回调用方按 tensor_table_offset 定位
    return result


def read_tensor_entries(f, n_tensors):
    """读取 tensor 索引表，返回 [(name, dtype, dims, offset, nbytes)]。

    offset 为文件中的绝对偏移（从文件头算起）。

    Args:
        f: 已定位到 tensor 表起始位置的文件对象。
        n_tensors: tensor 总数。

    Returns:
        元组列表，每个元素为 (name, dtype_code, dims_list, offset, nbytes)。
    """
    entries = []  # 存放所有 entry
    entry_size = struct.calcsize(ENTRY_FMT)  # 每个 entry 120 字节
    for _ in range(n_tensors):
        raw = f.read(entry_size)  # 读取一个 entry
        # 解包 entry 字段
        name_b, tdtype, ndim, d0, d1, d2, d3, offset, nbytes = struct.unpack(
            ENTRY_FMT, raw)
        # 解码 tensor 名字（截断到第一个零字节）
        name = name_b.split(b"\x00", 1)[0].decode("ascii")
        # 根据实际维度数截取有效 shape
        dims = [d0, d1, d2, d3][:ndim]
        entries.append((name, tdtype, dims, offset, nbytes))
    return entries


def dequant_i4_tensor(data: bytes, dims: list[int], group_size: int) -> np.ndarray:
    """将 I4 packed tensor 反量化为 fp32 [out_dim, in_dim]。

    反量化公式：value = (uint4_value - zero_point) × scale
    这与 HQQ 库和 C++ runtime 的反量化语义完全一致。

    磁盘布局（每行每组）：
        [scale_fp16(2B) | zero_fp16(2B) | packed_nibbles(group_size/2 B)]
    其中 packed_nibbles 中低 nibble 在前（little-endian nibble order）。

    Args:
        data: 量化后的原始字节数据。
        dims: [out_dim, in_dim] 形状列表。
        group_size: 量化组大小。

    Returns:
        形状为 [out_dim, in_dim] 的 fp32 numpy 数组。
    """
    out_dim, in_dim = dims[0], dims[1]  # 获取矩阵维度
    n_groups = (in_dim + group_size - 1) // group_size  # 每行的组数（向上取整）
    data_bytes_per_group = group_size // 2  # 每组 packed 数据的字节数
    group_total = 4 + data_bytes_per_group  # 每组总字节数 = 4B头部 + packed数据

    result = np.zeros((out_dim, in_dim), dtype=np.float32)  # 预分配输出矩阵
    buf = np.frombuffer(data, dtype=np.uint8)  # 将字节数据视为 uint8 数组

    for row in range(out_dim):  # 遍历每一行
        row_offset = row * n_groups * group_total  # 当前行在缓冲区中的起始偏移
        for g in range(n_groups):  # 遍历该行的每个量化组
            g_offset = row_offset + g * group_total  # 该组在缓冲区中的偏移
            # 读取 scale 和 zero（各 2 字节 fp16 little-endian）
            scale_bytes = buf[g_offset:g_offset + 2].tobytes()    # scale 的 2 字节
            zero_bytes = buf[g_offset + 2:g_offset + 4].tobytes()  # zero 的 2 字节
            # 将 fp16 字节转为 Python float
            scale = float(np.frombuffer(scale_bytes, dtype=np.float16)[0])
            zero = float(np.frombuffer(zero_bytes, dtype=np.float16)[0])

            col_base = g * group_size  # 该组对应的列起始索引
            # 遍历 packed 字节，每个字节包含两个 uint4 值
            for b in range(data_bytes_per_group):
                byte_val = int(buf[g_offset + 4 + b])  # 读取一个 packed 字节
                lo = byte_val & 0x0F          # 低 4 位（第一个 uint4 值）
                hi = (byte_val >> 4) & 0x0F   # 高 4 位（第二个 uint4 值）
                idx0 = col_base + 2 * b       # 第一个元素的列索引
                idx1 = col_base + 2 * b + 1   # 第二个元素的列索引
                # 反量化：value = (q - zero) × scale
                if idx0 < in_dim:
                    result[row, idx0] = (lo - zero) * scale  # 反量化第一个元素
                if idx1 < in_dim:
                    result[row, idx1] = (hi - zero) * scale  # 反量化第二个元素

    return result


def load_tensors(path: Path):
    """加载 tqwen 文件中所有 tensor，返回 {name: (dtype, dims, ndarray)}。

    自动处理 F32 和 I4 两种 dtype：
    - F32 tensor 直接 reshape 为 numpy 数组。
    - I4 tensor 先反量化再得到 fp32 numpy 数组。

    Args:
        path: .tqwen 文件路径。

    Returns:
        (tensors, group_size) 元组：
        - tensors: {name: (dtype_code, dims, numpy_array)} 字典。
        - group_size: 量化组大小（v1 时为 0）。
    """
    tensors = {}  # 存放所有 tensor
    with open(path, "rb") as f:
        # 读取并解析 header
        header = f.read(struct.calcsize(HEADER_FMT))
        n_tensors, file_dtype, group_size = parse_header_bytes(header)
        # tensor 表位置由 header 的 tensor_table_offset 给出（=192）
        table_offset = struct.unpack_from("<Q", header, 72)[0]
        f.seek(table_offset)  # 跳转到 tensor 表
        entries = read_tensor_entries(f, n_tensors)  # 读取所有 entry

        # 逐个加载 tensor 数据
        for name, tdtype, dims, offset, nbytes in entries:
            f.seek(offset)  # entry 里的 offset 是文件绝对偏移
            raw = f.read(nbytes)  # 读取原始字节数据
            if tdtype == DTYPE_F32:
                # fp32 tensor：直接解析并 reshape
                arr = np.frombuffer(raw, dtype=np.float32).reshape(dims)
            elif tdtype == DTYPE_I4:
                # INT4 tensor：反量化为 fp32
                arr = dequant_i4_tensor(raw, dims, group_size if group_size > 0 else 128)
            else:
                arr = None  # 未知 dtype 跳过
            tensors[name] = (tdtype, dims, arr)

    return tensors, group_size


def compare_tensors(fp32_tensors, i4_tensors):
    """逐 tensor 比较 fp32 参考与 INT4 反量化结果，打印精度指标。

    对每个 INT4 量化的 tensor，计算并打印三个指标：
    - MSE（均方误差）：整体量化噪声。
    - MaxErr（最大绝对误差）：最坏情况偏差。
    - CosSim（余弦相似度）：方向一致性。

    对于 tied embedding 模型，i4 文件中的 lm_head.weight 是 embed_tokens
    的量化副本，参照时使用 embed_tokens 的 fp32 权重。

    Args:
        fp32_tensors: fp32 参考模型的 tensor 字典。
        i4_tensors: INT4 量化模型的 tensor 字典。

    Returns:
        True 表示审计通过（平均 MSE <= 0.01），False 表示质量偏低。
    """
    # 打印表头
    print(f"\n{'Tensor':<50} {'MSE':>12} {'MaxErr':>12} {'CosSim':>10}")
    print("-" * 88)

    total_mse = 0.0  # 累计 MSE
    count = 0        # 已比较的 tensor 计数

    for name in sorted(i4_tensors.keys()):  # 按名字排序遍历
        i4_dtype, i4_dims, i4_arr = i4_tensors[name]
        if i4_dtype != DTYPE_I4:
            continue  # 只比较 INT4 量化的 tensor（跳过 fp32 小向量）
        ref_name = name  # 默认参照同名 tensor
        if name not in fp32_tensors:
            # tied 模型：i4 文件带独立的 lm_head.weight 量化副本，
            # fp32 参照是共享的 embed_tokens。
            if name == "lm_head.weight" and "model.embed_tokens.weight" in fp32_tensors:
                ref_name = "model.embed_tokens.weight"  # 使用 embed 作为参照
            else:
                print(f"  {name:<48} {'MISSING in fp32':>36}")  # 找不到参照
                continue

        fp32_dtype, fp32_dims, fp32_arr = fp32_tensors[ref_name]
        if fp32_arr is None:
            continue  # 跳过无法解析的 tensor

        # 计算差异
        diff = fp32_arr.flatten() - i4_arr.flatten()  # 逐元素差值
        mse = float(np.mean(diff ** 2))               # 均方误差
        max_err = float(np.max(np.abs(diff)))          # 最大绝对误差

        # 计算余弦相似度：cos(a,b) = dot(a,b) / (||a|| × ||b||)
        dot = float(np.dot(fp32_arr.flatten(), i4_arr.flatten()))  # 点积
        norm_a = float(np.linalg.norm(fp32_arr.flatten()))         # fp32 范数
        norm_b = float(np.linalg.norm(i4_arr.flatten()))           # i4 范数
        cos_sim = dot / (norm_a * norm_b + 1e-12)  # 加 epsilon 避免除零

        total_mse += mse  # 累加 MSE
        count += 1        # 计数 +1

        # 显示名字（如果是 tied lm_head，标注参照来源）
        disp = name if ref_name == name else f"{name} (vs embed)"
        print(f"  {disp:<48} {mse:>12.6e} {max_err:>12.6e} {cos_sim:>10.6f}")

    if count > 0:
        print("-" * 88)
        avg_mse = total_mse / count  # 计算平均 MSE
        print(f"  {'AVERAGE':<48} {avg_mse:>12.6e}")
        print(f"\n  共 {count} 个量化 tensor")
        if avg_mse > 0.01:
            # 平均 MSE 超过阈值，量化质量偏低
            print("  WARNING: 平均 MSE > 0.01，量化质量偏低")
            return False
    return True  # 审计通过


def main():
    """主入口函数：执行 INT4 量化精度审计。

    流程：
    1. 解析命令行参数（i4 文件路径、fp32 文件路径）。
    2. 校验文件存在性。
    3. 加载两个模型的所有 tensor。
    4. 逐 tensor 比较精度指标。
    5. 打印审计结论。
    """
    parser = argparse.ArgumentParser(description="INT4 量化精度审计")
    parser.add_argument("--model-i4", required=True, help="INT4 .tqwen 文件路径")  # INT4 模型
    parser.add_argument("--model-fp32", required=True, help="fp32 .tqwen 文件路径")  # fp32 参考
    parser.add_argument("--prompt-tokens", default="", help="(unused, 保持接口兼容)")  # 保留参数
    args = parser.parse_args()

    i4_path = Path(args.model_i4)    # INT4 文件路径
    fp32_path = Path(args.model_fp32)  # fp32 文件路径

    # 校验文件存在性
    if not i4_path.exists():
        print(f"ERROR: {i4_path} not found")
        sys.exit(1)
    if not fp32_path.exists():
        print(f"WARNING: {fp32_path} not found, 跳过精度对比")
        sys.exit(0)

    # 加载两个模型的 tensor
    print(f"加载 fp32 模型: {fp32_path}")
    fp32_tensors, _ = load_tensors(fp32_path)
    print(f"加载 I4 模型:   {i4_path}")
    i4_tensors, group_size = load_tensors(i4_path)
    print(f"量化组大小: {group_size}")

    # 执行精度比较
    ok = compare_tensors(fp32_tensors, i4_tensors)
    if not ok:
        sys.exit(1)  # 审计未通过
    print("\n  精度审计 PASS")  # 审计通过


if __name__ == "__main__":
    main()  # 脚本入口点
