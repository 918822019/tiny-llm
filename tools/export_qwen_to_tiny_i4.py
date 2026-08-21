#!/usr/bin/env python3
"""将 Qwen2.5（HF）权重导出为 INT4 量化的 .tqwen 文件。

非对称 uint4 [0,15]，per-group(128) scale+zero_point（fp16），interleaved packing。
小向量（norm/bias/embed）保持 fp32。RTN (round-to-nearest) 量化。

用法:
    python tools/export_qwen_to_tiny_i4.py \\
        --model Qwen/Qwen2.5-0.5B \\
        --out model_i4.tqwen \\
        --group-size 128

本脚本的核心流程：
    1. 复用 export_qwen_to_tiny.py 的基础设施（config 加载、tensor 名映射等）。
    2. 从 safetensors 加载 HF 权重并转为 fp32 numpy。
    3. 对大矩阵（linear projection）做 INT4 量化（支持 RTN 和 HQQ 两种算法），
       小向量（norm/bias/embed/conv1d/A_log/dt_bias）保持 fp32。
    4. 写出混合 dtype 的 .tqwen 文件：header 中标记主 dtype=I4，但每个 tensor
       entry 有自己的 dtype_code，runtime 按 entry 级别区分 I4 和 F32。

量化格式说明：
    - 每组 group_size 个元素共享一个 scale(fp16) 和 zero_point(fp16)。
    - 反量化公式：value = (uint4 - zero) * scale（HQQ 语义）。
    - 打包方式：两个 uint4 值拼成一个字节，低 nibble 在前（little-endian nibble order）。
    - 每组在磁盘上的布局：[scale_fp16(2B) | zero_fp16(2B) | packed_data(group_size/2 B)]。
    - 每行的总字节数 = n_groups × (4 + group_size/2)。

输入：HF 模型目录（含 config.json + model.safetensors*）。
输出：单个 .tqwen 二进制文件（INT4 量化版）。
关键设计：HQQ 量化通过 hqq 库实现 MSE 最优迭代优化；RTN 作为基线对照。
"""

from __future__ import annotations

# --- 标准库导入 ---
import argparse   # 命令行参数解析
import numpy as np  # 数组操作、量化计算
import os         # 文件大小查询
import struct     # 二进制打包/解包
import sys        # 错误退出、路径注入
from pathlib import Path  # 路径操作

# 复用现有 exporter 的基础设施（header 格式、对齐函数、config 加载等）
sys.path.insert(0, str(Path(__file__).parent))  # 将 tools/ 目录加入搜索路径
from export_qwen_to_tiny import (  # noqa: E402
    MAGIC, FORMAT_VERSION, DTYPE_F32, MAX_NAME,      # 常量：魔数、版本号、dtype、名字上限
    HEADER_FMT, ENTRY_FMT,                            # struct 格式字符串
    align_up, find_local_dir, load_config, plan_tensors,  # 工具函数
    print_table_summary,                              # 摘要打印
    MODEL_QWEN35,                                     # Qwen3.5 架构族标识
)

# INT4 dtype 编号，与 runtime/tiny_format.h 的 Dtype::I4 一致
DTYPE_I4 = 3
# 每个量化组的头部大小：scale(fp16, 2B) + zero(fp16, 2B) = 4 字节
GROUP_HEADER_BYTES = 4  # scale(fp16, 2B) + zero(fp16, 2B)

# v2 ext 格式：增加 quant_group_size（从 ext_reserved 的前 4 字节）
# 7 个 u32 + float + 2 个 u32 + u32(quant_group_size) + 20 字节 reserved
# 相比 f16 导出的 EXT_FMT，末尾多了 quant_group_size 字段
EXT_I4_FMT = "<7If2II20s"  # 64 字节
assert struct.calcsize(EXT_I4_FMT) == 64  # 编译期断言扩展块大小


def pack_v2_ext_i4(ext: dict, group_size: int) -> bytes:
    """将 v2 扩展字段 + 量化组大小打包成 64 字节。

    与 f16 版本的 pack_v2_ext 类似，但在 pad 字段位置改为存储
    quant_group_size，供 runtime 读取时知道量化分组粒度。

    Args:
        ext: 包含模型架构参数的字典。
        group_size: 量化组大小（如 64、128）。

    Returns:
        64 字节的 packed 二进制数据。
    """
    g = lambda k: ext.get(k, 0)  # noqa: E731  # 便捷取值函数，缺失返回 0
    return struct.pack(
        EXT_I4_FMT,                         # 按 I4 专用格式打包
        int(g("model_type")),               # 架构族标识
        int(g("linear_num_qk_heads")),      # 线性注意力 QK head 数
        int(g("linear_num_v_heads")),       # 线性注意力 V head 数
        int(g("linear_qk_head_dim")),       # 线性注意力 QK head 维度
        int(g("linear_v_head_dim")),        # 线性注意力 V head 维度
        int(g("linear_conv_kernel_dim")),   # depthwise conv1d 核大小
        int(g("full_attention_interval")),  # full attention 层间隔
        float(g("partial_rotary_factor")),  # 部分 RoPE 比例因子
        int(g("eos_token_id")),             # EOS token ID
        0,                                  # pad 字段（保留）
        int(group_size),                    # 量化组大小（I4 特有字段）
        b"\x00" * 20,                       # 尾部 20 字节保留区，全零填充
    )


def i4_row_bytes(in_dim: int, group_size: int) -> int:
    """计算一行 INT4 量化数据占用的字节数。

    每行对应一个输出通道（out_dim 的一行），包含若干个量化组，
    每组由 4 字节头部 + group_size/2 字节的 packed nibble 数据组成。

    Args:
        in_dim: 输入维度（该行的元素数）。
        group_size: 量化组大小。

    Returns:
        该行量化后的总字节数。
    """
    n_groups = (in_dim + group_size - 1) // group_size  # 向上取整计算组数
    return n_groups * (GROUP_HEADER_BYTES + group_size // 2)  # 每组 = 4B头 + gs/2 B数据


def should_quantize(name: str) -> bool:
    """判断给定 tensor 是否应该做 INT4 量化。

    大投影矩阵（q/k/v/o/gate/up/down_proj）走 INT4；
    小向量（norm/bias/embed/conv1d/A_log/dt_bias）保持 fp32，
    因为这些 tensor 对量化误差敏感或本身很小不值得量化。

    Args:
        name: .tqwen 中的 tensor 名字。

    Returns:
        True 表示应该量化，False 表示保持 fp32。
    """
    skip_patterns = [
        "embed_tokens",   # embedding 表：lookup 操作需要精确值
        "layernorm",      # LayerNorm/RMSNorm 权重：对小值敏感
        "norm.weight",    # 各种 norm 权重
        "norm_weight",    # norm 权重的另一种命名
        "bias",           # 偏置项：通常很小，量化收益低
        "conv1d",         # depthwise conv1d 权重：kernel 很小
        "A_log",          # SSM 状态矩阵 A 的对数：精度敏感
        "dt_bias",        # SSM dt 偏置：精度敏感
    ]
    for pat in skip_patterns:
        if pat in name:    # 如果名字包含任一跳过模式
            return False   # 不量化，保持 fp32
    return True  # 其余 tensor（主要是 linear projection）做 INT4 量化


def quantize_i4_group(data: np.ndarray) -> tuple[np.float16, np.float16, np.ndarray]:
    """对一组 float32 数据做非对称 RTN（Round-To-Nearest）INT4 量化。

    RTN 量化原理：
    1. 计算该组数据的 min/max 范围。
    2. scale = (max - min) / 15，将浮点范围映射到 [0, 15] 整数区间。
    3. zero_point = -min / scale，使得 uint4=0 对应 vmin。
    4. 量化：q = round(data / scale + zero_point)，clip 到 [0, 15]。
    5. scale 和 zero_point 都存为 fp16（与 C++ 反量化对齐）。

    注意：量化时使用 rounded fp16 的 scale/zero 值（而非原始 fp32 值），
    这保证 Python 侧量化结果与 C++ runtime 反量化后的数值完全一致。

    Args:
        data: 一维 float32 数组，长度 <= group_size。

    Returns:
        (scale_fp16, zero_fp16, quantized_uint8_array) 三元组。
        quantized 数组的元素值在 [0, 15] 范围内。
    """
    vmin = data.min()  # 组内最小值
    vmax = data.max()  # 组内最大值

    if vmax == vmin:
        # 特殊情况：所有元素相同，scale=0 避免除零
        scale = np.float16(0.0)    # scale 为零
        zero = np.float16(0.0)     # zero 为零
        quantized = np.zeros(len(data), dtype=np.uint8)  # 全部量化为 0
    else:
        # 标准非对称量化：将 [vmin, vmax] 映射到 [0, 15]
        scale_f32 = (vmax - vmin) / 15.0   # 每个 uint4 步长对应的浮点增量
        zero_f32 = -vmin / scale_f32       # uint4=0 maps to vmin（零点偏移）

        # 转为 fp16 存储（与 C++ runtime 的反量化精度对齐）
        scale = np.float16(scale_f32)
        zero = np.float16(zero_f32)

        # 用 rounded fp16 值做量化（与 C++ 反量化对齐）
        # 这确保 Python 量化 → C++ 反量化的往返误差最小
        s = float(scale)   # 将 fp16 scale 转回 float 用于计算
        z = float(zero)    # 将 fp16 zero 转回 float 用于计算
        if s == 0:
            quantized = np.zeros(len(data), dtype=np.uint8)  # scale 为零则全量化为 0
        else:
            # RTN 量化核心公式：q = round(x / scale + zero_point)
            q = np.round(data / s + z).astype(np.int32)
            # clip 到 uint4 有效范围 [0, 15]
            quantized = np.clip(q, 0, 15).astype(np.uint8)

    return scale, zero, quantized


def quantize_tensor_i4(tensor: np.ndarray, group_size: int) -> bytes:
    """将 [out_dim, in_dim] 的 fp32 tensor 量化为 interleaved INT4 格式（RTN 方法）。

    输出格式（每行每个组）：
        [scale_fp16(2B) | zero_fp16(2B) | packed_nibbles(group_size/2 B)]
    其中 packed_nibbles 中每两个 uint4 拼成一个字节，低 nibble 在前。

    Args:
        tensor: 形状为 [out_dim, in_dim] 的 fp32 numpy 数组。
        group_size: 量化组大小。

    Returns:
        量化后的字节串，长度为 out_dim × i4_row_bytes(in_dim, group_size)。
    """
    assert tensor.ndim == 2  # 必须是 2D 矩阵
    out_dim, in_dim = tensor.shape  # 获取矩阵维度
    row_bytes = i4_row_bytes(in_dim, group_size)  # 每行量化后的字节数
    result = bytearray(out_dim * row_bytes)  # 预分配输出缓冲区

    groups_per_row = (in_dim + group_size - 1) // group_size  # 每行的组数
    group_total = GROUP_HEADER_BYTES + group_size // 2  # 每组占用的总字节数

    for o in range(out_dim):  # 遍历每一行（输出通道）
        row_offset = o * row_bytes  # 当前行在输出缓冲区的起始偏移
        for g in range(groups_per_row):  # 遍历该行的每个量化组
            start = g * group_size                     # 组起始列索引
            end = min(start + group_size, in_dim)      # 组结束列索引（处理边界不满的情况）
            group_data = tensor[o, start:end].astype(np.float32)  # 提取该组数据并确保 fp32

            # 对该组做 RTN 量化，得到 scale、zero、量化后的 uint4 值
            scale, zero, quantized = quantize_i4_group(group_data)

            group_offset = row_offset + g * group_total  # 该组在输出缓冲区的偏移
            # 写 scale + zero（fp16 little-endian）
            result[group_offset:group_offset + 2] = scale.tobytes()      # 写入 scale（2B fp16 LE）
            result[group_offset + 2:group_offset + 4] = zero.tobytes()   # 写入 zero（2B fp16 LE）

            # pack nibbles: low nibble first（低 4 位在前，高 4 位在后）
            data_offset = group_offset + GROUP_HEADER_BYTES  # packed 数据的起始偏移
            n_elems = len(quantized)  # 该组实际元素数（最后一组可能不满 group_size）
            for i in range(0, n_elems, 2):  # 每次处理两个元素，拼成一个字节
                lo = quantized[i]                                    # 低 nibble（偶数索引）
                hi = quantized[i + 1] if i + 1 < n_elems else 0     # 高 nibble（奇数索引，越界补 0）
                result[data_offset + i // 2] = int(hi << 4) | int(lo)  # 拼接：高4位|低4位

    return bytes(result)  # 转为不可变字节串返回


def quantize_tensor_i4_hqq(tensor: np.ndarray, group_size: int) -> bytes:
    """HQQ（半二次优化，MSE 最优、免校准）INT4 量化。

    输出与 RTN 完全相同的 interleaved 格式：
    每组 [scale_fp16 | zero_fp16 | packed_uint4]，低 nibble 在前。

    HQQ vs RTN 的区别：
    - RTN 是简单的 min-max 线性映射 + 四舍五入，速度快但精度差。
    - HQQ 通过迭代优化寻找使 MSE(weight, dequant(quant(weight))) 最小的
      scale/zero 参数，无需校准数据，精度显著优于 RTN。

    hqq 库的反量化语义 (q - zero) × scale、分组方向（沿 in_dim、组序 row-major）
    与本仓库 C++ kernel 完全一致——探包验证过逐位重组 == hq.dequantize()——
    因此只需把 unpack 出的 q / scale / zero 直接 repack。

    Args:
        tensor: 形状为 [out_dim, in_dim] 的 fp32 numpy 数组。
        group_size: 量化组大小（HQQ 要求 in_dim % group_size == 0）。

    Returns:
        量化后的字节串，格式与 quantize_tensor_i4 完全相同。

    Raises:
        ValueError: in_dim 不能被 group_size 整除时抛出。
    """
    import torch  # 延迟导入：只有 method=hqq 才需要 torch
    from hqq.core.quantize import BaseQuantizeConfig, HQQLinear, Quantizer  # HQQ 量化库

    out_dim, in_dim = tensor.shape  # 获取矩阵维度
    if in_dim % group_size != 0:
        # HQQ 库要求 in_dim 必须被 group_size 整除
        raise ValueError(
            f"HQQ 要求 in_dim % group_size == 0（实际 {in_dim} % {group_size}）；"
            f"该 tensor 可退回 --method rtn")

    # 创建一个临时的 Linear 层来承载权重（HQQ 需要通过 HQQLinear 接口量化）
    linear = torch.nn.Linear(in_dim, out_dim, bias=False)
    # 将 numpy 权重拷贝到 Linear 层（确保 contiguous 内存布局）
    linear.weight.data = torch.from_numpy(np.ascontiguousarray(tensor))
    # 配置 HQQ 量化参数：4bit、指定组大小、不对 zero/scale 再做量化
    cfg = BaseQuantizeConfig(nbits=4, group_size=group_size,
                             quant_zero=False, quant_scale=False)
    # 执行 HQQ 量化（内部进行迭代优化寻找最优 scale/zero）
    hq = HQQLinear(linear, cfg, compute_dtype=torch.float32, device="cpu")

    # 从 HQQ 内部表示 unpack 出整数 q 值
    # q 的形状为 [n_groups_total, group_size]，组序 = row * groups_per_row + g
    q = Quantizer.unpack["4bit_u8"](hq.W_q, dtype=torch.float32)
    # 确保 q 值在 [0, 15] 范围内并转为 uint8 numpy
    q_np = q.round().clamp(0, 15).to(torch.uint8).numpy()
    # 提取 scale 和 zero，展平后转为 fp16 numpy（与磁盘格式一致）
    scale = hq.meta["scale"].float().flatten().numpy().astype(np.float16)
    zero = hq.meta["zero"].float().flatten().numpy().astype(np.float16)

    # 以下打包逻辑与 RTN 版本完全相同，只是数据来源不同
    row_bytes = i4_row_bytes(in_dim, group_size)  # 每行字节数
    result = bytearray(out_dim * row_bytes)       # 预分配输出缓冲区
    groups_per_row = in_dim // group_size          # 每行组数（HQQ 保证整除）
    group_total = GROUP_HEADER_BYTES + group_size // 2  # 每组总字节数
    for o in range(out_dim):  # 遍历每行
        row_offset = o * row_bytes  # 行偏移
        for g in range(groups_per_row):  # 遍历每组
            gi = o * groups_per_row + g  # 全局组索引（row-major 顺序）
            group_offset = row_offset + g * group_total  # 组在输出中的偏移
            # 写入 scale 和 zero（fp16 LE）
            result[group_offset:group_offset + 2] = scale[gi].tobytes()
            result[group_offset + 2:group_offset + 4] = zero[gi].tobytes()
            qv = q_np[gi]  # 该组的量化整数值数组
            data_offset = group_offset + GROUP_HEADER_BYTES  # packed 数据偏移
            # 两两打包 nibble：低 nibble 在前
            for i in range(0, group_size, 2):
                result[data_offset + i // 2] = int(qv[i + 1]) << 4 | int(qv[i])

    return bytes(result)


def write_tqwen_i4(out_path: str | Path, cfg: dict, tensors: dict,
                   group_size: int, version: int = 2, ext: dict | None = None,
                   method: str = "hqq") -> int:
    """写一个 INT4 混合 dtype 的 .tqwen 文件。

    大矩阵: INT4 量化（method=hqq 用 HQQ 优化量化，rtn 用 min-max 舍入）;
    小向量: fp32。

    与 f16 版本的 write_tqwen 的主要区别：
    - 每个 tensor 有独立的 dtype_code（I4 或 F32），而非全局统一。
    - 量化后的 tensor 不再是简单的 numpy tobytes，而是自定义 interleaved 格式。
    - header.dtype 设为 DTYPE_I4 作为默认/主标记。
    - v2 ext 中额外记录 quant_group_size。

    Args:
        out_path: 输出文件路径。
        cfg: 模型超参配置字典。
        tensors: {name: fp32_numpy_array} 的有序映射。
        group_size: 量化组大小。
        version: 格式版本号（默认 2）。
        ext: v2 扩展字段字典。
        method: 量化算法，"hqq" 或 "rtn"。

    Returns:
        写出文件的总字节数。
    """
    names = list(tensors.keys())  # 获取有序的 tensor 名列表

    # 计算 tensor 索引表的结束位置和数据区起始偏移
    table_end = struct.calcsize(HEADER_FMT) + len(names) * struct.calcsize(ENTRY_FMT)
    data_offset = align_up(table_end)  # 数据区 64B 对齐

    # 第一遍：分类 + 计算 layout（量化/保持 fp32，计算偏移和大小）
    entries = []  # (name, shape, offset, nbytes, dtype_code, packed_data)
    offset = data_offset  # 当前数据偏移

    print(f"Quantizing {len(names)} tensors (group_size={group_size})...")
    for name in names:
        arr = np.asarray(tensors[name], dtype=np.float32)  # 确保 fp32
        shape = list(arr.shape)  # 获取形状
        assert 1 <= len(shape) <= 4, f"{name}: ndim {len(shape)} not in [1,4]"  # 维度校验
        assert len(name) <= MAX_NAME, f"tensor name too long: {name}"  # 名字长度校验

        if arr.ndim == 2 and should_quantize(name):
            # 2D 矩阵且不在跳过列表中 → INT4 量化
            if method == "hqq":
                packed = quantize_tensor_i4_hqq(arr, group_size)  # HQQ 优化量化
            else:
                packed = quantize_tensor_i4(arr, group_size)      # RTN 基线量化
            nbytes = len(packed)       # 量化后的字节数
            dtype_code = DTYPE_I4      # 标记为 INT4
            print(f"  [I4] {name:<50} {str(shape):<18} -> {nbytes:>10} bytes")
        else:
            # 小向量或不适合量化的 tensor → 保持 fp32
            packed = arr.tobytes()     # 直接序列化 fp32 数据
            nbytes = len(packed)       # fp32 字节数
            dtype_code = DTYPE_F32     # 标记为 F32
            print(f"  [F32] {name:<49} {str(shape):<18} -> {nbytes:>10} bytes")

        entries.append((name, shape, offset, nbytes, dtype_code, packed))
        offset = align_up(offset + nbytes)  # 下一个 tensor 的偏移（64B 对齐）

    total_bytes = offset  # 文件总大小

    # 构造 header 的 reserved 区（含 v2 扩展 + quant_group_size）
    reserved = b"\x00" * 96
    if version >= 2:
        ext_bytes = pack_v2_ext_i4(ext or {}, group_size)  # 打包 v2 扩展（含量化组大小）
        reserved = ext_bytes + b"\x00" * (96 - len(ext_bytes))  # 不足 96B 补零

    # 打包 192 字节的文件头
    header = struct.pack(
        HEADER_FMT,
        MAGIC,                              # 魔数 "TINYQWEN"
        version,                            # 格式版本号
        DTYPE_I4,                           # 主 dtype = INT4（混合 dtype 文件的默认标记）
        int(cfg["n_layers"]),               # transformer 层数
        int(cfg["hidden_size"]),            # 隐藏层维度
        int(cfg["intermediate_size"]),      # MLP 中间维度
        int(cfg["n_heads"]),                # 注意力 head 数
        int(cfg["n_kv_heads"]),             # KV head 数
        int(cfg["head_dim"]),               # head 维度
        int(cfg["vocab_size"]),             # 词表大小
        int(cfg["max_seq_len"]),            # 最大序列长度
        int(cfg["tied"]),                   # 是否 tied embedding
        0,                                  # reserved_u32
        float(cfg["rms_norm_eps"]),         # RMSNorm epsilon
        float(cfg["rope_theta"]),           # RoPE base frequency
        len(names),                         # tensor 总数
        struct.calcsize(HEADER_FMT),        # tensor 表偏移
        data_offset,                        # 数据区起始偏移
        total_bytes,                        # 文件总大小
        reserved,                           # reserved 区（含 v2 扩展）
    )

    # 写入文件
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)  # 自动创建目录
    with open(out_path, "wb") as out:
        out.write(header)  # 写入 192B header
        # 写入 tensor 索引表
        for name, shape, off, nbytes, dtype_code, _ in entries:
            out.write(struct.pack(
                ENTRY_FMT,
                name.encode("ascii"),       # tensor 名字（ASCII，不足 64B 补零）
                dtype_code,                 # 该 tensor 的 dtype（I4 或 F32）
                len(shape),                 # 维度数
                *shape,                     # 各维度大小
                *[0] * (4 - len(shape)),    # 不足 4 维补零
                off,                        # 数据偏移
                nbytes,                     # 数据字节数
            ))
        # 补零到 data_offset（表区与数据区之间的对齐填充）
        out.write(b"\x00" * (data_offset - out.tell()))

        # 写入各 tensor 的实际数据
        for name, shape, off, nbytes, dtype_code, packed in entries:
            # 断言文件指针与预期偏移一致（检测对齐 bug）
            assert out.tell() == off, f"alignment bug at {name}: {out.tell()} != {off}"
            out.write(packed)  # 写入量化/fp32 数据
            # 每个 tensor 之后补零到 64B 边界
            out.write(b"\x00" * (align_up(off + nbytes) - (off + nbytes)))
        assert out.tell() == total_bytes  # 最终大小校验

    # 验证文件大小
    size = os.path.getsize(out_path)
    assert size == total_bytes, f"file size {size} != expected {total_bytes}"
    return total_bytes


# ---- 懒加载 safetensors -----------------------------------------------

class LazyTensors:
    """按需从 safetensors 文件加载并转换 tensor。

    与 export_qwen_to_tiny.py 中的 LazyTensors 不同，这里在初始化时就
    扫描所有 .safetensors 文件并建立 key->handle 索引，get() 时直接从
    缓存的 handle 读取并转为 fp32 numpy。

    Attributes:
        _handles: 已打开的 safe_open 句柄列表。
        _index: {tensor_name: safe_open_handle} 的映射。
    """

    def __init__(self, model_dir: Path):
        """初始化：扫描模型目录下所有 .safetensors 文件并建立索引。

        Args:
            model_dir: HF 模型目录路径。

        Raises:
            SystemExit: 缺少 safetensors 库时退出。
        """
        try:
            from safetensors import safe_open  # 延迟导入 safetensors
        except ImportError:
            sys.exit("error: pip install safetensors")
        self._handles = []   # 保存所有打开的文件句柄（防止 GC 关闭）
        self._index = {}     # tensor 名 -> 所在文件的 safe_open 句柄
        for st_file in sorted(model_dir.glob("*.safetensors")):
            # 用 torch 框架读：真实 Qwen 权重是 bf16，numpy 不认识 bf16；
            # get() 里统一转成 fp32（与 export_qwen_to_tiny.py 同款做法）。
            h = safe_open(str(st_file), framework="torch")  # 打开 safetensors 文件
            for key in h.keys():
                self._index[key] = h  # 记录每个 tensor 所在的文件句柄
            self._handles.append(h)   # 保持句柄引用

    def get(self, key: str) -> np.ndarray:
        """按名字加载 tensor 并转为 fp32 numpy 数组。

        Args:
            key: HF tensor 名字。

        Returns:
            fp32 numpy 数组。

        Raises:
            SystemExit: tensor 不存在时退出。
        """
        if key not in self._index:
            sys.exit(f"error: tensor '{key}' not found in safetensors")
        # 从 safetensors 读取 torch tensor → 转 fp32 → 转 numpy
        return self._index[key].get_tensor(key).float().numpy()


def main():
    """主入口函数：串联 INT4 量化导出的完整流程。

    流程：
    1. 解析命令行参数（模型路径、量化方法、组大小等）。
    2. 校验 group_size 合法性。
    3. 加载 HF config 并生成 tensor 映射计划。
    4. 组装 tinyqwen header 配置和 v2 扩展字段。
    5. 从 safetensors 加载所有权重并应用变换。
    6. 处理 tied embedding 的 lm_head 副本。
    7. 调用 write_tqwen_i4 写出量化后的 .tqwen 文件。
    8. 打印结果摘要。
    """
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="HF model dir")  # HF 模型目录
    p.add_argument("--out", required=True, help="output .tqwen path")  # 输出文件路径
    p.add_argument("--method", choices=["hqq", "rtn"], default="hqq",
                   help="量化算法：hqq（MSE 最优迭代，默认）/ rtn（min-max 舍入，回归对照）")
    p.add_argument("--group-size", type=int, default=64,
                   help="quantization group size（须 ≥32 且为 32 的倍数；默认 64）")
    p.add_argument("--max-seq-len", type=int, default=0, help="override max_seq_len")  # 覆盖最大序列长度
    p.add_argument("--no-lm-head-i4", action="store_true",
                   help="tied 模型默认把 lm_head 也量化导出（embed 的独立副本，"
                        "流量 544→~70MB/token）；此开关关闭，退回 fp32 lm_head（A/B 对照用）")
    args = p.parse_args()

    # 校验 group_size：NEON kernel 要求 ≥32 且为 32 的倍数
    if args.group_size < 32 or args.group_size % 32 != 0:
        sys.exit(f"error: --group-size 须 ≥32 且为 32 的倍数（NEON kernel 约束），"
                 f"实际 {args.group_size}")

    model_dir = find_local_dir(args.model)              # 定位并验证模型目录
    tcfg, model_type = load_config(model_dir)           # 加载 HF config
    tiny_names, src_of, transform_of = plan_tensors(tcfg, model_type)  # 生成 tensor 映射

    # 构造 tinyqwen header 配置字典
    max_seq_len = args.max_seq_len or tcfg.get("max_position_embeddings", 32768)
    tied = int(tcfg.get("tie_word_embeddings", True))  # tied embedding 标记
    tqwen_cfg = {
        "n_layers": tcfg["num_hidden_layers"],            # transformer 层数
        "hidden_size": tcfg["hidden_size"],               # 隐藏层维度
        "intermediate_size": tcfg["intermediate_size"],   # MLP 中间维度
        "n_heads": tcfg["num_attention_heads"],           # 注意力 head 数
        "n_kv_heads": tcfg["num_key_value_heads"],        # KV head 数
        "head_dim": tcfg["head_dim"],                     # head 维度
        "vocab_size": tcfg["vocab_size"],                 # 词表大小
        "max_seq_len": max_seq_len,                       # 最大序列长度
        "tied": tied,                                     # tied embedding 标记
        "rms_norm_eps": tcfg["rms_norm_eps"],             # RMSNorm epsilon
        "rope_theta": tcfg["rope_theta"],                 # RoPE base frequency
    }

    # 构造 v2 扩展字段
    ext = {"model_type": model_type}  # 基础字段：架构族
    if model_type == MODEL_QWEN35:
        # Qwen3.5 的 HF config 字段名与 runtime 内部不同：
        #   linear_num_key_heads   → linear_num_qk_heads
        #   linear_num_value_heads → linear_num_v_heads
        #   linear_key_head_dim    → linear_qk_head_dim
        #   linear_value_head_dim  → linear_v_head_dim
        # partial_rotary_factor / rope_theta 在 rope_parameters 子树里。
        # 与 export_qwen_to_tiny.py 的 load_config 同一套映射。
        rp = tcfg.get("rope_parameters") or {}  # 获取 rope_parameters 子树
        ext.update({
            "linear_num_qk_heads": tcfg.get("linear_num_key_heads", 0),       # QK head 数
            "linear_num_v_heads": tcfg.get("linear_num_value_heads", 0),      # V head 数
            "linear_qk_head_dim": tcfg.get("linear_key_head_dim", 0),         # QK head 维度
            "linear_v_head_dim": tcfg.get("linear_value_head_dim", 0),        # V head 维度
            "linear_conv_kernel_dim": tcfg.get("linear_conv_kernel_dim", 4),  # conv1d 核大小
            "full_attention_interval": tcfg.get("full_attention_interval", 4),  # full attn 间隔
            "partial_rotary_factor": rp.get("partial_rotary_factor",
                                             tcfg.get("partial_rotary_factor", 0.25)),  # 部分 RoPE
            "eos_token_id": tcfg.get("eos_token_id", 248044),                 # EOS token ID
        })

    # 从 safetensors 加载所有权重
    st = LazyTensors(model_dir)  # 初始化懒加载器
    tensors = {}  # 存放 {tiny_name: fp32_numpy} 的字典
    for tiny_name in tiny_names:
        hf_name = src_of[tiny_name]                    # 获取 HF 源 tensor 名
        arr = st.get(hf_name).astype(np.float32)       # 加载并转 fp32
        transform = transform_of[tiny_name]            # 获取变换函数
        if transform is not None:
            arr = transform(arr)                       # 应用变换（如 +1 折叠、squeeze）
        tensors[tiny_name] = arr                       # 存入字典

    # tied 模型的 lm_head：默认额外导出一份量化副本（与 embed 同权重）。
    # embed_tokens 本身保持 fp32 供 token lookup；lm_head.weight 是 2D 且不在
    # should_quantize 的 skip 列表里，写入循环会自动量化它。runtime 侧 tied
    # 时优先绑定 lm_head.weight（见 qwen_model.cpp），把每 token 544MB 的
    # fp32 流量换成 ~70MB i4。文件 +~70MB。
    if tied and not args.no_lm_head_i4:
        # 将 embed_tokens 的权重复制一份作为 lm_head（后续会被量化）
        tensors["lm_head.weight"] = tensors["model.embed_tokens.weight"]
        print("[lm_head] tied 模型：追加 lm_head.weight 量化副本（embed 保持 fp32 lookup）")

    # 调用写出函数，生成 INT4 量化的 .tqwen 文件
    total = write_tqwen_i4(args.out, tqwen_cfg, tensors, args.group_size,
                           version=FORMAT_VERSION, ext=ext, method=args.method)

    # 打印结果摘要
    mb = total / (1024 * 1024)  # 转换为 MB
    print(f"\nDone: {args.out} ({mb:.1f} MB)")
    print_table_summary(args.out)  # 打印 tensor 表摘要


if __name__ == "__main__":
    main()  # 脚本入口点
