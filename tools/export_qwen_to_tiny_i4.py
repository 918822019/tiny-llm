#!/usr/bin/env python3
"""将 Qwen2.5 / Qwen3.5（HF）权重导出为 INT4 量化的 .tqwen 文件。

非对称 uint4 [0,15]，per-group scale+zero_point（fp16），interleaved packing。
小向量（norm/bias/embed）保持 fp32。量化算法可选 HQQ（默认）/ RTN。

用法:
    python tools/export_qwen_to_tiny_i4.py \\
        --model models/Qwen3.5-4B \\
        --out model_qwen35_4b_i4.tqwen \\
        --method hqq --group-size 64

本脚本的核心流程：
    1. 复用 export_qwen_to_tiny.py 的基础设施（config 加载、tensor 名映射等）。
    2. 从 safetensors 按需加载权重并转为 fp32 numpy。
    3. 对大矩阵（linear projection）做 INT4 量化（HQQ / RTN），
       小向量（norm/bias/embed/conv1d/A_log/dt_bias）保持 fp32。
    4. 写出混合 dtype 的 .tqwen 文件：header 中标记主 dtype=I4，但每个 tensor
       entry 有自己的 dtype_code，runtime 按 entry 级别区分 I4 和 F32。

量化格式说明：
    - 每组 group_size 个元素共享一个 scale(fp16) 和 zero_point(fp16)。
    - 反量化公式：value = (uint4 - zero) * scale（HQQ 语义）。
    - 打包方式：两个 uint4 值拼成一个字节，低 nibble 在前（little-endian nibble order）。
    - 每组在磁盘上的布局：[scale_fp16(2B) | zero_fp16(2B) | packed_data(group_size/2 B)]。
    - 每行的总字节数 = n_groups × (4 + group_size/2)。

性能设计（v2 重构）：
    - 打包全 numpy 向量化（pack_i4_groups），替代三重纯 Python 循环——
      lm_head 量级张量从 ~10 分钟降到秒级。
    - RTN 量化向量化（按组轴批量 min/max/scale/zero），与旧逐组循环逐位一致。
    - 两遍流式写盘：第一遍只取 shape 定布局，第二遍边加载边量化边写，
      内存一次只驻留少量张量（旧版一次性持有全模型 fp32，4B ≈ 15GB）。
    - 量化进程池并行（--workers）。超大张量（lm_head/embed）"鲸鱼优先"提交，
      并按行拆分到多 worker——既是加速（消除长尾），也是内存约束（HQQ 整块
      量化 lm_head 单 worker 峰值 ~10GB，16GB 机器会 OOM）。RTN 拆分逐位可证；
      HQQ 拆分有 fp16 ULP 级漂移（质量等价，见 docs/quantization_guide.md）。

输入：HF 模型目录（含 config.json + model.safetensors*）。
输出：单个 .tqwen 二进制文件（INT4 量化版）。
"""

from __future__ import annotations

# --- 标准库导入 ---
import argparse   # 命令行参数解析
import os         # 文件大小查询 / CPU 核数
import struct     # 二进制打包/解包
import sys        # 错误退出、路径注入
import time       # 耗时统计
from pathlib import Path       # 路径操作

import numpy as np  # 数组操作、量化计算

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

# --- 并行/拆分参数 -------------------------------------------------------
# 元素数超过该阈值的张量按行拆分并行量化（lm_head/embed 量级，串行是长尾）
SHARD_ELEMS = 150_000_000
# 每个拆分块的目标元素数（64M fp32 = 256MB，内存友好）
SHARD_TARGET = 64_000_000


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


def pack_i4_groups(q: np.ndarray, scale: np.ndarray, zero: np.ndarray,
                   out_dim: int, groups_per_row: int, group_size: int) -> bytes:
    """向量化打包器：把 q/scale/zero 组装成 interleaved i4 磁盘格式。

    RTN / HQQ 两条量化路径共享这一步。替代原先的三重纯 Python 循环——
    lm_head 量级（数亿元素）从 ~10 分钟降到秒级，是导出提速的关键。

    磁盘布局（每行）：[group_0][group_1]...，每组 =
        [scale_fp16(2B) | zero_fp16(2B) | packed_nibbles(group_size/2 B)]
    nibble 顺序：低在前、高在后（与 C++ kernel 的 unpack 约定一致）。

    Args:
        q: uint8 数组 [out_dim * groups_per_row, group_size]，取值 [0,15]。
        scale: fp16 数组 [out_dim * groups_per_row]（组序 row-major）。
        zero: fp16 数组 [out_dim * groups_per_row]。
        out_dim: 行数（输出通道数）。
        groups_per_row: 每行的量化组数。
        group_size: 每组元素数。

    Returns:
        packed 字节串，长度 = out_dim * groups_per_row * (4 + group_size/2)。
    """
    group_total = GROUP_HEADER_BYTES + group_size // 2  # 每组总字节数
    # 一次性分配整块输出，按 [行, 组, 组内字节] 三维视图填充
    out = np.zeros((out_dim, groups_per_row, group_total), dtype=np.uint8)
    # scale/zero 按 fp16 小端字节序拆成 2 字节，写入每组 4B 头部
    sb = np.ascontiguousarray(scale).view(np.uint8).reshape(out_dim, groups_per_row, 2)
    zb = np.ascontiguousarray(zero).view(np.uint8).reshape(out_dim, groups_per_row, 2)
    out[:, :, 0] = sb[:, :, 0]  # scale 低字节
    out[:, :, 1] = sb[:, :, 1]  # scale 高字节
    out[:, :, 2] = zb[:, :, 0]  # zero 低字节
    out[:, :, 3] = zb[:, :, 1]  # zero 高字节
    # 两两 nibble 拼字节：偶数位进低 4 位，奇数位进高 4 位
    q2 = np.ascontiguousarray(q).reshape(out_dim, groups_per_row, group_size // 2, 2)
    out[:, :, GROUP_HEADER_BYTES:] = q2[..., 0] | (q2[..., 1] << np.uint8(4))
    return out.tobytes()


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


def quantize_i4_group(data: np.ndarray, symmetric: bool = False) -> tuple[np.float16, np.float16, np.ndarray]:
    """对一组 float32 数据做非对称 RTN（Round-To-Nearest）INT4 量化。

    symmetric=True 时强制 zero=8（对称量化），scale = max(|vmin|,|vmax|) / 7，
    与 llama.cpp Q4_0 同方案。反量化 = (q-8) * scale，无 zero 修正项。

    RTN 量化原理（非对称）：
    1. 计算该组数据的 min/max 范围。
    2. scale = (max - min) / 15，将浮点范围映射到 [0, 15] 整数区间。
    3. zero_point = -min / scale，使得 uint4=0 对应 vmin。
    4. 量化：q = round(data / scale + zero_point)，clip 到 [0, 15]。
    5. scale 和 zero_point 都存为 fp16（与 C++ 反量化对齐）。

    注意：量化时使用 rounded fp16 的 scale/zero 值（而非原始 fp32 值），
    这保证 Python 侧量化结果与 C++ runtime 反量化后的数值完全一致。

    Args:
        data: 一维 float32 数组，长度 <= group_size。
        symmetric: 是否对称量化（zero=8）。

    Returns:
        (scale_fp16, zero_fp16, quantized_uint8_array) 三元组。
        quantized 数组的元素值在 [0, 15] 范围内。
    """
    vmin = data.min()  # 组内最小值
    vmax = data.max()  # 组内最大值

    if symmetric:
        # 对称量化（llama.cpp Q4_0 同方案）：zero=8，scale = max(|vmin|,|vmax|)/7
        amax = max(abs(vmin), abs(vmax))
        if amax == 0.0:
            scale = np.float16(0.0)
            zero = np.float16(8.0)
            quantized = np.zeros(len(data), dtype=np.uint8)
        else:
            scale = np.float16(np.float32(amax) / np.float32(7.0))
            zero = np.float16(8.0)
            s = float(scale)
            if s == 0:
                quantized = np.zeros(len(data), dtype=np.uint8)
            else:
                q = np.round(data / s + 8.0).astype(np.int32)
                quantized = np.clip(q, 0, 15).astype(np.uint8)
        return scale, zero, quantized

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


def quantize_tensor_i4(tensor: np.ndarray, group_size: int, symmetric: bool = False) -> bytes:
    """将 [out_dim, in_dim] 的 fp32 tensor 量化为 interleaved INT4 格式（RTN 方法）。

    输出格式（每行每个组）：
        [scale_fp16(2B) | zero_fp16(2B) | packed_nibbles(group_size/2 B)]
    其中 packed_nibbles 中每两个 uint4 拼成一个字节，低 nibble 在前。

    in_dim 能被 group_size 整除时走 numpy 向量化快路径（占绝对多数情况）；
    不能整除时退回逐组循环慢路径（支持尾部不满组）。两条路径结果逐位一致。

    Args:
        tensor: 形状为 [out_dim, in_dim] 的 fp32 numpy 数组。
        group_size: 量化组大小。
        symmetric: 是否对称量化（zero=8）。

    Returns:
        量化后的字节串，长度为 out_dim × i4_row_bytes(in_dim, group_size)。
    """
    assert tensor.ndim == 2  # 必须是 2D 矩阵
    out_dim, in_dim = tensor.shape  # 获取矩阵维度
    if in_dim % group_size == 0:
        return _quantize_tensor_i4_vec(tensor, group_size, symmetric)  # 快路径：全向量化
    return _quantize_tensor_i4_loop(tensor, group_size, symmetric)     # 慢路径：尾部不满组


def _quantize_tensor_i4_vec(tensor: np.ndarray, group_size: int, symmetric: bool = False) -> bytes:
    """RTN 向量量化快路径（in_dim % group_size == 0）。

    与逐组循环版（_quantize_tensor_i4_loop）逐位一致：
    - scale/zero 先按 fp32 计算，再 round 到 fp16；
    - 量化用 rounded fp16 值；scale 在 fp16 下为 0 的组全量化为 0；
    - round → int32 → clip[0,15] → uint8（与循环版相同的取整链）。

    Args:
        tensor: [out_dim, in_dim] fp32，in_dim 可被 group_size 整除。
        group_size: 量化组大小。

    Returns:
        packed 字节串。
    """
    out_dim, in_dim = tensor.shape
    groups_per_row = in_dim // group_size  # 每行组数
    # 展平成 [总组数, group_size]，每组独立做 min-max RTN
    g = np.ascontiguousarray(tensor).reshape(out_dim * groups_per_row, group_size)
    vmin = g.min(axis=1)  # 每组最小值（fp32）
    vmax = g.max(axis=1)  # 每组最大值（fp32）
    same = vmax == vmin   # 退化组标记（全相同元素）

    if symmetric:
        # 对称量化（llama.cpp Q4_0 同方案）：zero=8，scale = max(|vmin|,|vmax|)/7
        amax = np.maximum(np.abs(vmin), np.abs(vmax))
        scale_f32 = np.where(amax == 0.0, np.float32(0.0), amax / np.float32(7.0))
        scale = scale_f32.astype(np.float16)
        zero = np.full(scale.shape, np.float16(8.0), dtype=np.float16)
        s = scale.astype(np.float32)
        z = np.float32(8.0)
        valid = s != 0
        q = np.zeros(g.shape, dtype=np.uint8)
        q[valid] = np.clip(np.round(g[valid] / s[valid, np.newaxis] + z), 0, 15).astype(np.uint8)
        return pack_i4_groups(q, scale, zero, out_dim, groups_per_row, group_size)

    # scale/zero 的 fp32 中间值（退化组 scale=zero=0，避免除零）
    scale_f32 = np.where(same, np.float32(0.0), (vmax - vmin) / np.float32(15.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        zero_f32 = np.where(same, np.float32(0.0), -vmin / scale_f32)

    # round 到 fp16 存储（与 C++ runtime 反量化精度对齐）
    scale = scale_f32.astype(np.float16)
    zero = zero_f32.astype(np.float16)

    # 用 rounded fp16 值量化；scale==0 的组（退化或 fp16 下溢）全零
    s = scale.astype(np.float32)
    z = zero.astype(np.float32)
    valid = ~same & (s != 0)  # 需要真实量化的组
    q = np.zeros(g.shape, dtype=np.uint8)
    if valid.any():
        sv = np.where(valid, s, np.float32(1.0))  # 无效组用 1.0 占位防除零
        # round → int32 → clip → uint8：与循环版完全相同的取整链
        qv = np.round(g / sv[:, None] + z[:, None]).astype(np.int32)
        q[valid] = np.clip(qv[valid], 0, 15).astype(np.uint8)

    return pack_i4_groups(q, scale, zero, out_dim, groups_per_row, group_size)


def _quantize_tensor_i4_loop(tensor: np.ndarray, group_size: int, symmetric: bool = False) -> bytes:
    """RTN 逐组循环版（支持尾部不满组；in_dim % group_size != 0 时用）。

    保留原始实现作为边界情况兜底，与向量化快路径结果逐位一致。
    """
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

            scale, zero, quantized = quantize_i4_group(group_data, symmetric)

            group_offset = row_offset + g * group_total
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

    # 向量化打包（替代原三重 Python 循环，格式逐位不变）
    groups_per_row = in_dim // group_size  # 每行组数（HQQ 保证整除）
    q_np = np.ascontiguousarray(q_np).reshape(out_dim * groups_per_row, group_size)
    return pack_i4_groups(q_np, scale, zero, out_dim, groups_per_row, group_size)


# ---- 并行量化 worker（模块级函数，ProcessPool spawn 后可 pickle）--------

def _worker_init(torch_threads: int, need_torch: bool) -> None:
    """worker 进程初始化：限制 torch 核内线程数，避免多进程超额订阅 CPU。

    Args:
        torch_threads: 每个 worker 允许的 torch 线程数。
        need_torch: 是否需要导入 torch（method=hqq 才需要）。
    """
    if need_torch and torch_threads > 0:
        try:
            import torch
            torch.set_num_threads(torch_threads)  # 限定每个 worker 的并行度
        except ImportError:
            pass  # 没有 torch 就无法 hqq，后续量化会自然报错


def _quantize_one(payload: tuple) -> bytes:
    """worker 端量化入口：(fp32_array, group_size, method, symmetric) -> packed bytes。"""
    arr, group_size, method, symmetric = payload
    if method == "hqq":
        return quantize_tensor_i4_hqq(arr, group_size)
    return quantize_tensor_i4(arr, group_size, symmetric)


def _shard_rows(arr: np.ndarray, shard_target: int) -> list:
    """把 [out_dim, in_dim] 张量按行拆成若干块（每块 contiguous 副本）。

    量化组沿行内划分、行间完全独立（RTN/HQQ 均成立），因此拆分后
    各块独立量化再按序拼接，结果与整块量化逐位一致。用于把
    lm_head/embed 这类超大张量摊到多个 worker 上，消除串行长尾。

    Args:
        arr: 2D fp32 数组。
        shard_target: 每块目标元素数。

    Returns:
        子数组列表（行序不变）。
    """
    rows_per_shard = max(1, shard_target // arr.shape[1])  # 每块行数
    shards = []
    for start in range(0, arr.shape[0], rows_per_shard):
        # ascontiguousarray 保证切片是独立连续内存（可安全跨进程序列化）
        shards.append(np.ascontiguousarray(arr[start:start + rows_per_shard]))
    return shards


def write_tqwen_i4(out_path: str | Path, cfg: dict, names: list, load,
                   group_size: int, version: int = 2, ext: dict | None = None,
                   method: str = "hqq", workers: int = 1, symmetric: bool = False) -> int:
    """写一个 INT4 混合 dtype 的 .tqwen 文件（流式两遍 + 可选并行量化）。

    与 f16 版本的 write_tqwen 的主要区别：
    - 每个 tensor 有独立的 dtype_code（I4 或 F32），而非全局统一。
    - 量化后的 tensor 不再是简单的 numpy tobytes，而是自定义 interleaved 格式。
    - header.dtype 设为 DTYPE_I4 作为默认/主标记。
    - v2 ext 中额外记录 quant_group_size。

    流式两遍设计（替代旧版"全模型 fp32 驻留内存"）：
    - 第一遍：逐个 load(name) 取 shape（数据即取即弃），定偏移布局；
    - 第二遍：再逐个 load → 量化 → 写盘；i4 量化可进程池并行，
      超大张量按行拆分到多 worker。内存峰值 ≈ 少量在途张量。

    Args:
        out_path: 输出文件路径。
        cfg: 模型超参配置字典。
        names: tensor 名的有序列表（写入顺序 = 文件内顺序）。
        load: 回调 load(name) -> fp32 numpy 数组（按需加载，调用方不驻留全模型）。
        group_size: 量化组大小。
        version: 格式版本号（默认 2）。
        ext: v2 扩展字段字典。
        method: 量化算法，"hqq" 或 "rtn"。
        workers: 量化并行进程数（1 = 串行，便于调试/逐位对照）。

    Returns:
        写出文件的总字节数。
    """
    t0 = time.time()  # 总耗时起点

    # ---- 第一遍：取 shape、定布局（内存一次只有一个张量）----
    shapes = {}  # name -> tuple(shape)
    for name in names:
        arr = load(name)  # 加载只为拿 shape
        assert 1 <= arr.ndim <= 4, f"{name}: ndim {arr.ndim} not in [1,4]"  # 维度校验
        assert len(name) <= MAX_NAME, f"tensor name too long: {name}"  # 名字长度校验
        shapes[name] = tuple(arr.shape)
        del arr  # 立即释放，不驻留

    # 分类 + 计算偏移/大小（i4/fp32 的大小都可由 shape 直接算出）
    entries = []  # (name, shape, offset, nbytes, dtype_code)
    table_end = struct.calcsize(HEADER_FMT) + len(names) * struct.calcsize(ENTRY_FMT)
    data_offset = align_up(table_end)  # 数据区 64B 对齐
    offset = data_offset
    print(f"Planning {len(names)} tensors (group_size={group_size}, "
          f"method={method}, workers={workers})...")
    for name in names:
        shape = shapes[name]
        if len(shape) == 2 and should_quantize(name):
            # 2D 矩阵且不在跳过列表 → INT4（大小 = 行数 × 每行字节）
            nbytes = i4_row_bytes(shape[1], group_size) * shape[0]
            dtype_code = DTYPE_I4
            tag = "[I4] "
        else:
            # 小向量或不适合量化 → fp32（大小 = 元素数 × 4B）
            n = 1
            for d in shape:
                n *= d
            nbytes = n * 4
            dtype_code = DTYPE_F32
            tag = "[F32]"
        entries.append((name, list(shape), offset, nbytes, dtype_code))
        print(f"  {tag} {name:<50} {str(list(shape)):<18} -> {nbytes:>10} bytes")
        offset = align_up(offset + nbytes)  # 下一个 tensor 偏移（64B 对齐）
    total_bytes = offset  # 文件总大小

    # 构造 header 的 reserved 区（含 v2 扩展 + quant_group_size）
    reserved = b"\x00" * 96
    if version >= 2:
        ext_bytes = pack_v2_ext_i4(ext or {}, group_size)  # 打包 v2 扩展
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

    # ---- 写文件：header + 索引表 + 数据区（第二遍流式）----
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)  # 自动创建目录
    with open(out_path, "wb") as out:
        out.write(header)  # 写入 192B header
        # 写入 tensor 索引表
        for name, shape, off, nbytes, dtype_code in entries:
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

        # 第二遍：加载 → 量化 → 写盘（并行与否由 workers 决定）
        _write_data_pass(out, entries, load, group_size, method, workers, symmetric)
        assert out.tell() == total_bytes  # 最终大小校验

    # 验证文件大小
    size = os.path.getsize(out_path)
    assert size == total_bytes, f"file size {size} != expected {total_bytes}"
    print(f"[export] 规划+量化+写盘总耗时 {time.time() - t0:.1f}s")
    return total_bytes


def _write_data_pass(out, entries: list, load, group_size: int,
                     method: str, workers: int, symmetric: bool = False) -> None:
    """第二遍数据写盘：按 entries 顺序写出每个 tensor 的数据。

    workers <= 1 时串行（加载 → 量化 → 写，便于调试与逐位对照）；
    workers > 1 时用进程池并行量化：
    - **鲸鱼优先**：超大张量（lm_head 量级）最先提交，与其余张量重叠执行，
      避免它排在最后成为纯串行长尾。写盘仍严格按 entries 顺序。
    - **行拆分**：超大张量按行拆成 ~64M 元素的块分给多 worker。必须拆分
      的另一个原因是内存——HQQ 整块量化 lm_head（635M 元素）单 worker
      峰值 ~10GB，16GB 机器会 OOM。RTN 拆分逐位可证；HQQ 拆分有 fp16
      ULP 级漂移（torch 归约顺序依赖行数，质量等价，见 quantization_guide.md）。

    Args:
        out: 已打开的输出文件（指针位于数据区起点）。
        entries: write_tqwen_i4 里计算的布局表。
        load: 按需加载回调。
        group_size / method / workers: 量化参数与并行度。
    """

    def write_padded(off: int, nbytes: int, data: bytes) -> None:
        """写数据并补零到 64B 对齐边界（断言偏移正确）。"""
        assert out.tell() == off, f"alignment bug: {out.tell()} != {off}"
        out.write(data)
        out.write(b"\x00" * (align_up(off + nbytes) - (off + nbytes)))

    # ---- 串行路径 ----
    if workers <= 1:
        for name, shape, off, nbytes, dtype_code in entries:
            arr = load(name)  # 按需加载
            if dtype_code == DTYPE_I4:
                data = _quantize_one((arr, group_size, method, symmetric))  # 量化
            else:
                data = np.ascontiguousarray(arr).tobytes()  # fp32 直接序列化
            assert len(data) == nbytes, f"{name}: size {len(data)} != {nbytes}"
            write_padded(off, nbytes, data)
        return

    # ---- 并行路径：进程池 + 鲸鱼优先提交 + 顺序写盘 ----
    from concurrent.futures import ProcessPoolExecutor

    cpu = os.cpu_count() or 4
    # 每个 worker 的 torch 线程数：总核数分摊，避免超额订阅
    torch_threads = max(1, cpu // (workers + 1))
    inflight = workers * 2  # 在途任务窗口上限（内存 = 窗口内 fp32 + 已完成 packed）

    # 提交顺序：鲸鱼（超大 i4 张量）按大小降序排最前，其余保持原顺序。
    # 这样 lm_head 最先开跑，与后面几十层的量化重叠，消除串行长尾。
    whale = [i for i, e in enumerate(entries)
             if e[4] == DTYPE_I4 and e[1][0] * e[1][1] > SHARD_ELEMS]
    whale.sort(key=lambda i: -entries[i][3])  # 最大的最先提交
    rest = [i for i in range(len(entries)) if i not in set(whale)]
    sub_order = whale + rest

    print(f"[export] 并行量化：workers={workers}，每 worker torch 线程≈{torch_threads}"
          + (f"，鲸鱼张量 {len(whale)} 个优先提交" if whale else ""))
    with ProcessPoolExecutor(max_workers=workers, initializer=_worker_init,
                             initargs=(torch_threads, method == "hqq")) as pool:
        pending = {}   # idx -> futures 列表 或 packed bytes（在途/已完成）
        sub_pos = 0    # sub_order 的提交游标
        n_inflight = 0  # 在途（已提交未取走结果）的任务数

        def submit_one(idx: int) -> None:
            """加载第 idx 个 tensor 并提交量化（或直接转 fp32 字节）。"""
            nonlocal n_inflight
            name, shape, off, nbytes, dtype_code = entries[idx]
            arr = load(name)  # 主线程加载（safetensors → fp32）
            if dtype_code != DTYPE_I4:
                pending[idx] = np.ascontiguousarray(arr).tobytes()
            else:
                n_elems = shape[0] * shape[1]
                # 超大张量按行拆分并行（量化组行内划分）。RTN 严格行独立、
                # 逐位可证；HQQ 拆分会有 fp16 ULP 级漂移（torch 归约顺序依赖
                # 行数，质量等价）——但 HQQ 鲸鱼不拆分会在单 worker 里占
                # ~10GB（fp32 输入+torch 副本+unpack 临时区），16GB 机器会
                # OOM，故两者都拆，ULP 漂移记录在文档里。
                if n_elems > SHARD_ELEMS and shape[1] % group_size == 0:
                    shards = _shard_rows(arr, SHARD_TARGET)
                else:
                    shards = [arr]
                # submit 时参数即被序列化进队列，主进程可尽快释放原数组
                pending[idx] = [pool.submit(_quantize_one, (s, group_size, method, symmetric))
                                for s in shards]
            del arr
            n_inflight += 1

        # 按文件顺序逐个写出；提交指针按需超前填充窗口
        for w in range(len(entries)):
            # 补提交：保持在途任务数 <= inflight（先提交的鲸鱼已在前）
            while sub_pos < len(sub_order) and n_inflight < inflight:
                submit_one(sub_order[sub_pos])
                sub_pos += 1
            # 取第 w 个的结果（可能还在算，阻塞等待）
            payload = pending.pop(w)
            if isinstance(payload, list):
                data = b"".join(f.result() for f in payload)  # 拆分块按序拼接
            else:
                data = payload
            n_inflight -= 1
            name, shape, off, nbytes, dtype_code = entries[w]
            assert len(data) == nbytes, f"{name}: size {len(data)} != {nbytes}"
            write_padded(off, nbytes, data)


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
    1. 解析命令行参数（模型路径、量化方法、组大小、并行度等）。
    2. 校验 group_size 合法性。
    3. 加载 HF config 并生成 tensor 映射计划。
    4. 组装 tinyqwen header 配置和 v2 扩展字段。
    5. 构造按需加载回调（load），处理 tied embedding 的 lm_head 副本。
    6. 调用 write_tqwen_i4 流式写出量化后的 .tqwen 文件。
    7. 打印结果摘要。
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
    p.add_argument("--workers", type=int, default=0,
                   help="量化并行进程数：0=自动（CPU 核数/3，1~4），1=串行。"
                        "注意：并行 worker 的 torch 线程数与串行不同，HQQ 归约"
                        "顺序变化会带来 fp16 ULP 级差异（质量等价，非逐位一致）；"
                        "需要字节级复现既有文件时用 --workers 1")
    p.add_argument("--symmetric", action="store_true",
                   help="对称量化：强制 zero=8，消除 per-group 修正项（解码 ~1.5-2× 提速）。"
                        "仅 --method rtn 有效；scale = max(|vmin|,|vmax|)/7")
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
        #   linear_value_head_dim  → linear_qk_head_dim
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

    # 从 safetensors 懒加载（句柄常驻，数据按需读取）
    st = LazyTensors(model_dir)  # 初始化懒加载器

    def load(tiny_name: str) -> np.ndarray:
        """按需加载回调：HF 源 tensor → fp32 numpy（含可选变换）。

        流式两遍写盘的核心：调用方（write_tqwen_i4）每 name 调两次
        （第一遍取 shape、第二遍取数据），内存始终只驻留少量张量。
        """
        arr = st.get(src_of[tiny_name])      # safetensors → fp32 numpy
        transform = transform_of[tiny_name]  # 可选变换（如 +1 折叠、squeeze）
        if transform is not None:
            arr = transform(arr)
        return np.asarray(arr, dtype=np.float32)  # 确保 fp32 + 规整类型

    names = list(tiny_names)  # 写入顺序 = plan_tensors 给出的顺序
    # tied 模型的 lm_head：默认额外导出一份量化副本（与 embed 同权重）。
    # embed_tokens 本身保持 fp32 供 token lookup；lm_head.weight 是 2D 且不在
    # should_quantize 的 skip 列表里，写入循环会自动量化它。runtime 侧 tied
    # 时优先绑定 lm_head.weight（见 qwen_model.cpp），把每 token 544MB 的
    # fp32 流量换成 ~70MB i4。文件 +~70MB。
    if tied and not args.no_lm_head_i4:
        # lm_head 复用 embed 的源 tensor 与变换（load 时读同一份权重）
        src_of["lm_head.weight"] = src_of["model.embed_tokens.weight"]
        transform_of["lm_head.weight"] = transform_of["model.embed_tokens.weight"]
        names.append("lm_head.weight")
        print("[lm_head] tied 模型：追加 lm_head.weight 量化副本（embed 保持 fp32 lookup）")

    # 并行度：0 = 自动（核数/3，限制 1~4；16GB 机器上 3 个 worker 较稳）
    workers = args.workers
    if workers == 0:
        workers = max(1, min(4, (os.cpu_count() or 4) // 3))

    # 调用写出函数，生成 INT4 量化的 .tqwen 文件
    t0 = time.time()
    total = write_tqwen_i4(args.out, tqwen_cfg, names, load, args.group_size,
                           version=FORMAT_VERSION, ext=ext, method=args.method,
                           workers=workers, symmetric=args.symmetric)

    # 打印结果摘要
    mb = total / (1024 * 1024)  # 转换为 MB
    print(f"\nDone: {args.out} ({mb:.1f} MB) in {time.time() - t0:.1f}s")
    print_table_summary(args.out)  # 打印 tensor 表摘要


if __name__ == "__main__":
    main()  # 脚本入口点
