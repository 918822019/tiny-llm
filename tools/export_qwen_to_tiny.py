#!/usr/bin/env python3
"""将 Qwen2.5（HF）权重导出为 tinyqwen 扁平二进制格式（.tqwen）。

二进制契约定义在 runtime/tiny_format.h 和 docs/weight_format.md，
下面的 struct 布局必须与该头文件保持同步。

用法:
    python tools/export_qwen_to_tiny.py \\
        --model Qwen/Qwen2.5-0.5B \\
        --out model.tqwen

说明:
    - v1 支持导出 float32（默认）或 float16（weight-only 半精度，文件减半，
      配 runtime 的 f16 matvec 实现使用；数值容差见 docs/optimization.md §5）。
    - linear 权重保持 HF 的 [out_dim, in_dim] 行主序，不转置。
    - 不处理 tokenizer，见 tools/tokenize_prompt.py。

本脚本的核心流程：
    1. 读取 HF config.json，解析模型超参（hidden_size、num_heads 等）。
    2. 根据 model_type（Qwen2.x / Qwen3.5）生成 tensor 名映射表，
       将 HF 权重名映射到 .tqwen 内部命名约定。
    3. 流式从 safetensors 分片读取权重，做必要的变换（zero-centered
       RMSNorm +1 折叠、conv1d squeeze 等），写入 .tqwen 二进制文件。
    4. 打印写出文件的摘要用于验收。

输入：HF 模型目录（含 config.json + model.safetensors*）。
输出：单个 .tqwen 二进制文件，包含 header + tensor 索引表 + 对齐后的权重数据。
关键设计：采用 LazyTensors 惰性加载，避免将整个模型载入内存；每个 tensor
仅在 write_tqwen 实际写入时才从磁盘读取并变换。
"""

from __future__ import annotations

# --- 标准库导入 ---
import argparse   # 命令行参数解析
import json       # 读取 HF config.json
import numpy as np  # 数组操作、dtype 转换
import os         # 文件大小查询
import struct     # 二进制打包/解包（header、tensor entry）
import sys        # 错误退出
from pathlib import Path  # 路径操作

# 必须与 runtime/tiny_format.h 保持一致 -------------------------------
# 文件魔数，用于快速识别 .tqwen 文件格式
MAGIC = b"TINYQWEN"
# 当前最高支持的格式版本号：
#   v2 = Qwen3.5 混合架构扩展字段（reserved 前 64B = TinyHeaderV2Ext）
#   v3 = MoE + GPTQ 扩展字段（reserved 后 32B = TinyHeaderV3Ext；V2Ext 布局不变）
#   v1/v2 文件仍可加载（FORMAT_VERSION_MIN=1，向后兼容）
FORMAT_VERSION = 3  # 当前最高版本
# 最低兼容版本号
FORMAT_VERSION_MIN = 1
# 所有数据区的对齐粒度（字节），保证 SIMD 加载时地址对齐
ALIGN = 64
# dtype 枚举值，与 C++ runtime/tiny_format.h 的 Dtype enum 一一对应
DTYPE_F32 = 0   # float32，每元素 4 字节
DTYPE_F16 = 1   # float16，每元素 2 字节（与 runtime/tiny_format.h 的 Dtype 枚举保持一致）
DTYPE_I4 = 3    # INT4 量化，仅 i4 导出脚本使用
DTYPE_VQ2 = 4   # 2-bit 块向量量化（码本+索引），仅 vq2 导出脚本使用
# tensor 名字最大长度（字节），超过则报错
MAX_NAME = 64

# 架构族标识（TinyHeaderV2Ext.model_type 字段取值）。
MODEL_QWEN2 = 0    # Qwen2.x 系列（包括 Qwen2.5、Qwen3 纯文本）
MODEL_QWEN35 = 1   # Qwen3.5 混合架构（linear attention + full attention 交替）

# dtype 字符串 -> (编号, 每元素字节数, numpy 类型名) 的映射表
DTYPES = {
    "f32": (DTYPE_F32, 4, "float32"),   # float32: 4 字节/元素
    "f16": (DTYPE_F16, 2, "float16"),   # float16: 2 字节/元素
}

# header 二进制格式：小端序，8s=魔数, 12I=12个u32, ff=2个float, 4Q=4个u64, 96s=reserved
HEADER_FMT = "<8s12Iff4Q96s"  # 总大小 192 字节
# tensor 索引条目格式：64s=名字, II=dtype+ndim, 4Q=shape(4维)+offset+nbytes
ENTRY_FMT = "<64sII4QQQ"  # 总大小 120 字节
# v2 扩展块（TinyHeaderV2Ext）：放进 header.reserved 的前 64 字节。
# 字段顺序须与 C++ 结构体一致：7 个 u32 -> float partial_rotary_factor ->
# u32 eos_token_id -> u32 pad -> u32 quant_group_size -> 20 字节 reserved。
# 用于存储 Qwen3.5 混合架构特有的线性注意力参数与 INT4 组大小
EXT_FMT = "<7If3I20s"  # 总大小 64 字节

# 编译期断言：确保 Python 侧 struct 格式与 C++ 头文件大小完全匹配
assert struct.calcsize(HEADER_FMT) == 192   # header 必须恰好 192 字节
assert struct.calcsize(ENTRY_FMT) == 120    # 每个 tensor entry 必须恰好 120 字节
assert struct.calcsize(EXT_FMT) == 64       # v2 扩展块必须恰好 64 字节


# ---------------------------------------------------------------------


def align_up(x: int, align: int = ALIGN) -> int:
    """将 x 向上对齐到 align 的整数倍。

    用于计算数据区偏移，保证每个 tensor 的起始地址都是 64 字节对齐，
    方便 C++ runtime 用 SIMD 指令直接加载。

    Args:
        x: 需要对齐的值（字节偏移或大小）。
        align: 对齐粒度，默认 64 字节。

    Returns:
        >= x 的最小 align 倍数。
    """
    return (x + align - 1) // align * align  # 整数除法实现向上取整对齐


def pack_v2_ext(ext: dict) -> bytes:
    """把 v2 扩展字段打包成 64 字节（TinyHeaderV2Ext）。ext 缺省字段填 0。

    该扩展块存储在 header.reserved 的前 64 字节中，仅在 format version >= 2
    （即 Qwen3.5 混合架构）时使用。包含线性注意力的 head 配置、卷积核大小、
    full attention 间隔、partial rotary factor 和 eos token id。

    Args:
        ext: 包含扩展字段的字典，缺失的键默认为 0。

    Returns:
        64 字节的 packed 二进制数据。
    """
    g = lambda k: ext.get(k, 0)  # noqa: E731  # 便捷取值函数，缺失返回 0
    return struct.pack(
        EXT_FMT,                          # 按 EXT_FMT 格式打包
        int(g("model_type")),             # 架构族：0=qwen2, 1=qwen3_5
        int(g("linear_num_qk_heads")),    # 线性注意力 QK head 数量
        int(g("linear_num_v_heads")),     # 线性注意力 V head 数量
        int(g("linear_qk_head_dim")),     # 线性注意力 QK head 维度
        int(g("linear_v_head_dim")),      # 线性注意力 V head 维度
        int(g("linear_conv_kernel_dim")), # 线性注意力 depthwise conv1d 核大小
        int(g("full_attention_interval")),  # full attention 层的间隔（每隔几层一个）
        float(g("partial_rotary_factor")),  # 部分 RoPE 的比例因子
        int(g("eos_token_id")),           # EOS token ID
        0,                                # pad 字段，保留对齐
        int(g("quant_group_size")),       # INT4 每组元素数（0 = 未量化）
        b"\x00" * 20,                     # 尾部 20 字节保留区，全零填充
    )


def unpack_v2_ext(reserved: bytes) -> dict:
    """从 header.reserved 解出 v2 扩展字段（对齐/读取工具用）。

    这是 pack_v2_ext 的逆操作，用于回读已写出的 .tqwen 文件时
    还原 Qwen3.5 特有的配置参数。

    Args:
        reserved: header 中 96 字节的 reserved 字段（只使用前 64 字节）。

    Returns:
        包含所有 v2 扩展字段的字典。
    """
    # 从 reserved 的前 64 字节解包所有 v2 扩展字段
    (model_type, qk_h, v_h, qk_d, v_d, conv, interval,
     rotary, eos, _pad, quant_gs, _rsv) = struct.unpack(
        EXT_FMT, reserved[: struct.calcsize(EXT_FMT)])
    return {
        "model_type": model_type,                    # 架构族标识
        "linear_num_qk_heads": qk_h,                 # 线性注意力 QK head 数
        "linear_num_v_heads": v_h,                   # 线性注意力 V head 数
        "linear_qk_head_dim": qk_d,                  # 线性注意力 QK head 维度
        "linear_v_head_dim": v_d,                    # 线性注意力 V head 维度
        "linear_conv_kernel_dim": conv,              # depthwise conv1d 核大小
        "full_attention_interval": interval,          # full attention 层间隔
        "partial_rotary_factor": rotary,             # 部分 RoPE 比例
        "eos_token_id": eos,                         # EOS token ID
        "quant_group_size": quant_gs,                # INT4 每组元素数（0 = 未量化）
    }


GPTQ_MAGIC = 0x47505451  # 'GPTQ'（与 runtime/tiny_format.h 的 kGptqMagic 一致）
GPTQ_PACK = 8            # 每个 int32 装 8 个 uint4（bits=4）


def repack_gptq_from_hf(qweight, qzeros, scales, g_idx, group_size: int) -> bytes:
    """把 HF/AutoGPTQ 的 GPTQ 张量重打包成 runtime 的 in-band 块。

    形状契约（真 AutoGPTQ checkpoint 实测，见 tools/validate_gptq_numeric.py）：
      qweight int32 [in//8, out]  — 一个 word 打包 8 个连续 in_dim，低 nibble=最小下标
      qzeros  int32 [ng, out//8]  — 一个 word 打包 8 个连续 out_dim
      scales  fp16  [ng, out]
      g_idx   int32 [in]          — contiguous 时可省略

    两处必须转换，否则 C++ dequant 静默算错：
      1. qzeros 从 int4-packed 解包成逐元素 fp16 [ng, out]
      2. 存的值 +1 —— AutoGPTQ 存的是 zp-1（sym/bits=4 下 nibble 恒为 7，真 zp=8）。
         不修正会让 dequant 整体偏一个量化步长，MSE 恶化 8~10×。

    返回 in-band 块字节：[header 8B][scales][qzeros][g_idx 可选][qweight]。
    """
    qw = qweight.astype(np.uint32)
    in_dim = qw.shape[0] * GPTQ_PACK
    out_dim = qw.shape[1]
    if scales.shape != (in_dim // group_size, out_dim):
        raise ValueError(f"scales shape {scales.shape} != "
                         f"({in_dim // group_size}, {out_dim})")
    if in_dim % group_size != 0 or in_dim % GPTQ_PACK != 0:
        raise ValueError(f"in_dim {in_dim} must be divisible by group_size and {GPTQ_PACK}")

    # qzeros: [ng, out//8] 每 word 8 个连续 out → [ng, out] 逐元素
    qz = qzeros.astype(np.uint32)
    if qz.shape != (in_dim // group_size, out_dim // GPTQ_PACK):
        raise ValueError(f"qzeros shape {qz.shape} != "
                         f"({in_dim // group_size}, {out_dim // GPTQ_PACK})")
    nib = np.stack([(qz >> (k * 4)) & 0xF for k in range(GPTQ_PACK)], axis=2)
    z_true = nib.reshape(qz.shape[0], out_dim).astype(np.float32) + 1.0

    # g_idx：contiguous（= col//group_size）时省略，flags bit0=0
    has_g_idx = False
    if g_idx is not None:
        gi = g_idx.astype(np.int64)
        has_g_idx = not np.array_equal(gi, np.arange(in_dim) // group_size)

    buf = bytearray()
    buf += struct.pack("<II", GPTQ_MAGIC, 1 if has_g_idx else 0)
    buf += scales.astype(np.float16).tobytes()
    buf += z_true.astype(np.float16).tobytes()
    if has_g_idx:
        buf += g_idx.astype(np.uint32).tobytes()
    buf += qw.tobytes()
    return bytes(buf)


def write_tqwen(out_path: str | Path, cfg: dict, tensors: "dict[str, object]",
                dtype: str = "f32", version: int = 1, ext: dict | None = None) -> int:
    """写一个 .tqwen 文件，返回文件总字节数。

    这是核心写出函数。按照 docs/weight_format.md 定义的二进制布局，依次写入：
    1. 192 字节的 header（含魔数、版本、模型超参、reserved/v2扩展）。
    2. N × 120 字节的 tensor 索引表（名字、dtype、shape、offset、nbytes）。
    3. 补零到 64 字节边界。
    4. 逐个 tensor 的权重数据，每个 tensor 之后也补零到 64 字节边界。

    Args:
        out_path: 输出文件路径。
        cfg: 模型配置字典，必需键包括 n_layers, hidden_size, intermediate_size,
             n_heads, n_kv_heads, head_dim, vocab_size, max_seq_len, tied,
             rms_norm_eps, rope_theta。
        tensors: 有序映射 name -> 数组对象，要求有 .shape 属性以及
            .astype(<dtype>).tobytes() 方法（numpy 数组可直接使用）。
        dtype: "f32"（默认）或 "f16"；决定 header/tensor 表的 dtype 字段与元素大小。
        version: 1（Qwen2.x）或 2（混合架构，需配 ext）。
        ext: v2 扩展字段 dict（见 pack_v2_ext）；version>=2 时写进 header.reserved。

    Returns:
        写出文件的总字节数。

    Raises:
        SystemExit: 版本号不支持、tensor 维度不在 [1,4]、名字过长、文件大小不一致时退出。
    """
    # 校验版本号在合法范围内
    if version < FORMAT_VERSION_MIN or version > FORMAT_VERSION:
        sys.exit(f"error: unsupported format version {version}")
    # 根据 dtype 字符串获取对应的编号、元素大小、numpy 类型名
    dtype_code, elem_size, np_dtype = DTYPES[dtype]
    # 取出所有 tensor 名字的有序列表（保持插入顺序）
    names = list(tensors.keys())

    # 计算 tensor 索引表的结束位置 = header大小 + N × entry大小
    table_end = struct.calcsize(HEADER_FMT) + len(names) * struct.calcsize(ENTRY_FMT)
    # 数据区起始偏移：表结束后向上对齐到 64 字节边界
    data_offset = align_up(table_end)

    # 第一遍：只取 shape，计算每个 tensor 的对齐偏移，得到文件总大小。
    # 这一遍不读取实际数据，只做空间规划。
    entries = []       # 存放 (name, shape, offset, nbytes) 四元组
    offset = data_offset  # 当前数据写入偏移，从 data_offset 开始
    for name in names:
        # 获取 tensor 的形状，转为 int 列表
        shape = [int(d) for d in tensors[name].shape]
        # 校验维度数在 [1, 4] 范围内（权重矩阵最多 4 维）
        if not (1 <= len(shape) <= 4):
            sys.exit(f"error: {name}: ndim {len(shape)} not in [1, 4]")
        # 校验名字长度不超过 MAX_NAME（64 字节）
        if len(name) > MAX_NAME:
            sys.exit(f"error: tensor name too long: {name}")
        # 计算元素总数
        numel = 1
        for d in shape:
            numel *= d   # 累乘各维度得到总元素数
        # 计算该 tensor 占用的字节数 = 元素数 × 每元素字节数
        nbytes = numel * elem_size
        # 记录该 tensor 的元信息
        entries.append((name, shape, offset, nbytes))
        # 下一个 tensor 的偏移 = 当前偏移 + 当前大小，再向上对齐到 64 字节
        offset = align_up(offset + nbytes)
    # 所有 tensor 排完后的总偏移即为文件总大小
    total_bytes = offset

    # 构造 reserved 区：v2 时前 64 字节放扩展块，其余补 0；v1 全 0。
    reserved = b"\x00" * 96   # 默认全零的 96 字节 reserved
    if version >= 2:
        # v2 格式：将扩展字段打包后放入 reserved 前部，剩余补零
        ext_bytes = pack_v2_ext(ext or {})
        reserved = ext_bytes + b"\x00" * (96 - len(ext_bytes))

    # 按 HEADER_FMT 打包 192 字节的文件头
    header = struct.pack(
        HEADER_FMT,
        MAGIC,                              # 8s: 文件魔数 "TINYQWEN"
        version,                            # I: 格式版本号
        dtype_code,                         # I: 默认 dtype 编号
        int(cfg["n_layers"]),               # I: transformer 层数
        int(cfg["hidden_size"]),            # I: 隐藏层维度
        int(cfg["intermediate_size"]),      # I: MLP 中间层维度
        int(cfg["n_heads"]),                # I: 注意力 head 数
        int(cfg["n_kv_heads"]),             # I: KV head 数（GQA）
        int(cfg["head_dim"]),               # I: 每个 head 的维度
        int(cfg["vocab_size"]),             # I: 词表大小
        int(cfg["max_seq_len"]),            # I: 最大序列长度
        int(cfg["tied"]),                   # I: embedding/lm_head 是否共享权重
        0,                                  # I: reserved_u32（保留字段）
        float(cfg["rms_norm_eps"]),         # f: RMSNorm epsilon
        float(cfg["rope_theta"]),           # f: RoPE base frequency
        len(names),                         # Q: tensor 总数
        struct.calcsize(HEADER_FMT),        # Q: tensor 表偏移（= header 大小）
        data_offset,                        # Q: 数据区起始偏移
        total_bytes,                        # Q: 文件总大小
        reserved,                           # 96s: reserved 区（含 v2 扩展）
    )

    # 写入顺序与磁盘布局一致（docs/weight_format.md）：
    #   header -> tensor 表 -> 补零对齐 -> 64B 对齐的数据区。
    out_path = Path(out_path)
    # 自动创建输出目录
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as out:
        # 第一步：写入 192 字节 header
        out.write(header)
        # 第二步：逐个写入 tensor 索引条目（每条 120 字节）
        for name, shape, off, nbytes in entries:
            out.write(struct.pack(
                ENTRY_FMT,
                name.encode("ascii"),       # 64s: tensor 名字（ASCII 编码，不足补零）
                dtype_code,                 # I: dtype 编号
                len(shape),                 # I: 维度数
                *shape,                     # 4Q 的前 len(shape) 个：各维度大小
                *[0] * (4 - len(shape)),    # 4Q 的剩余位：不足 4 维补零
                off,                        # Q: 数据在文件中的字节偏移
                nbytes,                     # Q: 数据字节数
            ))
        # 第三步：表区末尾到第一个数据偏移之间补零，保证数据区 64B 对齐
        out.write(b"\x00" * (data_offset - out.tell()))

        # 第四步：逐个写入 tensor 的实际权重数据
        for name, shape, off, nbytes in entries:
            # 断言当前文件指针位置与该 tensor 的预期偏移一致（检测对齐 bug）
            assert out.tell() == off, f"alignment bug at {name}"
            # 将数组转为目标 dtype 后获取原始字节
            data = tensors[name].astype(np_dtype, copy=False).tobytes()
            # 断言实际字节数与预期一致（检测 size 计算 bug）
            assert len(data) == nbytes, f"size bug at {name}"
            # 写入权重数据
            out.write(data)
            # 每个 payload 之后补零到下一个 64B 边界，保证下一个 tensor 也对齐
            out.write(b"\x00" * (align_up(off + nbytes) - (off + nbytes)))
        # 最终断言：写入的总字节数与预计算的 total_bytes 一致
        assert out.tell() == total_bytes

    # 验证实际文件大小与预期一致
    size = os.path.getsize(out_path)
    if size != total_bytes:
        sys.exit(f"error: file size {size} != expected {total_bytes}")
    return total_bytes  # 返回文件总字节数


def print_table_summary(out_path: str | Path) -> None:
    """回读刚写出文件的 header + tensor 表，打印摘要（验收用）。

    以表格形式打印每个 tensor 的名字、形状、dtype、文件偏移和字节数，
    方便人工检查导出结果是否正确。对于 v2 格式还会打印 Qwen3.5 特有的
    扩展字段信息。

    Args:
        out_path: 要读取的 .tqwen 文件路径。
    """
    with open(out_path, "rb") as f:
        # 读取 192 字节的 header
        header = f.read(struct.calcsize(HEADER_FMT))
        # 解包 header 的所有字段
        fields = struct.unpack(HEADER_FMT, header)
        magic, version, dtype = fields[0], fields[1], fields[2]  # 提取魔数、版本、dtype
        # 校验魔数正确
        assert magic == MAGIC
        # 校验版本号在合法范围
        assert FORMAT_VERSION_MIN <= version <= FORMAT_VERSION, f"bad version {version}"
        # 校验 dtype 是已知值
        assert dtype in (DTYPE_F32, DTYPE_F16, DTYPE_I4, DTYPE_VQ2)
        # I4 / VQ2 文件是混合 dtype（大矩阵量化、小向量 f32），逐 tensor 取自己的标签。
        dtype_names = {DTYPE_F32: "f32", DTYPE_F16: "f16", DTYPE_I4: "i4", DTYPE_VQ2: "vq2"}
        # fields 下标: 0 magic, 1 version, 2 dtype, 3..12 十个 u32,
        #   13 eps, 14 theta, 15 tensor_count, 16 tensor_table_offset,
        #   17 data_offset, 18 total, 19 reserved
        if version >= 2:
            # v2 格式：从 reserved 字段解包扩展信息并打印
            ext = struct.unpack(EXT_FMT, fields[19][: struct.calcsize(EXT_FMT)])
            model_type = ext[0]  # 架构族标识
            # 将 model_type 数字转为可读标签
            label = {MODEL_QWEN2: "qwen2", MODEL_QWEN35: "qwen3_5"}.get(model_type, "?")
            print(f"format v{version} model_type={label} "
                  f"attn_interval={ext[6]} rotary={ext[7]:g} eos={ext[8]}")
            if model_type == MODEL_QWEN35:
                # Qwen3.5 额外打印线性注意力的 head 配置
                print(f"  linear: qk_heads={ext[1]} v_heads={ext[2]} "
                      f"qk_dim={ext[3]} v_dim={ext[4]} conv={ext[5]}")
        # 获取 tensor 总数
        tensor_count = fields[15]
        # 获取 tensor 表在文件中的偏移
        table_offset = fields[16]
        # 跳转到 tensor 表起始位置
        f.seek(table_offset)
        # 打印表头
        print(f"{'name':<56} {'shape':<22} {'dtype':<6} {'offset':>12} {'nbytes':>14}")
        # 逐个读取并打印 tensor 条目
        for _ in range(tensor_count):
            # 读取 120 字节的 entry 并解包
            name_b, t_dtype, ndim, s0, s1, s2, s3, off, nbytes = struct.unpack(
                ENTRY_FMT, f.read(struct.calcsize(ENTRY_FMT)))
            # 去除名字尾部的零字节并解码为 ASCII 字符串
            name = name_b.rstrip(b"\x00").decode("ascii")
            # 根据实际维度数截取有效的 shape 分量
            shape = [s0, s1, s2, s3][:ndim]
            # 将 dtype 编号转为可读字符串
            label = dtype_names.get(t_dtype, f"?{t_dtype}")
            # 打印一行 tensor 信息
            print(f"{name:<56} {str(shape):<22} {label:<6} {off:>12} {nbytes:>14}")


# ---- HF 模型收集 ------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """解析命令行参数。

    Returns:
        解析后的参数命名空间，包含 model、out、max_seq_len、dtype。
    """
    p = argparse.ArgumentParser(description=__doc__)  # 用模块 docstring 作为帮助文本
    p.add_argument("--model", required=True,
                   help="HF model dir or repo id (must contain config.json)")  # HF 模型路径
    p.add_argument("--out", required=True, help="output .tqwen path")  # 输出文件路径
    p.add_argument("--max-seq-len", type=int, default=0,
                   help="override max_seq_len in header (0 = use config)")  # 覆盖最大序列长度
    p.add_argument("--dtype", choices=sorted(DTYPES), default="f32",
                   help="权重存储精度：f32（默认）或 f16（文件减半，配 f16 matvec 用）")
    return p.parse_args()


def find_local_dir(model: str) -> Path:
    """解析本地模型目录。repo id 的下载留给用户自己执行
    （如 `huggingface-cli download`），让本脚本保持轻依赖。

    Args:
        model: 用户传入的模型路径或 repo id。

    Returns:
        验证通过的本地模型目录 Path。

    Raises:
        SystemExit: 如果指定路径不是包含 config.json 的有效目录。
    """
    path = Path(model)
    # 检查是否为有效目录且包含 config.json
    if path.is_dir() and (path / "config.json").exists():
        return path
    # 无效路径时报错并提示下载命令
    sys.exit(f"error: {model} is not a local dir with config.json. "
             f"Download it first, e.g. `huggingface-cli download {model} --local-dir {model}`")


def detect_model_type(cfg: dict) -> int:
    """从 HF config 判定架构族。Qwen3.5 是多模态壳（ForConditionalGeneration），
    文本塔参数在 text_config 里，model_type 为 qwen3_5。

    Args:
        cfg: HF config.json 解析后的完整字典。

    Returns:
        MODEL_QWEN2 (0) 或 MODEL_QWEN35 (1)。

    Raises:
        SystemExit: model_type 不在已知列表中时退出。
    """
    mt = cfg.get("model_type", "")  # 从 config 获取 model_type 字符串
    if mt in ("qwen3_5", "qwen3_5_moe"):
        return MODEL_QWEN35   # Qwen3.5 系列（含 MoE 变体）
    if mt in ("qwen2", "qwen2_5", "qwen3"):
        return MODEL_QWEN2    # Qwen2.x / Qwen3 纯文本系列
    # 未知架构报错退出
    sys.exit(f"error: unsupported model_type '{mt}' (expected qwen2* / qwen3_5)")


def text_cfg_of(cfg: dict, model_type: int) -> dict:
    """取出纯文本塔的配置子树。Qwen3.5 在 text_config，Qwen2.x 在顶层。

    Qwen3.5 是条件生成模型（多模态壳），其文本部分的超参嵌套在
    config["text_config"] 下；而 Qwen2.x 直接在顶层。

    Args:
        cfg: 完整的 HF config 字典。
        model_type: 架构族标识。

    Returns:
        文本塔的配置字典。

    Raises:
        SystemExit: Qwen3.5 缺少 text_config 时退出。
    """
    if model_type == MODEL_QWEN35:
        # Qwen3.5：从嵌套的 text_config 取文本塔配置
        if "text_config" not in cfg:
            sys.exit("error: qwen3_5 config missing text_config")
        return cfg["text_config"]
    # Qwen2.x：配置直接在顶层
    return cfg


def load_config(model_dir: Path) -> tuple[dict, int]:
    """读取 HF config.json，返回 (文本塔配置, model_type)。

    校验导出所需字段；head_dim / rope_theta 缺省时推导。Qwen3.5 的
    rope_theta 在 rope_parameters 里。

    具体步骤：
    1. 读取并解析 config.json。
    2. 检测架构族（Qwen2.x / Qwen3.5）。
    3. 提取文本塔配置子树。
    4. 校验必需字段是否存在。
    5. 推导缺失的 head_dim（= hidden_size / num_attention_heads）。
    6. 查找 rope_theta（Qwen2.x 顶层直给，Qwen3.5 在 rope_parameters 里）。

    Args:
        model_dir: HF 模型目录路径。

    Returns:
        (tcfg, model_type) 元组，tcfg 是补全后的文本塔配置字典。

    Raises:
        SystemExit: 缺少必需字段或无法推导 head_dim/rope_theta 时退出。
    """
    # 读取并解析 config.json
    cfg = json.loads((model_dir / "config.json").read_text())
    # 检测架构族
    model_type = detect_model_type(cfg)
    # 提取文本塔配置（拷贝一份以免修改原始 dict）
    tcfg = dict(text_cfg_of(cfg, model_type))

    # 列出导出所必需的 config 字段
    required = ["hidden_size", "intermediate_size", "num_hidden_layers",
                "num_attention_heads", "num_key_value_heads", "vocab_size",
                "rms_norm_eps", "max_position_embeddings"]
    # 检查是否有缺失字段
    missing = [k for k in required if k not in tcfg]
    if missing:
        sys.exit(f"error: config missing fields: {missing}")

    # 尝试获取 head_dim；HF config 可能不显式给出，需要从 hidden_size / heads 推导
    head_dim = tcfg.get("head_dim")
    if head_dim is None:
        # 标准推导：head_dim = hidden_size / num_attention_heads
        head_dim = tcfg["hidden_size"] // tcfg["num_attention_heads"]
    tcfg["head_dim"] = head_dim  # 写回配置
    # 校验 GQA 约束：attention heads 必须是 kv heads 的整数倍
    if tcfg["num_attention_heads"] % tcfg["num_key_value_heads"] != 0:
        sys.exit("error: num_attention_heads % num_key_value_heads != 0")

    # rope_theta：Qwen2.x 顶层直给；Qwen3.5 在 rope_parameters.rope_theta。
    if "rope_theta" not in tcfg:
        # 尝试从 rope_parameters 子字典获取
        rp = tcfg.get("rope_parameters") or {}
        if "rope_theta" not in rp:
            sys.exit("error: cannot find rope_theta in config")
        tcfg["rope_theta"] = rp["rope_theta"]  # 从嵌套位置提取并写回顶层
    return tcfg, model_type


def _fold_one(a):
    """zero-centered RMSNorm 的 +1 折叠变换。

    Qwen3.5 使用 zero-centered RMSNorm，权重初始化为 0 附近，
    实际缩放因子为 (1 + weight)。导出时将 +1 折进权重，使 runtime
    可以直接用 weight 做缩放，省去运行时的加法。

    Args:
        a: 输入数组（numpy 或 torch tensor）。

    Returns:
        a + 1.0 的结果数组。
    """
    return a + 1.0  # 将 zero-centered 偏移折入权重


def _squeeze_conv(a):
    """将 HF depthwise conv1d 权重从 3D squeeze 为 2D。

    HF 存储的 depthwise conv1d 权重形状为 [out_channels, 1, kernel_size]，
    中间的 1 是 in_channels（depthwise 卷积 in_ch=out_ch，这里简化为 1）。
    tinyqwen runtime 期望 [out_channels, kernel_size] 的 2D 布局。

    Args:
        a: 形状为 [out, 1, kernel] 的数组。

    Returns:
        形状为 [out, kernel] 的 2D 数组。

    Raises:
        AssertionError: 输入不是 3D 或第二维不为 1。
    """
    arr = np.asarray(a)  # 确保是 numpy 数组
    # 校验输入形状：必须是 3D 且中间维度为 1
    assert arr.ndim == 3 and arr.shape[1] == 1, f"unexpected conv shape {arr.shape}"
    # reshape 去掉中间的 1 维度：[out, 1, kernel] -> [out, kernel]
    return arr.reshape(arr.shape[0], arr.shape[2])


def plan_tensors(tcfg: dict, model_type: int):
    """构造导出计划：返回 (tiny_names, src_of, transform_of)。

    根据模型架构族（Qwen2.x / Qwen3.5）生成三份映射：
    - tiny_names: 写入 .tqwen 的 tensor 名字有序列表（== C++ loader 期望的顺序）。
    - src_of[tiny_name]: 对应 HF safetensors 里的源 tensor 名。
    - transform_of[tiny_name]: 可选的数组变换函数（_fold_one / _squeeze_conv / None）。

    Qwen3.5 的 HF 权重带 model.language_model. 前缀，写入时统一去掉
    language_model. 以复用 Qwen2.x 的命名约定。zero-centered RMSNorm
    （input/post/model/q_norm/k_norm）在导出时把 +1 折进权重，
    linear_attn.norm 是普通 RMSNormGated 不折。

    Args:
        tcfg: 文本塔配置字典。
        model_type: 架构族标识（MODEL_QWEN2 或 MODEL_QWEN35）。

    Returns:
        (names, src_of, transform_of) 三元组。
    """
    names: list[str] = []         # .tqwen 中的 tensor 名字有序列表
    src_of: dict[str, str] = {}   # tiny名 -> HF源名的映射
    transform_of: dict[str, object] = {}  # tiny名 -> 变换函数的映射

    def add(tiny: str, hf: str, transform=None):
        """注册一个 tensor 映射条目。

        Args:
            tiny: .tqwen 中的目标名字。
            hf: HF safetensors 中的源名字。
            transform: 可选的数组变换函数。
        """
        names.append(tiny)           # 追加到有序列表
        src_of[tiny] = hf            # 记录源名映射
        transform_of[tiny] = transform  # 记录变换函数（None 表示无需变换）

    if model_type == MODEL_QWEN35:
        # ===== Qwen3.5 混合架构的 tensor 映射 =====
        prefix = "model.language_model."  # HF 中 Qwen3.5 权重的公共前缀
        # embedding 层
        add("model.embed_tokens.weight", prefix + "embed_tokens.weight")
        # 获取每层的类型列表（linear_attention / full_attention）
        layer_types = tcfg.get("layer_types") or []
        for i in range(tcfg["num_hidden_layers"]):
            lp = prefix + f"layers.{i}."   # HF 侧第 i 层的前缀
            tp = f"model.layers.{i}."      # .tqwen 侧第 i 层的前缀（去掉 language_model）
            # 判断该层是 linear attention 还是 full attention
            is_linear = layer_types[i] == "linear_attention" if layer_types else \
                ((i + 1) % tcfg.get("full_attention_interval", 4) != 0)
            # input_layernorm 是 zero-centered RMSNorm，需要 +1 折叠
            add(tp + "input_layernorm.weight", lp + "input_layernorm.weight", _fold_one)
            if is_linear:
                # ---- 线性注意力层的 tensor 映射 ----
                la, la_t = lp + "linear_attn.", tp + "linear_attn."
                add(la_t + "in_proj_qkv.weight", la + "in_proj_qkv.weight")  # QKV 合并投影
                add(la_t + "in_proj_z.weight", la + "in_proj_z.weight")      # z 门控投影
                add(la_t + "in_proj_b.weight", la + "in_proj_b.weight")      # b 偏置投影
                add(la_t + "in_proj_a.weight", la + "in_proj_a.weight")      # a 参数投影
                add(la_t + "out_proj.weight", la + "out_proj.weight")        # 输出投影
                add(la_t + "conv1d.weight", la + "conv1d.weight", _squeeze_conv)  # depthwise conv，需 squeeze
                add(la_t + "A_log", la + "A_log")                            # SSM 状态矩阵 A 的对数
                add(la_t + "dt_bias", la + "dt_bias")                        # dt 偏置
                add(la_t + "norm.weight", la + "norm.weight")                # 普通 RMSNormGated，不折 +1
            else:
                # ---- 全注意力层的 tensor 映射 ----
                sa, sa_t = lp + "self_attn.", tp + "self_attn."
                add(sa_t + "q_proj.weight", sa + "q_proj.weight")    # Q 投影权重
                add(sa_t + "k_proj.weight", sa + "k_proj.weight")    # K 投影权重
                add(sa_t + "v_proj.weight", sa + "v_proj.weight")    # V 投影权重
                add(sa_t + "o_proj.weight", sa + "o_proj.weight")    # O 投影权重
                add(sa_t + "q_norm.weight", sa + "q_norm.weight", _fold_one)  # Q RMSNorm，zero-centered 折 +1
                add(sa_t + "k_norm.weight", sa + "k_norm.weight", _fold_one)  # K RMSNorm，zero-centered 折 +1
            # post_attention_layernorm 也是 zero-centered，需折 +1
            add(tp + "post_attention_layernorm.weight",
                lp + "post_attention_layernorm.weight", _fold_one)
            # MLP 三个投影：gate、up、down
            add(tp + "mlp.gate_proj.weight", lp + "mlp.gate_proj.weight")  # SwiGLU gate
            add(tp + "mlp.up_proj.weight", lp + "mlp.up_proj.weight")      # SwiGLU up
            add(tp + "mlp.down_proj.weight", lp + "mlp.down_proj.weight")  # down projection
        # 最终 RMSNorm（zero-centered，折 +1）
        add("model.norm.weight", prefix + "norm.weight", _fold_one)
        if not tcfg.get("tie_word_embeddings", False):
            # Qwen3.5 0.8B 为 tied embeddings，正常不会走到这里。
            # 非 tied 时单独导出 lm_head 权重
            add("lm_head.weight", "lm_head.weight")
    else:
        # ===== Qwen2.x / Qwen3 稠密 标准 Transformer 的 tensor 映射 =====
        # 两者层结构相同，差异只在两处可选权重（runtime 按权重存在性识别，无需格式标志）：
        #   - attention bias：Qwen2.x 有，Qwen3 稠密 attention_bias=false 没有
        #   - QK per-head RMSNorm：Qwen3 稠密有，Qwen2.x 没有。注意 Qwen3 的
        #     q_norm/k_norm 是标准 RMSNorm，**不**折 +1（区别于 Qwen3.5 的 zero-centered）
        is_qwen3_dense = tcfg.get("model_type") == "qwen3"
        attn_bias = bool(tcfg.get("attention_bias", not is_qwen3_dense))
        add("model.embed_tokens.weight", "model.embed_tokens.weight")  # embedding 层
        for i in range(tcfg["num_hidden_layers"]):
            p = f"model.layers.{i}."  # 第 i 层的前缀
            add(p + "input_layernorm.weight", p + "input_layernorm.weight")      # 输入 RMSNorm
            add(p + "self_attn.q_proj.weight", p + "self_attn.q_proj.weight")    # Q 投影权重
            add(p + "self_attn.k_proj.weight", p + "self_attn.k_proj.weight")    # K 投影权重
            add(p + "self_attn.v_proj.weight", p + "self_attn.v_proj.weight")    # V 投影权重
            if attn_bias:
                add(p + "self_attn.q_proj.bias", p + "self_attn.q_proj.bias")    # Q 投影偏置
                add(p + "self_attn.k_proj.bias", p + "self_attn.k_proj.bias")    # K 投影偏置
                add(p + "self_attn.v_proj.bias", p + "self_attn.v_proj.bias")    # V 投影偏置
            if is_qwen3_dense:
                add(p + "self_attn.q_norm.weight", p + "self_attn.q_norm.weight")  # Q RMSNorm，不折 +1
                add(p + "self_attn.k_norm.weight", p + "self_attn.k_norm.weight")  # K RMSNorm，不折 +1
            add(p + "self_attn.o_proj.weight", p + "self_attn.o_proj.weight")    # O 投影权重
            add(p + "post_attention_layernorm.weight", p + "post_attention_layernorm.weight")  # 后 RMSNorm
            add(p + "mlp.gate_proj.weight", p + "mlp.gate_proj.weight")          # MLP gate
            add(p + "mlp.up_proj.weight", p + "mlp.up_proj.weight")              # MLP up
            add(p + "mlp.down_proj.weight", p + "mlp.down_proj.weight")          # MLP down
        add("model.norm.weight", "model.norm.weight")  # 最终 RMSNorm
        if not tcfg.get("tie_word_embeddings", False):
            # 非 tied 时导出 lm_head 权重
            add("lm_head.weight", "lm_head.weight")
    return names, src_of, transform_of


def build_shard_map(model_dir: Path, src_names: list[str]) -> dict[str, Path]:
    """把每个 HF 源 tensor 名映射到包含它的 safetensors 分片文件。

    HF 大模型通常将权重分成多个 .safetensors 分片文件，并用
    model.safetensors.index.json 记录每个 tensor 所在的分片。
    如果只有一个 model.safetensors 文件（小模型），则所有 tensor 都指向它。

    Args:
        model_dir: HF 模型目录。
        src_names: 需要查找的 HF tensor 名列表。

    Returns:
        {tensor_name: safetensors_file_path} 的映射字典。

    Raises:
        SystemExit: 找不到任何 safetensors 文件时退出。
    """
    index_file = model_dir / "model.safetensors.index.json"  # 分片索引文件
    if index_file.exists():
        # 有多分片索引：读取 weight_map 建立 tensor->shard 映射
        index = json.loads(index_file.read_text())["weight_map"]
        return {n: model_dir / index[n] for n in src_names}
    single = model_dir / "model.safetensors"  # 单文件情况
    if single.exists():
        # 所有 tensor 都在同一个文件中
        return {n: single for n in src_names}
    # 既没有索引也没有单文件，报错退出
    sys.exit("error: no model.safetensors[.index.json] found")


def main() -> None:
    """主入口函数：串联整个导出流程。

    流程：
    1. 解析命令行参数。
    2. 定位本地模型目录。
    3. 加载并校验 HF config。
    4. 生成 tensor 名映射计划。
    5. 构建 safetensors 分片映射。
    6. 构造 LazyTensors 惰性加载器。
    7. 调用 write_tqwen 写出 .tqwen 文件。
    8. 打印摘要信息。
    """
    args = parse_args()                             # 解析命令行参数
    model_dir = find_local_dir(args.model)          # 定位并验证本地模型目录
    cfg, model_type = load_config(model_dir)        # 加载 HF config 并检测架构族
    names, src_of, transform_of = plan_tensors(cfg, model_type)  # 生成 tensor 映射计划
    shard_map = build_shard_map(model_dir, list(src_of.values()))  # 构建分片文件映射

    from safetensors import safe_open  # 延迟导入：晚失败、报错清晰（避免无 torch 环境启动报错）

    opened: dict[Path, object] = {}  # 缓存已打开的 safetensors 文件句柄

    def get_shard(path: Path):
        """获取指定分片文件的 safe_open 句柄（带缓存）。

        Args:
            path: safetensors 文件路径。

        Returns:
            safe_open 上下文管理器对象。
        """
        if path not in opened:
            # 用 torch 框架读：真实 Qwen 权重是 bf16，numpy 不认识 bf16；
            # 读出来后在 LazyTensors 里统一转成 fp32。
            opened[path] = safe_open(str(path), framework="torch")
        return opened[path]

    # 先校验所有源 tensor 都存在（任何缺失都在写文件之前报错）。
    # 这样避免写到一半才发现缺 tensor 导致浪费时间和产生不完整文件。
    for tiny in names:
        src = src_of[tiny]                       # 获取 HF 源 tensor 名
        f = get_shard(shard_map[src])            # 获取对应的 safetensors 句柄
        if src not in f.keys():                  # 检查 tensor 是否存在
            sys.exit(f"error: tensor {src} not found in {shard_map[src]}")

    # 组装写出 .tqwen 所需的 header 配置字典
    header_cfg = {
        "n_layers": cfg["num_hidden_layers"],            # transformer 层数
        "hidden_size": cfg["hidden_size"],               # 隐藏层维度
        "intermediate_size": cfg["intermediate_size"],   # MLP 中间维度
        "n_heads": cfg["num_attention_heads"],           # 注意力 head 数
        "n_kv_heads": cfg["num_key_value_heads"],        # KV head 数（GQA）
        "head_dim": cfg["head_dim"],                     # head 维度
        "vocab_size": cfg["vocab_size"],                 # 词表大小
        "max_seq_len": args.max_seq_len or cfg["max_position_embeddings"],  # 最大序列长度
        "tied": 1 if cfg.get("tie_word_embeddings", False) else 0,  # 是否 tied embedding
        "rms_norm_eps": cfg["rms_norm_eps"],             # RMSNorm epsilon
        "rope_theta": cfg["rope_theta"],                 # RoPE base frequency
    }

    # v2 扩展字段（仅 qwen3_5 混合架构需要）。
    version = 1      # 默认 v1 格式
    ext = None       # 默认无扩展
    if model_type == MODEL_QWEN35:
        version = 2  # Qwen3.5 使用 v2 格式
        eos = cfg.get("eos_token_id", 0)  # 获取 EOS token ID
        if isinstance(eos, (list, tuple)):  # 多 eos 取第一个
            eos = eos[0]
        ext = {
            "model_type": MODEL_QWEN35,                    # 架构族标识
            "linear_num_qk_heads": cfg["linear_num_key_heads"],    # 线性注意力 QK head 数
            "linear_num_v_heads": cfg["linear_num_value_heads"],   # 线性注意力 V head 数
            "linear_qk_head_dim": cfg["linear_key_head_dim"],      # 线性注意力 QK head 维度
            "linear_v_head_dim": cfg["linear_value_head_dim"],     # 线性注意力 V head 维度
            "linear_conv_kernel_dim": cfg["linear_conv_kernel_dim"],  # conv1d 核大小
            "full_attention_interval": cfg.get("full_attention_interval", 4),  # full attn 间隔
            "partial_rotary_factor": cfg.get("partial_rotary_factor",
                                             (cfg.get("rope_parameters") or {})
                                             .get("partial_rotary_factor", 1.0)),  # 部分 RoPE 因子
            "eos_token_id": int(eos),                      # EOS token ID
        }

    # 逐个流式读取 tensor，保持内存占用平稳：write_tqwen 需要一个映射，
    # 所以这里传给它一个惰性 dict，访问时才从对应分片加载源 tensor 并做变换。
    # 注意每个 tensor 会被读两次（shape 一遍 + payload 一遍）——
    # 相比把 2GB fp32 模型整个放进内存，这是可接受的代价。
    np_dtype = DTYPES[args.dtype][2]  # 获取目标 numpy dtype 字符串

    class LazyTensors(dict):
        """惰性 tensor 字典：仅在 __getitem__ 被调用时才从磁盘加载并变换。

        继承 dict 是为了满足 write_tqwen 对映射接口的要求（.keys(), .shape 等）。
        通过 fromkeys(names) 初始化时只设置 key，value 为 None；
        实际数据在 write_tqwen 遍历写入时按需加载。
        """
        def __getitem__(self, key):
            """按需加载并变换 tensor。

            Args:
                key: .tqwen 中的 tensor 名字。

            Returns:
                目标 dtype 的 numpy 数组。
            """
            src = src_of[key]                                    # 获取 HF 源 tensor 名
            tensor = get_shard(shard_map[src]).get_tensor(src)   # 从 safetensors 读取 torch tensor
            # bf16/fp16 -> fp32 numpy；f16 导出时再降到 float16
            # （round-to-nearest-even，numpy astype 的默认舍入）。
            arr = tensor.float().numpy()                         # 先转 fp32 numpy
            tr = transform_of.get(key)                           # 获取变换函数（如有）
            if tr is not None:
                arr = tr(arr)                                    # 应用变换（如 +1 折叠、squeeze）
            return np.asarray(arr).astype(np_dtype, copy=False)  # 转为目标 dtype

    # 用 LazyTensors.fromkeys(names) 创建只有 key 的惰性字典，传给 write_tqwen
    total = write_tqwen(args.out, header_cfg, LazyTensors.fromkeys(names), args.dtype,
                        version=version, ext=ext)

    # 打印导出结果的摘要信息
    mt_label = "qwen3_5" if model_type == MODEL_QWEN35 else "qwen2"  # 架构族标签
    print(f"model dir  : {model_dir}")           # 模型目录
    print(f"model type : {mt_label} (format v{version})")  # 架构族和格式版本
    print(f"dtype      : {args.dtype}")          # 导出精度
    print(f"config     : layers={header_cfg['n_layers']} hidden={header_cfg['hidden_size']} "
          f"inter={header_cfg['intermediate_size']} heads={header_cfg['n_heads']} "
          f"kv_heads={header_cfg['n_kv_heads']} head_dim={header_cfg['head_dim']} "
          f"vocab={header_cfg['vocab_size']} tied={header_cfg['tied']}")  # 模型超参摘要
    print_table_summary(args.out)                # 打印 tensor 表摘要
    print(f"out        : {args.out}")            # 输出文件路径
    print(f"file size  : {total} bytes ({total / 2 ** 20:.1f} MB) OK")  # 文件大小


if __name__ == "__main__":
    main()  # 脚本入口点
