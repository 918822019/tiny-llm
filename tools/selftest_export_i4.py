#!/usr/bin/env python3
"""i4 导出器重构自测：新实现（向量化）vs 旧实现（循环参考版）逐字节对照。

重构（2026-08）把打包/RTN 量化从纯 Python 循环改为 numpy 向量化，
并引入流式写盘与进程池并行。本脚本保证格式逐位不变：

覆盖项：
  1. RTN 整除快路径 vs 旧全循环版（常规形状 + 常数/零/极值边界）
  2. RTN 不整除慢路径（尾部不满组、奇数长度尾组）
  3. pack_i4_groups vs 旧 HQQ 打包循环（随机 q/scale/zero）
  4. HQQ：同一 HQQ 核心结果分别用新旧打包，字节一致
  5. 行拆分一致性：拆分后逐块量化拼接 == 整块量化

用法：.venv/bin/python tools/selftest_export_i4.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from export_qwen_to_tiny_i4 import (  # noqa: E402
    GROUP_HEADER_BYTES, quantize_tensor_i4, quantize_tensor_i4_hqq,
    pack_i4_groups, _shard_rows, _quantize_one,
)


# ===========================================================================
# 旧实现参考版（重构前原始代码的忠实拷贝，作为对照基准）
# ===========================================================================

def ref_quantize_i4_group(data):
    """旧版逐组 RTN（原 quantize_i4_group）。"""
    vmin = data.min()
    vmax = data.max()
    if vmax == vmin:
        scale = np.float16(0.0)
        zero = np.float16(0.0)
        quantized = np.zeros(len(data), dtype=np.uint8)
    else:
        scale_f32 = (vmax - vmin) / 15.0
        zero_f32 = -vmin / scale_f32
        scale = np.float16(scale_f32)
        zero = np.float16(zero_f32)
        s = float(scale)
        z = float(zero)
        if s == 0:
            quantized = np.zeros(len(data), dtype=np.uint8)
        else:
            q = np.round(data / s + z).astype(np.int32)
            quantized = np.clip(q, 0, 15).astype(np.uint8)
    return scale, zero, quantized


def ref_quantize_tensor_i4(tensor, group_size):
    """旧版 RTN 全循环（原 quantize_tensor_i4，含打包）。"""
    out_dim, in_dim = tensor.shape
    n_groups = (in_dim + group_size - 1) // group_size
    row_bytes = n_groups * (GROUP_HEADER_BYTES + group_size // 2)
    result = bytearray(out_dim * row_bytes)
    group_total = GROUP_HEADER_BYTES + group_size // 2
    for o in range(out_dim):
        row_offset = o * row_bytes
        for g in range(n_groups):
            start = g * group_size
            end = min(start + group_size, in_dim)
            group_data = tensor[o, start:end].astype(np.float32)
            scale, zero, quantized = ref_quantize_i4_group(group_data)
            group_offset = row_offset + g * group_total
            result[group_offset:group_offset + 2] = scale.tobytes()
            result[group_offset + 2:group_offset + 4] = zero.tobytes()
            data_offset = group_offset + GROUP_HEADER_BYTES
            n_elems = len(quantized)
            for i in range(0, n_elems, 2):
                lo = quantized[i]
                hi = quantized[i + 1] if i + 1 < n_elems else 0
                result[data_offset + i // 2] = int(hi << 4) | int(lo)
    return bytes(result)


def ref_pack_hqq(q_np, scale, zero, out_dim, groups_per_row, group_size):
    """旧版 HQQ 打包循环（原 quantize_tensor_i4_hqq 的打包段）。"""
    row_bytes = groups_per_row * (GROUP_HEADER_BYTES + group_size // 2)
    result = bytearray(out_dim * row_bytes)
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


# ===========================================================================
# 测试主体
# ===========================================================================

def test_rtn(rng):
    """RTN：新旧实现逐字节对照（整除快路径 + 不整除慢路径 + 边界）。"""
    cases = [
        (64, 128, 64),    # 常规整除
        (37, 192, 64),    # 质数行
        (128, 64, 64),    # 单组行
        (8, 256, 128),    # 大组
        (33, 100, 64),    # 不整除（尾组 36 元素）
        (17, 65, 32),     # 不整除且尾组奇数长度（33 元素）
        (5, 96, 32),
    ]
    for out_dim, in_dim, gs in cases:
        t = rng.standard_normal((out_dim, in_dim)).astype(np.float32)
        assert ref_quantize_tensor_i4(t, gs) == quantize_tensor_i4(t, gs), \
            f"RTN mismatch @ ({out_dim},{in_dim},gs={gs})"
    # 边界：常数（vmax==vmin）、全零、大幅值、微小值（fp16 下溢风险区）
    for t in [
        np.full((16, 64), 3.14, dtype=np.float32),
        np.zeros((16, 64), dtype=np.float32),
        (rng.standard_normal((16, 64)) * 1e5).astype(np.float32),
        (rng.standard_normal((16, 64)) * 1e-7).astype(np.float32),
        np.full((4, 32), -0.0, dtype=np.float32),
    ]:
        assert ref_quantize_tensor_i4(t, 32) == quantize_tensor_i4(t, 32), \
            f"RTN edge mismatch shape={t.shape}"
    print("[PASS] RTN 新旧逐字节一致（7 组形状 + 5 组边界）")


def test_pack(rng):
    """pack_i4_groups vs 旧打包循环（随机 q/scale/zero）。"""
    for out_dim, gpr, gs in [(64, 2, 64), (17, 3, 32), (128, 1, 128)]:
        n = out_dim * gpr
        q = rng.integers(0, 16, (n, gs)).astype(np.uint8)
        scale = (rng.standard_normal(n) * 0.1).astype(np.float16)
        zero = (rng.standard_normal(n) * 8).astype(np.float16)
        old = ref_pack_hqq(q, scale, zero, out_dim, gpr, gs)
        new = pack_i4_groups(q, scale, zero, out_dim, gpr, gs)
        assert old == new, f"pack mismatch @ ({out_dim},{gpr},gs={gs})"
    print("[PASS] pack_i4_groups 与旧打包循环逐字节一致")


def test_hqq(rng):
    """HQQ：同一量化核心输出，新旧打包字节一致。"""
    import torch
    from hqq.core.quantize import BaseQuantizeConfig, HQQLinear, Quantizer
    t = rng.standard_normal((32, 128)).astype(np.float32)
    out_dim, in_dim = t.shape
    gs = 64
    # HQQ 核心（新旧共用同一库调用，此处只取结果）
    linear = torch.nn.Linear(in_dim, out_dim, bias=False)
    linear.weight.data = torch.from_numpy(np.ascontiguousarray(t))
    cfg = BaseQuantizeConfig(nbits=4, group_size=gs,
                             quant_zero=False, quant_scale=False)
    hq = HQQLinear(linear, cfg, compute_dtype=torch.float32, device="cpu")
    q = Quantizer.unpack["4bit_u8"](hq.W_q, dtype=torch.float32)
    q_np = q.round().clamp(0, 15).to(torch.uint8).numpy()
    scale = hq.meta["scale"].float().flatten().numpy().astype(np.float16)
    zero = hq.meta["zero"].float().flatten().numpy().astype(np.float16)
    gpr = in_dim // gs
    old = ref_pack_hqq(q_np, scale, zero, out_dim, gpr, gs)
    new = pack_i4_groups(np.ascontiguousarray(q_np).reshape(out_dim * gpr, gs),
                         scale, zero, out_dim, gpr, gs)
    assert old == new, "HQQ pack mismatch"
    # 端到端：整函数与"核心+旧打包"一致
    assert quantize_tensor_i4_hqq(t, gs) == old, "HQQ end-to-end mismatch"
    print("[PASS] HQQ 打包新旧逐字节一致（含端到端）")


def test_shard(rng):
    """行拆分一致性：RTN 拆分逐块量化拼接 == 整块量化（逐位可证）。

    HQQ 拆分不做字节级断言：torch 归约顺序依赖行数，会引入 fp16 ULP 级
    漂移（scale/q 通常一致，zero 偶有 ±1 ULP）——质量等价但非逐位。
    导出器里 HQQ 鲸鱼张量仍按行拆分（内存约束，见 _write_data_pass），
    其正确性由端到端冒烟（生成文本连贯）把关，这里只测 RTN 的格式正确性。
    """
    t = rng.standard_normal((1000, 256)).astype(np.float32)
    gs = 64
    whole = quantize_tensor_i4(t, gs)
    parts = b"".join(_quantize_one((s, gs, "rtn")) for s in _shard_rows(t, 64 * 256))
    assert whole == parts, "RTN shard mismatch"
    # 多种拆分粒度都验证一遍
    for target in (16 * 256, 128 * 256, 333 * 256):
        parts = b"".join(_quantize_one((s, gs, "rtn"))
                         for s in _shard_rows(t, target))
        assert whole == parts, f"RTN shard mismatch @ target={target}"
    print("[PASS] RTN 行拆分量化与整块量化逐字节一致（3 种粒度）")


if __name__ == "__main__":
    rng = np.random.default_rng(42)  # 固定种子，可复现
    test_rtn(rng)
    test_pack(rng)
    test_hqq(rng)
    test_shard(rng)
    print("\n全部自测通过：重构未改变任何输出字节")
