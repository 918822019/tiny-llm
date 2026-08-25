#!/usr/bin/env python3
"""把一个 VQ2 .tqwen 的 embed 从 f16/f32 量化为紧凑 INT4（补齐体积短板）。

布局与 runtime/tiny_format.h 的 i4 packing 一致：
  每组 [scale_fp16(2B) | zero_fp16(2B) | packed_uint4(gs/2 B)]，低 nibble 在前；
  非对称 min-max：scale=(max-min)/15, zero=round(-min/scale), q=round(w/scale+zero)。
反量化 (q-zero)*scale，对应 C++ dequant_i4_row / matvec_i4。

同时把 header 的 quant_group_size 写入 v2 ext（偏移 96+40），供运行时读取。
输入须是 header dtype=VQ2 的文件（线性层已是 VQ2）；embed 变为 dtype=I4。

用法:
    python tools/quantize_embed_i4.py --in model_vq2.tqwen --out model_vq2_ei4.tqwen \
        [--group-size 64]
"""
import argparse, struct
import numpy as np

ALIGN = 64
HEADER_FMT = "<8s12Iff4Q96s"
ENTRY_FMT = "<64sII4QQQ"
DTYPE_I4 = 3   # tiny_format.h: Dtype::kI4=3（2 是预留 kI8）
QUANT_GS_OFFSET = 96 + 40  # TinyHeaderV2Ext.quant_group_size 在 header 内的偏移


def align_up(x, a=ALIGN): return (x + a - 1) // a * a


def quantize_embed_i4(W: np.ndarray, gs: int):
    """W [vocab, dim] fp32 → i4 packed bytes（逐行逐组）。"""
    vocab, dim = W.shape
    groups = (dim + gs - 1) // gs
    group_total = 4 + gs // 2
    out = np.zeros(vocab * groups * group_total, dtype=np.uint8)
    Wf = W.astype(np.float32)
    for r in range(vocab):
        row_off = r * groups * group_total
        for g in range(groups):
            s, e = g * gs, min((g + 1) * gs, dim)
            blk = Wf[r, s:e]
            vmin, vmax = float(blk.min()), float(blk.max())
            if vmax - vmin < 1e-12:
                scale_f, zero_f = 0.0, 0.0
                q = np.zeros(blk.shape, dtype=np.int32)
            else:
                scale_f = (vmax - vmin) / 15.0
                zero_f = int(np.clip(round(-vmin / scale_f), 0, 15))
                q = np.clip(np.round(blk / scale_f + zero_f), 0, 15).astype(np.int32)
            go = row_off + g * group_total
            scale_h = np.float16(scale_f).view(np.uint16).item()
            zero_h = np.float16(float(zero_f)).view(np.uint16).item()
            struct.pack_into("<H", out, go, scale_h)
            struct.pack_into("<H", out, go + 2, zero_h)
            packed = q.astype(np.uint8)
            n = len(packed)
            for i in range(0, n, 2):
                lo = packed[i]
                hi = packed[i + 1] if i + 1 < n else 0
                out[go + 4 + i // 2] = lo | (hi << 4)
    return out.tobytes()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--group-size", type=int, default=64)
    args = ap.parse_args()
    gs = args.group_size

    data = open(args.inp, "rb").read()
    hdr = bytearray(data[:192])
    dtype = struct.unpack_from("<I", hdr, 12)[0]
    assert dtype == 4, f"输入须为 VQ2 文件（dtype=4），实际 {dtype}"
    tc, tto, doff, total = struct.unpack_from("<4Q", hdr, 64)

    # 解析张量表
    tensors, order = {}, []
    off = tto
    for _ in range(tc):
        e = data[off:off + 120]; off += 120
        name = e[:64].split(b"\x00")[0].decode()
        dt, nd = struct.unpack_from("<II", e, 64)
        shape = list(struct.unpack_from("<4Q", e, 72))[:nd]
        toff, nb = struct.unpack_from("<QQ", e, 104)
        tensors[name] = {"dtype": dt, "shape": shape, "bytes": data[toff:toff + nb]}
        order.append(name)

    # 量化 embed
    ename = "model.embed_tokens.weight"
    assert ename in tensors, "缺 embed_tokens"
    t = tensors[ename]
    assert t["dtype"] in (0, 1), f"embed 当前 dtype={t['dtype']}，需 f32(0)/f16(1)"
    vocab, dim = t["shape"]
    raw = t["bytes"]
    W = (np.frombuffer(raw, np.float32) if t["dtype"] == 0
         else np.frombuffer(raw, np.float16).astype(np.float32)).reshape(vocab, dim)
    assert dim % gs == 0 or True  # 允许尾部不足一组
    i4_bytes = quantize_embed_i4(W, gs)
    groups = (dim + gs - 1) // gs
    assert len(i4_bytes) == vocab * groups * (4 + gs // 2)
    tensors[ename] = {"dtype": DTYPE_I4, "shape": [vocab, dim], "bytes": i4_bytes}

    # header：写 quant_group_size
    struct.pack_into("<I", hdr, QUANT_GS_OFFSET, gs)

    # 重写布局
    table_end = 192 + len(order) * 120
    cur = align_up(table_end)
    entries = []
    for name in order:
        nb = len(tensors[name]["bytes"])
        entries.append((name, tensors[name]["dtype"], tensors[name]["shape"], cur, nb))
        cur = align_up(cur + nb)
    total_new = cur
    struct.pack_into("<4Q", hdr, 64, len(order), 192, align_up(table_end), total_new)

    with open(args.out, "wb") as f:
        f.write(bytes(hdr))
        for name, dt, shape, offc, nb in entries:
            sh = list(shape) + [0] * (4 - len(shape))
            f.write(struct.pack(ENTRY_FMT, name.encode(), dt, len(shape), *sh, offc, nb))
        f.write(b"\x00" * (align_up(table_end) - f.tell()))
        for name, dt, shape, offc, nb in entries:
            assert f.tell() == offc
            f.write(tensors[name]["bytes"])
            f.write(b"\x00" * (align_up(offc + nb) - (offc + nb)))
        assert f.tell() == total_new
    old_mb = len(data) / 1048576
    new_mb = total_new / 1048576
    print(f"OK embed INT4(gs={gs}): {args.out}  {old_mb:.2f}MB -> {new_mb:.2f}MB "
          f"(embed {vocab}x{dim}: f16/f32 -> i4)")


if __name__ == "__main__":
    main()
