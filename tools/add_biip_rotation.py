#!/usr/bin/env python3
"""对一个 .tqwen 模型的线性子层权重施加 BiIP 旋转，并写入旋转参数张量。

用途：
  1. 合成自抵消测试——对未量化模型施加旋转，验证 tiny-llm 运行时旋转后输出不变。
  2. 桥接导出器的旋转部分原型（真实配方里旋转在 kronq 量化前完成，这里等效复现）。

旋转约定（与 kronq/rotation_manager.py 一致）：
  权重正变换：W_rot = blockHadamard( (W * scale) ⊙ sign )   沿输入维(列)
  运行时逆变换：x_rot = blockHadamard( (x / scale) ⊙ sign )
  两者配对自抵消：x_rot @ W_rot^T == x @ W^T。

用法:
    python tools/add_biip_rotation.py --in in.tqwen --out out.tqwen [--seed 42] [--no-scale]
"""
import argparse, struct, math
import numpy as np

ALIGN = 64
HEADER_FMT = "<8s12Iff4Q96s"
ENTRY_FMT = "<64sII4QQQ"
SUBLAYERS = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
             "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]


def align_up(x, a=ALIGN):
    return (x + a - 1) // a * a


def find_block(dim):
    for bs in [256, 128, 64, 32, 16, 8, 4, 2]:
        if dim % bs == 0:
            return bs
    return 1


def hadamard(n):
    H = np.array([[1.0]])
    while H.shape[0] < n:
        H = np.block([[H, H], [H, -H]])
    return H / math.sqrt(n)


def block_hadamard(W, block_size, sign):
    """W: [out, in]；沿列(输入维)：W*sign 后分块 @Hᵀ。"""
    out_f, in_f = W.shape
    nb = in_f // block_size
    W = W * sign[None, :]
    W = W.reshape(out_f, nb, block_size)
    H = hadamard(block_size)
    W = W @ H.T
    return W.reshape(out_f, in_f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-scale", action="store_true", help="不加 scaleH（测纯 Hadamard）")
    args = ap.parse_args()

    data = open(args.inp, "rb").read()
    hdr = data[:192]
    magic = hdr[:8]
    assert magic == b"TINYQWEN"
    version, dtype = struct.unpack_from("<II", hdr, 8)
    cfg_fields = struct.unpack_from("<10I", hdr, 16)
    rms, theta = struct.unpack_from("<ff", hdr, 56)
    tc, tto, doff, total = struct.unpack_from("<4Q", hdr, 64)
    reserved = hdr[96:192]

    # 解析张量表
    tensors = {}   # name -> dict(dtype, shape, bytes)
    order = []
    off = tto
    for _ in range(tc):
        e = data[off:off + 120]; off += 120
        name = e[:64].split(b"\x00")[0].decode()
        dt, ndim = struct.unpack_from("<II", e, 64)
        shape = list(struct.unpack_from("<4Q", e, 72))[:ndim]
        toff, nb = struct.unpack_from("<QQ", e, 104)
        tensors[name] = {"dtype": dt, "shape": shape, "bytes": data[toff:toff + nb]}
        order.append(name)

    rng = np.random.default_rng(args.seed)
    n_rot = 0
    for i in range(cfg_fields[0]):  # n_layers
        p = f"model.layers.{i}."
        for sl in SUBLAYERS:
            wname = p + sl + ".weight"
            if wname not in tensors:
                continue
            t = tensors[wname]
            assert t["dtype"] == 0, "仅支持 f32 模型的旋转注入（测试用）"
            W = np.frombuffer(t["bytes"], dtype=np.float32).reshape(t["shape"]).copy()
            out_f, in_f = W.shape
            bs = find_block(in_f)
            # 生成 sign(±1) 与可选 scale
            sign = (rng.integers(0, 2, size=in_f).astype(np.float32) * 2 - 1)
            prefix = p + sl
            if args.no_scale:
                W_rot = block_hadamard(W, bs, sign)
                tensors[prefix + ".rot_sign"] = {"dtype": 0, "shape": [in_f],
                                                 "bytes": sign.astype(np.float32).tobytes()}
            else:
                scale = rng.uniform(0.5, 2.0, size=in_f).astype(np.float32)
                W_rot = block_hadamard(W * scale[None, :], bs, sign)
                tensors[prefix + ".rot_sign"] = {"dtype": 0, "shape": [in_f],
                                                 "bytes": sign.astype(np.float32).tobytes()}
                tensors[prefix + ".rot_scale"] = {"dtype": 0, "shape": [in_f],
                                                  "bytes": scale.astype(np.float32).tobytes()}
            tensors[wname]["bytes"] = W_rot.astype(np.float32).tobytes()
            order += [n for n in (prefix + ".rot_sign", prefix + ".rot_scale")
                      if n in tensors and n not in order]
            n_rot += 1

    # 重写 .tqwen
    table_end = 192 + len(order) * 120
    cur = align_up(table_end)
    entries = []
    for name in order:
        t = tensors[name]
        nb = len(t["bytes"])
        entries.append((name, t["dtype"], t["shape"], cur, nb))
        cur = align_up(cur + nb)
    total_new = cur

    new_hdr = bytearray(struct.pack(
        HEADER_FMT, b"TINYQWEN", version, dtype, *cfg_fields, rms, theta,
        len(order), 192, align_up(table_end), total_new, reserved))
    with open(args.out, "wb") as f:
        f.write(bytes(new_hdr))
        for name, dt, shape, offc, nb in entries:
            sh = list(shape) + [0] * (4 - len(shape))
            f.write(struct.pack(ENTRY_FMT, name.encode("ascii"), dt, len(shape), *sh, offc, nb))
        f.write(b"\x00" * (align_up(table_end) - f.tell()))
        for name, dt, shape, offc, nb in entries:
            assert f.tell() == offc
            f.write(tensors[name]["bytes"])
            f.write(b"\x00" * (align_up(offc + nb) - (offc + nb)))
        assert f.tell() == total_new
    print(f"OK 旋转 {n_rot} 个子层 -> {args.out} ({total_new} bytes)")


if __name__ == "__main__":
    main()
