#!/usr/bin/env python3
"""合成"旋转 + VQ2"假模型（迷你版桥接），端到端验证 tiny-llm 能加载并运行
桥接导出器将产出的确切格式：VQ2 线性层(码本+索引) + 旋转参数 + f32 embed/norm。

做法：随机小 Qwen2 权重 → 对 7 类子层施加 BiIP 旋转 → 在旋转空间做块 k-means
(K=256, d=4) → 写 .tqwen。tiny-llm 加载后运行时旋转应让输出≈未旋转参考。
"""
import struct, math
import numpy as np

ALIGN = 64
HEADER_FMT = "<8s12Iff4Q96s"
ENTRY_FMT = "<64sII4QQQ"
DTYPE_F32, DTYPE_VQ2 = 0, 4
K, D = 256, 4
CB_BYTES = K * D * 2
SUBS = [("self_attn.q_proj", "q"), ("self_attn.k_proj", "k"), ("self_attn.v_proj", "v"),
        ("self_attn.o_proj", "o"), ("mlp.gate_proj", "gate"), ("mlp.up_proj", "up"),
        ("mlp.down_proj", "down")]


def align_up(x, a=ALIGN): return (x + a - 1) // a * a
def find_block(dim):
    for bs in [256, 128, 64, 32, 16, 8, 4, 2]:
        if dim % bs == 0: return bs
    return 1
def hadamard(n):
    H = np.array([[1.0]])
    while H.shape[0] < n: H = np.block([[H, H], [H, -H]])
    return H / math.sqrt(n)
def block_hadamard(W, bs, sign):
    out_f, in_f = W.shape; nb = in_f // bs
    W = (W * sign[None, :]).reshape(out_f, nb, bs) @ hadamard(bs).T
    return W.reshape(out_f, in_f)
def kmeans_blocks(W, k=K, d=D, iters=8, seed=0):
    out_f, in_f = W.shape
    blocks = W.reshape(-1, d).astype(np.float32)
    rng = np.random.default_rng(seed)
    cent = blocks[rng.choice(blocks.shape[0], size=k,
                             replace=(blocks.shape[0] < k))].copy()
    for _ in range(iters):
        # 子采样加速
        n = blocks.shape[0]
        sub = blocks[rng.choice(n, size=min(n, 50000), replace=False)] if n > 50000 else blocks
        d2 = ((sub[:, None, :] - cent[None, :, :]) ** 2).sum(-1)
        a = d2.argmin(-1)
        sums = np.zeros((k, d), np.float64); cnt = np.zeros(k, np.int64)
        np.add.at(sums, a, sub); np.add.at(cnt, a, 1)
        ne = cnt > 0; nc = cent.copy()
        nc[ne] = (sums[ne] / cnt[ne, None]).astype(np.float32)
        if not ne.all(): nc[~ne] = blocks[rng.choice(n, size=(~ne).sum(), replace=True)]
        cent = nc
    d2 = ((blocks[:, None, :] - cent[None, :, :]) ** 2).sum(-1)
    idx = d2.argmin(-1).astype(np.uint8).reshape(out_f, in_f // d)
    return cent.astype(np.float16), idx


def main(out_path, seed=0):
    rng = np.random.default_rng(seed)
    n_layers, H, I, V = 2, 16, 32, 64
    n_heads, n_kv, hd = 4, 2, 4
    qd, kvd = n_heads * hd, n_kv * hd

    tensors, order = {}, []
    def put(name, dt, shape, blob):
        tensors[name] = {"dtype": dt, "shape": list(shape), "bytes": blob}
        order.append(name)
    def w(rows, cols): return rng.normal(0, 0.1, (rows, cols)).astype(np.float32)
    def f32(a): return np.ascontiguousarray(a.astype(np.float32)).tobytes()

    put("model.embed_tokens.weight", DTYPE_F32, [V, H], f32(w(V, H)))
    for li in range(n_layers):
        p = f"model.layers.{li}."
        put(p + "input_layernorm.weight", DTYPE_F32, [H], f32(np.ones(H, np.float32)))
        put(p + "post_attention_layernorm.weight", DTYPE_F32, [H], f32(np.ones(H, np.float32)))
        shapes = {"q": (qd, H), "k": (kvd, H), "v": (kvd, H), "o": (H, qd),
                  "gate": (I, H), "up": (I, H), "down": (H, I)}
        for frag, key in SUBS:
            rows, cols = shapes[key]
            W = w(rows, cols)
            bs = find_block(cols)
            sign = (rng.integers(0, 2, size=cols).astype(np.float32) * 2 - 1)
            scale = rng.uniform(0.5, 2.0, size=cols).astype(np.float32)
            W_rot = block_hadamard(W * scale[None, :], bs, sign)   # 旋转空间
            cent, idx = kmeans_blocks(W_rot, seed=seed + li)        # 旋转空间 VQ2
            blob = np.ascontiguousarray(cent).tobytes() + np.ascontiguousarray(idx).tobytes()
            put(p + frag + ".weight", DTYPE_VQ2, [rows, cols], blob)
            if key in ("q", "k", "v"):  # bias
                put(p + frag + ".bias", DTYPE_F32, [rows], f32(rng.normal(0, 0.01, rows).astype(np.float32)))
            put(p + frag + ".rot_sign", DTYPE_F32, [cols], f32(sign))
            put(p + frag + ".rot_scale", DTYPE_F32, [cols], f32(scale))
    put("model.norm.weight", DTYPE_F32, [H], f32(np.ones(H, np.float32)))

    cfg = dict(n_layers=n_layers, hidden_size=H, intermediate_size=I, n_heads=n_heads,
               n_kv_heads=n_kv, head_dim=hd, vocab_size=V, max_seq_len=32, tied=1,
               rms_norm_eps=1e-6, rope_theta=10000.0)
    table_end = 192 + len(order) * 120
    cur = align_up(table_end); entries = []
    for name in order:
        nb = len(tensors[name]["bytes"]); entries.append((name, tensors[name]["dtype"],
                                                          tensors[name]["shape"], cur, nb))
        cur = align_up(cur + nb)
    total = cur
    hdr = struct.pack(HEADER_FMT, b"TINYQWEN", 2, DTYPE_VQ2, cfg["n_layers"], cfg["hidden_size"],
                      cfg["intermediate_size"], cfg["n_heads"], cfg["n_kv_heads"], cfg["head_dim"],
                      cfg["vocab_size"], cfg["max_seq_len"], cfg["tied"], 0, cfg["rms_norm_eps"],
                      cfg["rope_theta"], len(order), 192, align_up(table_end), total, b"\x00" * 96)
    with open(out_path, "wb") as f:
        f.write(hdr)
        for name, dt, shape, off, nb in entries:
            sh = list(shape) + [0] * (4 - len(shape))
            f.write(struct.pack(ENTRY_FMT, name.encode(), dt, len(shape), *sh, off, nb))
        f.write(b"\x00" * (align_up(table_end) - f.tell()))
        for name, dt, shape, off, nb in entries:
            assert f.tell() == off
            f.write(tensors[name]["bytes"]); f.write(b"\x00" * (align_up(off + nb) - (off + nb)))
        assert f.tell() == total
    print(f"OK 旋转VQ2 假模型 -> {out_path} ({total} B, {len(order)} tensors)")


if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else "/tmp/fake_rotvq2.tqwen")
