"""MemComModel reconstruction (shape-faithful) for QNN/NPU compatibility + perf test.

Rebuilt from: checkpoint state_dict shapes + mcfg + user-provided architecture diagram.
ASSUMPTIONS (documented, may differ from training code's exact semantics but ops/shapes/FLOPs match):
  1. BMC: W_kv(2304) splits into k(1152)/v(1152), each viewed as [9,2,64];
     per BMC head h: S_h = g_h*S_h + i_h*( k[h,0](x)v[h,0] + k[h,1](x)v[h,1] )
     gates g,i ([9,64]) broadcast over the 2nd state dim; S clamped to [-2000,2000].
     readout: o_h = sum_j q[h,j] @ S_h ; o [9,64] -> flatten 576 -> W_o.
     g = 0.99*sigmoid(W_g x + b) (forget_gate_max), i = sigmoid(W_i x + b);
     bilinear_norm=True -> L2-normalize k,v before outer product.
  2. ATTN layers [4,9,14,19]: GQA 18Q/2KV, standard RoPE (theta 1e4, d=64),
     cos/sin precomputed on CPU and passed as graph inputs; KV cache [2,256,64]
     rolling (concat+slice) - decode perf test, not exact generation semantics.
  3. RMSNorm eps 1e-6; SwiGLU FFN (gate_up 6220 = 2x3110); tie_embeddings False.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

D = 1152
N_LAYERS = 20
ATTN_LAYERS = {4, 9, 14, 19}
CACHE_LEN = 256
KV_HEADS = 2
Q_HEADS = 18
BMC_HEADS = 9
DK = 64
VOCAB = 50000
FFN = 3110
EPS = 1e-6


def rmsnorm(x, w):
    ms = (x * x).mean(-1, keepdim=True)
    return x / torch.sqrt(ms + EPS) * w


def rotate_half(t):
    t1, t2 = t.chunk(2, dim=-1)
    return torch.cat([-t2, t1], dim=-1)


def l2_normalize(x):
    # F.normalize equivalent via HTP-fp16-native ops (ReduceMean/Sqrt/Div).
    # F.normalize exports ReduceL2+Clip+Expand which the converter keeps FP32
    # and which misbehave on real HTP (norm computed wrong -> state explosion).
    ms = (x * x).mean(-1, keepdim=True)
    return x / torch.sqrt(ms * DK + 1e-6)


def apply_rope(t, cos, sin):
    # t: [..., 64], cos/sin: [64]
    return t * cos + rotate_half(t) * sin


class RMSNorm(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x):
        return rmsnorm(x, self.weight)


class SwiGLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_up = nn.Linear(D, 2 * FFN, bias=False)
        self.down = nn.Linear(FFN, D, bias=False)

    def forward(self, x):
        g, u = self.gate_up(x).chunk(2, dim=-1)
        return self.down(F.silu(g) * u)


class BilinearMemoryCell(nn.Module):
    """Decode-step recurrence: S_new = g*S + i*(k(x)v), no chunked form needed."""

    def __init__(self):
        super().__init__()
        self.W_q = nn.Linear(D, D, bias=False)
        self.W_kv = nn.Linear(D, 2 * D, bias=False)
        self.W_g = nn.Linear(D, BMC_HEADS * DK, bias=True)
        self.W_i = nn.Linear(D, BMC_HEADS * DK, bias=True)
        self.W_o = nn.Linear(BMC_HEADS * DK, D, bias=False)

    def forward(self, x, S):
        # x: [1,1152]; S: [9,64,64]
        q = self.W_q(x).view(BMC_HEADS, 2, DK)
        kv = self.W_kv(x)
        k = l2_normalize(kv[:, :D].view(BMC_HEADS, 2, DK))
        v = l2_normalize(kv[:, D:].view(BMC_HEADS, 2, DK))
        g = (0.99 * torch.sigmoid(self.W_g(x))).view(BMC_HEADS, DK)
        i = torch.sigmoid(self.W_i(x)).view(BMC_HEADS, DK)
        outer = k.unsqueeze(-1) * v.unsqueeze(-2)        # [9,2,64,64]
        outer = outer.sum(dim=1)                          # [9,64,64]
        S_new = (g.unsqueeze(-1) * S + i.unsqueeze(-1) * outer).clamp(-2000.0, 2000.0)
        o = torch.matmul(q, S_new).sum(dim=1)             # [9,2,64]->[9,64]
        return self.W_o(o.reshape(1, BMC_HEADS * DK)), S_new


class GQAAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.W_q = nn.Linear(D, D, bias=False)
        self.W_k = nn.Linear(D, KV_HEADS * DK, bias=False)
        self.W_v = nn.Linear(D, KV_HEADS * DK, bias=False)
        self.W_o = nn.Linear(D, D, bias=False)

    def forward(self, x, cos, sin, kc, vc):
        # x: [1,1152]; kc/vc: [2,256,64]
        q = self.W_q(x).view(Q_HEADS, DK)
        k_new = self.W_k(x).view(KV_HEADS, DK)
        v_new = self.W_v(x).view(KV_HEADS, DK)
        q = apply_rope(q, cos, sin).view(KV_HEADS, Q_HEADS // KV_HEADS, DK)
        k_new = apply_rope(k_new, cos, sin).unsqueeze(1)  # [2,1,64]
        K = torch.cat([kc[:, 1:], k_new], dim=1)          # [2,256,64] rolling
        V = torch.cat([vc[:, 1:], v_new.unsqueeze(1)], dim=1)
        scores = torch.matmul(q, K.transpose(-2, -1)) / (DK ** 0.5)  # [2,9,256]
        attn = torch.softmax(scores, dim=-1)
        o = torch.matmul(attn, V)                         # [2,9,64]
        return self.W_o(o.reshape(1, D)), K, V


class MemComBlock(nn.Module):
    def __init__(self, idx):
        super().__init__()
        self.norm1 = RMSNorm(D)
        self.norm2 = RMSNorm(D)
        self.sublayer = GQAAttention() if idx in ATTN_LAYERS else BilinearMemoryCell()
        self.ffn = SwiGLU()


class MemComModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB, D)
        self.blocks = nn.ModuleList([MemComBlock(i) for i in range(N_LAYERS)])
        self.final_norm = RMSNorm(D)
        self.lm_head = nn.Linear(D, VOCAB, bias=False)

    def forward(self, ids, cos, sin, *states_and_caches):
        # ids:[1] int32; cos/sin:[64]; then 16x S[9,64,64], 4x kc[2,256,64], 4x vc[2,256,64]
        S_in = states_and_caches[:16]
        kc_in = states_and_caches[16:20]
        vc_in = states_and_caches[20:24]
        x = self.embed(ids.long())                        # [1,1152], follows weight dtype
        S_out, kc_out, vc_out = [], [], []
        bi = ci = 0
        for idx, blk in enumerate(self.blocks):
            h = blk.norm1(x)
            if idx in ATTN_LAYERS:
                sub, K, V = blk.sublayer(h, cos, sin, kc_in[ci], vc_in[ci])
                kc_out.append(K); vc_out.append(V); ci += 1
            else:
                sub, S_new = blk.sublayer(h, S_in[bi])
                S_out.append(S_new); bi += 1
            x = x + sub
            x = x + blk.ffn(blk.norm2(x))
        logits = self.lm_head(self.final_norm(x))         # [1,50000]
        return logits, *S_out, *kc_out, *vc_out


def load_weights(model, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu", mmap=True, weights_only=False)
    sd = ckpt["model"]
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"loaded: {len(sd)} tensors; missing={len(missing)} unexpected={len(unexpected)}")
    if missing:
        print("MISSING:", missing[:10])
    if unexpected:
        print("UNEXPECTED:", unexpected[:10])
    assert not missing and not unexpected, "key mismatch!"
    return model


if __name__ == "__main__":
    import sys
    m = MemComModel()
    n = sum(p.numel() for p in m.parameters())
    print(f"params: {n/1e6:.1f}M")
    load_weights(m, sys.argv[1])
    print("OK: strict-equivalent load passed - shape assumptions verified")
