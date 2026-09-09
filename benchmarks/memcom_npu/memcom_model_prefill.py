"""MemCom-0.3B prefill model — chunked parallel processing for NPU.

Same architecture as memcom_model.py but with forward_prefill() that processes
C tokens in one pass. BMC layers use batched matmuls + unrolled state scan
(ONNX unrolls the loop, but the matmuls dominate and are batched → HMX utilization).
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
CHUNK_SIZE = 8


def rmsnorm(x, w):
    ms = (x * x).mean(-1, keepdim=True)
    return x / torch.sqrt(ms + EPS) * w


def rotate_half(t):
    t1, t2 = t.chunk(2, dim=-1)
    return torch.cat([-t2, t1], dim=-1)


def l2_normalize(x):
    ms = (x * x).mean(-1, keepdim=True)
    return x / torch.sqrt(ms * DK + 1e-6)


def apply_rope(t, cos, sin):
    if t.dim() == 3 and cos.dim() == 2:
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
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

    def forward_prefill(self, x):
        return self.forward(x)


class BilinearMemoryCell(nn.Module):
    def __init__(self):
        super().__init__()
        self.W_q = nn.Linear(D, D, bias=False)
        self.W_kv = nn.Linear(D, 2 * D, bias=False)
        self.W_g = nn.Linear(D, BMC_HEADS * DK, bias=True)
        self.W_i = nn.Linear(D, BMC_HEADS * DK, bias=True)
        self.W_o = nn.Linear(BMC_HEADS * DK, D, bias=False)

    def forward(self, x, S):
        q = self.W_q(x).view(BMC_HEADS, 2, DK)
        kv = self.W_kv(x)
        k = l2_normalize(kv[:, :D].view(BMC_HEADS, 2, DK))
        v = l2_normalize(kv[:, D:].view(BMC_HEADS, 2, DK))
        g = (0.99 * torch.sigmoid(self.W_g(x))).view(BMC_HEADS, DK)
        i = torch.sigmoid(self.W_i(x)).view(BMC_HEADS, DK)
        outer = k.unsqueeze(-1) * v.unsqueeze(-2)
        outer = outer.sum(dim=1)
        S_new = (g.unsqueeze(-1) * S + i.unsqueeze(-1) * outer).clamp(-2000.0, 2000.0)
        o = torch.matmul(q, S_new).sum(dim=1)
        return self.W_o(o.reshape(1, BMC_HEADS * DK)), S_new

    def forward_prefill(self, x, S_prev):
        C = x.shape[0]
        q = self.W_q(x).view(C, BMC_HEADS, 2, DK)
        kv = self.W_kv(x)
        k = l2_normalize(kv[:, :D].view(C, BMC_HEADS, 2, DK))
        v = l2_normalize(kv[:, D:].view(C, BMC_HEADS, 2, DK))
        g = (0.99 * torch.sigmoid(self.W_g(x))).view(C, BMC_HEADS, DK)
        i_g = torch.sigmoid(self.W_i(x)).view(C, BMC_HEADS, DK)
        outer = (k.unsqueeze(-1) * v.unsqueeze(-2)).sum(dim=2)
        S = S_prev
        S_all = []
        for t in range(CHUNK_SIZE):
            S = (g[t].unsqueeze(-1) * S + i_g[t].unsqueeze(-1) * outer[t]).clamp(-2000.0, 2000.0)
            S_all.append(S)
        S_stack = torch.stack(S_all)
        o = torch.matmul(q, S_stack)
        o = o.sum(dim=2)
        return self.W_o(o.reshape(C, BMC_HEADS * DK)), S


class GQAAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.W_q = nn.Linear(D, D, bias=False)
        self.W_k = nn.Linear(D, KV_HEADS * DK, bias=False)
        self.W_v = nn.Linear(D, KV_HEADS * DK, bias=False)
        self.W_o = nn.Linear(D, D, bias=False)

    def forward(self, x, cos, sin, kc, vc):
        q = self.W_q(x).view(Q_HEADS, DK)
        k_new = self.W_k(x).view(KV_HEADS, DK)
        v_new = self.W_v(x).view(KV_HEADS, DK)
        q = apply_rope(q, cos, sin).view(KV_HEADS, Q_HEADS // KV_HEADS, DK)
        k_new = apply_rope(k_new, cos, sin).unsqueeze(1)
        K = torch.cat([kc[:, 1:], k_new], dim=1)
        V = torch.cat([vc[:, 1:], v_new.unsqueeze(1)], dim=1)
        scores = torch.matmul(q, K.transpose(-2, -1)) / (DK ** 0.5)
        attn = torch.softmax(scores, dim=-1)
        o = torch.matmul(attn, V)
        return self.W_o(o.reshape(1, D)), K, V

    def forward_prefill(self, x, cos, sin, kc, vc):
        C = x.shape[0]
        q = self.W_q(x).view(C, Q_HEADS, DK)
        k_new = self.W_k(x).view(C, KV_HEADS, DK)
        v_new = self.W_v(x).view(C, KV_HEADS, DK)
        q = apply_rope(q, cos, sin)
        k_new = apply_rope(k_new, cos, sin)
        K = torch.cat([kc[:, C:], k_new.transpose(0, 1)], dim=1)
        V = torch.cat([vc[:, C:], v_new.transpose(0, 1)], dim=1)
        q = q.view(C, KV_HEADS, Q_HEADS // KV_HEADS, DK).permute(1, 2, 0, 3)
        scores = torch.matmul(q, K.unsqueeze(1).transpose(-2, -1)) / (DK ** 0.5)
        rows = torch.arange(C).unsqueeze(1)
        cols = torch.arange(CACHE_LEN).unsqueeze(0)
        mask = cols < (CACHE_LEN - C + rows + 1)
        scores = scores.masked_fill(~mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        attn = torch.softmax(scores, dim=-1)
        o = torch.matmul(attn, V.unsqueeze(1))
        o = o.permute(2, 0, 1, 3).reshape(C, D)
        return self.W_o(o), K, V


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
        S_in = states_and_caches[:16]
        kc_in = states_and_caches[16:20]
        vc_in = states_and_caches[20:24]
        x = self.embed(ids.long())
        S_out, kc_out, vc_out = [], [], []
        bi = ci = 0
        for idx, blk in enumerate(self.blocks):
            h = blk.norm1(x)
            if idx in ATTN_LAYERS:
                sub, K, V = blk.sublayer(h, cos, sin, kc_in[ci], vc_in[ci])
                kc_out.append(K)
                vc_out.append(V)
                ci += 1
            else:
                sub, S_new = blk.sublayer(h, S_in[bi])
                S_out.append(S_new)
                bi += 1
            x = x + sub
            x = x + blk.ffn(blk.norm2(x))
        logits = self.lm_head(self.final_norm(x))
        return logits, *S_out, *kc_out, *vc_out

    def forward_prefill(self, ids, cos, sin, *states_and_caches):
        C = ids.shape[0]
        S_in = list(states_and_caches[:16])
        kc_in = list(states_and_caches[16:20])
        vc_in = list(states_and_caches[20:24])
        x = self.embed(ids.long())
        S_out, kc_out, vc_out = [], [], []
        bi = ci = 0
        for idx, blk in enumerate(self.blocks):
            h = blk.norm1(x)
            if idx in ATTN_LAYERS:
                sub, K, V = blk.sublayer.forward_prefill(h, cos, sin, kc_in[ci], vc_in[ci])
                kc_out.append(K)
                vc_out.append(V)
                ci += 1
            else:
                sub, S_new = blk.sublayer.forward_prefill(h, S_in[bi])
                S_out.append(S_new)
                bi += 1
            x = x + sub
            x = x + blk.ffn.forward_prefill(blk.norm2(x))
        logits = self.lm_head(self.final_norm(x))
        return logits[-1:], *S_out, *kc_out, *vc_out


def load_weights(model, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu", mmap=True, weights_only=False)
    sd = ckpt["model"]
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"loaded: {len(sd)} tensors; missing={len(missing)} unexpected={len(unexpected)}")
    assert not missing and not unexpected, "key mismatch!"
    return model
