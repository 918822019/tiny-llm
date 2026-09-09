import numpy as np, torch
from memcom_model import MemComModel, load_weights

m = MemComModel()
load_weights(m, "/work/memcom-step22000.pt")
m.eval()

D = "/work/memcom/inputs/"
def rd(name, shape, dt=np.float32):
    return np.fromfile(D + name + ".raw", dtype=dt).reshape(shape)

ids = torch.from_numpy(rd("ids", (1,), np.int32))
cos = torch.from_numpy(rd("cos", (64,)))
sin = torch.from_numpy(rd("sin", (64,)))
S = [torch.from_numpy(rd(f"s{i}", (9, 64, 64))) for i in range(16)]
# graph/pytorch layout: raw is (2,64,256); pytorch forward expects [2,256,64]
kc = [torch.from_numpy(rd(f"kc{i}", (2, 64, 256))).transpose(1, 2).contiguous() for i in range(4)]
vc = [torch.from_numpy(rd(f"vc{i}", (2, 64, 256))).transpose(1, 2).contiguous() for i in range(4)]

with torch.no_grad():
    out = m(ids, cos, sin, *S, *kc, *vc)
logits = out[0].numpy()          # [1,50000]
s0 = out[1].numpy()              # s_out0 [9,64,64]

p_l = np.fromfile("/work/memcom/phone_out/logits.raw", dtype=np.float32)
x_l = np.fromfile("/work/memcom/output_x86/Result_0/logits.raw", dtype=np.float32)
p_s = np.fromfile("/work/memcom/phone_out/s_out0.raw", dtype=np.float32)
x_s = np.fromfile("/work/memcom/output_x86/Result_0/s_out0.raw", dtype=np.float32)

ref_l = logits.reshape(-1); ref_s = s0.reshape(-1)
print(f"REF   logits: absmax={np.abs(ref_l).max():.2f} top1={ref_l.argmax()}")
print(f"PHONE logits: absmax={np.abs(p_l).max():.2f} top1={p_l.argmax()}")
print(f"X86   logits: absmax={np.abs(x_l).max():.2f} top1={x_l.argmax()}")
print(f"REF   s_out0: absmax={np.abs(ref_s).max():.2f}")
print(f"PHONE s_out0: absmax={np.abs(p_s).max():.2f}")
print(f"X86   s_out0: absmax={np.abs(x_s).max():.2f}")
print(f"phone-vs-ref logits meanabs={np.abs(p_l-ref_l).mean():.3f} | x86-vs-ref logits meanabs={np.abs(x_l-ref_l).mean():.3f}")
print(f"phone-vs-ref s_out meanabs={np.abs(p_s-ref_s).mean():.3f} | x86-vs-ref s_out meanabs={np.abs(x_s-ref_s).mean():.3f}")
