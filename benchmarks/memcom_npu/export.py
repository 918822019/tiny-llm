import sys, torch
from memcom_model import MemComModel, load_weights, CACHE_LEN

m = MemComModel()
load_weights(m, "/work/memcom-step22000.pt")
m.eval()
m.half()  # fp16: halves RAM + proto; HTP-native dtype anyway

ids = torch.zeros(1, dtype=torch.int32)
cos = (torch.randn(64) * 0.1).half()
sin = (torch.randn(64) * 0.1).half()
S = [(torch.randn(9, 64, 64) * 0.05).half() for _ in range(16)]
kc = [(torch.randn(2, CACHE_LEN, 64) * 0.1).half() for _ in range(4)]
vc = [(torch.randn(2, CACHE_LEN, 64) * 0.1).half() for _ in range(4)]
inputs = (ids, cos, sin, *S, *kc, *vc)

names_in = ["ids", "cos", "sin"] + [f"s{i}" for i in range(16)] + \
           [f"kc{i}" for i in range(4)] + [f"vc{i}" for i in range(4)]
names_out = ["logits"] + [f"s_out{i}" for i in range(16)] + \
            [f"kc_out{i}" for i in range(4)] + [f"vc_out{i}" for i in range(4)]

with torch.no_grad():
    out = m(*inputs)
    print("sanity forward OK, logits:", tuple(out[0].shape))

    torch.onnx.export(
        m, inputs, "/work/memcom/decode.onnx",
        input_names=names_in, output_names=names_out,
        opset_version=17, do_constant_folding=True, dynamo=False,
    )
print("export done")
