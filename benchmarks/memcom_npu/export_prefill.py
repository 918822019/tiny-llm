"""Export chunked parallel prefill ONNX for memcom-0.3b."""
import sys, torch
from memcom_model_prefill import MemComModel, load_weights, CACHE_LEN, CHUNK_SIZE

m = MemComModel()
load_weights(m, "/work/memcom-step22000.pt")
m.eval()
m.half()

C = CHUNK_SIZE
print(f"Exporting prefill ONNX with chunk_size={C}")

class PrefillWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, *args):
        return self.model.forward_prefill(*args)

wrapper = PrefillWrapper(m)

ids = torch.zeros(C, dtype=torch.int32)
cos = (torch.randn(C, 64) * 0.1).half()
sin = (torch.randn(C, 64) * 0.1).half()
S = [(torch.randn(9, 64, 64) * 0.05).half() for _ in range(16)]
kc = [(torch.randn(2, CACHE_LEN, 64) * 0.1).half() for _ in range(4)]
vc = [(torch.randn(2, CACHE_LEN, 64) * 0.1).half() for _ in range(4)]
inputs = (ids, cos, sin, *S, *kc, *vc)

names_in = ["ids", "cos", "sin"] + [f"s{i}" for i in range(16)] + \
           [f"kc{i}" for i in range(4)] + [f"vc{i}" for i in range(4)]
names_out = ["logits"] + [f"s_out{i}" for i in range(16)] + \
            [f"kc_out{i}" for i in range(4)] + [f"vc_out{i}" for i in range(4)]

with torch.no_grad():
    out = wrapper(*inputs)
    print(f"sanity forward_prefill OK, logits: {tuple(out[0].shape)}")
    for i, o in enumerate(out):
        print(f"  output[{i}] ({names_out[i] if i < len(names_out) else '?'}): {tuple(o.shape)} {o.dtype}")

    torch.onnx.export(
        wrapper, inputs, "/work/memcom/prefill.onnx",
        input_names=names_in, output_names=names_out,
        opset_version=17, do_constant_folding=True, dynamo=False,
    )
print("prefill ONNX export done")
