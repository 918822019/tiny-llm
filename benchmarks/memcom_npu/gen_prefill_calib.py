"""Generate calibration inputs for prefill (chunk_size=8) at correct shapes."""
import numpy as np, os
os.makedirs("/work/memcom/prefill_inputs", exist_ok=True)
C = 8

# ids[C] int32
np.random.randint(0, 50000, C, dtype=np.int32).tofile("/work/memcom/prefill_inputs/ids.raw")
# cos[C, 64] float32
(np.random.randn(C, 64).astype(np.float32) * 0.1).tofile("/work/memcom/prefill_inputs/cos.raw")
# sin[C, 64] float32
(np.random.randn(C, 64).astype(np.float32) * 0.1).tofile("/work/memcom/prefill_inputs/sin.raw")
# s0..s15 [9, 64, 64] float32
for i in range(16):
    (np.random.randn(9, 64, 64).astype(np.float32) * 0.05).tofile(f"/work/memcom/prefill_inputs/s{i}.raw")
# kc0..kc3 [2, 256, 64] float32
for i in range(4):
    (np.random.randn(2, 256, 64).astype(np.float32) * 0.1).tofile(f"/work/memcom/prefill_inputs/kc{i}.raw")
# vc0..vc3 [2, 256, 64] float32
for i in range(4):
    (np.random.randn(2, 256, 64).astype(np.float32) * 0.1).tofile(f"/work/memcom/prefill_inputs/vc{i}.raw")

parts = ["ids:=prefill_inputs/ids.raw", "cos:=prefill_inputs/cos.raw", "sin:=prefill_inputs/sin.raw"]
parts += [f"s{i}:=prefill_inputs/s{i}.raw" for i in range(16)]
parts += [f"kc{i}:=prefill_inputs/kc{i}.raw" for i in range(4)]
parts += [f"vc{i}:=prefill_inputs/vc{i}.raw" for i in range(4)]
open("/work/memcom/prefill_input_list.txt", "w").write(" ".join(parts) + "\n")
print(f"Generated {len(parts)} calibration inputs for chunk_size={C}")
