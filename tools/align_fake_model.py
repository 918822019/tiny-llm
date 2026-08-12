#!/usr/bin/env python3
"""Numerical alignment check: tinyqwen C++ runtime vs PyTorch (HF Qwen2).

Builds the tiny random-weight model (tools/make_fake_model.py), runs the C++
binary for greedy decode with full logits dumped (--dump-logits, fp32 exact),
then loads the SAME weights into a HF Qwen2 model (fp32, eager attention) and
compares logits at EVERY position (prefill + decode).

This exercises the whole forward path (RMSNorm/RoPE/q-k-v bias/GQA/SwiGLU/
tied lm_head) without downloading the real model. For the real-model workflow
see docs/pytorch_alignment.md.

Usage:
    python tools/align_fake_model.py [--binary build/runtime/tinyqwen]
"""

from __future__ import annotations

import argparse
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from export_qwen_to_tiny import HEADER_FMT, ENTRY_FMT  # noqa: E402

PROMPT = [3, 7, 11, 2]
MAX_NEW = 6
TOL = 1e-5


def read_tqwen(path: Path):
    import numpy as np

    with open(path, "rb") as f:
        fields = struct.unpack(HEADER_FMT, f.read(struct.calcsize(HEADER_FMT)))
        cfg = {
            "n_layers": fields[3], "hidden_size": fields[4],
            "intermediate_size": fields[5], "n_heads": fields[6],
            "n_kv_heads": fields[7], "head_dim": fields[8],
            "vocab_size": fields[9], "max_seq_len": fields[10],
            "tied": fields[11], "rms_norm_eps": fields[13], "rope_theta": fields[14],
        }
        count, table_offset = fields[15], fields[16]
        tensors = {}
        f.seek(table_offset)
        for _ in range(count):
            name_b, dtype, ndim, s0, s1, s2, s3, off, nbytes = struct.unpack(
                ENTRY_FMT, f.read(struct.calcsize(ENTRY_FMT)))
            assert dtype == 0
            name = name_b.rstrip(b"\x00").decode("ascii")
            shape = [s0, s1, s2, s3][:ndim]
            pos = f.tell()
            f.seek(off)
            tensors[name] = np.frombuffer(f.read(nbytes), dtype=np.float32).reshape(shape).copy()
            f.seek(pos)
    return cfg, tensors


def run_cpp(binary: Path, model: Path, tmp: Path):
    """Run the C++ binary and collect its exact logits.

    Returns (generated_ids, logits) where logits[i] is the fp32 row produced
    after consuming sequence position i — row order == position order, which
    is the --dump-logits contract of runtime/main.cpp.
    """
    import numpy as np

    logits_path = tmp / "logits.bin"
    cmd = [str(binary), "--model", str(model),
           "--tokens", ",".join(map(str, PROMPT)),
           "--max-new-tokens", str(MAX_NEW),
           "--max-seq-len", "32",
           "--eos", "-1",
           "--dump-logits", str(logits_path)]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    gen_ids = [int(line.split()[2]) for line in out.splitlines()
               if line.startswith("gen ")]
    assert len(gen_ids) == MAX_NEW, gen_ids
    vocab = None
    rows = np.fromfile(logits_path, dtype=np.float32)
    # forwards = len(PROMPT) prefill + (MAX_NEW - 1) decode advances
    n_rows = len(PROMPT) + MAX_NEW - 1
    assert rows.size % n_rows == 0, (rows.size, n_rows)
    vocab = rows.size // n_rows
    return gen_ids, rows.reshape(n_rows, vocab)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--binary", default="build/runtime/tinyqwen")
    args = p.parse_args()

    import numpy as np
    import torch
    from transformers import Qwen2Config, Qwen2ForCausalLM

    with tempfile.TemporaryDirectory() as tmp_d:
        tmp = Path(tmp_d)
        model_path = tmp / "fake.tqwen"
        subprocess.run([sys.executable, "tools/make_fake_model.py",
                        "--out", str(model_path)], check=True, capture_output=True)
        cfg, tensors = read_tqwen(model_path)
        gen_ids, cpp_logits = run_cpp(Path(args.binary), model_path, tmp)

    hf_cfg = Qwen2Config(
        hidden_size=cfg["hidden_size"],
        intermediate_size=cfg["intermediate_size"],
        num_hidden_layers=cfg["n_layers"],
        num_attention_heads=cfg["n_heads"],
        num_key_value_heads=cfg["n_kv_heads"],
        head_dim=cfg["head_dim"],
        vocab_size=cfg["vocab_size"],
        max_position_embeddings=cfg["max_seq_len"],
        rms_norm_eps=cfg["rms_norm_eps"],
        rope_theta=cfg["rope_theta"],
        tie_word_embeddings=bool(cfg["tied"]),
        hidden_act="silu",
    )
    hf_cfg._attn_implementation = "eager"
    model = Qwen2ForCausalLM(hf_cfg).eval()
    missing, unexpected = model.load_state_dict(
        {k: torch.from_numpy(v) for k, v in tensors.items()}, strict=False)
    allowed_missing = {"lm_head.weight"} if cfg["tied"] else set()
    assert set(missing) <= allowed_missing, missing
    assert not unexpected, unexpected

    # C++ forward rows: positions 0..P-1 are prefill, then decode positions.
    # Greedy token g_s = argmax(logits[P - 1 + s]).
    P = len(PROMPT)
    # HF needs the same token schedule as input; the last generated token is
    # never consumed by any compared position, so drop it.
    full_seq = PROMPT + gen_ids[: MAX_NEW - 1]
    input_ids = torch.tensor([full_seq], dtype=torch.long)
    with torch.no_grad():
        pos = torch.arange(len(full_seq), dtype=torch.long).unsqueeze(0)
        ref = model(input_ids=input_ids, position_ids=pos,
                    use_cache=False).logits[0].numpy()
    assert ref.shape == cpp_logits.shape, (ref.shape, cpp_logits.shape)

    worst = 0.0
    for i in range(len(full_seq)):
        err = float(np.abs(cpp_logits[i] - ref[i]).max())
        worst = max(worst, err)
        print(f"pos {i}: max_abs_err={err:.3e}")
        assert err < TOL, f"position {i} diverges: {err}"

    for s, g in enumerate(gen_ids):
        row = P - 1 + s
        assert int(cpp_logits[row].argmax()) == g, f"cpp internal argmax mismatch at {s}"
        assert int(ref[row].argmax()) == g, f"step {s}: cpp={g} ref={int(ref[row].argmax())}"
    print(f"generated: {gen_ids} (matches HF argmax at every step)")
    print(f"WORST max_abs_err: {worst:.3e} (tol {TOL})")
    print("ALIGNMENT OK")


if __name__ == "__main__":
    main()
