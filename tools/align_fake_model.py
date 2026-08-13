#!/usr/bin/env python3
"""数值对齐检查：tinyqwen C++ runtime vs PyTorch（HF Qwen2）。

流程：用 tools/make_fake_model.py 生成随机权重小模型，跑 C++ binary 做
greedy decode 并用 --dump-logits 导出全量 logits（fp32 精确值），然后把
同一份权重灌进 HF Qwen2 模型（fp32、eager attention），逐位置
（prefill + decode）比对 logits。

这条链路完整覆盖 RMSNorm / q-k-v bias / RoPE / GQA / SwiGLU / tied lm_head，
且不需要下载真模型。真模型的对齐流程见 docs/pytorch_alignment.md。

用法:
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
    """Python 侧的最小 .tqwen 读取器（与 C++ loader 同一格式契约）。"""
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
    """跑 C++ binary，收集精确 logits。

    返回 (generated_ids, logits)，其中 logits[i] 是消费完序列位置 i 后
    产生的 fp32 行——行序 == 位置序，这是 runtime/main.cpp 的
    --dump-logits 契约。
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
    # forward 次数 = len(PROMPT) 次 prefill + (MAX_NEW - 1) 次 decode 推进
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

    # 用完全相同的配置和权重构造 HF 参考模型。
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
    # tied 模型里 lm_head.weight 与 embed 共享，允许缺省。
    allowed_missing = {"lm_head.weight"} if cfg["tied"] else set()
    assert set(missing) <= allowed_missing, missing
    assert not unexpected, unexpected

    # C++ 的 forward 行：位置 0..P-1 是 prefill，之后是 decode 位置。
    # greedy token g_s = argmax(logits[P - 1 + s])。
    P = len(PROMPT)
    # HF 需要与 C++ 相同的 token 序列作为输入；最后一个生成 token
    # 不会出现在任何被比较位置的输入里，所以去掉。
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
