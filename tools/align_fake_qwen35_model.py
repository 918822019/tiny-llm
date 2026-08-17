#!/usr/bin/env python3
"""数值对齐检查：tinyqwen C++ runtime（Qwen3.5 混合架构）vs PyTorch（HF）。

流程与 align_fake_model.py 相同：make_fake_qwen35_model.py 生成随机权重小模型
（同时产出 HF state_dict），跑 C++ binary greedy decode 并 --dump-logits 导出
全量 logits，再把同一份权重灌进 HF Qwen3_5ForCausalLM（fp32、eager、torch
兜底 kernel），逐位置（prefill + decode）比对 logits。

这条链路覆盖 GDN 递归/conv1d/l2norm/门控 RMSNorm、full attention 的 QK-norm/
partial RoPE/输出门、zero-centered RMSNorm 折叠、tied lm_head——且不需要下载真模型。

用法:
    python tools/align_fake_qwen35_model.py [--binary build/runtime/tinyqwen]
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

PROMPT = [3, 7, 11, 2]
MAX_NEW = 6
TOL = 1e-5


def run_cpp(binary: Path, model: Path, tmp: Path):
    """跑 C++ binary，收集精确 logits（行序 == 位置序，同 align_fake_model.py）。"""
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
    rows = np.fromfile(logits_path, dtype=np.float32)
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

    from make_fake_qwen35_model import make_config
    from transformers import Qwen3_5ForCausalLM

    with tempfile.TemporaryDirectory() as tmp_d:
        tmp = Path(tmp_d)
        model_path = tmp / "fake35.tqwen"
        hf_path = tmp / "fake35_hf.pt"
        subprocess.run([sys.executable, "tools/make_fake_qwen35_model.py",
                        "--out", str(model_path), "--hf-out", str(hf_path)],
                       check=True, capture_output=True)
        gen_ids, cpp_logits = run_cpp(Path(args.binary), model_path, tmp)
        hf_state = torch.load(hf_path, weights_only=True)

    # 用完全相同的配置和权重构造 HF 参考模型（torch 兜底 kernel，无 fla 依赖）。
    cfg = make_config()
    cfg._attn_implementation = "eager"
    model = Qwen3_5ForCausalLM(cfg).eval()
    # tied embeddings 下默认 copy 式 load 不会写入 embed，需用 assign=True
    # 再 tie_weights() 恢复 lm_head<->embed 的绑定。
    missing, unexpected = model.load_state_dict(hf_state, strict=False, assign=True)
    model.tie_weights()
    allowed_missing = {"lm_head.weight"} if cfg.tie_word_embeddings else set()
    assert set(missing) <= allowed_missing, missing
    assert not unexpected, unexpected
    assert torch.allclose(model.model.embed_tokens.weight,
                          hf_state["model.embed_tokens.weight"]), "embed not loaded"

    # C++ 行：0..P-1 是 prefill，之后是 decode；greedy g_s = argmax(row P-1+s)。
    P = len(PROMPT)
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
        tag = "prefill" if i < P else "decode "
        print(f"pos {i} ({tag}): max_abs_err={err:.3e}")
        assert err < TOL, f"position {i} diverges: {err}"

    for s, g in enumerate(gen_ids):
        row = P - 1 + s
        assert int(cpp_logits[row].argmax()) == g, f"cpp internal argmax mismatch at {s}"
        assert int(ref[row].argmax()) == g, \
            f"step {s}: cpp={g} ref={int(ref[row].argmax())}"
    print(f"generated: {gen_ids} (matches HF argmax at every step)")
    print(f"WORST max_abs_err: {worst:.3e} (tol {TOL})")
    print("ALIGNMENT OK")


if __name__ == "__main__":
    main()
