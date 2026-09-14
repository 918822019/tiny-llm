#!/usr/bin/env python3
"""Export a DFlash/DFlare safetensors checkpoint to tinyqwen's flat format.

The v2 header is intentionally reused: ``eos_token_id`` stores the DFlash mask
token id, while ``dflash.target_layer_ids`` carries the target residual layers.
The dedicated DFlash loader interprets those fields; ordinary QwenModel never
loads this file.
"""

import argparse
import json
from collections import OrderedDict
from pathlib import Path

import numpy as np
from safetensors import safe_open

from export_qwen_to_tiny import MODEL_QWEN2, write_tqwen


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="DFlash checkpoint directory")
    parser.add_argument("--out", required=True, help="output .tqwen/.tdflash path")
    args = parser.parse_args()

    root = Path(args.model)
    cfg = json.loads((root / "config.json").read_text())
    dc = cfg.get("dflash_config", {})
    if dc.get("fusion_type") != "dflare":
        raise SystemExit("error: this exporter currently requires fusion_type=dflare")
    if int(dc.get("markov_rank", 0)) <= 0:
        raise SystemExit("error: this exporter requires the Markov head")

    tensors = OrderedDict()
    with safe_open(str(root / "model.safetensors"), framework="numpy") as f:
        for name in f.keys():
            tensors[name] = np.asarray(f.get_tensor(name))
    tensors["dflash.target_layer_ids"] = np.asarray(dc["target_layer_ids"], dtype=np.float16)

    tiny_cfg = {
        "n_layers": cfg["num_hidden_layers"],
        "hidden_size": cfg["hidden_size"],
        "intermediate_size": cfg["intermediate_size"],
        "n_heads": cfg["num_attention_heads"],
        "n_kv_heads": cfg["num_key_value_heads"],
        "head_dim": cfg["head_dim"],
        "vocab_size": cfg["vocab_size"],
        "max_seq_len": cfg["max_position_embeddings"],
        "tied": 0,
        "rms_norm_eps": cfg["rms_norm_eps"],
        "rope_theta": cfg["rope_theta"],
    }
    ext = {
        "model_type": MODEL_QWEN2,
        "eos_token_id": int(dc["mask_token_id"]),
    }
    size = write_tqwen(args.out, tiny_cfg, tensors, dtype="f16", version=2, ext=ext)
    print(f"exported DFlash FP16: {args.out} ({size / (1024**2):.1f} MiB)")
    print(f"block_size={cfg['block_size']} markov_rank={dc['markov_rank']} "
          f"target_layers={dc['target_layer_ids']}")


if __name__ == "__main__":
    main()
