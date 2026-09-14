#!/usr/bin/env python3
"""Convert a SpecForge/SGLang EAGLE3 checkpoint to tinyqwen FP16.

The exported file contains the one-layer EAGLE3 network, a compact
draft-id -> target-id vocabulary map, and the three target residual stream
indices used to train the checkpoint. Matrix and norm weights are converted
from BF16 to FP16. Token-id metadata uses an exact two-limb FP16 encoding so
the file remains uniform-dtype without losing ids above FP16's integer range.

Example:
    .venv/bin/python tools/export_eagle3_to_tiny.py \
        --model models/SGLang-EAGLE3-Qwen3-0.6B-SpecForge \
        --out eagle3_qwen3_06b_specforge_f16.tqwen
"""

from __future__ import annotations

import argparse
import json
import struct
from collections import OrderedDict
from pathlib import Path

import numpy as np
from safetensors import safe_open

from export_qwen_to_tiny import (
    ALIGN,
    DTYPE_F16,
    ENTRY_FMT,
    HEADER_FMT,
    MAGIC,
    MAX_NAME,
    MODEL_QWEN2,
    align_up,
    pack_v2_ext,
)


EXPECTED_WEIGHTS = OrderedDict(
    [
        ("fc.weight", (1024, 3072)),
        ("midlayer.hidden_norm.weight", (1024,)),
        ("midlayer.input_layernorm.weight", (1024,)),
        ("midlayer.self_attn.q_proj.weight", (2048, 2048)),
        ("midlayer.self_attn.k_proj.weight", (1024, 2048)),
        ("midlayer.self_attn.v_proj.weight", (1024, 2048)),
        ("midlayer.self_attn.o_proj.weight", (1024, 2048)),
        ("midlayer.post_attention_layernorm.weight", (1024,)),
        ("midlayer.mlp.gate_proj.weight", (3072, 1024)),
        ("midlayer.mlp.up_proj.weight", (3072, 1024)),
        ("midlayer.mlp.down_proj.weight", (1024, 3072)),
        ("norm.weight", (1024,)),
        ("lm_head.weight", (32000, 1024)),
    ]
)


def _parse_layers(text: str) -> list[int]:
    try:
        layers = [int(piece.strip()) for piece in text.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("target layers must be comma-separated integers") from exc
    if len(layers) != 3 or any(layer < 0 for layer in layers):
        raise argparse.ArgumentTypeError("EAGLE3 requires exactly three non-negative target layers")
    if len(set(layers)) != 3 or layers != sorted(layers):
        raise argparse.ArgumentTypeError("target layers must be distinct and increasing")
    return layers


def _write_tqwen(
    out_path: Path,
    cfg: dict,
    tensors: "OrderedDict[str, np.ndarray]",
) -> int:
    """Write a v2 tinyqwen file whose tensors are uniformly FP16."""
    names = list(tensors)
    table_end = struct.calcsize(HEADER_FMT) + len(names) * struct.calcsize(ENTRY_FMT)
    data_offset = align_up(table_end, ALIGN)
    entries = []
    offset = data_offset

    for name, array in tensors.items():
        if len(name.encode("ascii")) > MAX_NAME:
            raise ValueError(f"tensor name is too long: {name}")
        if not 1 <= array.ndim <= 4:
            raise ValueError(f"{name}: ndim {array.ndim} is outside [1, 4]")
        if array.dtype == np.float16:
            dtype_code = DTYPE_F16
        else:
            raise ValueError(f"{name}: unsupported exported dtype {array.dtype}")
        nbytes = int(array.nbytes)
        shape = tuple(int(dim) for dim in array.shape)
        entries.append((name, dtype_code, shape, offset, nbytes))
        offset = align_up(offset + nbytes, ALIGN)

    total_bytes = offset
    ext = pack_v2_ext(
        {
            "model_type": MODEL_QWEN2,
            "partial_rotary_factor": 1.0,
            "eos_token_id": int(cfg["eos_token_id"]),
        }
    )
    reserved = ext + b"\x00" * (96 - len(ext))
    header = struct.pack(
        HEADER_FMT,
        MAGIC,
        2,
        DTYPE_F16,
        int(cfg["num_hidden_layers"]),
        int(cfg["hidden_size"]),
        int(cfg["intermediate_size"]),
        int(cfg["num_attention_heads"]),
        int(cfg["num_key_value_heads"]),
        int(cfg["head_dim"]),
        int(cfg["vocab_size"]),
        int(cfg["max_position_embeddings"]),
        0,
        0,
        float(cfg["rms_norm_eps"]),
        float(cfg["rope_theta"]),
        len(entries),
        struct.calcsize(HEADER_FMT),
        data_offset,
        total_bytes,
        reserved,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as out:
        out.write(header)
        for name, dtype_code, shape, tensor_offset, nbytes in entries:
            out.write(
                struct.pack(
                    ENTRY_FMT,
                    name.encode("ascii"),
                    dtype_code,
                    len(shape),
                    *shape,
                    *([0] * (4 - len(shape))),
                    tensor_offset,
                    nbytes,
                )
            )
        out.write(b"\x00" * (data_offset - out.tell()))
        for name, _dtype_code, _shape, tensor_offset, nbytes in entries:
            if out.tell() != tensor_offset:
                raise AssertionError(f"alignment bug at {name}")
            payload = tensors[name].tobytes(order="C")
            if len(payload) != nbytes:
                raise AssertionError(f"size bug at {name}")
            out.write(payload)
            out.write(b"\x00" * (align_up(tensor_offset + nbytes, ALIGN) - out.tell()))
    if out_path.stat().st_size != total_bytes:
        raise AssertionError("exported file size does not match header")
    return total_bytes


def export(model_dir: Path, out_path: Path, target_layers: list[int]) -> int:
    config_path = model_dir / "config.json"
    weight_path = model_dir / "model.safetensors"
    if not config_path.is_file() or not weight_path.is_file():
        raise FileNotFoundError("model directory must contain config.json and model.safetensors")
    cfg = json.loads(config_path.read_text(encoding="utf-8"))

    if cfg.get("architectures") != ["LlamaForCausalLMEagle3"]:
        raise ValueError(f"unsupported architecture: {cfg.get('architectures')}")
    required_cfg = {
        "num_hidden_layers": 1,
        "hidden_size": 1024,
        "intermediate_size": 3072,
        "num_attention_heads": 16,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "draft_vocab_size": 32000,
    }
    for key, expected in required_cfg.items():
        if cfg.get(key) != expected:
            raise ValueError(f"unsupported {key}: got {cfg.get(key)!r}, expected {expected}")
    if cfg.get("rope_scaling") not in (None, {}):
        raise ValueError("rope_scaling is not supported by this EAGLE3 exporter")
    if cfg.get("attention_bias", False) or cfg.get("mlp_bias", False):
        raise ValueError("biased EAGLE3 layers are not supported")
    if cfg.get("fc_norm", False):
        raise ValueError("fc_norm checkpoints are not supported yet")

    tensors: "OrderedDict[str, np.ndarray]" = OrderedDict()
    with safe_open(str(weight_path), framework="pt", device="cpu") as weights:
        keys = set(weights.keys())
        expected_keys = set(EXPECTED_WEIGHTS) | {"d2t", "t2d"}
        missing = expected_keys - keys
        unexpected = keys - expected_keys
        if missing:
            raise ValueError(f"missing checkpoint tensors: {sorted(missing)}")
        if unexpected:
            raise ValueError(f"unexpected checkpoint tensors: {sorted(unexpected)}")

        # Shape expectations are derived from config, while the ordered names
        # above make the output deterministic and easy to inspect.
        h = int(cfg["hidden_size"])
        i = int(cfg["intermediate_size"])
        q = int(cfg["num_attention_heads"] * cfg["head_dim"])
        kv = int(cfg["num_key_value_heads"] * cfg["head_dim"])
        dv = int(cfg["draft_vocab_size"])
        shapes = dict(EXPECTED_WEIGHTS)
        shapes.update(
            {
                "fc.weight": (h, 3 * h),
                "midlayer.hidden_norm.weight": (h,),
                "midlayer.input_layernorm.weight": (h,),
                "midlayer.self_attn.q_proj.weight": (q, 2 * h),
                "midlayer.self_attn.k_proj.weight": (kv, 2 * h),
                "midlayer.self_attn.v_proj.weight": (kv, 2 * h),
                "midlayer.self_attn.o_proj.weight": (h, q),
                "midlayer.post_attention_layernorm.weight": (h,),
                "midlayer.mlp.gate_proj.weight": (i, h),
                "midlayer.mlp.up_proj.weight": (i, h),
                "midlayer.mlp.down_proj.weight": (h, i),
                "norm.weight": (h,),
                "lm_head.weight": (dv, h),
            }
        )
        for name in EXPECTED_WEIGHTS:
            tensor = weights.get_tensor(name)
            if tuple(tensor.shape) != shapes[name]:
                raise ValueError(f"{name}: shape {tuple(tensor.shape)} != {shapes[name]}")
            tensors[name] = tensor.float().numpy().astype(np.float16)

        d2t = weights.get_tensor("d2t").cpu().numpy().astype(np.int64)
        t2d = weights.get_tensor("t2d").cpu().numpy().astype(np.bool_)
        if d2t.shape != (dv,) or t2d.shape != (int(cfg["vocab_size"]),):
            raise ValueError("d2t/t2d vocabulary metadata has an invalid shape")
        mapping = np.arange(dv, dtype=np.int64) + d2t
        if mapping.min() < 0 or mapping.max() >= int(cfg["vocab_size"]):
            raise ValueError("draft-to-target mapping contains an out-of-range token id")
        if np.unique(mapping).size != dv:
            raise ValueError("draft-to-target mapping is not one-to-one")
        mask_ids = np.flatnonzero(t2d)
        if mask_ids.size != dv or not np.array_equal(mask_ids, np.sort(mapping)):
            raise ValueError("t2d and d2t vocabulary metadata disagree")

    # A single FP16 cannot represent the target vocabulary's 151k-range ids.
    # Split every id into base-256 limbs; both limbs are integers <= 593 and
    # therefore exactly representable in FP16. This preserves uniform-dtype.
    mapping_limbs = np.stack((mapping // 256, mapping % 256), axis=1)
    tensors["eagle3.draft_to_target"] = mapping_limbs.astype(np.float16)
    tensors["eagle3.target_layer_ids"] = np.asarray(target_layers, dtype=np.float16)
    return _write_tqwen(out_path, cfg, tensors)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path, help="EAGLE3 HF checkpoint directory")
    parser.add_argument("--out", required=True, type=Path, help="output .tqwen path")
    parser.add_argument(
        "--target-layers",
        type=_parse_layers,
        default=_parse_layers("1,13,24"),
        help="three zero-based target layers captured after each layer (default: 1,13,24)",
    )
    args = parser.parse_args()
    try:
        size = export(args.model, args.out, args.target_layers)
    except (FileNotFoundError, ValueError, OSError) as exc:
        parser.error(str(exc))
    print(f"wrote {args.out} ({size / 1048576:.2f} MiB, FP16 EAGLE3)")
    print(f"target layers: {','.join(map(str, args.target_layers))}")


if __name__ == "__main__":
    main()
