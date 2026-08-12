#!/usr/bin/env python3
"""Export Qwen2.5 (HF) weights to the tinyqwen flat binary format (.tqwen).

The binary contract is defined in runtime/tiny_format.h and docs/weight_format.md.
Keep the struct layouts below in sync with that header.

Usage:
    python tools/export_qwen_to_tiny.py \
        --model Qwen/Qwen2.5-0.5B \
        --out model.tqwen

Notes:
    - v1 exports float32 only.
    - Linear weights keep the HF [out_dim, in_dim] row-major layout (no transpose).
    - Tokenizer is NOT handled here; see tools/tokenize_prompt.py.
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys
from pathlib import Path

# Must match runtime/tiny_format.h -------------------------------------------
MAGIC = b"TINYQWEN"
FORMAT_VERSION = 1
ALIGN = 64
DTYPE_F32 = 0
MAX_NAME = 64

HEADER_FMT = "<8s12Iff4Q96s"  # 192 bytes
ENTRY_FMT = "<64sII4QQQ"      # 120 bytes

assert struct.calcsize(HEADER_FMT) == 192
assert struct.calcsize(ENTRY_FMT) == 120
# ---------------------------------------------------------------------------


def align_up(x: int, align: int = ALIGN) -> int:
    return (x + align - 1) // align * align


def write_tqwen(out_path: str | Path, cfg: dict, tensors: "dict[str, object]") -> int:
    """Write a .tqwen file. Returns total file size in bytes.

    cfg keys (all required):
        n_layers, hidden_size, intermediate_size, n_heads, n_kv_heads,
        head_dim, vocab_size, max_seq_len, tied (0/1), rms_norm_eps, rope_theta
    tensors: ordered mapping name -> array-like with .shape and
        .astype("float32").tobytes() (numpy arrays work).
    """
    names = list(tensors.keys())

    table_end = struct.calcsize(HEADER_FMT) + len(names) * struct.calcsize(ENTRY_FMT)
    data_offset = align_up(table_end)

    entries = []
    offset = data_offset
    for name in names:
        shape = [int(d) for d in tensors[name].shape]
        if not (1 <= len(shape) <= 4):
            sys.exit(f"error: {name}: ndim {len(shape)} not in [1, 4]")
        if len(name) > MAX_NAME:
            sys.exit(f"error: tensor name too long: {name}")
        numel = 1
        for d in shape:
            numel *= d
        nbytes = numel * 4
        entries.append((name, shape, offset, nbytes))
        offset = align_up(offset + nbytes)
    total_bytes = offset

    header = struct.pack(
        HEADER_FMT,
        MAGIC,
        FORMAT_VERSION,
        DTYPE_F32,
        int(cfg["n_layers"]),
        int(cfg["hidden_size"]),
        int(cfg["intermediate_size"]),
        int(cfg["n_heads"]),
        int(cfg["n_kv_heads"]),
        int(cfg["head_dim"]),
        int(cfg["vocab_size"]),
        int(cfg["max_seq_len"]),
        int(cfg["tied"]),
        0,  # reserved_u32
        float(cfg["rms_norm_eps"]),
        float(cfg["rope_theta"]),
        len(names),
        struct.calcsize(HEADER_FMT),
        data_offset,
        total_bytes,
        b"\x00" * 96,
    )

    # Write order mirrors the on-disk layout (docs/weight_format.md):
    #   header -> tensor table -> zero padding -> 64B-aligned payloads.
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as out:
        out.write(header)
        for name, shape, off, nbytes in entries:
            out.write(struct.pack(
                ENTRY_FMT,
                name.encode("ascii"),
                DTYPE_F32,
                len(shape),
                *shape,
                *[0] * (4 - len(shape)),
                off,
                nbytes,
            ))
        # Zero-fill the gap between table end and the first aligned payload.
        out.write(b"\x00" * (data_offset - out.tell()))

        for name, shape, off, nbytes in entries:
            assert out.tell() == off, f"alignment bug at {name}"
            data = tensors[name].astype("float32", copy=False).tobytes()
            assert len(data) == nbytes, f"size bug at {name}"
            out.write(data)
            # Pad each payload up to the next 64B boundary.
            out.write(b"\x00" * (align_up(off + nbytes) - (off + nbytes)))
        assert out.tell() == total_bytes

    size = os.path.getsize(out_path)
    if size != total_bytes:
        sys.exit(f"error: file size {size} != expected {total_bytes}")
    return total_bytes


def print_table_summary(out_path: str | Path) -> None:
    """Re-read the written file's header + tensor table and print a summary."""
    with open(out_path, "rb") as f:
        header = f.read(struct.calcsize(HEADER_FMT))
        fields = struct.unpack(HEADER_FMT, header)
        magic, version, dtype = fields[0], fields[1], fields[2]
        assert magic == MAGIC and version == FORMAT_VERSION and dtype == DTYPE_F32
        # fields: 0 magic, 1 version, 2 dtype, 3..12 ten u32s, 13 eps, 14 theta,
        #         15 tensor_count, 16 tensor_table_offset, 17 data_offset, 18 total
        tensor_count = fields[15]
        table_offset = fields[16]
        f.seek(table_offset)
        print(f"{'name':<56} {'shape':<22} {'dtype':<6} {'offset':>12} {'nbytes':>14}")
        for _ in range(tensor_count):
            name_b, t_dtype, ndim, s0, s1, s2, s3, off, nbytes = struct.unpack(
                ENTRY_FMT, f.read(struct.calcsize(ENTRY_FMT)))
            name = name_b.rstrip(b"\x00").decode("ascii")
            shape = [s0, s1, s2, s3][:ndim]
            print(f"{name:<56} {str(shape):<22} {'f32':<6} {off:>12} {nbytes:>14}")


# ---- HF model collection ---------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True,
                   help="HF model dir or repo id (must contain config.json)")
    p.add_argument("--out", required=True, help="output .tqwen path")
    p.add_argument("--max-seq-len", type=int, default=0,
                   help="override max_seq_len in header (0 = use config)")
    return p.parse_args()


def find_local_dir(model: str) -> Path:
    """Resolve a local model directory. Repo-id download is left to the user
    (e.g. `huggingface-cli download`) so this script stays dependency-light."""
    path = Path(model)
    if path.is_dir() and (path / "config.json").exists():
        return path
    sys.exit(f"error: {model} is not a local dir with config.json. "
             f"Download it first, e.g. `huggingface-cli download {model} --local-dir {model}`")


def load_config(model_dir: Path) -> dict:
    cfg = json.loads((model_dir / "config.json").read_text())
    required = ["hidden_size", "intermediate_size", "num_hidden_layers",
                "num_attention_heads", "num_key_value_heads", "vocab_size",
                "rms_norm_eps", "rope_theta", "max_position_embeddings"]
    missing = [k for k in required if k not in cfg]
    if missing:
        sys.exit(f"error: config.json missing fields: {missing}")
    head_dim = cfg.get("head_dim")
    if head_dim is None:
        head_dim = cfg["hidden_size"] // cfg["num_attention_heads"]
    cfg["head_dim"] = head_dim
    if cfg["num_attention_heads"] % cfg["num_key_value_heads"] != 0:
        sys.exit("error: num_attention_heads % num_key_value_heads != 0")
    return cfg


def tensor_names(cfg: dict) -> list[str]:
    names = ["model.embed_tokens.weight"]
    for i in range(cfg["num_hidden_layers"]):
        names += [
            f"model.layers.{i}.input_layernorm.weight",
            f"model.layers.{i}.self_attn.q_proj.weight",
            f"model.layers.{i}.self_attn.k_proj.weight",
            f"model.layers.{i}.self_attn.v_proj.weight",
            f"model.layers.{i}.self_attn.q_proj.bias",
            f"model.layers.{i}.self_attn.k_proj.bias",
            f"model.layers.{i}.self_attn.v_proj.bias",
            f"model.layers.{i}.self_attn.o_proj.weight",
            f"model.layers.{i}.post_attention_layernorm.weight",
            f"model.layers.{i}.mlp.gate_proj.weight",
            f"model.layers.{i}.mlp.up_proj.weight",
            f"model.layers.{i}.mlp.down_proj.weight",
        ]
    names.append("model.norm.weight")
    if not cfg.get("tie_word_embeddings", False):
        names.append("lm_head.weight")
    return names


def build_shard_map(model_dir: Path, names: list[str]) -> dict[str, Path]:
    """Map each tensor name to the safetensors shard that contains it."""
    index_file = model_dir / "model.safetensors.index.json"
    if index_file.exists():
        index = json.loads(index_file.read_text())["weight_map"]
        return {n: model_dir / index[n] for n in names}
    single = model_dir / "model.safetensors"
    if single.exists():
        return {n: single for n in names}
    sys.exit("error: no model.safetensors[.index.json] found")


def main() -> None:
    args = parse_args()
    model_dir = find_local_dir(args.model)
    cfg = load_config(model_dir)
    names = tensor_names(cfg)
    shard_map = build_shard_map(model_dir, names)

    from safetensors import safe_open  # lazy import: fail late, fail clearly

    opened: dict[Path, object] = {}

    def get_shard(path: Path):
        if path not in opened:
            opened[path] = safe_open(str(path), framework="numpy")
        return opened[path]

    # Validate presence and collect shapes first (fail before writing anything).
    for name in names:
        f = get_shard(shard_map[name])
        if name not in f.keys():
            sys.exit(f"error: tensor {name} not found in {shard_map[name]}")

    header_cfg = {
        "n_layers": cfg["num_hidden_layers"],
        "hidden_size": cfg["hidden_size"],
        "intermediate_size": cfg["intermediate_size"],
        "n_heads": cfg["num_attention_heads"],
        "n_kv_heads": cfg["num_key_value_heads"],
        "head_dim": cfg["head_dim"],
        "vocab_size": cfg["vocab_size"],
        "max_seq_len": args.max_seq_len or cfg["max_position_embeddings"],
        "tied": 1 if cfg.get("tie_word_embeddings", False) else 0,
        "rms_norm_eps": cfg["rms_norm_eps"],
        "rope_theta": cfg["rope_theta"],
    }

    # Stream one tensor at a time to keep memory flat: write_tqwen needs all
    # tensors as a mapping, so for very large models we hand it a lazy dict
    # that loads a tensor from its safetensors shard on access. Note each
    # tensor is read twice (shape pass + payload pass) — acceptable trade-off
    # vs. holding a 2GB fp32 model in RAM.
    class LazyTensors(dict):
        def __getitem__(self, key):
            return get_shard(shard_map[key]).get_tensor(key)

    total = write_tqwen(args.out, header_cfg, LazyTensors.fromkeys(names))

    print(f"model dir  : {model_dir}")
    print(f"config     : layers={header_cfg['n_layers']} hidden={header_cfg['hidden_size']} "
          f"inter={header_cfg['intermediate_size']} heads={header_cfg['n_heads']} "
          f"kv_heads={header_cfg['n_kv_heads']} head_dim={header_cfg['head_dim']} "
          f"vocab={header_cfg['vocab_size']} tied={header_cfg['tied']}")
    print_table_summary(args.out)
    print(f"out        : {args.out}")
    print(f"file size  : {total} bytes ({total / 2**20:.1f} MB) OK")


if __name__ == "__main__":
    main()
