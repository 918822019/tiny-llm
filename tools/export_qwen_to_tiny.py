#!/usr/bin/env python3
"""将 Qwen2.5（HF）权重导出为 tinyqwen 扁平二进制格式（.tqwen）。

二进制契约定义在 runtime/tiny_format.h 和 docs/weight_format.md，
下面的 struct 布局必须与该头文件保持同步。

用法:
    python tools/export_qwen_to_tiny.py \
        --model Qwen/Qwen2.5-0.5B \
        --out model.tqwen

说明:
    - v1 只导出 float32。
    - linear 权重保持 HF 的 [out_dim, in_dim] 行主序，不转置。
    - 不处理 tokenizer，见 tools/tokenize_prompt.py。
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys
from pathlib import Path

# 必须与 runtime/tiny_format.h 保持一致 -------------------------------
MAGIC = b"TINYQWEN"
FORMAT_VERSION = 1
ALIGN = 64
DTYPE_F32 = 0
MAX_NAME = 64

HEADER_FMT = "<8s12Iff4Q96s"  # 192 字节
ENTRY_FMT = "<64sII4QQQ"  # 120 字节

assert struct.calcsize(HEADER_FMT) == 192
assert struct.calcsize(ENTRY_FMT) == 120


# ---------------------------------------------------------------------


def align_up(x: int, align: int = ALIGN) -> int:
    return (x + align - 1) // align * align


def write_tqwen(out_path: str | Path, cfg: dict, tensors: "dict[str, object]") -> int:
    """写一个 .tqwen 文件，返回文件总字节数。

    cfg 必需的键：
        n_layers, hidden_size, intermediate_size, n_heads, n_kv_heads,
        head_dim, vocab_size, max_seq_len, tied (0/1), rms_norm_eps, rope_theta
    tensors: 有序映射 name -> 数组对象，要求有 .shape 属性以及
        .astype("float32").tobytes() 方法（numpy 数组可直接使用）。
    """
    names = list(tensors.keys())

    table_end = struct.calcsize(HEADER_FMT) + len(names) * struct.calcsize(ENTRY_FMT)
    data_offset = align_up(table_end)

    # 第一遍：只取 shape，计算每个 tensor 的对齐偏移，得到文件总大小。
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

    # 写入顺序与磁盘布局一致（docs/weight_format.md）：
    #   header -> tensor 表 -> 补零对齐 -> 64B 对齐的数据区。
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
        # 表区末尾到第一个数据偏移之间补零。
        out.write(b"\x00" * (data_offset - out.tell()))

        for name, shape, off, nbytes in entries:
            assert out.tell() == off, f"alignment bug at {name}"
            data = tensors[name].astype("float32", copy=False).tobytes()
            assert len(data) == nbytes, f"size bug at {name}"
            out.write(data)
            # 每个 payload 之后补零到下一个 64B 边界。
            out.write(b"\x00" * (align_up(off + nbytes) - (off + nbytes)))
        assert out.tell() == total_bytes

    size = os.path.getsize(out_path)
    if size != total_bytes:
        sys.exit(f"error: file size {size} != expected {total_bytes}")
    return total_bytes


def print_table_summary(out_path: str | Path) -> None:
    """回读刚写出文件的 header + tensor 表，打印摘要（验收用）。"""
    with open(out_path, "rb") as f:
        header = f.read(struct.calcsize(HEADER_FMT))
        fields = struct.unpack(HEADER_FMT, header)
        magic, version, dtype = fields[0], fields[1], fields[2]
        assert magic == MAGIC and version == FORMAT_VERSION and dtype == DTYPE_F32
        # fields 下标: 0 magic, 1 version, 2 dtype, 3..12 十个 u32,
        #   13 eps, 14 theta, 15 tensor_count, 16 tensor_table_offset,
        #   17 data_offset, 18 total
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


# ---- HF 模型收集 ------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True,
                   help="HF model dir or repo id (must contain config.json)")
    p.add_argument("--out", required=True, help="output .tqwen path")
    p.add_argument("--max-seq-len", type=int, default=0,
                   help="override max_seq_len in header (0 = use config)")
    return p.parse_args()


def find_local_dir(model: str) -> Path:
    """解析本地模型目录。repo id 的下载留给用户自己执行
    （如 `huggingface-cli download`），让本脚本保持轻依赖。"""
    path = Path(model)
    if path.is_dir() and (path / "config.json").exists():
        return path
    sys.exit(f"error: {model} is not a local dir with config.json. "
             f"Download it first, e.g. `huggingface-cli download {model} --local-dir {model}`")


def load_config(model_dir: Path) -> dict:
    """读取 HF config.json 并校验导出所需字段；head_dim 缺省时推导。"""
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
    """按 HF 命名列出需要导出的全部 tensor（q/k/v 含 bias）。"""
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
    """把每个 tensor 名字映射到包含它的 safetensors 分片文件。"""
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

    from safetensors import safe_open  # 延迟导入：晚失败、报错清晰

    opened: dict[Path, object] = {}

    def get_shard(path: Path):
        if path not in opened:
            # 用 torch 框架读：真实 Qwen 权重是 bf16，numpy 不认识 bf16；
            # 读出来后在 LazyTensors 里统一转成 fp32。
            opened[path] = safe_open(str(path), framework="torch")
        return opened[path]

    # 先校验所有 tensor 都存在（任何缺失都在写文件之前报错）。
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

    # 逐个流式读取 tensor，保持内存占用平稳：write_tqwen 需要一个映射，
    # 所以这里传给它一个惰性 dict，访问时才从对应分片加载 tensor。
    # 注意每个 tensor 会被读两次（shape 一遍 + payload 一遍）——
    # 相比把 2GB fp32 模型整个放进内存，这是可接受的代价。
    class LazyTensors(dict):
        def __getitem__(self, key):
            tensor = get_shard(shard_map[key]).get_tensor(key)
            return tensor.float().numpy()  # bf16/fp16 -> fp32 numpy

    total = write_tqwen(args.out, header_cfg, LazyTensors.fromkeys(names))

    print(f"model dir  : {model_dir}")
    print(f"config     : layers={header_cfg['n_layers']} hidden={header_cfg['hidden_size']} "
          f"inter={header_cfg['intermediate_size']} heads={header_cfg['n_heads']} "
          f"kv_heads={header_cfg['n_kv_heads']} head_dim={header_cfg['head_dim']} "
          f"vocab={header_cfg['vocab_size']} tied={header_cfg['tied']}")
    print_table_summary(args.out)
    print(f"out        : {args.out}")
    print(f"file size  : {total} bytes ({total / 2 ** 20:.1f} MB) OK")


if __name__ == "__main__":
    main()
