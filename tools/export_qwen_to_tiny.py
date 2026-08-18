#!/usr/bin/env python3
"""将 Qwen2.5（HF）权重导出为 tinyqwen 扁平二进制格式（.tqwen）。

二进制契约定义在 runtime/tiny_format.h 和 docs/weight_format.md，
下面的 struct 布局必须与该头文件保持同步。

用法:
    python tools/export_qwen_to_tiny.py \
        --model Qwen/Qwen2.5-0.5B \
        --out model.tqwen

说明:
    - v1 支持导出 float32（默认）或 float16（weight-only 半精度，文件减半，
      配 runtime 的 f16 matvec 实现使用；数值容差见 docs/optimization.md §5）。
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

import numpy as np

# 必须与 runtime/tiny_format.h 保持一致 -------------------------------
MAGIC = b"TINYQWEN"
FORMAT_VERSION = 2  # 当前最高版本（v1 文件布局不变，按 model_type 选择）
FORMAT_VERSION_MIN = 1
ALIGN = 64
DTYPE_F32 = 0
DTYPE_F16 = 1  # 与 runtime/tiny_format.h 的 Dtype 枚举保持一致
DTYPE_I4 = 3
MAX_NAME = 64

# 架构族（TinyHeaderV2Ext.model_type）。
MODEL_QWEN2 = 0
MODEL_QWEN35 = 1

# dtype -> (编号, 每元素字节数, numpy 类型名)
DTYPES = {
    "f32": (DTYPE_F32, 4, "float32"),
    "f16": (DTYPE_F16, 2, "float16"),
}

HEADER_FMT = "<8s12Iff4Q96s"  # 192 字节
ENTRY_FMT = "<64sII4QQQ"  # 120 字节
# v2 扩展块（TinyHeaderV2Ext）：放进 header.reserved 的前 64 字节。
# 字段顺序须与 C++ 结构体一致：7 个 u32 -> float partial_rotary_factor ->
# u32 eos_token_id -> u32 pad -> 24 字节 reserved。
EXT_FMT = "<7If2I24s"  # 64 字节

assert struct.calcsize(HEADER_FMT) == 192
assert struct.calcsize(ENTRY_FMT) == 120
assert struct.calcsize(EXT_FMT) == 64


# ---------------------------------------------------------------------


def align_up(x: int, align: int = ALIGN) -> int:
    return (x + align - 1) // align * align


def pack_v2_ext(ext: dict) -> bytes:
    """把 v2 扩展字段打包成 64 字节（TinyHeaderV2Ext）。ext 缺省字段填 0。"""
    g = lambda k: ext.get(k, 0)  # noqa: E731
    return struct.pack(
        EXT_FMT,
        int(g("model_type")),
        int(g("linear_num_qk_heads")),
        int(g("linear_num_v_heads")),
        int(g("linear_qk_head_dim")),
        int(g("linear_v_head_dim")),
        int(g("linear_conv_kernel_dim")),
        int(g("full_attention_interval")),
        float(g("partial_rotary_factor")),
        int(g("eos_token_id")),
        0,  # pad
        b"\x00" * 24,
    )


def unpack_v2_ext(reserved: bytes) -> dict:
    """从 header.reserved 解出 v2 扩展字段（对齐/读取工具用）。"""
    (model_type, qk_h, v_h, qk_d, v_d, conv, interval,
     rotary, eos, _pad, _rsv) = struct.unpack(EXT_FMT, reserved[: struct.calcsize(EXT_FMT)])
    return {
        "model_type": model_type,
        "linear_num_qk_heads": qk_h, "linear_num_v_heads": v_h,
        "linear_qk_head_dim": qk_d, "linear_v_head_dim": v_d,
        "linear_conv_kernel_dim": conv, "full_attention_interval": interval,
        "partial_rotary_factor": rotary, "eos_token_id": eos,
    }


def write_tqwen(out_path: str | Path, cfg: dict, tensors: "dict[str, object]",
                dtype: str = "f32", version: int = 1, ext: dict | None = None) -> int:
    """写一个 .tqwen 文件，返回文件总字节数。

    cfg 必需的键：
        n_layers, hidden_size, intermediate_size, n_heads, n_kv_heads,
        head_dim, vocab_size, max_seq_len, tied (0/1), rms_norm_eps, rope_theta
    tensors: 有序映射 name -> 数组对象，要求有 .shape 属性以及
        .astype(<dtype>).tobytes() 方法（numpy 数组可直接使用）。
    dtype: "f32"（默认）或 "f16"；决定 header/tensor 表的 dtype 字段与元素大小。
    version: 1（Qwen2.x）或 2（混合架构，需配 ext）。
    ext: v2 扩展字段 dict（见 pack_v2_ext）；version>=2 时写进 header.reserved。
    """
    if version < FORMAT_VERSION_MIN or version > FORMAT_VERSION:
        sys.exit(f"error: unsupported format version {version}")
    dtype_code, elem_size, np_dtype = DTYPES[dtype]
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
        nbytes = numel * elem_size
        entries.append((name, shape, offset, nbytes))
        offset = align_up(offset + nbytes)
    total_bytes = offset

    # reserved 区：v2 时前 64 字节放扩展块，其余补 0；v1 全 0。
    reserved = b"\x00" * 96
    if version >= 2:
        ext_bytes = pack_v2_ext(ext or {})
        reserved = ext_bytes + b"\x00" * (96 - len(ext_bytes))

    header = struct.pack(
        HEADER_FMT,
        MAGIC,
        version,
        dtype_code,
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
        reserved,
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
                dtype_code,
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
            data = tensors[name].astype(np_dtype, copy=False).tobytes()
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
        assert magic == MAGIC
        assert FORMAT_VERSION_MIN <= version <= FORMAT_VERSION, f"bad version {version}"
        assert dtype in (DTYPE_F32, DTYPE_F16, DTYPE_I4)
        # I4 文件是混合 dtype（大矩阵 i4、小向量 f32），逐 tensor 取自己的标签。
        dtype_names = {DTYPE_F32: "f32", DTYPE_F16: "f16", DTYPE_I4: "i4"}
        # fields 下标: 0 magic, 1 version, 2 dtype, 3..12 十个 u32,
        #   13 eps, 14 theta, 15 tensor_count, 16 tensor_table_offset,
        #   17 data_offset, 18 total, 19 reserved
        if version >= 2:
            ext = struct.unpack(EXT_FMT, fields[19][: struct.calcsize(EXT_FMT)])
            model_type = ext[0]
            label = {MODEL_QWEN2: "qwen2", MODEL_QWEN35: "qwen3_5"}.get(model_type, "?")
            print(f"format v{version} model_type={label} "
                  f"attn_interval={ext[6]} rotary={ext[7]:g} eos={ext[8]}")
            if model_type == MODEL_QWEN35:
                print(f"  linear: qk_heads={ext[1]} v_heads={ext[2]} "
                      f"qk_dim={ext[3]} v_dim={ext[4]} conv={ext[5]}")
        tensor_count = fields[15]
        table_offset = fields[16]
        f.seek(table_offset)
        print(f"{'name':<56} {'shape':<22} {'dtype':<6} {'offset':>12} {'nbytes':>14}")
        for _ in range(tensor_count):
            name_b, t_dtype, ndim, s0, s1, s2, s3, off, nbytes = struct.unpack(
                ENTRY_FMT, f.read(struct.calcsize(ENTRY_FMT)))
            name = name_b.rstrip(b"\x00").decode("ascii")
            shape = [s0, s1, s2, s3][:ndim]
            label = dtype_names.get(t_dtype, f"?{t_dtype}")
            print(f"{name:<56} {str(shape):<22} {label:<6} {off:>12} {nbytes:>14}")


# ---- HF 模型收集 ------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True,
                   help="HF model dir or repo id (must contain config.json)")
    p.add_argument("--out", required=True, help="output .tqwen path")
    p.add_argument("--max-seq-len", type=int, default=0,
                   help="override max_seq_len in header (0 = use config)")
    p.add_argument("--dtype", choices=sorted(DTYPES), default="f32",
                   help="权重存储精度：f32（默认）或 f16（文件减半，配 f16 matvec 用）")
    return p.parse_args()


def find_local_dir(model: str) -> Path:
    """解析本地模型目录。repo id 的下载留给用户自己执行
    （如 `huggingface-cli download`），让本脚本保持轻依赖。"""
    path = Path(model)
    if path.is_dir() and (path / "config.json").exists():
        return path
    sys.exit(f"error: {model} is not a local dir with config.json. "
             f"Download it first, e.g. `huggingface-cli download {model} --local-dir {model}`")


def detect_model_type(cfg: dict) -> int:
    """从 HF config 判定架构族。Qwen3.5 是多模态壳（ForConditionalGeneration），
    文本塔参数在 text_config 里，model_type 为 qwen3_5。"""
    mt = cfg.get("model_type", "")
    if mt in ("qwen3_5", "qwen3_5_moe"):
        return MODEL_QWEN35
    if mt in ("qwen2", "qwen2_5", "qwen3"):
        return MODEL_QWEN2
    sys.exit(f"error: unsupported model_type '{mt}' (expected qwen2* / qwen3_5)")


def text_cfg_of(cfg: dict, model_type: int) -> dict:
    """取出纯文本塔的配置子树。Qwen3.5 在 text_config，Qwen2.x 在顶层。"""
    if model_type == MODEL_QWEN35:
        if "text_config" not in cfg:
            sys.exit("error: qwen3_5 config missing text_config")
        return cfg["text_config"]
    return cfg


def load_config(model_dir: Path) -> tuple[dict, int]:
    """读取 HF config.json，返回 (文本塔配置, model_type)。

    校验导出所需字段；head_dim / rope_theta 缺省时推导。Qwen3.5 的
    rope_theta 在 rope_parameters 里。"""
    cfg = json.loads((model_dir / "config.json").read_text())
    model_type = detect_model_type(cfg)
    tcfg = dict(text_cfg_of(cfg, model_type))

    required = ["hidden_size", "intermediate_size", "num_hidden_layers",
                "num_attention_heads", "num_key_value_heads", "vocab_size",
                "rms_norm_eps", "max_position_embeddings"]
    missing = [k for k in required if k not in tcfg]
    if missing:
        sys.exit(f"error: config missing fields: {missing}")

    head_dim = tcfg.get("head_dim")
    if head_dim is None:
        head_dim = tcfg["hidden_size"] // tcfg["num_attention_heads"]
    tcfg["head_dim"] = head_dim
    if tcfg["num_attention_heads"] % tcfg["num_key_value_heads"] != 0:
        sys.exit("error: num_attention_heads % num_key_value_heads != 0")

    # rope_theta：Qwen2.x 顶层直给；Qwen3.5 在 rope_parameters.rope_theta。
    if "rope_theta" not in tcfg:
        rp = tcfg.get("rope_parameters") or {}
        if "rope_theta" not in rp:
            sys.exit("error: cannot find rope_theta in config")
        tcfg["rope_theta"] = rp["rope_theta"]
    return tcfg, model_type


def _fold_one(a):
    return a + 1.0


def _squeeze_conv(a):
    # HF depthwise conv1d 权重 [out, 1, kernel] -> [out, kernel]
    arr = np.asarray(a)
    assert arr.ndim == 3 and arr.shape[1] == 1, f"unexpected conv shape {arr.shape}"
    return arr.reshape(arr.shape[0], arr.shape[2])


def plan_tensors(tcfg: dict, model_type: int):
    """构造导出计划：返回 (tiny_names, src_of, transform_of)。

    tiny_names: 写入 .tqwen 的名字序（== C++ loader 期望的名字）。
    src_of[tiny] : HF safetensors 里的源 tensor 名。
    transform_of[tiny]: 可选的数组变换（fold +1 / squeeze conv / None）。

    Qwen3.5 的 HF 权重带 model.language_model. 前缀，写入时统一去掉
    language_model. 以复用 Qwen2.x 的命名约定。zero-centered RMSNorm
    （input/post/model/q_norm/k_norm）在导出时把 +1 折进权重，
    linear_attn.norm 是普通 RMSNormGated 不折。
    """
    names: list[str] = []
    src_of: dict[str, str] = {}
    transform_of: dict[str, object] = {}

    def add(tiny: str, hf: str, transform=None):
        names.append(tiny)
        src_of[tiny] = hf
        transform_of[tiny] = transform

    if model_type == MODEL_QWEN35:
        prefix = "model.language_model."
        add("model.embed_tokens.weight", prefix + "embed_tokens.weight")
        layer_types = tcfg.get("layer_types") or []
        for i in range(tcfg["num_hidden_layers"]):
            lp = prefix + f"layers.{i}."
            tp = f"model.layers.{i}."
            is_linear = layer_types[i] == "linear_attention" if layer_types else \
                ((i + 1) % tcfg.get("full_attention_interval", 4) != 0)
            add(tp + "input_layernorm.weight", lp + "input_layernorm.weight", _fold_one)
            if is_linear:
                la, la_t = lp + "linear_attn.", tp + "linear_attn."
                add(la_t + "in_proj_qkv.weight", la + "in_proj_qkv.weight")
                add(la_t + "in_proj_z.weight", la + "in_proj_z.weight")
                add(la_t + "in_proj_b.weight", la + "in_proj_b.weight")
                add(la_t + "in_proj_a.weight", la + "in_proj_a.weight")
                add(la_t + "out_proj.weight", la + "out_proj.weight")
                add(la_t + "conv1d.weight", la + "conv1d.weight", _squeeze_conv)
                add(la_t + "A_log", la + "A_log")
                add(la_t + "dt_bias", la + "dt_bias")
                add(la_t + "norm.weight", la + "norm.weight")
            else:
                sa, sa_t = lp + "self_attn.", tp + "self_attn."
                add(sa_t + "q_proj.weight", sa + "q_proj.weight")
                add(sa_t + "k_proj.weight", sa + "k_proj.weight")
                add(sa_t + "v_proj.weight", sa + "v_proj.weight")
                add(sa_t + "o_proj.weight", sa + "o_proj.weight")
                add(sa_t + "q_norm.weight", sa + "q_norm.weight", _fold_one)
                add(sa_t + "k_norm.weight", sa + "k_norm.weight", _fold_one)
            add(tp + "post_attention_layernorm.weight",
                lp + "post_attention_layernorm.weight", _fold_one)
            add(tp + "mlp.gate_proj.weight", lp + "mlp.gate_proj.weight")
            add(tp + "mlp.up_proj.weight", lp + "mlp.up_proj.weight")
            add(tp + "mlp.down_proj.weight", lp + "mlp.down_proj.weight")
        add("model.norm.weight", prefix + "norm.weight", _fold_one)
        if not tcfg.get("tie_word_embeddings", False):
            # Qwen3.5 0.8B 为 tied embeddings，正常不会走到这里。
            add("lm_head.weight", "lm_head.weight")
    else:
        add("model.embed_tokens.weight", "model.embed_tokens.weight")
        for i in range(tcfg["num_hidden_layers"]):
            p = f"model.layers.{i}."
            add(p + "input_layernorm.weight", p + "input_layernorm.weight")
            add(p + "self_attn.q_proj.weight", p + "self_attn.q_proj.weight")
            add(p + "self_attn.k_proj.weight", p + "self_attn.k_proj.weight")
            add(p + "self_attn.v_proj.weight", p + "self_attn.v_proj.weight")
            add(p + "self_attn.q_proj.bias", p + "self_attn.q_proj.bias")
            add(p + "self_attn.k_proj.bias", p + "self_attn.k_proj.bias")
            add(p + "self_attn.v_proj.bias", p + "self_attn.v_proj.bias")
            add(p + "self_attn.o_proj.weight", p + "self_attn.o_proj.weight")
            add(p + "post_attention_layernorm.weight", p + "post_attention_layernorm.weight")
            add(p + "mlp.gate_proj.weight", p + "mlp.gate_proj.weight")
            add(p + "mlp.up_proj.weight", p + "mlp.up_proj.weight")
            add(p + "mlp.down_proj.weight", p + "mlp.down_proj.weight")
        add("model.norm.weight", "model.norm.weight")
        if not tcfg.get("tie_word_embeddings", False):
            add("lm_head.weight", "lm_head.weight")
    return names, src_of, transform_of


def build_shard_map(model_dir: Path, src_names: list[str]) -> dict[str, Path]:
    """把每个 HF 源 tensor 名映射到包含它的 safetensors 分片文件。"""
    index_file = model_dir / "model.safetensors.index.json"
    if index_file.exists():
        index = json.loads(index_file.read_text())["weight_map"]
        return {n: model_dir / index[n] for n in src_names}
    single = model_dir / "model.safetensors"
    if single.exists():
        return {n: single for n in src_names}
    sys.exit("error: no model.safetensors[.index.json] found")


def main() -> None:
    args = parse_args()
    model_dir = find_local_dir(args.model)
    cfg, model_type = load_config(model_dir)
    names, src_of, transform_of = plan_tensors(cfg, model_type)
    shard_map = build_shard_map(model_dir, list(src_of.values()))

    from safetensors import safe_open  # 延迟导入：晚失败、报错清晰

    opened: dict[Path, object] = {}

    def get_shard(path: Path):
        if path not in opened:
            # 用 torch 框架读：真实 Qwen 权重是 bf16，numpy 不认识 bf16；
            # 读出来后在 LazyTensors 里统一转成 fp32。
            opened[path] = safe_open(str(path), framework="torch")
        return opened[path]

    # 先校验所有源 tensor 都存在（任何缺失都在写文件之前报错）。
    for tiny in names:
        src = src_of[tiny]
        f = get_shard(shard_map[src])
        if src not in f.keys():
            sys.exit(f"error: tensor {src} not found in {shard_map[src]}")

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

    # v2 扩展字段（仅 qwen3_5）。
    version = 1
    ext = None
    if model_type == MODEL_QWEN35:
        version = 2
        eos = cfg.get("eos_token_id", 0)
        if isinstance(eos, (list, tuple)):  # 多 eos 取第一个
            eos = eos[0]
        ext = {
            "model_type": MODEL_QWEN35,
            "linear_num_qk_heads": cfg["linear_num_key_heads"],
            "linear_num_v_heads": cfg["linear_num_value_heads"],
            "linear_qk_head_dim": cfg["linear_key_head_dim"],
            "linear_v_head_dim": cfg["linear_value_head_dim"],
            "linear_conv_kernel_dim": cfg["linear_conv_kernel_dim"],
            "full_attention_interval": cfg.get("full_attention_interval", 4),
            "partial_rotary_factor": cfg.get("partial_rotary_factor",
                                             (cfg.get("rope_parameters") or {})
                                             .get("partial_rotary_factor", 1.0)),
            "eos_token_id": int(eos),
        }

    # 逐个流式读取 tensor，保持内存占用平稳：write_tqwen 需要一个映射，
    # 所以这里传给它一个惰性 dict，访问时才从对应分片加载源 tensor 并做变换。
    # 注意每个 tensor 会被读两次（shape 一遍 + payload 一遍）——
    # 相比把 2GB fp32 模型整个放进内存，这是可接受的代价。
    np_dtype = DTYPES[args.dtype][2]

    class LazyTensors(dict):
        def __getitem__(self, key):
            src = src_of[key]
            tensor = get_shard(shard_map[src]).get_tensor(src)
            # bf16/fp16 -> fp32 numpy；f16 导出时再降到 float16
            # （round-to-nearest-even，numpy astype 的默认舍入）。
            arr = tensor.float().numpy()
            tr = transform_of.get(key)
            if tr is not None:
                arr = tr(arr)
            return np.asarray(arr).astype(np_dtype, copy=False)

    total = write_tqwen(args.out, header_cfg, LazyTensors.fromkeys(names), args.dtype,
                        version=version, ext=ext)

    mt_label = "qwen3_5" if model_type == MODEL_QWEN35 else "qwen2"
    print(f"model dir  : {model_dir}")
    print(f"model type : {mt_label} (format v{version})")
    print(f"dtype      : {args.dtype}")
    print(f"config     : layers={header_cfg['n_layers']} hidden={header_cfg['hidden_size']} "
          f"inter={header_cfg['intermediate_size']} heads={header_cfg['n_heads']} "
          f"kv_heads={header_cfg['n_kv_heads']} head_dim={header_cfg['head_dim']} "
          f"vocab={header_cfg['vocab_size']} tied={header_cfg['tied']}")
    print_table_summary(args.out)
    print(f"out        : {args.out}")
    print(f"file size  : {total} bytes ({total / 2 ** 20:.1f} MB) OK")


if __name__ == "__main__":
    main()
