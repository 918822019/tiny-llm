#!/usr/bin/env python3
"""生成随机权重的 Qwen3.5 混合架构小 .tqwen（v2）文件，供 C++ runtime 冒烟测试。

与 make_fake_model.py（Qwen2.x）同思路：配置刻意很小，让 loader、混合层
forward、GDN 递归状态、partial RoPE 整条链路毫秒级跑完。权重随机，生成结果
无语义，只验证管道与 v2 二进制格式。

同时把 HF 格式的 state_dict 存成 .pt，供 align_fake_qwen35_model.py 直接灌进
HF Qwen3_5ForCausalLM 做数值对齐（避免 tiny<->HF 往返引入歧义）。

用法:
    python tools/make_fake_qwen35_model.py --out /tmp/fake35.tqwen \
        --hf-out /tmp/fake35_hf.pt [--seed 0]
"""

from __future__ import annotations

import argparse
import numpy as np

from export_qwen_to_tiny import write_tqwen, MODEL_QWEN35

# 极小但结构完整的 Qwen3.5 配置：4 层 = 3 linear + 1 full（interval=4）。
# head_dim=16、partial_rotary_factor=0.25 -> rotary_dim=4（偶数，合法）。
FAKE_CFG = dict(
    hidden_size=16,
    intermediate_size=32,
    num_hidden_layers=4,
    num_attention_heads=2,
    num_key_value_heads=1,
    head_dim=16,
    vocab_size=64,
    max_position_embeddings=32,
    rms_norm_eps=1e-6,
    linear_num_key_heads=2,
    linear_num_value_heads=2,
    linear_key_head_dim=8,
    linear_value_head_dim=8,
    linear_conv_kernel_dim=4,
    full_attention_interval=4,
    tie_word_embeddings=True,
    attention_bias=False,
    rope_theta=10000.0,
    partial_rotary_factor=0.25,
    eos_token_id=63,
)


def _is_zero_centered_norm(name: str) -> bool:
    """Qwen3.5 里用 Qwen3_5RMSNorm（(1+w)*norm）的权重：导出/造fake 时要 +1。
    linear_attn.norm 是 RMSNormGated（标准 w*norm），不在此列。"""
    if name == "model.norm.weight":
        return True
    return any(name.endswith(s) for s in (
        "input_layernorm.weight", "post_attention_layernorm.weight",
        "self_attn.q_norm.weight", "self_attn.k_norm.weight"))


def make_config():
    """构造与 FAKE_CFG 对应的 Qwen3_5TextConfig（对齐脚本复用，保证一致）。"""
    from transformers import Qwen3_5TextConfig
    return Qwen3_5TextConfig(
        hidden_size=FAKE_CFG["hidden_size"],
        intermediate_size=FAKE_CFG["intermediate_size"],
        num_hidden_layers=FAKE_CFG["num_hidden_layers"],
        num_attention_heads=FAKE_CFG["num_attention_heads"],
        num_key_value_heads=FAKE_CFG["num_key_value_heads"],
        head_dim=FAKE_CFG["head_dim"],
        vocab_size=FAKE_CFG["vocab_size"],
        max_position_embeddings=FAKE_CFG["max_position_embeddings"],
        rms_norm_eps=FAKE_CFG["rms_norm_eps"],
        linear_num_key_heads=FAKE_CFG["linear_num_key_heads"],
        linear_num_value_heads=FAKE_CFG["linear_num_value_heads"],
        linear_key_head_dim=FAKE_CFG["linear_key_head_dim"],
        linear_value_head_dim=FAKE_CFG["linear_value_head_dim"],
        linear_conv_kernel_dim=FAKE_CFG["linear_conv_kernel_dim"],
        full_attention_interval=FAKE_CFG["full_attention_interval"],
        tie_word_embeddings=FAKE_CFG["tie_word_embeddings"],
        attention_bias=FAKE_CFG["attention_bias"],
        rope_parameters={"rope_type": "default",
                         "rope_theta": FAKE_CFG["rope_theta"],
                         "partial_rotary_factor": FAKE_CFG["partial_rotary_factor"]},
    )


def build_hf_state(seed: int) -> "dict[str, np.ndarray]":
    """构造受控的随机 HF state_dict（形状取自真实实例化，保证无误）。"""
    from transformers import Qwen3_5ForCausalLM

    cfg = make_config()
    model = Qwen3_5ForCausalLM(cfg)
    layer_types = cfg.layer_types

    rng = np.random.default_rng(seed)

    def small(shape):
        return (rng.standard_normal(shape) * 0.1).astype(np.float32)

    state = {}
    for name, t in model.state_dict().items():
        shape = tuple(t.shape)
        if _is_zero_centered_norm(name):
            arr = np.zeros(shape, np.float32)  # (1+0)=1，中性
        elif name.endswith("linear_attn.norm.weight"):
            arr = np.ones(shape, np.float32)   # RMSNormGated，标准缩放
        elif name.endswith("linear_attn.A_log"):
            arr = np.log(rng.uniform(0.5, 2.0, shape)).astype(np.float32)
        elif name.endswith("linear_attn.dt_bias"):
            arr = np.zeros(shape, np.float32)
        elif name.endswith("linear_attn.conv1d.weight"):
            arr = small(shape)
        else:
            arr = small(shape)
        state[name] = arr
    # 记录层类型供调试/对齐脚本参考。
    state["__layer_types__"] = np.array(layer_types)
    return state, cfg


def hf_to_tiny(state: "dict[str, np.ndarray]") -> "dict[str, np.ndarray]":
    """HF state_dict -> .tqwen tensor 映射（与 exporter 的变换一致）。"""
    tiny = {}
    for name, arr in state.items():
        if name.startswith("__"):
            continue
        if _is_zero_centered_norm(name):
            tiny[name] = arr + 1.0
        elif name.endswith("linear_attn.conv1d.weight"):
            tiny[name] = arr.reshape(arr.shape[0], arr.shape[2])  # [d,1,k]->[d,k]
        else:
            tiny[name] = arr
    return tiny


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="fake35.tqwen")
    p.add_argument("--hf-out", default="fake35_hf.pt")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    import torch

    state, cfg = build_hf_state(args.seed)
    layer_types = list(state.pop("__layer_types__"))

    # 存 HF 权重（对齐脚本直接 load_state_dict）。
    torch.save({k: torch.from_numpy(v) for k, v in state.items()}, args.hf_out)

    tiny = hf_to_tiny(state)

    header_cfg = {
        "n_layers": FAKE_CFG["num_hidden_layers"],
        "hidden_size": FAKE_CFG["hidden_size"],
        "intermediate_size": FAKE_CFG["intermediate_size"],
        "n_heads": FAKE_CFG["num_attention_heads"],
        "n_kv_heads": FAKE_CFG["num_key_value_heads"],
        "head_dim": FAKE_CFG["head_dim"],
        "vocab_size": FAKE_CFG["vocab_size"],
        "max_seq_len": FAKE_CFG["max_position_embeddings"],
        "tied": 1 if FAKE_CFG["tie_word_embeddings"] else 0,
        "rms_norm_eps": FAKE_CFG["rms_norm_eps"],
        "rope_theta": FAKE_CFG["rope_theta"],
    }
    ext = {
        "model_type": MODEL_QWEN35,
        "linear_num_qk_heads": FAKE_CFG["linear_num_key_heads"],
        "linear_num_v_heads": FAKE_CFG["linear_num_value_heads"],
        "linear_qk_head_dim": FAKE_CFG["linear_key_head_dim"],
        "linear_v_head_dim": FAKE_CFG["linear_value_head_dim"],
        "linear_conv_kernel_dim": FAKE_CFG["linear_conv_kernel_dim"],
        "full_attention_interval": FAKE_CFG["full_attention_interval"],
        "partial_rotary_factor": FAKE_CFG["partial_rotary_factor"],
        "eos_token_id": FAKE_CFG["eos_token_id"],
    }
    total = write_tqwen(args.out, header_cfg, tiny, "f32", version=2, ext=ext)
    print(f"wrote {args.out}: {total} bytes, {len(tiny)} tensors, "
          f"layer_types={layer_types}")
    print(f"wrote {args.hf_out}: HF state_dict for alignment")


if __name__ == "__main__":
    main()
