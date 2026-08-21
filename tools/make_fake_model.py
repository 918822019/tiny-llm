#!/usr/bin/env python3
"""生成随机权重的小 .tqwen 文件，用于 C++ runtime 的冒烟测试。

配置刻意很小（hidden=16，2 层），让 loader、forward、KV cache、profiler
整条链路在毫秒级跑完。权重是随机的，生成结果没有语义，只用于验证
管道和二进制格式。

用法:
    python tools/make_fake_model.py --out /tmp/fake.tqwen [--seed 0]

本脚本生成的 fake 模型特点：
    - 极小配置：hidden=16, 2层, vocab=64，确保快速执行。
    - GQA 结构：4 个 Q head 共享 2 个 KV head（与真实 Qwen2.5 同构）。
    - RMSNorm 权重设为 1（中性缩放），避免随机 norm 权重导致数值溢出。
    - 其余权重从 N(0, 0.1²) 采样，保证良态范围。
    - 支持 tied/untied embedding 模式。

输入：无外部依赖（纯 numpy 随机生成）。
输出：单个 .tqwen 二进制文件。
"""

from __future__ import annotations

# --- 标准库导入 ---
import argparse   # 命令行参数解析
import numpy as np  # 随机数生成、数组操作

# 复用 export_qwen_to_tiny 的写出函数
from export_qwen_to_tiny import write_tqwen


def main() -> None:
    """主入口函数：生成随机权重的 fake Qwen2.5 小模型并写出 .tqwen 文件。

    流程：
    1. 解析命令行参数（输出路径、随机种子、tied 模式）。
    2. 定义极小的模型配置。
    3. 用随机数生成所有权重 tensor。
    4. 调用 write_tqwen 写出 .tqwen 文件。
    """
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="fake.tqwen")      # 输出文件路径
    p.add_argument("--seed", type=int, default=0)      # 随机种子（可复现）
    p.add_argument("--tied", type=int, default=1)      # 是否 tied embedding（1=tied, 0=untied）
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)  # 创建可复现的随机数生成器

    # 与真实 Qwen2.5 同构但极小的配置（GQA：4 个 q head 共享 2 个 kv head）。
    cfg = {
        "n_layers": 2,              # transformer 层数（极少，加速测试）
        "hidden_size": 16,          # 隐藏层维度（极小）
        "intermediate_size": 32,    # MLP 中间维度（通常为 hidden 的 2-4 倍）
        "n_heads": 4,               # 注意力 Q head 数
        "n_kv_heads": 2,            # KV head 数（GQA：4/2=2 个 Q head 共享 1 个 KV head）
        "head_dim": 4,              # 每个 head 的维度
        "vocab_size": 64,           # 词表大小
        "max_seq_len": 32,          # 最大序列长度
        "tied": args.tied,          # embedding/lm_head 是否共享权重
        "rms_norm_eps": 1e-6,       # RMSNorm epsilon
        "rope_theta": 10000.0,      # RoPE base frequency
    }
    H, I = cfg["hidden_size"], cfg["intermediate_size"]  # 快捷变量
    qd = cfg["n_heads"] * cfg["head_dim"]     # Q 投影总维度 = n_heads × head_dim
    kvd = cfg["n_kv_heads"] * cfg["head_dim"]  # KV 投影总维度 = n_kv_heads × head_dim
    V = cfg["vocab_size"]  # 词表大小

    def w(shape):
        """生成指定形状的随机权重数组。

        从标准正态分布采样后乘以 0.1，使权重值在 [-0.3, 0.3] 范围内，
        保证前向推理时激活值不会溢出。

        Args:
            shape: 目标形状的元组。

        Returns:
            fp32 numpy 数组。
        """
        return (rng.standard_normal(shape) * 0.1).astype(np.float32)

    # ===== 构建所有 tensor =====
    tensors = {"model.embed_tokens.weight": w((V, H))}  # embedding 表：[vocab, hidden]
    for i in range(cfg["n_layers"]):
        pfx = f"model.layers.{i}."  # 第 i 层的名字前缀
        # RMSNorm 权重取 1（中性缩放）：让激活保持在良态范围内，
        # 对齐测试检验的是数学实现，而不是溢出行为。
        tensors[pfx + "input_layernorm.weight"] = np.ones(H, dtype=np.float32)  # 输入 RMSNorm
        tensors[pfx + "self_attn.q_proj.weight"] = w((qd, H))   # Q 投影：[q_dim, hidden]
        tensors[pfx + "self_attn.k_proj.weight"] = w((kvd, H))  # K 投影：[kv_dim, hidden]
        tensors[pfx + "self_attn.v_proj.weight"] = w((kvd, H))  # V 投影：[kv_dim, hidden]
        # Qwen2/2.5 的 attention 带 q/k/v bias（attention_bias=True）。
        tensors[pfx + "self_attn.q_proj.bias"] = w((qd,))    # Q 偏置：[q_dim]
        tensors[pfx + "self_attn.k_proj.bias"] = w((kvd,))   # K 偏置：[kv_dim]
        tensors[pfx + "self_attn.v_proj.bias"] = w((kvd,))   # V 偏置：[kv_dim]
        tensors[pfx + "self_attn.o_proj.weight"] = w((H, qd))  # O 投影：[hidden, q_dim]
        tensors[pfx + "post_attention_layernorm.weight"] = np.ones(H, dtype=np.float32)  # 后 RMSNorm
        tensors[pfx + "mlp.gate_proj.weight"] = w((I, H))    # MLP gate：[inter, hidden]
        tensors[pfx + "mlp.up_proj.weight"] = w((I, H))      # MLP up：[inter, hidden]
        tensors[pfx + "mlp.down_proj.weight"] = w((H, I))    # MLP down：[hidden, inter]
    tensors["model.norm.weight"] = np.ones(H, dtype=np.float32)  # 最终 RMSNorm（全 1）
    if not args.tied:
        # 非 tied 模式：单独生成 lm_head 权重
        tensors["lm_head.weight"] = w((V, H))  # lm_head：[vocab, hidden]

    # 调用 write_tqwen 写出 .tqwen 文件
    total = write_tqwen(args.out, cfg, tensors)
    print(f"wrote {args.out}: {total} bytes, {len(tensors)} tensors")  # 打印结果摘要


if __name__ == "__main__":
    main()  # 脚本入口点
