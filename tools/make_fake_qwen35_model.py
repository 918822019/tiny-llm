#!/usr/bin/env python3
"""生成随机权重的 Qwen3.5 混合架构小 .tqwen（v2）文件，供 C++ runtime 冒烟测试。

与 make_fake_model.py（Qwen2.x）同思路：配置刻意很小，让 loader、混合层
forward、GDN 递归状态、partial RoPE 整条链路毫秒级跑完。权重随机，生成结果
无语义，只验证管道与 v2 二进制格式。

同时把 HF 格式的 state_dict 存成 .pt，供 align_fake_qwen35_model.py 直接灌进
HF Qwen3_5ForCausalLM 做数值对齐（避免 tiny<->HF 往返引入歧义）。

用法:
    python tools/make_fake_qwen35_model.py --out /tmp/fake35.tqwen \\
        --hf-out /tmp/fake35_hf.pt [--seed 0]

核心设计要点：
    1. 先生成 HF 格式的 state_dict（通过实例化 Qwen3_5ForCausalLM 获取正确的
       tensor 名字和形状），再转换为 .tqwen 格式。这保证了两侧权重完全一致。
    2. zero-centered RMSNorm 权重在 HF 侧初始化为 0（实际缩放 = 1+0 = 1），
       导出到 .tqwen 时 +1 折叠（与真实导出器行为一致）。
    3. conv1d 权重在 HF 侧是 [out, 1, kernel] 的 3D 形状，导出时 squeeze 为
       [out, kernel] 的 2D 形状。
    4. tied embedding 模型不写独立的 lm_head.weight（runtime 会自动绑定 embed）。

输入：无外部依赖（纯 numpy 随机 + transformers 获取 tensor 结构）。
输出：.tqwen 文件 + HF state_dict .pt 文件。
"""

from __future__ import annotations

# --- 标准库导入 ---
import argparse   # 命令行参数解析
import numpy as np  # 随机数生成、数组操作

# 复用 export_qwen_to_tiny 的写出函数和架构族常量
from export_qwen_to_tiny import write_tqwen, MODEL_QWEN35

# 极小但结构完整的 Qwen3.5 配置：4 层 = 3 linear + 1 full（interval=4）。
# head_dim=16、partial_rotary_factor=0.25 -> rotary_dim=4（偶数，合法）。
FAKE_CFG = dict(
    hidden_size=16,                    # 隐藏层维度（极小）
    intermediate_size=32,              # MLP 中间维度
    num_hidden_layers=4,               # 4 层：3 个 linear attention + 1 个 full attention
    num_attention_heads=2,             # full attention 的 Q head 数
    num_key_value_heads=1,             # full attention 的 KV head 数（GQA）
    head_dim=16,                       # full attention 每个 head 的维度
    vocab_size=64,                     # 词表大小
    max_position_embeddings=64,        # 最大位置编码长度（批量 prefill 对齐需 >=32 prompt）
    rms_norm_eps=1e-6,                 # RMSNorm epsilon
    linear_num_key_heads=2,            # 线性注意力 QK head 数
    linear_num_value_heads=2,          # 线性注意力 V head 数
    linear_key_head_dim=8,             # 线性注意力 QK head 维度
    linear_value_head_dim=8,           # 线性注意力 V head 维度
    linear_conv_kernel_dim=4,          # depthwise conv1d 核大小
    full_attention_interval=4,         # 每 4 层一个 full attention（第 4 层）
    tie_word_embeddings=True,          # tied embedding（embed 和 lm_head 共享权重）
    attention_bias=False,              # Qwen3.5 不使用 attention bias
    rope_theta=10000.0,                # RoPE base frequency
    partial_rotary_factor=0.25,        # 部分 RoPE：只对 25% 的 head 维度应用旋转编码
    eos_token_id=63,                   # EOS token ID
)


def _is_zero_centered_norm(name: str) -> bool:
    """判断给定 tensor 名是否是 zero-centered RMSNorm 权重。

    Qwen3.5 里用 Qwen3_5RMSNorm（(1+w)*norm）的权重：导出/造 fake 时要 +1。
    linear_attn.norm 是 RMSNormGated（标准 w*norm），不在此列。

    zero-centered RMSNorm 包括：
    - model.norm.weight（最终 norm）
    - input_layernorm.weight（每层输入 norm）
    - post_attention_layernorm.weight（每层后 norm）
    - self_attn.q_norm.weight（full attention 的 Q norm）
    - self_attn.k_norm.weight（full attention 的 K norm）

    Args:
        name: tensor 名字。

    Returns:
        True 表示是 zero-centered RMSNorm，需要 +1 折叠。
    """
    if name == "model.norm.weight":
        return True  # 最终 norm 是 zero-centered
    # 检查是否以已知的 zero-centered norm 后缀结尾
    return any(name.endswith(s) for s in (
        "input_layernorm.weight",              # 输入层 norm
        "post_attention_layernorm.weight",     # 后注意力 norm
        "self_attn.q_norm.weight",             # Q norm
        "self_attn.k_norm.weight"))            # K norm


def make_config():
    """构造与 FAKE_CFG 对应的 Qwen3_5TextConfig（对齐脚本复用，保证一致）。

    返回 HF transformers 的 Qwen3_5TextConfig 对象，其字段值与 FAKE_CFG
    完全对应。rope_theta 和 partial_rotary_factor 放在 rope_parameters 子树中，
    这是 Qwen3.5 HF config 的标准布局。

    Returns:
        Qwen3_5TextConfig 实例。
    """
    from transformers import Qwen3_5TextConfig  # 延迟导入
    return Qwen3_5TextConfig(
        hidden_size=FAKE_CFG["hidden_size"],                           # 隐藏层维度
        intermediate_size=FAKE_CFG["intermediate_size"],               # MLP 中间维度
        num_hidden_layers=FAKE_CFG["num_hidden_layers"],               # 层数
        num_attention_heads=FAKE_CFG["num_attention_heads"],           # Q head 数
        num_key_value_heads=FAKE_CFG["num_key_value_heads"],           # KV head 数
        head_dim=FAKE_CFG["head_dim"],                                 # head 维度
        vocab_size=FAKE_CFG["vocab_size"],                             # 词表大小
        max_position_embeddings=FAKE_CFG["max_position_embeddings"],   # 最大位置编码
        rms_norm_eps=FAKE_CFG["rms_norm_eps"],                         # RMSNorm eps
        linear_num_key_heads=FAKE_CFG["linear_num_key_heads"],         # 线性 QK head 数
        linear_num_value_heads=FAKE_CFG["linear_num_value_heads"],     # 线性 V head 数
        linear_key_head_dim=FAKE_CFG["linear_key_head_dim"],           # 线性 QK head 维度
        linear_value_head_dim=FAKE_CFG["linear_value_head_dim"],       # 线性 V head 维度
        linear_conv_kernel_dim=FAKE_CFG["linear_conv_kernel_dim"],     # conv1d 核大小
        full_attention_interval=FAKE_CFG["full_attention_interval"],   # full attn 间隔
        tie_word_embeddings=FAKE_CFG["tie_word_embeddings"],           # tied embedding
        attention_bias=FAKE_CFG["attention_bias"],                     # attention bias
        # rope 参数放在嵌套字典中（Qwen3.5 HF config 的标准格式）
        rope_parameters={"rope_type": "default",
                         "rope_theta": FAKE_CFG["rope_theta"],
                         "partial_rotary_factor": FAKE_CFG["partial_rotary_factor"]},
    )


def build_hf_state(seed: int) -> "dict[str, np.ndarray]":
    """构造受控的随机 HF state_dict（形状取自真实实例化，保证无误）。

    通过实例化 Qwen3_5ForCausalLM 获取所有 tensor 的名字和形状，
    然后用随机数填充。不同类型的 tensor 使用不同的初始化策略：
    - zero-centered norm：全零（实际缩放 = 1+0 = 1，中性）
    - RMSNormGated（linear_attn.norm）：全一（标准缩放）
    - A_log：log(uniform(0.5, 2.0))，模拟真实的 SSM 状态矩阵
    - dt_bias：全零
    - 其余：N(0, 0.1²) 随机采样

    Args:
        seed: 随机种子。

    Returns:
        (state, cfg) 元组：
        - state: {tensor_name: numpy_array} 的 HF state_dict。
        - cfg: Qwen3_5TextConfig 实例。
    """
    from transformers import Qwen3_5ForCausalLM  # 延迟导入

    cfg = make_config()  # 获取 HF config
    model = Qwen3_5ForCausalLM(cfg)  # 实例化模型以获取正确的 tensor 结构
    layer_types = cfg.layer_types  # 获取每层的类型列表

    rng = np.random.default_rng(seed)  # 创建可复现的随机数生成器

    def small(shape):
        """生成 N(0, 0.1²) 的随机 fp32 数组。"""
        return (rng.standard_normal(shape) * 0.1).astype(np.float32)

    state = {}  # HF state_dict
    for name, t in model.state_dict().items():
        shape = tuple(t.shape)  # 获取该 tensor 的形状
        if _is_zero_centered_norm(name):
            arr = np.zeros(shape, np.float32)  # (1+0)=1，中性缩放
        elif name.endswith("linear_attn.norm.weight"):
            arr = np.ones(shape, np.float32)  # RMSNormGated，标准缩放（w*norm）
        elif name.endswith("linear_attn.A_log"):
            # SSM 状态矩阵 A 的对数：从 uniform(0.5, 2.0) 取 log，模拟真实分布
            arr = np.log(rng.uniform(0.5, 2.0, shape)).astype(np.float32)
        elif name.endswith("linear_attn.dt_bias"):
            arr = np.zeros(shape, np.float32)  # dt 偏置初始化为零
        elif name.endswith("linear_attn.conv1d.weight"):
            arr = small(shape)  # conv1d 权重：小随机值
        else:
            arr = small(shape)  # 其余权重：小随机值
        state[name] = arr
    # 记录层类型供调试/对齐脚本参考（用特殊前缀 __ 区分于正常 tensor）
    state["__layer_types__"] = np.array(layer_types)
    return state, cfg


def hf_to_tiny(state: "dict[str, np.ndarray]") -> "dict[str, np.ndarray]":
    """HF state_dict -> .tqwen tensor 映射（与 exporter 的变换一致）。

    对 HF 权重执行与 export_qwen_to_tiny.py 相同的变换：
    - zero-centered RMSNorm 权重 +1 折叠
    - conv1d 权重从 [out, 1, kernel] squeeze 为 [out, kernel]
    - 跳过内部元数据键（以 __ 开头）

    Args:
        state: HF state_dict（含 __layer_types__ 元数据）。

    Returns:
        .tqwen 格式的 tensor 字典。
    """
    tiny = {}  # .tqwen 格式的 tensor 字典
    for name, arr in state.items():
        if name.startswith("__"):
            continue  # 跳过内部元数据键
        if _is_zero_centered_norm(name):
            tiny[name] = arr + 1.0  # zero-centered norm：+1 折叠
        elif name.endswith("linear_attn.conv1d.weight"):
            # conv1d: [out, 1, kernel] -> [out, kernel]（squeeze 中间维度）
            tiny[name] = arr.reshape(arr.shape[0], arr.shape[2])
        else:
            tiny[name] = arr  # 其余 tensor 直接复制
    return tiny


def main() -> None:
    """主入口函数：生成 Qwen3.5 fake 模型并写出 .tqwen 和 HF .pt 文件。

    流程：
    1. 构造随机 HF state_dict（利用 Qwen3_5ForCausalLM 获取正确结构）。
    2. 保存 HF state_dict 为 .pt 文件（供对齐脚本使用）。
    3. 将 HF state_dict 转换为 .tqwen 格式（+1 折叠、squeeze conv1d）。
    4. 处理 tied embedding：移除独立的 lm_head.weight。
    5. 组装 header 配置和 v2 扩展字段。
    6. 调用 write_tqwen 写出 .tqwen 文件。
    """
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="fake35.tqwen")      # .tqwen 输出路径
    p.add_argument("--hf-out", default="fake35_hf.pt")   # HF state_dict 输出路径
    p.add_argument("--seed", type=int, default=0)        # 随机种子
    args = p.parse_args()

    import torch  # 延迟导入 torch（用于保存 .pt 文件）

    # 构建随机 HF state_dict
    state, cfg = build_hf_state(args.seed)
    # 提取并移除层类型元数据
    layer_types = list(state.pop("__layer_types__"))

    # 存 HF 权重（对齐脚本直接 load_state_dict）。
    # 将 numpy 数组转为 torch tensor 后保存
    torch.save({k: torch.from_numpy(v) for k, v in state.items()}, args.hf_out)

    # 将 HF state_dict 转换为 .tqwen 格式
    tiny = hf_to_tiny(state)

    # 忠实模拟真实导出器：tied 模型不写独立 lm_head.weight（见
    # export_qwen_to_tiny.py 的 `if not tie_word_embeddings` 守卫）。此前这里
    # 把 HF state_dict 里 lm_head.weight（tied 模型也会列出该 key）原样写出，
    # 其随机数据与 embed 不同，会让 runtime 的 tied 绑定逻辑（检测到独立
    # lm_head 就优先用它）拿错权重，导致 C++ vs HF 的 logits 偏差。
    if FAKE_CFG["tie_word_embeddings"]:
        tiny.pop("lm_head.weight", None)  # tied 模型移除独立 lm_head

    # 组装 .tqwen header 所需的配置字典
    header_cfg = {
        "n_layers": FAKE_CFG["num_hidden_layers"],            # transformer 层数
        "hidden_size": FAKE_CFG["hidden_size"],               # 隐藏层维度
        "intermediate_size": FAKE_CFG["intermediate_size"],   # MLP 中间维度
        "n_heads": FAKE_CFG["num_attention_heads"],           # Q head 数
        "n_kv_heads": FAKE_CFG["num_key_value_heads"],        # KV head 数
        "head_dim": FAKE_CFG["head_dim"],                     # head 维度
        "vocab_size": FAKE_CFG["vocab_size"],                 # 词表大小
        "max_seq_len": FAKE_CFG["max_position_embeddings"],   # 最大序列长度
        "tied": 1 if FAKE_CFG["tie_word_embeddings"] else 0,  # tied embedding 标记
        "rms_norm_eps": FAKE_CFG["rms_norm_eps"],             # RMSNorm epsilon
        "rope_theta": FAKE_CFG["rope_theta"],                 # RoPE base frequency
    }
    # 组装 v2 扩展字段（Qwen3.5 混合架构特有参数）
    ext = {
        "model_type": MODEL_QWEN35,                           # 架构族标识
        "linear_num_qk_heads": FAKE_CFG["linear_num_key_heads"],     # 线性 QK head 数
        "linear_num_v_heads": FAKE_CFG["linear_num_value_heads"],    # 线性 V head 数
        "linear_qk_head_dim": FAKE_CFG["linear_key_head_dim"],       # 线性 QK head 维度
        "linear_v_head_dim": FAKE_CFG["linear_value_head_dim"],      # 线性 V head 维度
        "linear_conv_kernel_dim": FAKE_CFG["linear_conv_kernel_dim"],  # conv1d 核大小
        "full_attention_interval": FAKE_CFG["full_attention_interval"],  # full attn 间隔
        "partial_rotary_factor": FAKE_CFG["partial_rotary_factor"],    # 部分 RoPE 因子
        "eos_token_id": FAKE_CFG["eos_token_id"],                      # EOS token ID
    }
    # 写出 .tqwen 文件（v2 格式，fp32 精度）
    total = write_tqwen(args.out, header_cfg, tiny, "f32", version=2, ext=ext)
    print(f"wrote {args.out}: {total} bytes, {len(tiny)} tensors, "
          f"layer_types={layer_types}")
    print(f"wrote {args.hf_out}: HF state_dict for alignment")


if __name__ == "__main__":
    main()  # 脚本入口点
