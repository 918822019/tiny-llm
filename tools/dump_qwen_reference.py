#!/usr/bin/env python3
"""dump PyTorch/HF 参考激活值，用于 tinyqwen C++ runtime 的数值对齐。

对固定的 token 序列（单次前向、eager attention、fp32、cpu）dump：
    embed_out                     [seq, hidden]
    layer_0_attn_norm             input_layernorm 输出              [seq, hidden]
    layer_0_q / k / v             q/k/v_proj 输出（RoPE 之前）      [seq, n_heads*head_dim] 等
    layer_0_attn_out              self_attn 输出（o_proj 之后）     [seq, hidden]
    layer_0_post_attn_residual    post_attention_layernorm 的输入   [seq, hidden]
    layer_0_ffn_norm              post_attention_layernorm 输出     [seq, hidden]
    layer_0_gate / layer_0_up     mlp gate/up 投影输出              [seq, inter]
    layer_0_ffn_out               mlp down_proj 输出                [seq, hidden]
    layer_0_output                decoder layer 0 输出              [seq, hidden]
    final_norm                    model.norm 输出                   [seq, hidden]
    logits                        [seq, vocab]
    topk_values / topk_indices    最后一个位置的 top-k              [k]

旁边的 <out>.meta.json 记录复现本次运行所需的全部信息
（tokens、dtype、attention 实现等）。对齐流程见 docs/pytorch_alignment.md。

用法:
    python tools/dump_qwen_reference.py \\
        --model Qwen/Qwen2.5-0.5B \\
        --tokens-json prompt_tokens.json \\
        --out ref.npz --topk 10

核心设计思路：
    1. 使用 PyTorch forward hook/pre_hook 机制在模型内部各关键节点捕获中间激活。
    2. 所有计算强制 fp32 + cpu + eager attention，确保确定性且与 C++ runtime 可比。
    3. dump 的 key 名是 docs/pytorch_alignment.md 定义的契约，C++ 侧按同名 dump。
    4. 同时保存 .meta.json 元数据文件，记录复现条件。

输入：HF 模型目录 + token 序列（JSON 或逗号分隔）。
输出：.npz 文件（含所有中间激活）+ .meta.json 元数据文件。
"""

from __future__ import annotations

# --- 标准库导入 ---
import argparse   # 命令行参数解析
import json       # 读取 token JSON、写出 meta JSON
import numpy as np  # 数组操作、npz 保存
from pathlib import Path  # 路径操作


def parse_args() -> argparse.Namespace:
    """解析命令行参数。

    Returns:
        解析后的参数命名空间，包含 model、tokens/tokens_json、out、topk、layer。
    """
    p = argparse.ArgumentParser(description=__doc__)  # 用模块 docstring 作为帮助
    p.add_argument("--model", required=True, help="HF model dir or repo id")  # HF 模型路径
    # tokens 来源：二选一（互斥组）
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--tokens-json", help="JSON produced by tools/tokenize_prompt.py")  # token JSON 文件
    src.add_argument("--tokens", help="comma separated token ids, e.g. 1,2,3")  # 逗号分隔 token ID
    p.add_argument("--out", default="ref.npz")  # 输出 npz 文件路径
    p.add_argument("--topk", type=int, default=10)  # top-k 数量
    p.add_argument("--layer", type=int, default=0,
                   help="which layer to dump in detail (default 0)")  # 要详细 dump 的层号
    return p.parse_args()


def load_tokens(args: argparse.Namespace) -> list[int]:
    """从命令行参数加载 token 序列。

    支持两种来源：
    - --tokens: 逗号分隔的整数字符串（如 "1,2,3"）。
    - --tokens-json: tokenize_prompt.py 生成的 JSON 文件。

    Args:
        args: 解析后的命令行参数。

    Returns:
        token ID 列表。
    """
    if args.tokens:
        # 从逗号分隔字符串解析
        return [int(t) for t in args.tokens.split(",") if t.strip()]
    # 从 JSON 文件读取
    data = json.loads(Path(args.tokens_json).read_text())
    return [int(t) for t in data["tokens"]]


def main() -> None:
    """主入口函数：执行 HF 参考模型的单次前向推理并 dump 所有中间激活。

    流程：
    1. 加载 token 序列。
    2. 以 fp32/cpu/eager 模式加载 HF 模型。
    3. 注册 forward hook/pre_hook 捕获中间激活。
    4. 执行单次前向推理。
    5. 移除所有 hook。
    6. 保存 logits、top-k 和所有捕获的中间激活到 .npz。
    7. 保存元数据到 .meta.json。
    """
    args = parse_args()  # 解析命令行参数
    tokens = load_tokens(args)  # 加载 token 序列
    print(f"tokens ({len(tokens)}): {tokens}")  # 打印 token 信息

    import torch  # 延迟导入 torch
    from transformers import AutoModelForCausalLM  # HF 自动模型加载

    # 确定性参考实现：fp32、cpu、eager attention、eval 模式。
    # eager attention 避免 flash/sdpa 优化引入数值差异；
    # fp32 + cpu 保证与 C++ runtime 的精度基准一致。
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float32,       # 强制 fp32
        attn_implementation="eager",     # 强制 eager attention
    ).eval()  # eval 模式（关闭 dropout 等训练专用行为）

    L = args.layer  # 要详细 dump 的层号
    lm = model.model  # 获取底层 transformer 模型（不含 lm_head）
    captured: dict[str, np.ndarray] = {}  # 存放所有捕获的中间激活

    def save(name: str, tensor: torch.Tensor) -> None:
        """将 torch tensor 转为 fp32 numpy 并存入 captured 字典。

        Args:
            name: 激活值的名字（对应 .npz 中的 key）。
            tensor: 要保存的 torch tensor。
        """
        captured[name] = tensor.detach().float().cpu().numpy()  # detach → float → cpu → numpy

    hooks = []  # 存放注册的 hook 句柄（最后统一移除）

    # reg：捕获子模块的输出。HF 的 attention / decoder layer 返回元组，
    # 这里统一解包取 hidden-state 张量。
    def reg(module: torch.nn.Module, name: str) -> None:
        """注册 forward hook 捕获模块输出。

        Args:
            module: 要 hook 的 PyTorch 模块。
            name: 捕获数据的存储 key。
        """
        hooks.append(module.register_forward_hook(
            lambda _m, _i, out, n=name: save(n, out[0] if isinstance(out, tuple) else out)))

    # reg_pre：捕获子模块的输入——用于取 post-attention 残差状态，
    # 它恰好就是 post_attention_layernorm 的输入。
    def reg_pre(module: torch.nn.Module, name: str) -> None:
        """注册 forward pre-hook 捕获模块输入。

        Args:
            module: 要 hook 的 PyTorch 模块。
            name: 捕获数据的存储 key。
        """
        hooks.append(module.register_forward_pre_hook(
            lambda _m, inp, n=name: save(n, inp[0])))

    # hook 的位置与 docs/qwen_forward.md 的 op 一一对应；dump 的 key 名
    # 是 docs/pytorch_alignment.md 使用的契约。
    # 注意：q/k/v dump 的是 RoPE 之前的值（投影输出），与 C++ 侧
    # q/k/v 投影刚结束时的 buffer 状态对应。
    layer0 = lm.layers[L]  # 获取要 dump 的目标层

    # ===== 注册所有 hook =====
    # embedding 输出（手动调用而非 hook，因为 embed_tokens 不是独立 forward 的子模块）
    save("embed_out", lm.embed_tokens(torch.tensor(tokens)))
    # input_layernorm 输出
    reg(layer0.input_layernorm, f"layer_{L}_attn_norm")
    # Q/K/V 投影输出（RoPE 之前）
    reg(layer0.self_attn.q_proj, f"layer_{L}_q")
    reg(layer0.self_attn.k_proj, f"layer_{L}_k")
    reg(layer0.self_attn.v_proj, f"layer_{L}_v")
    # self_attn 整体输出（o_proj 之后）
    reg(layer0.self_attn, f"layer_{L}_attn_out")
    # post_attention_layernorm 的输入 = attention 输出 + 残差
    reg_pre(layer0.post_attention_layernorm, f"layer_{L}_post_attn_residual")
    # post_attention_layernorm 输出
    reg(layer0.post_attention_layernorm, f"layer_{L}_ffn_norm")
    # MLP gate/up 投影输出
    reg(layer0.mlp.gate_proj, f"layer_{L}_gate")
    reg(layer0.mlp.up_proj, f"layer_{L}_up")
    # MLP down_proj 输出（即 FFN 输出）
    reg(layer0.mlp.down_proj, f"layer_{L}_ffn_out")
    # decoder layer 整体输出
    reg(layer0, f"layer_{L}_output")
    # 最终 RMSNorm 输出
    reg(lm.norm, "final_norm")

    # 构造输入 tensor
    input_ids = torch.tensor([tokens], dtype=torch.long)  # [1, seq_len]
    with torch.no_grad():  # 禁用梯度
        # 显式 position_ids 保证与 C++ runtime 可比：C++ 侧
        # pos == KV cache 长度（0 起始，调度完全一致）。
        position_ids = torch.arange(len(tokens), dtype=torch.long).unsqueeze(0)  # [1, seq_len]
        # 执行前向推理（use_cache=False 不使用 KV cache，纯一次性前向）
        out = model(input_ids=input_ids, position_ids=position_ids, use_cache=False)

    # 移除所有 hook（避免影响后续操作）
    for h in hooks:
        h.remove()

    # 提取 logits 并保存
    logits = out.logits[0]  # [seq, vocab]（去掉 batch 维度）
    save("logits", logits)
    # 计算最后一个位置的 top-k
    top = torch.topk(logits[-1], args.topk)
    captured["topk_values"] = top.values.numpy()    # top-k 的值
    captured["topk_indices"] = top.indices.numpy()  # top-k 的 token ID

    # 保存所有捕获的激活到 .npz 文件
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)  # 自动创建目录
    np.savez(out_path, **captured)  # 以 keyword arguments 方式保存

    # 构造并保存元数据 JSON
    meta = {
        "model": args.model,                      # 模型标识
        "tokens": tokens,                         # 输入 token 序列
        "n_tokens": len(tokens),                  # token 数量
        "dtype": "float32",                       # 计算精度
        "device": "cpu",                          # 计算设备
        "attn_implementation": "eager",           # attention 实现
        "layer_dumped": L,                        # 详细 dump 的层号
        "topk": args.topk,                        # top-k 数量
        "note": "single forward pass with explicit position_ids = arange; "
                "compare C++ token-by-token outputs at matching positions",  # 使用说明
    }
    meta_path = out_path.with_suffix(out_path.suffix + ".meta.json")  # meta 文件路径
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n")  # 写出 JSON

    # 打印结果摘要
    print(f"saved {len(captured)} tensors -> {out_path}")
    print(f"meta -> {meta_path}")
    print("last-position top-k:",
          list(zip(captured['topk_indices'].tolist(),
                   np.round(captured['topk_values'], 4).tolist())))


if __name__ == "__main__":
    main()  # 脚本入口点
