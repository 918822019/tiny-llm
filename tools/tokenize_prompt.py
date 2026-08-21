#!/usr/bin/env python3
"""把 prompt 转成 token ids，供 tinyqwen CLI 使用。

v1 把 tokenizer 逻辑留在 Python；C++ runtime 只消费 token ids。
PyTorch reference dump（tools/dump_qwen_reference.py）必须使用同一份
tokenizer 配置，否则对齐结果没有意义。

用法:
    # 基本分词（不带 chat template）
    python tools/tokenize_prompt.py \\
        --model Qwen/Qwen2.5-0.5B \\
        --prompt "你好" \\
        --out prompt_tokens.json
    # 带 chat template（推荐用于 instruct 模型）
    python tools/tokenize_prompt.py \\
        --model models/Qwen2.5-0.5B \\
        --prompt "中国的首都是哪里？" \\
        --chat \\
        --out prompt_tokens.json

输出 JSON:
    {"model": ..., "prompt": ..., "chat": true,
     "tokens": [151644, ...], "n_tokens": N}

输入：
    - HF 模型目录或 repo id（提供 tokenizer）
    - prompt 文本字符串

输出：
    - JSON 文件包含 token id 列表（下游 C++ CLI 和 reference dump 共用）
    - 同时打印到 stdout 方便快速查看
"""

# 启用延迟注解求值
from __future__ import annotations

# ---- 标准库导入 ----
import argparse       # 命令行参数解析
import json           # JSON 序列化
from pathlib import Path  # 路径操作


def parse_args() -> argparse.Namespace:
    """解析命令行参数。

    Returns:
        argparse.Namespace: 解析后的参数对象
    """
    p = argparse.ArgumentParser(description=__doc__)
    # HF 模型目录或 repo id（如 models/Qwen2.5-0.5B 或 Qwen/Qwen2.5-0.5B）
    p.add_argument("--model", required=True, help="HF model dir or repo id")
    # prompt 文本
    p.add_argument("--prompt", required=True)
    # 是否使用 chat template 包裹 prompt
    p.add_argument("--chat", action="store_true",
                   help="wrap prompt with the model chat template "
                        "(recommended for Qwen2.5 instruct models)")
    # system prompt（仅 --chat 模式生效）
    p.add_argument("--system", default="You are Qwen, created by Alibaba Cloud. "
                                       "You are a helpful assistant.",
                   help="system prompt used only with --chat")
    # 输出文件路径
    p.add_argument("--out", default="prompt_tokens.json")
    return p.parse_args()


def main() -> None:
    """tokenize_prompt.py 的主入口函数。

    加载 tokenizer → 分词（可选 chat template）→ 输出 JSON。
    """
    args = parse_args()

    # 延迟导入 transformers（避免影响其他工具的启动速度）
    from transformers import AutoTokenizer

    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    if args.chat:
        # ---- Chat template 模式 ----
        # 构建消息列表（system + user）
        messages = []
        if args.system:
            messages.append({"role": "system", "content": args.system})
        messages.append({"role": "user", "content": args.prompt})
        # 应用 chat template 并直接返回 token id 列表
        tokens = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,   # 添加 assistant 回复前缀
            tokenize=True,                # 返回 token ids 而非文本
            return_dict=False,            # 强制返回纯 token id 列表（而非 BatchEncoding）
        )
        # 兜底：个别版本仍可能返回 dict/张量，统一转成一维 int 列表
        if not isinstance(tokens, (list, tuple)):
            tokens = tokens["input_ids"] if "input_ids" in tokens else tokens
        if hasattr(tokens, "tolist"):
            tokens = tokens.tolist()  # numpy/tensor → list
        tokens = list(tokens)
        # 可能带 batch 维 [[...]]，展平成一维
        if tokens and isinstance(tokens[0], (list, tuple)):
            tokens = list(tokens[0])
    else:
        # ---- 原始分词模式（不加特殊 token）----
        # Qwen2 的 tokenizer 没有 BOS；不要悄悄添加特殊 token
        tokens = tokenizer(args.prompt, add_special_tokens=False)["input_ids"]

    # "tokens" 数组是下游唯一消费的字段：C++ CLI
    # （runtime/main.cpp 的 parse_tokens_json）和 tools/dump_qwen_reference.py
    # 必须看到完全相同的 ids，否则对齐比较没有意义。
    payload = {
        "model": args.model,       # 使用的模型标识
        "prompt": args.prompt,     # 原始 prompt 文本
        "chat": bool(args.chat),   # 是否使用了 chat template
        "tokens": tokens,          # token id 列表（核心字段）
        "n_tokens": len(tokens),   # token 数量
    }
    # 写入输出文件
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)  # 确保父目录存在
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    # 同时打印到 stdout
    print(json.dumps(payload, ensure_ascii=False))


# 脚本直接运行入口
if __name__ == "__main__":
    main()
