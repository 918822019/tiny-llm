#!/usr/bin/env python3
"""把 prompt 转成 token ids，供 tinyqwen CLI 使用。

v1 把 tokenizer 逻辑留在 Python；C++ runtime 只消费 token ids。
PyTorch reference dump（tools/dump_qwen_reference.py）必须使用同一份
tokenizer 配置，否则对齐结果没有意义。

用法:
    python tools/tokenize_prompt.py \
        --model Qwen/Qwen2.5-0.5B \
        --prompt "你好" \
        --chat \
        --out prompt_tokens.json

输出 JSON:
    {"model": ..., "prompt": ..., "chat": true,
     "tokens": [151644, ...], "n_tokens": N}
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, help="HF model dir or repo id")
    p.add_argument("--prompt", required=True)
    p.add_argument("--chat", action="store_true",
                   help="wrap prompt with the model chat template "
                        "(recommended for Qwen2.5 instruct models)")
    p.add_argument("--system", default="You are Qwen, created by Alibaba Cloud. "
                   "You are a helpful assistant.",
                   help="system prompt used only with --chat")
    p.add_argument("--out", default="prompt_tokens.json")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    if args.chat:
        messages = []
        if args.system:
            messages.append({"role": "system", "content": args.system})
        messages.append({"role": "user", "content": args.prompt})
        tokens = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
        )
        tokens = list(tokens)
    else:
        # Qwen2 的 tokenizer 没有 BOS；不要悄悄添加特殊 token。
        tokens = tokenizer(args.prompt, add_special_tokens=False)["input_ids"]

    # "tokens" 数组是下游唯一消费的字段：C++ CLI
    #（runtime/main.cpp 的 parse_tokens_json）和 tools/dump_qwen_reference.py
    # 必须看到完全相同的 ids，否则对齐比较没有意义。
    payload = {
        "model": args.model,
        "prompt": args.prompt,
        "chat": bool(args.chat),
        "tokens": tokens,
        "n_tokens": len(tokens),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
