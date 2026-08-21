#!/usr/bin/env python3
"""用 ModelScope 中文诗词数据集驱动 tinyqwen 做续写演示。

数据集 modelscope/chinese-poetry-collection：单列 text1，每行一首诗。
玩法：取每首诗的前两句作 prompt，让 runtime 续写，再和原诗后两句对照。
这是一个定性 demo，用于直观感受模型的中文生成能力，不做性能测量。

用法：
    # 默认取 3 首诗，每首生成 24 个 token
    python tools/run_poetry_demo.py [--num 3] [--max-new-tokens 24]
    # 指定数据集路径
    python tools/run_poetry_demo.py --csv datasets/chinese-poetry-collection/test.csv

输入：
    - datasets/chinese-poetry-collection/test.csv（CSV 格式诗词数据集）
    - models/Qwen2.5-0.5B（HF tokenizer 目录）
    - build/runtime/tinyqwen（C++ runtime）
    - model.tqwen（量化模型文件）

输出：
    - 终端逐首打印 prompt、模型续写、原诗后两句的对比
"""

# 启用延迟注解求值
from __future__ import annotations

# ---- 标准库导入 ----
import argparse       # 命令行参数解析
import csv            # CSV 文件读取（诗词数据集）
import subprocess     # 调用 tinyqwen C++ runtime
from pathlib import Path  # 路径操作

# ---- 常量 ----
MODEL_DIR = "models/Qwen2.5-0.5B"   # HF tokenizer 所在目录
BIN = "build/runtime/tinyqwen"       # C++ runtime 可执行文件路径
MODEL_TQWEN = "model.tqwen"         # .tqwen 模型文件路径


def load_poems(csv_path: str, num: int) -> list[str]:
    """从 CSV 数据集中加载指定数量的诗词文本。

    Args:
        csv_path: CSV 文件路径（需含 text1 列）
        num: 要加载的诗的数量上限

    Returns:
        list[str]: 诗词文本列表（已过滤过短的条目）
    """
    poems = []
    # utf-8-sig 编码去掉 BOM（Windows 导出的 CSV 常带 BOM）
    with open(csv_path, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)  # 按字典方式读取（自动识别表头）
        for row in reader:
            text = (row.get("text1") or "").strip()  # 取 text1 列并去空白
            if len(text) >= 10:  # 过滤过短的文本（至少 10 字符才算有效诗）
                poems.append(text)
            if len(poems) >= num:  # 达到数量上限则停止
                break
    return poems


def split_prompt(poem: str) -> tuple[str, str]:
    """按第一个句号将诗拆成 prompt（前两句）和参考（后两句）。

    Args:
        poem: 完整诗文（如 "白日依山尽，黄河入海流。欲穷千里目，更上一层楼。"）

    Returns:
        tuple[str, str]: (prompt 部分含句号, 参考部分)；无句号时返回 (全文, "")
    """
    if "。" in poem:
        head, _, tail = poem.partition("。")  # 按第一个句号分割
        return head + "。", tail  # prompt 保留句号
    return poem, ""  # 没有句号时整段作为 prompt


def main() -> None:
    """run_poetry_demo.py 的主入口函数。

    加载诗词 → 逐首 tokenize → 调 runtime 续写 → 打印对比结果。
    """
    # 创建参数解析器
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="datasets/chinese-poetry-collection/test.csv")  # 数据集路径
    ap.add_argument("--num", type=int, default=3, help="取多少首诗")
    ap.add_argument("--max-new-tokens", type=int, default=24)   # 每首生成的 token 数
    ap.add_argument("--max-seq-len", type=int, default=96)      # KV cache 容量
    args = ap.parse_args()

    # 延迟导入 transformers（仅本脚本需要，避免影响其他工具的启动速度）
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)  # 加载 Qwen2.5 tokenizer

    # 加载诗词
    poems = load_poems(args.csv, args.num)
    print(f"从 {args.csv} 取了 {len(poems)} 首诗\n" + "=" * 60)

    # 逐首处理
    for idx, poem in enumerate(poems):
        prompt, reference = split_prompt(poem)  # 拆分 prompt 和参考
        ids = tok(prompt, add_special_tokens=False)["input_ids"]  # tokenize prompt
        tokens_csv = ",".join(map(str, ids))  # 转为逗号分隔的 token id 字符串

        # 构造 tinyqwen CLI 命令并执行
        cmd = [
            BIN, "--model", MODEL_TQWEN,          # binary 和模型
            "--tokens", tokens_csv,               # prompt token ids
            "--max-new-tokens", str(args.max_new_tokens),  # 生成长度
            "--max-seq-len", str(args.max_seq_len),       # KV 容量
            "--eos", "-1",                         # 禁用 EOS，保证生成固定数量
        ]
        r = subprocess.run(cmd, capture_output=True, text=True, check=True)

        # 从 stdout 中解析生成的 token id 列表
        gen_ids = []
        for line in r.stdout.splitlines():
            if line.startswith("generated_ids:"):
                # generated_ids: 后面的空格分隔整数
                gen_ids = [int(t) for t in line.split()[1:]]
        # 将 token id 解码回文本
        gen_text = tok.decode(gen_ids, skip_special_tokens=True)

        # 打印对比结果
        print(f"\n【{idx + 1}】prompt（前两句）:\n  {prompt}")
        print(f"模型续写:\n  {gen_text}")
        if reference:
            print(f"原诗后两句:\n  {reference}")
        print("-" * 60)


# 脚本直接运行入口
if __name__ == "__main__":
    main()
