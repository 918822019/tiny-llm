#!/usr/bin/env python3
"""数值对齐检查：tinyqwen C++ runtime（Qwen3.5 混合架构）vs PyTorch（HF）。

流程与 align_fake_model.py 相同：make_fake_qwen35_model.py 生成随机权重小模型
（同时产出 HF state_dict），跑 C++ binary greedy decode 并 --dump-logits 导出
全量 logits，再把同一份权重灌进 HF Qwen3_5ForCausalLM（fp32、eager、torch
兜底 kernel），逐位置（prefill + decode）比对 logits。

这条链路覆盖 GDN 递归/conv1d/l2norm/门控 RMSNorm、full attention 的 QK-norm/
partial RoPE/输出门、zero-centered RMSNorm 折叠、tied lm_head——且不需要下载真模型。

用法:
    python tools/align_fake_qwen35_model.py [--binary build/runtime/tinyqwen]

核心设计思路：
    1. 调用 make_fake_qwen35_model.py 生成一个小型随机 Qwen3.5 fake 模型，
       同时产出 .tqwen 文件（给 C++ runtime）和 .pt 文件（给 HF Python）。
    2. C++ runtime 执行 greedy decode 并 dump 每步 logits。
    3. Python 侧用 HF transformers 的 Qwen3_5ForCausalLM 加载同一份权重，
       以 eager attention + torch 兜底 kernel 前向推理。
    4. 逐位置比较两侧 logits，验证最大绝对误差 < TOL。
    5. 验证每步 argmax 一致。

与 align_fake_model.py 的关键区别：
    - Qwen3.5 是混合架构（linear_attention + full_attention 交替），
      涉及 GDN（Gated Delta Network）算子、conv1d、partial RoPE 等。
    - 使用 zero-centered RMSNorm，导出时已折叠 +1。
    - tied embeddings 下需用 assign=True 加载权重再 tie_weights()。
    - HF 参考模型需要 Qwen3_5ForCausalLM（非 Qwen2ForCausalLM）。

输入：C++ binary 路径（默认 build/runtime/tinyqwen）。
输出：控制台打印逐位置误差和对齐结果。
"""

from __future__ import annotations

# --- 标准库导入 ---
import argparse     # 命令行参数解析
import subprocess   # 调用 C++ binary 和 make_fake_qwen35_model.py
import sys          # 路径注入
import tempfile     # 临时目录
from pathlib import Path  # 路径操作

# 将 tools/ 加入搜索路径
sys.path.insert(0, str(Path(__file__).parent))

# 固定的 prompt token 序列（用于对齐测试的输入）
PROMPT = [3, 7, 11, 2]
# greedy decode 生成的最大新 token 数
MAX_NEW = 6
# 数值对齐容差
TOL = 1e-5


def run_cpp(binary: Path, model: Path, tmp: Path):
    """跑 C++ binary，收集精确 logits（行序 == 位置序，同 align_fake_model.py）。

    调用 tinyqwen C++ runtime 对 Qwen3.5 fake 模型执行 greedy decode，
    并通过 --dump-logits 将每个 forward 步的全量 logits 写入二进制文件。

    Args:
        binary: C++ runtime 可执行文件路径。
        model: .tqwen 模型文件路径。
        tmp: 临时目录路径。

    Returns:
        (gen_ids, logits_array) 元组：
        - gen_ids: 生成的 token ID 列表（长度 MAX_NEW）。
        - logits_array: 形状为 [n_rows, vocab_size] 的 fp32 numpy 数组。
    """
    import numpy as np  # 延迟导入 numpy

    logits_path = tmp / "logits.bin"  # logits dump 文件路径
    # 构造 C++ runtime 命令行参数
    cmd = [str(binary), "--model", str(model),
           "--tokens", ",".join(map(str, PROMPT)),   # 输入 prompt tokens
           "--max-new-tokens", str(MAX_NEW),          # 最大生成数
           "--max-seq-len", "32",                     # 最大序列长度
           "--eos", "-1",                             # 禁用 EOS 检测
           # 逐 token prefill：批量 GEMM prefill 不做逐位置 dump，拿不到 prompt
           # 各位置的 logits；--verbose 强制逐 token 路径，恢复 dump 契约
           "--verbose",
           "--dump-logits", str(logits_path)]         # logits 输出文件
    # 运行 C++ binary 并捕获 stdout
    out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    # 从 stdout 解析生成的 token ID
    gen_ids = [int(line.split()[2]) for line in out.splitlines()
               if line.startswith("gen ")]
    assert len(gen_ids) == MAX_NEW, gen_ids  # 校验生成数量
    # 读取 logits 二进制文件
    rows = np.fromfile(logits_path, dtype=np.float32)
    # forward 次数 = prompt 长度 + (MAX_NEW - 1) 次 decode
    n_rows = len(PROMPT) + MAX_NEW - 1
    assert rows.size % n_rows == 0, (rows.size, n_rows)  # 校验整除
    vocab = rows.size // n_rows  # 推算词表大小
    return gen_ids, rows.reshape(n_rows, vocab)  # reshape 为 [n_rows, vocab]


def main() -> None:
    """主入口函数：执行 Qwen3.5 混合架构的 C++ vs HF 数值对齐验证。

    流程：
    1. 在临时目录中生成 fake Qwen3.5 模型（.tqwen + .pt）。
    2. 运行 C++ runtime 获取 logits 和生成序列。
    3. 加载 HF state_dict 并用 Qwen3_5ForCausalLM 构造参考模型。
    4. HF 前向推理获取参考 logits。
    5. 逐位置比较两侧 logits 的最大绝对误差。
    6. 验证每步 argmax 一致性。
    """
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--binary", default="build/runtime/tinyqwen")  # C++ binary 路径
    args = p.parse_args()

    import numpy as np   # 数值计算
    import torch         # PyTorch tensor 操作

    from make_fake_qwen35_model import make_config  # 导入 fake 模型的配置工厂函数
    from transformers import Qwen3_5ForCausalLM     # HF Qwen3.5 因果语言模型

    # 在临时目录中生成 fake 模型并运行 C++ binary
    with tempfile.TemporaryDirectory() as tmp_d:
        tmp = Path(tmp_d)
        model_path = tmp / "fake35.tqwen"    # .tqwen 格式（给 C++ runtime）
        hf_path = tmp / "fake35_hf.pt"       # PyTorch state_dict（给 HF Python）
        # 调用 make_fake_qwen35_model.py 同时生成两种格式的 fake 模型
        subprocess.run([sys.executable, "tools/make_fake_qwen35_model.py",
                        "--out", str(model_path), "--hf-out", str(hf_path)],
                       check=True, capture_output=True)
        # 运行 C++ runtime 获取生成结果和 logits
        gen_ids, cpp_logits = run_cpp(Path(args.binary), model_path, tmp)
        # 加载 HF state_dict
        hf_state = torch.load(hf_path, weights_only=True)

    # 用完全相同的配置和权重构造 HF 参考模型（torch 兜底 kernel，无 fla 依赖）。
    cfg = make_config()  # 获取与 fake 模型一致的 HF config
    cfg._attn_implementation = "eager"  # 强制 eager attention，避免 flash/sdpa 数值差异
    model = Qwen3_5ForCausalLM(cfg).eval()  # 创建模型并设为 eval 模式
    # tied embeddings 下默认 copy 式 load 不会写入 embed，需用 assign=True
    # 再 tie_weights() 恢复 lm_head<->embed 的绑定。
    missing, unexpected = model.load_state_dict(hf_state, strict=False, assign=True)
    model.tie_weights()  # 恢复 tied embedding 的权重共享
    # 校验缺失 key 在允许范围内（tied 时 lm_head.weight 允许缺失）
    allowed_missing = {"lm_head.weight"} if cfg.tie_word_embeddings else set()
    assert set(missing) <= allowed_missing, missing
    assert not unexpected, unexpected  # 不应有多余 key
    # 额外校验 embed_tokens 确实被正确加载
    assert torch.allclose(model.model.embed_tokens.weight,
                          hf_state["model.embed_tokens.weight"]), "embed not loaded"

    # C++ 行：0..P-1 是 prefill，之后是 decode；greedy g_s = argmax(row P-1+s)。
    P = len(PROMPT)  # prompt 长度
    # 构造完整输入序列：prompt + 前 MAX_NEW-1 个生成 token
    full_seq = PROMPT + gen_ids[: MAX_NEW - 1]
    input_ids = torch.tensor([full_seq], dtype=torch.long)  # 转为 torch tensor
    with torch.no_grad():  # 禁用梯度
        # 手动指定 position_ids 确保与 C++ 一致
        pos = torch.arange(len(full_seq), dtype=torch.long).unsqueeze(0)
        # HF 前向推理获取参考 logits
        ref = model(input_ids=input_ids, position_ids=pos,
                    use_cache=False).logits[0].numpy()
    # 校验形状一致
    assert ref.shape == cpp_logits.shape, (ref.shape, cpp_logits.shape)

    # 逐位置比较 logits 的最大绝对误差
    worst = 0.0  # 全局最差误差
    for i in range(len(full_seq)):
        err = float(np.abs(cpp_logits[i] - ref[i]).max())  # 该位置最大绝对误差
        worst = max(worst, err)  # 更新全局最差
        tag = "prefill" if i < P else "decode "  # 标记 prefill/decode 阶段
        print(f"pos {i} ({tag}): max_abs_err={err:.3e}")
        assert err < TOL, f"position {i} diverges: {err}"  # 超容差断言失败

    # 验证每步 greedy decode 的 argmax 在两侧完全一致
    for s, g in enumerate(gen_ids):
        row = P - 1 + s  # logits 矩阵中的对应行号
        # C++ 内部 argmax 校验
        assert int(cpp_logits[row].argmax()) == g, f"cpp internal argmax mismatch at {s}"
        # HF argmax 校验
        assert int(ref[row].argmax()) == g, \
            f"step {s}: cpp={g} ref={int(ref[row].argmax())}"
    # 打印最终结果
    print(f"generated: {gen_ids} (matches HF argmax at every step)")
    print(f"WORST max_abs_err: {worst:.3e} (tol {TOL})")
    print("ALIGNMENT OK")  # 对齐通过


if __name__ == "__main__":
    main()  # 脚本入口点
