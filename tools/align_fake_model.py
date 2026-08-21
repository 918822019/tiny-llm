#!/usr/bin/env python3
"""数值对齐检查：tinyqwen C++ runtime vs PyTorch（HF Qwen2）。

流程：用 tools/make_fake_model.py 生成随机权重小模型，跑 C++ binary 做
greedy decode 并用 --dump-logits 导出全量 logits（fp32 精确值），然后把
同一份权重灌进 HF Qwen2 模型（fp32、eager attention），逐位置
（prefill + decode）比对 logits。

这条链路完整覆盖 RMSNorm / q-k-v bias / RoPE / GQA / SwiGLU / tied lm_head，
且不需要下载真模型。真模型的对齐流程见 docs/pytorch_alignment.md。

用法:
    python tools/align_fake_model.py [--binary build/runtime/tinyqwen]

核心设计思路：
    1. 生成一个小型随机 fake 模型（.tqwen 格式），确保 C++ runtime 和
       Python HF 模型使用完全相同的权重。
    2. C++ runtime 执行 greedy decode，同时 dump 每个位置的 logits。
    3. Python 侧用 HF transformers 加载同样的权重，以 eager attention
       （避免 flash attention 引入额外数值差异）前向推理。
    4. 逐位置比较 C++ 和 HF 的 logits，最大绝对误差必须 < TOL。
    5. 验证每一步的 argmax（生成的 token ID）在两侧完全一致。

输入：C++ binary 路径（默认 build/runtime/tinyqwen）。
输出：控制台打印逐位置误差和最终对齐结果。
"""

from __future__ import annotations

# --- 标准库导入 ---
import argparse     # 命令行参数解析
import struct       # 二进制解包（读取 .tqwen header/entry）
import subprocess   # 调用 C++ binary 和 make_fake_model.py
import sys          # 路径注入、解释器路径
import tempfile     # 临时目录（存放 fake 模型和 logits dump）
from pathlib import Path  # 路径操作

# 将 tools/ 加入搜索路径，复用 export_qwen_to_tiny 的格式常量
sys.path.insert(0, str(Path(__file__).parent))
from export_qwen_to_tiny import HEADER_FMT, ENTRY_FMT  # noqa: E402  # .tqwen 文件格式定义

# 固定的 prompt token 序列（用于对齐测试的输入）
PROMPT = [3, 7, 11, 2]
# greedy decode 生成的最大新 token 数
MAX_NEW = 6
# 数值对齐容差：C++ 与 HF logits 的最大允许绝对误差
TOL = 1e-5


def read_tqwen(path: Path):
    """Python 侧的最小 .tqwen 读取器（与 C++ loader 同一格式契约）。

    从 .tqwen 文件中读取模型配置和所有权重 tensor，返回 Python 字典形式，
    供后续构造 HF 参考模型时使用。只支持 fp32 dtype（dtype_code == 0）。

    Args:
        path: .tqwen 文件路径。

    Returns:
        (cfg, tensors) 元组：
        - cfg: 包含 n_layers, hidden_size 等超参的字典。
        - tensors: {tensor_name: numpy_fp32_array} 的字典。
    """
    import numpy as np  # 延迟导入 numpy

    with open(path, "rb") as f:
        # 读取并解包 192 字节的 header
        fields = struct.unpack(HEADER_FMT, f.read(struct.calcsize(HEADER_FMT)))
        # 从 header 字段提取模型配置
        cfg = {
            "n_layers": fields[3],           # transformer 层数
            "hidden_size": fields[4],        # 隐藏层维度
            "intermediate_size": fields[5],  # MLP 中间维度
            "n_heads": fields[6],            # 注意力 head 数
            "n_kv_heads": fields[7],         # KV head 数（GQA）
            "head_dim": fields[8],           # head 维度
            "vocab_size": fields[9],         # 词表大小
            "max_seq_len": fields[10],       # 最大序列长度
            "tied": fields[11],              # 是否 tied embedding
            "rms_norm_eps": fields[13],      # RMSNorm epsilon
            "rope_theta": fields[14],        # RoPE base frequency
        }
        count, table_offset = fields[15], fields[16]  # tensor 总数和索引表偏移
        tensors = {}  # 存放读取到的 tensor
        f.seek(table_offset)  # 跳转到 tensor 索引表
        for _ in range(count):
            # 读取 120 字节的 tensor entry
            name_b, dtype, ndim, s0, s1, s2, s3, off, nbytes = struct.unpack(
                ENTRY_FMT, f.read(struct.calcsize(ENTRY_FMT)))
            assert dtype == 0  # 只支持 fp32（fake 模型都是 fp32）
            # 解码 tensor 名字（去除尾部零字节）
            name = name_b.rstrip(b"\x00").decode("ascii")
            # 根据实际维度数截取有效 shape
            shape = [s0, s1, s2, s3][:ndim]
            pos = f.tell()      # 保存当前文件指针位置（读完数据后要回来继续读下一个 entry）
            f.seek(off)         # 跳转到该 tensor 的数据偏移
            # 读取 nbytes 字节并按 fp32 解析，reshape 为目标形状，copy 脱离 mmap
            tensors[name] = np.frombuffer(f.read(nbytes), dtype=np.float32).reshape(shape).copy()
            f.seek(pos)         # 恢复到 entry 表的下一个位置
    return cfg, tensors


def run_cpp(binary: Path, model: Path, tmp: Path):
    """跑 C++ binary，收集精确 logits。

    调用 tinyqwen C++ runtime 执行 greedy decode，并通过 --dump-logits
    将每个 forward 步的全量 logits 写入二进制文件。

    返回 (generated_ids, logits)，其中 logits[i] 是消费完序列位置 i 后
    产生的 fp32 行——行序 == 位置序，这是 runtime/main.cpp 的
    --dump-logits 契约。

    Args:
        binary: C++ runtime 可执行文件路径。
        model: .tqwen 模型文件路径。
        tmp: 临时目录路径（用于存放 logits dump 文件）。

    Returns:
        (gen_ids, logits_array) 元组：
        - gen_ids: 生成的 token ID 列表（长度 MAX_NEW）。
        - logits_array: 形状为 [n_rows, vocab_size] 的 fp32 numpy 数组，
          n_rows = len(PROMPT) + MAX_NEW - 1。
    """
    import numpy as np  # 延迟导入 numpy

    logits_path = tmp / "logits.bin"  # logits dump 文件路径
    # 构造 C++ runtime 的命令行参数
    cmd = [str(binary), "--model", str(model),
           "--tokens", ",".join(map(str, PROMPT)),   # 输入 prompt tokens
           "--max-new-tokens", str(MAX_NEW),          # 最大生成 token 数
           "--max-seq-len", "32",                     # 最大序列长度
           "--eos", "-1",                             # 禁用 EOS（-1 表示不检测）
           # 逐 token prefill：批量 GEMM prefill 不做逐位置 dump，拿不到 prompt
           # 各位置的 logits；--verbose 强制逐 token 路径，恢复 dump 契约
           "--verbose",
           "--dump-logits", str(logits_path)]         # logits 输出文件
    # 运行 C++ binary 并捕获 stdout
    out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    # 从 stdout 解析生成的 token ID（格式："gen <step> <token_id>"）
    gen_ids = [int(line.split()[2]) for line in out.splitlines()
               if line.startswith("gen ")]
    assert len(gen_ids) == MAX_NEW, gen_ids  # 校验生成数量
    vocab = None  # 词表大小（稍后从 logits 推算）
    # 从二进制文件读取所有 fp32 logits 值
    rows = np.fromfile(logits_path, dtype=np.float32)
    # forward 次数 = len(PROMPT) 次 prefill + (MAX_NEW - 1) 次 decode 推进
    # 注意：最后一个生成 token 不需要再 forward，所以 decode 步数是 MAX_NEW - 1
    n_rows = len(PROMPT) + MAX_NEW - 1
    assert rows.size % n_rows == 0, (rows.size, n_rows)  # 校验总元素数能被行数整除
    vocab = rows.size // n_rows  # 推算词表大小
    return gen_ids, rows.reshape(n_rows, vocab)  # reshape 为 [n_rows, vocab]


def main() -> None:
    """主入口函数：执行完整的 C++ vs HF 数值对齐验证。

    流程：
    1. 生成 fake 模型并读取其配置和权重。
    2. 运行 C++ runtime 获取 logits 和生成序列。
    3. 用相同权重构造 HF Qwen2 参考模型。
    4. HF 前向推理获取参考 logits。
    5. 逐位置比较两侧 logits 的最大绝对误差。
    6. 验证每步 argmax 一致性。
    """
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--binary", default="build/runtime/tinyqwen")  # C++ binary 路径
    args = p.parse_args()

    import numpy as np   # 数值计算
    import torch         # PyTorch tensor 操作
    from transformers import Qwen2Config, Qwen2ForCausalLM  # HF Qwen2 模型

    # 在临时目录中生成 fake 模型并运行 C++ binary
    with tempfile.TemporaryDirectory() as tmp_d:
        tmp = Path(tmp_d)
        model_path = tmp / "fake.tqwen"  # fake 模型的临时路径
        # 调用 make_fake_model.py 生成随机权重的小模型
        subprocess.run([sys.executable, "tools/make_fake_model.py",
                        "--out", str(model_path)], check=True, capture_output=True)
        # 读取 fake 模型的配置和权重
        cfg, tensors = read_tqwen(model_path)
        # 运行 C++ runtime 获取生成结果和 logits
        gen_ids, cpp_logits = run_cpp(Path(args.binary), model_path, tmp)

    # 用完全相同的配置和权重构造 HF 参考模型。
    # 配置必须与 fake 模型完全一致，确保两侧计算图等价。
    hf_cfg = Qwen2Config(
        hidden_size=cfg["hidden_size"],                   # 隐藏层维度
        intermediate_size=cfg["intermediate_size"],       # MLP 中间维度
        num_hidden_layers=cfg["n_layers"],                # transformer 层数
        num_attention_heads=cfg["n_heads"],               # 注意力 head 数
        num_key_value_heads=cfg["n_kv_heads"],            # KV head 数
        head_dim=cfg["head_dim"],                         # head 维度
        vocab_size=cfg["vocab_size"],                     # 词表大小
        max_position_embeddings=cfg["max_seq_len"],       # 最大位置编码长度
        rms_norm_eps=cfg["rms_norm_eps"],                 # RMSNorm epsilon
        rope_theta=cfg["rope_theta"],                     # RoPE base frequency
        tie_word_embeddings=bool(cfg["tied"]),            # 是否共享 embed/lm_head
        hidden_act="silu",                                # 激活函数（SwiGLU 用 SiLU）
    )
    # 强制使用 eager attention（非 flash/sdpa），避免优化实现引入数值差异
    hf_cfg._attn_implementation = "eager"
    model = Qwen2ForCausalLM(hf_cfg).eval()  # 创建模型并设为 eval 模式
    # 将 numpy 权重转为 torch tensor 并加载到 HF 模型
    missing, unexpected = model.load_state_dict(
        {k: torch.from_numpy(v) for k, v in tensors.items()}, strict=False)
    # tied 模型里 lm_head.weight 与 embed 共享，允许缺省。
    allowed_missing = {"lm_head.weight"} if cfg["tied"] else set()
    assert set(missing) <= allowed_missing, missing  # 校验缺失的 key 在允许范围内
    assert not unexpected, unexpected  # 不应有多余的 key

    # C++ 的 forward 行：位置 0..P-1 是 prefill，之后是 decode 位置。
    # greedy token g_s = argmax(logits[P - 1 + s])。
    P = len(PROMPT)  # prompt 长度
    # HF 需要与 C++ 相同的 token 序列作为输入；最后一个生成 token
    # 不会出现在任何被比较位置的输入里，所以去掉。
    full_seq = PROMPT + gen_ids[: MAX_NEW - 1]  # 构造完整的输入序列
    input_ids = torch.tensor([full_seq], dtype=torch.long)  # 转为 torch tensor
    with torch.no_grad():  # 禁用梯度（纯推理）
        # 手动指定 position_ids，确保与 C++ runtime 的位置编码一致
        pos = torch.arange(len(full_seq), dtype=torch.long).unsqueeze(0)
        # HF 前向推理，获取所有位置的 logits
        ref = model(input_ids=input_ids, position_ids=pos,
                    use_cache=False).logits[0].numpy()  # 取 batch=0 并转 numpy
    # 校验 HF 和 C++ 的 logits 形状一致
    assert ref.shape == cpp_logits.shape, (ref.shape, cpp_logits.shape)

    # 逐位置比较 logits 的最大绝对误差
    worst = 0.0  # 记录全局最差误差
    for i in range(len(full_seq)):
        err = float(np.abs(cpp_logits[i] - ref[i]).max())  # 该位置的最大绝对误差
        worst = max(worst, err)  # 更新全局最差
        print(f"pos {i}: max_abs_err={err:.3e}")  # 打印该位置误差
        assert err < TOL, f"position {i} diverges: {err}"  # 超过容差则断言失败

    # 验证每步 greedy decode 的 argmax 在两侧完全一致
    for s, g in enumerate(gen_ids):
        row = P - 1 + s  # 对应 logits 矩阵中的行号
        # C++ 内部 argmax 应与生成 token 一致
        assert int(cpp_logits[row].argmax()) == g, f"cpp internal argmax mismatch at {s}"
        # HF argmax 也应与 C++ 生成 token 一致
        assert int(ref[row].argmax()) == g, f"step {s}: cpp={g} ref={int(ref[row].argmax())}"
    # 打印最终结果
    print(f"generated: {gen_ids} (matches HF argmax at every step)")
    print(f"WORST max_abs_err: {worst:.3e} (tol {TOL})")
    print("ALIGNMENT OK")  # 对齐通过


if __name__ == "__main__":
    main()  # 脚本入口点
