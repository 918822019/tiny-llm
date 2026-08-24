#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
export_qwen_to_tiny_vq2.py — 把 HuggingFace Qwen 权重导出为 .tqwen 的 VQ2(2-bit)格式。

KronQ-VQ 2-bit 部署洁净点的最小自含实现（块向量量化）：
    对每个大线性子层权重，把它按每 d=4 个连续权重切成 d 维向量块，做
    K=256 的向量 k-means（Lloyd 迭代），得到：
      - 码本：[K=256, d=4] fp16 = 2048 字节，in-band 存在张量数据区头部；
      - 索引：每块一个 uint8（每 d=4 个权重 1 字节），行主序。
    码率 = log2(K)/d = log2(256)/4 = 2 bit/权重。
    反量化 = 查表取码本行（一个 4 维向量），零算术、每块恰 1 字节、免 bit-pack。

与 INT4 导出器(export_qwen_to_tiny_i4.py)的异同：
    - 同样流式两遍写盘（第一遍取 shape 定布局，第二遍量化写盘）；
    - 量化器换成 per-sublayer 向量 k-means（免训练，无需校准数据）；
    - 大张量(embed/lm_head)保持 fp32 lookup/投影（与 i4 一致），因此无需
      i4 那套"鲸鱼张量行拆分 + 进程池并行"——被量化的线性层都是中等尺寸，
      串行即可，代码大幅简化；
    - header.dtype = 4 (kVQ2)；v2 ext 用基座 pack_v2_ext（VQ2 不需要
      quant_group_size，K=256 / d=4 为编译期常量）。

用法:
    python tools/export_qwen_to_tiny_vq2.py \
        --model models/Qwen3.5-0.8B --out model_qwen35_08b_vq2.tqwen

精度提示：
    v1 用朴素向量 k-means（无旋转 / 无 GPTQ 误差补偿 / 单遍码本），2-bit 下
    PPL 可能明显退化——本导出器的目标是打通格式→内核→测算全链路并验证
    "部署洁净性"，精度优化（TwoPass 码本精化 / BiIP 旋转）是后续工作。
"""

import argparse
import struct
import sys
import time
from pathlib import Path

import numpy as np

# 复用基座导出器的格式契约与工具（与 i4 导出器同一套基础设施）
sys.path.insert(0, str(Path(__file__).parent))
from export_qwen_to_tiny import (  # noqa: E402
    MAGIC, FORMAT_VERSION, DTYPE_F32, DTYPE_F16, MAX_NAME,
    HEADER_FMT, ENTRY_FMT,
    align_up, find_local_dir, load_config, plan_tensors,
    print_table_summary, pack_v2_ext, MODEL_QWEN35,
)
# 复用 i4 导出器的懒加载与"该不该量化"判定
from export_qwen_to_tiny_i4 import LazyTensors, should_quantize  # noqa: E402

# VQ2 dtype 编号，与 runtime/tiny_format.h 的 Dtype::kVQ2 一致
DTYPE_VQ2 = 4
# 块向量量化参数：K=256 码本条目、d=4 块大小 → 码率 = log2(K)/d = 2 bit/权重
VQ2_K = 256
VQ2_BLOCK = 4
VQ2_CODEBOOK_BYTES = VQ2_K * VQ2_BLOCK * 2  # [K, d] fp16 = 2048 字节


def vq2_tensor_nbytes(rows: int, cols: int) -> int:
    """VQ2 张量字节数 = 码本(2048B) + 索引(rows * cols/4 字节)。与 C++ vq2_tensor_bytes 一致。"""
    return VQ2_CODEBOOK_BYTES + rows * (cols // VQ2_BLOCK)


def _assign_nearest(blocks: np.ndarray, centroids: np.ndarray,
                    chunk: int = 1 << 18) -> np.ndarray:
    """把每个 d 维块分配到最近质心（平方欧氏距离）。分块避免 (N, K) 内存爆炸。"""
    c = centroids.astype(np.float32)
    n = blocks.shape[0]
    out = np.empty(n, dtype=np.uint8)
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        seg = blocks[s:e, None, :]                      # (m, 1, d)
        d2 = ((seg - c[None, :, :]) ** 2).sum(-1)       # (m, K)
        out[s:e] = np.argmin(d2, axis=1).astype(np.uint8)
    return out


def vq2_vector_kmeans(tensor: np.ndarray, k: int = VQ2_K, block: int = VQ2_BLOCK,
                      n_iter: int = 12, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """对一个 [rows, cols] 权重矩阵做块向量 k-means。

    把每 block 个连续权重当作一个 block 维向量，Lloyd 迭代拟合 K 个质心。
    返回 (fp16 码本 [K, block], uint8 索引 [rows, cols/block])。
    """
    rows, cols = tensor.shape
    assert cols % block == 0, f"cols {cols} 须被 block {block} 整除"
    blocks = np.ascontiguousarray(tensor).astype(np.float32).reshape(-1, block)  # (N, d)
    n_blocks = blocks.shape[0]
    rng = np.random.default_rng(seed)
    pick = rng.choice(n_blocks, size=k, replace=(n_blocks < k))
    centroids = blocks[pick].astype(np.float32).copy()  # (K, d)

    # 加速：Lloyd 迭代期只在子采样块上拟合质心（k-means 对子采样鲁棒），
    # 收敛后最后才做一次全量分配。大张量从 O(n_iter·N·K) 降到 ~O(1·N·K)。
    subsample = 200_000
    if n_blocks > subsample:
        sub_idx = rng.choice(n_blocks, size=subsample, replace=False)
        fit_blocks = blocks[sub_idx]
    else:
        fit_blocks = blocks

    for _ in range(n_iter):
        assign_sub = _assign_nearest(fit_blocks, centroids)
        sums = np.zeros((k, block), dtype=np.float64)
        counts = np.zeros(k, dtype=np.int64)
        np.add.at(sums, assign_sub, fit_blocks.astype(np.float64))
        np.add.at(counts, assign_sub, 1)
        nonempty = counts > 0
        new_centroids = centroids.copy()
        new_centroids[nonempty] = (sums[nonempty] / counts[nonempty, None]).astype(np.float32)
        if not nonempty.all():  # 空簇：随机数据块重置，避免死码本
            empty_idx = np.where(~nonempty)[0]
            new_centroids[empty_idx] = blocks[rng.choice(n_blocks, size=len(empty_idx),
                                                         replace=True)]
        if np.array_equal(new_centroids, centroids):
            centroids = new_centroids
            break
        centroids = new_centroids

    # 最终全量分配（用收敛后的质心）
    assign = _assign_nearest(blocks, centroids)
    codebook_fp16 = centroids.astype(np.float16)                    # (K, d)
    indices = assign.astype(np.uint8).reshape(rows, cols // block)  # (rows, cols/d)
    return codebook_fp16, indices


def pack_vq2_tensor(tensor: np.ndarray, k: int, block: int, n_iter: int, seed: int) -> bytes:
    """量化并打包成 [码本 [K,d] 2048B][uint8 索引]，与 C++ 内核读取布局一致。"""
    codebook_fp16, indices = vq2_vector_kmeans(tensor, k=k, block=block,
                                               n_iter=n_iter, seed=seed)
    return (np.ascontiguousarray(codebook_fp16).tobytes() +
            np.ascontiguousarray(indices).tobytes())


def write_tqwen_vq2(out_path, cfg, names, load, k=VQ2_K, block=VQ2_BLOCK,
                    n_iter=12, seed=0, version=FORMAT_VERSION, ext=None) -> int:
    """写一个 VQ2 混合 dtype 的 .tqwen 文件（流式两遍，串行量化）。"""
    t0 = time.time()

    # ---- 第一遍：取 shape、定布局 ----
    shapes = {}
    for name in names:
        arr = load(name)
        assert 1 <= arr.ndim <= 4, f"{name}: ndim {arr.ndim} not in [1,4]"
        assert len(name) <= MAX_NAME, f"tensor name too long: {name}"
        shapes[name] = tuple(arr.shape)
        del arr

    entries = []  # (name, shape, offset, nbytes, dtype_code)
    table_end = struct.calcsize(HEADER_FMT) + len(names) * struct.calcsize(ENTRY_FMT)
    data_offset = align_up(table_end)
    offset = data_offset
    print(f"Planning {len(names)} tensors (VQ2 K={k}, block={block}, "
          f"kmeans_iters={n_iter})...")
    for name in names:
        shape = shapes[name]
        # 只有 2D、该量化、且列数能被块大小整除的张量才走 VQ2；否则保 fp32
        if len(shape) == 2 and should_quantize(name) and shape[1] % block == 0:
            nbytes = vq2_tensor_nbytes(shape[0], shape[1])
            dtype_code = DTYPE_VQ2
            tag = "[VQ2]"
        elif "embed_tokens" in name:
            # embed 查表 + tied-lm_head：存 f16（把总体积大头砍半；runtime 按
            # embed_dtype_ 路由 f16 查表与 matvec_f16）
            n = 1
            for d in shape:
                n *= d
            nbytes = n * 2
            dtype_code = DTYPE_F16
            tag = "[F16]"
        else:
            n = 1
            for d in shape:
                n *= d
            nbytes = n * 4
            dtype_code = DTYPE_F32
            tag = "[F32]"
        entries.append((name, list(shape), offset, nbytes, dtype_code))
        print(f"  {tag} {name:<52} {str(list(shape)):<20} -> {nbytes:>12} bytes")
        offset = align_up(offset + nbytes)
    total_bytes = offset

    reserved = b"\x00" * 96
    if version >= 2:
        ext_bytes = pack_v2_ext(ext or {})
        reserved = ext_bytes + b"\x00" * (96 - len(ext_bytes))

    header = struct.pack(
        HEADER_FMT,
        MAGIC, version, DTYPE_VQ2,
        int(cfg["n_layers"]), int(cfg["hidden_size"]), int(cfg["intermediate_size"]),
        int(cfg["n_heads"]), int(cfg["n_kv_heads"]), int(cfg["head_dim"]),
        int(cfg["vocab_size"]), int(cfg["max_seq_len"]), int(cfg["tied"]),
        0,
        float(cfg["rms_norm_eps"]), float(cfg["rope_theta"]),
        len(names), struct.calcsize(HEADER_FMT), data_offset, total_bytes,
        reserved,
    )

    # ---- 写文件：header + 索引表 + 数据区（第二遍流式）----
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as out:
        out.write(header)
        for name, shape, off, nbytes, dtype_code in entries:
            out.write(struct.pack(
                ENTRY_FMT,
                name.encode("ascii"), dtype_code, len(shape),
                *shape, *[0] * (4 - len(shape)),
                off, nbytes,
            ))
        out.write(b"\x00" * (data_offset - out.tell()))

        for name, shape, off, nbytes, dtype_code in entries:
            assert out.tell() == off, f"offset drift at {name}"
            arr = load(name)
            if dtype_code == DTYPE_VQ2:
                blob = pack_vq2_tensor(arr, k=k, block=block, n_iter=n_iter, seed=seed)
            elif dtype_code == DTYPE_F16:
                blob = np.ascontiguousarray(arr.astype(np.float16)).tobytes()
            else:
                blob = np.ascontiguousarray(arr.astype(np.float32)).tobytes()
            assert len(blob) == nbytes, f"{name}: blob {len(blob)} != nbytes {nbytes}"
            out.write(blob)
            pad = align_up(off + nbytes) - (off + nbytes)
            out.write(b"\x00" * pad)
            del arr
        assert out.tell() == total_bytes, f"final size {out.tell()} != {total_bytes}"

    return total_bytes


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="HF model dir")
    p.add_argument("--out", required=True, help="output .tqwen path")
    p.add_argument("--k", type=int, default=VQ2_K, help="码本条目数（默认 256）")
    p.add_argument("--block", type=int, default=VQ2_BLOCK, help="块大小 d（默认 4 → 2bit）")
    p.add_argument("--kmeans-iters", type=int, default=12, help="k-means Lloyd 迭代次数")
    p.add_argument("--seed", type=int, default=0, help="k-means 随机种子（可复现）")
    p.add_argument("--max-seq-len", type=int, default=0, help="override max_seq_len")
    args = p.parse_args()

    if args.k < 2 or args.k > 256:
        sys.exit(f"error: --k 须在 [2,256]（>256 超出 uint8 索引），实际 {args.k}")
    if args.block not in (1, 2, 4, 8):
        sys.exit(f"error: --block 须为 1/2/4/8，实际 {args.block}")
    bits = int(round(np.log2(args.k) / args.block * args.block))  # 仅提示用
    print(f"[VQ2] K={args.k}, block={args.block} → 码率 = log2({args.k})/{args.block} "
          f"= {np.log2(args.k) / args.block:.2f} bit/权重")

    model_dir = find_local_dir(args.model)
    tcfg, model_type = load_config(model_dir)
    tiny_names, src_of, transform_of = plan_tensors(tcfg, model_type)

    max_seq_len = args.max_seq_len or tcfg.get("max_position_embeddings", 32768)
    tied = int(tcfg.get("tie_word_embeddings", True))
    tqwen_cfg = {
        "n_layers": tcfg["num_hidden_layers"],
        "hidden_size": tcfg["hidden_size"],
        "intermediate_size": tcfg["intermediate_size"],
        "n_heads": tcfg["num_attention_heads"],
        "n_kv_heads": tcfg["num_key_value_heads"],
        "head_dim": tcfg["head_dim"],
        "vocab_size": tcfg["vocab_size"],
        "max_seq_len": max_seq_len,
        "tied": tied,
        "rms_norm_eps": tcfg["rms_norm_eps"],
        "rope_theta": tcfg["rope_theta"],
    }

    ext = {"model_type": model_type}
    if model_type == MODEL_QWEN35:
        rp = tcfg.get("rope_parameters") or {}
        ext.update({
            "linear_num_qk_heads": tcfg.get("linear_num_key_heads", 0),
            "linear_num_v_heads": tcfg.get("linear_num_value_heads", 0),
            "linear_qk_head_dim": tcfg.get("linear_key_head_dim", 0),
            "linear_v_head_dim": tcfg.get("linear_value_head_dim", 0),
            "linear_conv_kernel_dim": tcfg.get("linear_conv_kernel_dim", 4),
            "full_attention_interval": tcfg.get("full_attention_interval", 4),
            "partial_rotary_factor": rp.get("partial_rotary_factor",
                                            tcfg.get("partial_rotary_factor", 0.25)),
            "eos_token_id": tcfg.get("eos_token_id", 248044),
        })

    st = LazyTensors(model_dir)

    def load(tiny_name: str) -> np.ndarray:
        arr = st.get(src_of[tiny_name])
        transform = transform_of[tiny_name]
        if transform is not None:
            arr = transform(arr)
        return np.asarray(arr, dtype=np.float32)

    names = list(tiny_names)
    # VQ2 v1：tied 模型的 lm_head 直接共享 fp32 embed（runtime 侧 lm_head_is_f32_），
    # 不额外导出量化副本（与 i4 的可选 lm_head 量化不同，保持最简）。

    t0 = time.time()
    total = write_tqwen_vq2(args.out, tqwen_cfg, names, load, k=args.k, block=args.block,
                            n_iter=args.kmeans_iters, seed=args.seed,
                            version=FORMAT_VERSION, ext=ext)
    mb = total / (1024 * 1024)
    print(f"\nDone: {args.out} ({mb:.1f} MB) in {time.time() - t0:.1f}s")
    print_table_summary(args.out)


if __name__ == "__main__":
    main()
