#!/usr/bin/env python3
# align_fake_qwen35_moe_model.py — MoE SSD 卸载机制正确性对齐
#
# 正确性锚点：C++ resident 模式（专家常驻内存）的 logits 必须与 SSD 模式
# （ExpertStore pread + LRU，slots=0 全 miss 与 slots=N 缓存）逐位一致。
# 这是"激活专家按需加载进内存计算"命题的核心验证——两种模式用同一份
# GPTQ 权重，差异只在专家访问路径（resident 指针 vs SSD pread）。
#
# 不依赖 HF/transformers（Qwen3.5 MoE 在 HF 可能未上 main），也不依赖 numpy
# 参考前向——resident 模式本身即正确性 oracle（与既有 dense 路径同源）。
#
# 流程：生成 fake MoE .tqwen → 跑 resident/SSD0/SSD4 三遍（--dump-logits）
#       → 逐字节比对 logits + 生成 token 一致。
import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PY = sys.executable
BIN = os.path.join(ROOT, "build", "runtime", "tinyqwen")
GEN = os.path.join(HERE, "make_fake_qwen35_moe_model.py")

PROMPT = [3, 7, 11, 2]
TOL = 0.0  # 逐位一致（同二进制）


def run(model, extra, logits_path):
    cmd = [BIN, "--model", model,
           "--tokens", ",".join(str(t) for t in PROMPT),
           "--max-new-tokens", "8", "--max-seq-len", "32",
           "--matvec-impl", "ref", "--ops-impl", "ref",
           "--dump-logits", logits_path] + extra
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        sys.stderr.write(r.stderr)
        raise SystemExit(f"run failed (rc={r.returncode}): {' '.join(extra)}")
    # 解析 generated_ids
    gen = []
    for line in r.stdout.splitlines():
        if line.startswith("generated_ids:"):
            gen = [int(x) for x in line.split(":", 1)[1].split()]
    return gen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/fake_moe.tqwen")
    ap.add_argument("--slots", type=int, default=4)
    args = ap.parse_args()

    if not os.path.exists(GEN):
        raise SystemExit(f"missing generator: {GEN}")
    if not os.path.exists(BIN):
        raise SystemExit(f"missing binary: {BIN}（先 cmake --build build）")

    subprocess.run([PY, GEN, "--out", args.out], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

    lr = os.path.join("/tmp", "align_moe_resident.bin")
    l0 = os.path.join("/tmp", "align_moe_ssd0.bin")
    lN = os.path.join("/tmp", "align_moe_ssdN.bin")

    gen_r = run(args.out, [], lr)
    gen_0 = run(args.out, ["--moe-ssd", "--moe-expert-cache-slots", "0"], l0)
    gen_N = run(args.out, ["--moe-ssd", "--moe-expert-cache-slots", str(args.slots)], lN)

    br = open(lr, "rb").read()
    b0 = open(l0, "rb").read()
    bN = open(lN, "rb").read()

    ok = True
    if br != b0:
        print(f"[FAIL] resident vs ssd-slots0 logits differ ({len(br)} vs {len(b0)} bytes)")
        ok = False
    else:
        print("[ OK ] resident == ssd-slots0 (全 miss) logits 逐位一致")
    if br != bN:
        print(f"[FAIL] resident vs ssd-slots{args.slots} logits differ")
        ok = False
    else:
        print(f"[ OK ] resident == ssd-slots{args.slots} (缓存) logits 逐位一致")
    if gen_r != gen_0 or gen_r != gen_N:
        print(f"[FAIL] generated ids differ: r={gen_r} 0={gen_0} N={gen_N}")
        ok = False
    else:
        print(f"[ OK ] generated ids 一致: {gen_r}")

    print("\n结论：SSD 专家卸载（pread + LRU）与 resident 指针路径逐位一致——"
          "激活专家按需加载进内存计算的正确性成立。")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
