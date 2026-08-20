#!/usr/bin/env bash
# CUDA 后端 A/B 测试脚本
#
# 用法：
#   ./scripts/bench_cuda_backend.sh
#
# 需要 CUDA 构建：
#   cmake -S . -B build-cuda -DCMAKE_BUILD_TYPE=Release
#   cmake --build build-cuda -j
#
# 测试内容：
#   1. CPU 后端（baseline）
#   2. CUDA 后端（--backend cuda）
#   3. GPU decode engine（--engine cuda）
#
# 注意：--backend cuda 是每个算子单独调用 CUDA kernel，性能不如 --engine cuda。
#      --engine cuda 是整段 forward 一起跑，性能最好。

set -euo pipefail
cd "$(dirname "$0")/.."

if [ ! -f build-cuda/runtime/tinyqwen ]; then
    echo "error: CUDA build not found. Please run:"
    echo "  cmake -S . -B build-cuda -DCMAKE_BUILD_TYPE=Release"
    echo "  cmake --build build-cuda -j"
    exit 1
fi

MODEL="model.tqwen"
TOKENS="prompt_tokens.json"

if [ ! -f "$MODEL" ] || [ ! -f "$TOKENS" ]; then
    echo "error: model or tokens not found. Please run:"
    echo "  python tools/export_qwen_to_tiny.py --model Qwen/Qwen2.5-0.5B --out $MODEL"
    echo "  python tools/tokenize_prompt.py --model Qwen/Qwen2.5-0.5B --prompt '你好' --chat --out $TOKENS"
    exit 1
fi

echo "=== A/B 测试：CUDA 后端 ==="
echo ""

# 1. CPU 后端（baseline）
echo "=== [1/3] CPU 后端（baseline）==="
./scripts/record.sh cpu_backend --skip-verify

echo ""

# 2. CUDA 后端（每个算子单独调用）
echo "=== [2/3] CUDA 后端（--backend cuda）==="
./scripts/record.sh cuda_backend --skip-verify --extra-args "--backend cuda"

echo ""

# 3. GPU decode engine（整段 forward 一起跑）
echo "=== [3/3] GPU decode engine（--engine cuda）==="
./scripts/record.sh gpu_engine --skip-verify --extra-args "--engine cuda"

echo ""
echo "=== 测试完成 ==="
echo "查看结果：docs/optimization_log.md"
