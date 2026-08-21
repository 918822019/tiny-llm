#!/usr/bin/env bash
# ============================================================================
# bench_cuda_backend.sh — CUDA 后端 A/B 测试脚本
# ============================================================================
# 用途：依次运行三种后端配置的性能测试，用于对比 CPU、CUDA 逐算子调用、
#       GPU decode engine（整段 forward）之间的性能差异。
#
# 用法：
#   ./scripts/bench_cuda_backend.sh
#
# 前置条件（需要先完成 CUDA 构建）：
#   cmake -S . -B build-cuda -DCMAKE_BUILD_TYPE=Release
#   cmake --build build-cuda -j
#
# 测试内容（3 轮）：
#   1. CPU 后端（baseline）— 纯 CPU 推理作为参照基线
#   2. CUDA 后端（--backend cuda）— 每个算子单独调用 CUDA kernel
#   3. GPU decode engine（--engine cuda）— 整段 forward 在 GPU 上跑
#
# 注意：
#   --backend cuda 是每个算子单独调用 CUDA kernel，存在频繁的 host-device
#   数据传输开销，性能不如 --engine cuda。
#   --engine cuda 是整段 forward 一起跑在 GPU 上，减少数据搬运，性能最好。
#
# 依赖环境：
#   - build-cuda/runtime/tinyqwen 已编译
#   - model.tqwen 和 prompt_tokens.json 已生成
#   - scripts/record.sh 可用
# ============================================================================

# set -e: 任何命令失败立即退出；-u: 引用未定义变量报错；-o pipefail: 管道中任一命令失败则整体失败
set -euo pipefail

# 切换到项目根目录
cd "$(dirname "$0")/.."

# 检查 CUDA 构建产物是否存在，不存在则打印构建指引并退出
if [ ! -f build-cuda/runtime/tinyqwen ]; then
    echo "error: CUDA build not found. Please run:"
    echo "  cmake -S . -B build-cuda -DCMAKE_BUILD_TYPE=Release"
    echo "  cmake --build build-cuda -j"
    exit 1
fi

# 模型文件路径
MODEL="model.tqwen"
# prompt token 文件路径（由 tools/tokenize_prompt.py 生成）
TOKENS="prompt_tokens.json"

# 检查模型文件和 token 文件是否都存在
if [ ! -f "$MODEL" ] || [ ! -f "$TOKENS" ]; then
    echo "error: model or tokens not found. Please run:"
    echo "  python tools/export_qwen_to_tiny.py --model Qwen/Qwen2.5-0.5B --out $MODEL"
    echo "  python tools/tokenize_prompt.py --model Qwen/Qwen2.5-0.5B --prompt '你好' --chat --out $TOKENS"
    exit 1
fi

# 打印测试标题
echo "=== A/B 测试：CUDA 后端 ==="
echo ""

# ---- 第 1 轮：CPU 后端（baseline）----
echo "=== [1/3] CPU 后端（baseline）==="
# 调用 record.sh 进行测速+记录，--skip-verify 跳过正确性门禁（A/B 测试只关注性能）
./scripts/record.sh cpu_backend --skip-verify

echo ""

# ---- 第 2 轮：CUDA 后端（每个算子单独调用 CUDA kernel）----
echo "=== [2/3] CUDA 后端（--backend cuda）==="
# --extra-args 将 "--backend cuda" 透传给 runtime，启用逐算子 CUDA 模式
./scripts/record.sh cuda_backend --skip-verify --extra-args "--backend cuda"

echo ""

# ---- 第 3 轮：GPU decode engine（整段 forward 在 GPU 上跑）----
echo "=== [3/3] GPU decode engine（--engine cuda）==="
# --extra-args 将 "--engine cuda" 透传给 runtime，启用整段 GPU forward 模式
./scripts/record.sh gpu_engine --skip-verify --extra-args "--engine cuda"

echo ""
# 打印完成提示及结果查看位置
echo "=== 测试完成 ==="
echo "查看结果：docs/optimization_log.md"
