// ============================================================================
// rope_ref.cpp — RoPE 旋转位置编码的参考实现（标量版）
// ============================================================================
// 本文件实现 Rotary Position Embedding (RoPE) 旋转位置编码。
//
// 数学定义：
//   将向量 x 的每对分量 (x[i], x[i + d/2]) 视为二维平面上的点，旋转角度 θ_i：
//     x'[i]       = x[i] * cos(θ_i) - x[i + d/2] * sin(θ_i)
//     x'[i + d/2] = x[i + d/2] * cos(θ_i) + x[i] * sin(θ_i)
//   其中 θ_i = pos * inv_freq[i]，inv_freq[i] = base^(-2i/d)
//   d = head_dim，pos = token 在序列中的位置索引，base = theta（通常 10000）。
//
// 为什么需要 RoPE？
//   Attention 本身是排列不变的——打乱输入顺序它算出来一样。但语言是有顺序的。
//   RoPE 的办法是按 token 位置把 q/k 向量旋转一个角度，位置不同旋转角度不同，
//   模型就间接感知到了"谁在前谁在后"。而且相对位置信息自然编码在旋转差中。
//
// 在 Transformer 中的位置：
//   每个 decoder layer 中，在 Q/K 线性投影之后、attention 计算之前施加 RoPE。
//   V 不需要旋转（只有 Q 和 K 参与位置编码）。
//
// ref 与 neon 版本的关系：
//   - 本文件是标量参考实现，cos/sin 用 std::cos/std::sin 保证精度。
//   - rope_neon.cpp 仅将"施加旋转"的内循环向量化（vmulq + vsubq/vaddq），
//     cos/sin 表的计算与 ref 逐字相同（标量 libm），保证喂给旋转的数据一致。
//
// 关键约定（必须和 HuggingFace Qwen2 的 "rotate_half" 一致）：
//   x[i] 的旋转搭档是 x[i + head_dim/2]——即 head 向量按前后两半拆分配对，
//   而不是奇偶交错配对。搞错这点，输出"看起来正常"但会和 PyTorch 悄悄偏离。
//
// 优化要点：
//   1. cos/sin 表每个位置只算一次，所有 head 共享（head_dim/2 次三角函数调用）。
//   2. 就地修改 q/k，不需要额外输出缓冲区。
//   3. GQA 中 q 有 n_heads 个头、k 有 n_kv_heads 个头，分别遍历旋转。
// ============================================================================

#include "ref_ops.h" // 声明 rope_ref 等参考算子的头文件

#include <cmath>   // std::pow, std::cos, std::sin
#include <cstdio>  // std::fprintf
#include <cstdlib> // std::abort

namespace tinyqwen {
  // =========================================================================
  // rope_ref — RoPE 旋转位置编码参考实现
  // =========================================================================
  // 功能：对 q 和 k 的所有 head 就地施加旋转位置编码
  // 参数：
  //   q          — query 张量，形状 [n_heads, head_dim]，就地修改
  //   k          — key 张量，形状 [n_kv_heads, head_dim]，就地修改
  //   n_heads    — query 的注意力头数量
  //   n_kv_heads — key 的注意力头数量（GQA 中 <= n_heads）
  //   head_dim   — 每个注意力头的维度（Qwen2.5-0.5B 为 64）
  //   pos        — 当前 token 在序列中的位置索引（从 0 开始）
  //   theta      — RoPE 基频（Qwen 默认为 10000.0）
  void rope_ref(float *q, float *k, int n_heads, int n_kv_heads, int head_dim, int pos,
                float theta) {
    // half = head_dim / 2：配对的"前半"长度
    // 旋转配对方式：x[i] 与 x[i + half] 配对（rotate_half 约定）
    const int half = head_dim / 2;
    // 最大支持的 half 值（head_dim <= 512），远超任何 Qwen 配置
    constexpr int kMaxHalf = 256;
    // 合法性检查：half 必须在 [1, 256] 范围内
    if (half <= 0 || half > kMaxHalf) {
      std::fprintf(stderr, "rope_ref: unsupported head_dim %d\n", head_dim);
      std::abort(); // 不支持的 head_dim，终止程序
    }

    // ---- 预计算 cos/sin 表 ----
    // 所有 head 共享同一组 cos/sin 值（因为旋转角度只取决于位置和频率，与 head 无关）
    // cs[i] = cos(pos * inv_freq[i])，sn[i] = sin(pos * inv_freq[i])
    float cs[kMaxHalf]; // cos 值数组
    float sn[kMaxHalf]; // sin 值数组
    for (int i = 0; i < half; ++i) {
      // inv_freq[i] = theta^(-2i/head_dim)：不同配对使用不同的"频率"
      // 低维（i 小）转得快、高维（i 大）转得慢，这样能编码丰富的位置信息
      const float exponent = -static_cast<float>(2 * i) / static_cast<float>(head_dim);
      // 计算逆频率 = base^exponent
      const float inv_freq = std::pow(theta, exponent);
      // 旋转角度 = 位置 × 逆频率
      const float angle = static_cast<float>(pos) * inv_freq;
      // 缓存 cos 和 sin 值
      cs[i] = std::cos(angle);
      sn[i] = std::sin(angle);
    }

    // ---- 定义旋转 lambda：对单个 head 的所有 (x0, x1) 配对施加二维旋转 ----
    // 这是标准的二维旋转公式：
    //   x0' = x0 * cos(θ) - x1 * sin(θ)
    //   x1' = x1 * cos(θ) + x0 * sin(θ)
    // 就地修改 x 数组
    const auto apply = [&](float *x) {
      for (int i = 0; i < half; ++i) {
        const float x0 = x[i];           // 前半部分的第 i 个分量
        const float x1 = x[i + half];    // 后半部分的第 i 个分量（旋转搭档）
        const float c = cs[i];           // 该配对的 cos 值
        const float s = sn[i];           // 该配对的 sin 值
        // 二维旋转公式：新 x0 = x0*cos - x1*sin
        const float rotated0 = x0 * c - x1 * s;
        // 二维旋转公式：新 x1 = x1*cos + x0*sin
        const float rotated1 = x1 * c + x0 * s;
        // 写回旋转后的值（就地修改）
        x[i] = rotated0;
        x[i + half] = rotated1;
      }
    };

    // ---- 对 q 的每个 head 施加旋转 ----
    for (int h = 0; h < n_heads; ++h) {
      apply(q + static_cast<size_t>(h) * head_dim); // q[h] 的起始地址
    }
    // ---- 对 k 的每个 head 施加旋转 ----
    // 注意：v 不需要旋转（只有 q 和 k 参与位置编码）
    for (int h = 0; h < n_kv_heads; ++h) {
      apply(k + static_cast<size_t>(h) * head_dim); // k[h] 的起始地址
    }
  }
} // namespace tinyqwen
