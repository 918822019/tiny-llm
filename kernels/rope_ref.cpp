// RoPE 旋转位置编码（Rotary Position Embedding）。
//
// 为什么需要它？attention 本身"不认识顺序"——打乱输入顺序它算出来一样。
// 但语言是有顺序的。RoPE 的办法是：按 token 的位置，把 q/k 向量旋转一个
// 角度。位置不同，旋转角度不同，模型就间接感知到了"谁在前谁在后"。
//
// 具体做法：把每对分量 (x0, x1) 看成平面上的一个点，旋转一个角度 angle：
//   x0' = x0 * cos(angle) - x1 * sin(angle)
//   x1' = x1 * cos(angle) + x0 * sin(angle)
// 这就是二维旋转公式。angle = pos * inv_freq，位置 pos 越大转得越多。
//
// 关键约定（必须和 HF Qwen2 的 "rotate_half" 一致）：
//   x[i] 的旋转搭档是 x[i + head_dim/2]——即 head 向量按前后两半拆分配对，
//   而不是奇偶交错配对。搞错这点，输出"看起来正常"但会和 PyTorch 悄悄偏离。
//
// cos/sin 每个位置只算一次、所有 head 共享，与 HF apply_rotary_pos_emb 一致。

#include "ref_ops.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace tinyqwen {

void rope_ref(float* q, float* k, int n_heads, int n_kv_heads, int head_dim, int pos,
              float theta) {
  const int half = head_dim / 2;  // 配对的"前半"长度
  constexpr int kMaxHalf = 256;   // 支持 head_dim <= 512，远超任何 Qwen 配置
  if (half <= 0 || half > kMaxHalf) {
    std::fprintf(stderr, "rope_ref: unsupported head_dim %d\n", head_dim);
    std::abort();
  }

  // 先算好本位置所有配对要用的 cos/sin。
  // inv_freq[i] = theta ^ (-2i / head_dim)：不同配对用不同"频率"，
  // 低维转得快、高维转得慢，这样能编码丰富的位置信息。
  float cs[kMaxHalf];
  float sn[kMaxHalf];
  for (int i = 0; i < half; ++i) {
    const float exponent = -static_cast<float>(2 * i) / static_cast<float>(head_dim);
    const float inv_freq = std::pow(theta, exponent);
    const float angle = static_cast<float>(pos) * inv_freq;
    cs[i] = std::cos(angle);
    sn[i] = std::sin(angle);
  }

  // 对单个 head 的所有 (x0, x1) 配对施加二维旋转（就地修改）。
  const auto apply = [&](float* x) {
    for (int i = 0; i < half; ++i) {
      const float x0 = x[i];
      const float x1 = x[i + half];
      const float c = cs[i];
      const float s = sn[i];
      const float rotated0 = x0 * c - x1 * s;
      const float rotated1 = x1 * c + x0 * s;
      x[i] = rotated0;
      x[i + half] = rotated1;
    }
  };

  // q 的每个 head、k 的每个 head 都要旋转（v 不需要）。
  for (int h = 0; h < n_heads; ++h) {
    apply(q + static_cast<size_t>(h) * head_dim);
  }
  for (int h = 0; h < n_kv_heads; ++h) {
    apply(k + static_cast<size_t>(h) * head_dim);
  }
}

}  // namespace tinyqwen
