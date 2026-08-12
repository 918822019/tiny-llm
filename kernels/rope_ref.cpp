// 旋转位置编码 RoPE（reference，in-place）。
//
// 配对接定（必须与 HF Qwen2 的 "rotate_half" 一致）：
//   x[i] 的旋转配对是 x[i + head_dim/2]——head 向量按前后两半拆分，
//   而不是交错配对。搞错这一点输出依然"看起来正常"，但会与 PyTorch
//   悄悄偏离。
//
// cos/sin 每个位置只算一次（所有 head 共享），与 HF 的
// apply_rotary_pos_emb 语义一致。

#include "ref_ops.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace tinyqwen {

void rope_ref(float* q, float* k, int n_heads, int n_kv_heads, int head_dim, int pos,
              float theta) {
  const int half = head_dim / 2;
  constexpr int kMaxHalf = 256;  // 支持 head_dim <= 512，远超任何 Qwen 配置
  if (half <= 0 || half > kMaxHalf) {
    std::fprintf(stderr, "rope_ref: unsupported head_dim %d\n", head_dim);
    std::abort();
  }

  // inv_freq[i] = theta ^ (-2i / head_dim)；angle = pos * inv_freq[i]
  float cs[kMaxHalf];
  float sn[kMaxHalf];
  for (int i = 0; i < half; ++i) {
    const float inv_freq =
        std::pow(theta, -static_cast<float>(2 * i) / static_cast<float>(head_dim));
    const float angle = static_cast<float>(pos) * inv_freq;
    cs[i] = std::cos(angle);
    sn[i] = std::sin(angle);
  }

  // 对单个 head 的所有 (x0, x1) 配对施加二维旋转。
  const auto apply = [&](float* x) {
    for (int i = 0; i < half; ++i) {
      const float x0 = x[i];
      const float x1 = x[i + half];
      x[i] = x0 * cs[i] - x1 * sn[i];
      x[i + half] = x1 * cs[i] + x0 * sn[i];
    }
  };

  for (int h = 0; h < n_heads; ++h) apply(q + static_cast<size_t>(h) * head_dim);
  for (int h = 0; h < n_kv_heads; ++h) apply(k + static_cast<size_t>(h) * head_dim);
}

}  // namespace tinyqwen
