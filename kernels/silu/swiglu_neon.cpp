// SwiGLU 融合的 NEON 版（aarch64）：gate[i] = silu(gate[i]) * up[i]，单遍。
//
// ref（swiglu_ref）逐元素调 std::exp——expf 是标量慢路径（几十周期），4864 维
// 逐元素跑是这块 176µs 的主因。这里把 exp 向量化：
//     exp(x) = 2^(x·log2e) = 2^n · 2^f   （n 整数、f ∈ [-0.5, 0.5]）
//   - 2^n：把 (n+127)<<23 直接当 float 指数位（位操作，一条指令）；
//   - 2^f：e^(f·ln2) 的 6 阶泰勒（在 [-0.5,0.5] 上相对误差 ~1e-7），Horner 求值。
// 然后 silu = gate/(1+exp)（vdivq，与 ref 同款除法形式，对大负数安全），再乘 up。
//
// 数值差异：exp 的多项式逼近 + float 位拼 2^n，相对 swiglu_ref（std::exp）约
// 1e-7 相对误差；单测按容差门禁（rel 1e-4，含 |x|≤100 溢出边界）对齐。
// 贪心决策不受影响（golden token 逐位一致）。

#include "dispatch.h" // TINYQWEN_SWIGLU_VARIANT 自注册宏
#include "ref_ops.h"

#if defined(__aarch64__) || defined(_M_ARM64)

#include <arm_neon.h>

#include <cmath>

namespace tinyqwen {
  namespace {
    constexpr float kLog2e = 1.4426950408889634f; // 1/ln2
    constexpr float kLn2 = 0.6931471805599453f;

    // 向量化的 exp(x)。对越界指数钳位到 [-126,127]，避免 inf/nan，
    // 与 silu 在 ±大数处趋于 0/x 的语义一致。
    inline float32x4_t vexpq_f32(float32x4_t x) {
      // y = x * log2e；n = round(y)；f = y - n。
      const float32x4_t vlog2e = vdupq_n_f32(kLog2e);
      const float32x4_t y = vmulq_f32(x, vlog2e);
      // round-to-nearest：加 1.5*2^23 再减回（利用 float 尾数截断实现舍入）。
      const float32x4_t magic = vdupq_n_f32(12582912.0f); // 1.5 * 2^23
      const float32x4_t n_f = vsubq_f32(vaddq_f32(y, magic), magic);
      const float32x4_t f = vsubq_f32(y, n_f);

      // 2^n：(n + 127) << 23  reinterpret 成 float（钳位防溢出）。
      int32x4_t n_i = vcvtq_s32_f32(n_f);
      n_i = vmaxq_s32(n_i, vdupq_n_s32(-126));
      n_i = vminq_s32(n_i, vdupq_n_s32(127));
      const int32x4_t exp_bits = vshlq_n_s32(vaddq_s32(n_i, vdupq_n_s32(127)), 23);
      const float32x4_t pow2n = vreinterpretq_f32_s32(exp_bits);

      // 2^f = e^(f·ln2)，6 阶泰勒 + Horner（f∈[-0.5,0.5]，误差 ~1e-7）。
      const float32x4_t t = vmulq_f32(f, vdupq_n_f32(kLn2));
      float32x4_t p = vdupq_n_f32(1.0f / 720.0f);
      p = vfmaq_f32(vdupq_n_f32(1.0f / 120.0f), t, p);
      p = vfmaq_f32(vdupq_n_f32(1.0f / 24.0f), t, p);
      p = vfmaq_f32(vdupq_n_f32(1.0f / 6.0f), t, p);
      p = vfmaq_f32(vdupq_n_f32(0.5f), t, p);
      p = vfmaq_f32(vdupq_n_f32(1.0f), t, p);
      p = vfmaq_f32(vdupq_n_f32(1.0f), t, p);

      return vmulq_f32(pow2n, p);
    }

    void swiglu_neon(float *gate, const float *up, int n) {
      const float32x4_t one = vdupq_n_f32(1.0f);
      int i = 0;
      for (; i + 4 <= n; i += 4) {
        const float32x4_t g = vld1q_f32(gate + i);
        const float32x4_t u = vld1q_f32(up + i);
        const float32x4_t e = vexpq_f32(vnegq_f32(g)); // exp(-gate)
        const float32x4_t denom = vaddq_f32(one, e);   // 1 + exp(-gate)
        const float32x4_t silu = vdivq_f32(g, denom);  // gate / denom
        vst1q_f32(gate + i, vmulq_f32(silu, u));       // * up
      }
      for (; i < n; ++i) { // 标量尾段（与 swiglu_ref 同款公式）
        const float denom = 1.0f + __builtin_expf(-gate[i]);
        const float s = gate[i] / denom;
        gate[i] = s * up[i];
      }
    }
  } // namespace

  // 自注册进 swiglu 注册表，名 "neon"。仅 aarch64 构建存在。
  TINYQWEN_SWIGLU_VARIANT(swiglu_neon, "neon");
} // namespace tinyqwen

#endif // defined(__aarch64__) || defined(_M_ARM64)
