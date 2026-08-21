// ============================================================================
// swiglu_neon.cpp — SwiGLU 融合算子的 NEON SIMD 优化版（aarch64）
// ============================================================================
// 本文件实现 SwiGLU 融合算子的 ARM NEON 向量化版本：
//   gate[i] = silu(gate[i]) * up[i]（就地，单遍扫描）
//
// 数学定义与 swiglu_ref.cpp 完全相同：
//   SwiGLU(gate, up) = SiLU(gate) ⊙ up = (gate / (1 + exp(-gate))) ⊙ up
//
// 在 Transformer 中的位置：
//   FFN 的核心计算，位于 gate_proj/up_proj 之后、down_proj 之前。
//
// ref 与 neon 版本的关系：
//   - 算法语义一致，但 exp 的实现不同：ref 用 std::exp（标量 libm），
//     本版用自定义 vexpq_f32（NEON 向量化多项式逼近）。
//   - 数值差异 ~1e-7 相对误差，单测按容差门禁（rel 1e-4）对齐。
//   - 贪心决策不受影响（golden token 逐位一致）。
//
// 优化要点：
//   1. 向量化 exp(x) = 2^(x·log2e) = 2^n · 2^f：
//      - 2^n：把整数 n 编码到 float 的指数位（位操作，一条指令）
//      - 2^f：e^(f·ln2) 的 6 阶泰勒展开，Horner 求值（f ∈ [-0.5, 0.5]）
//   2. 主循环每次处理 4 个元素（一个 float32x4），吞吐量是标量的 4 倍。
//   3. SiLU 使用除法形式（vdivq），与 ref 同款，对大负数安全。
//   4. ref 逐元素调 std::exp 是性能瓶颈（几十周期/次），向量化后大幅提速。
// ============================================================================

#include "dispatch.h" // TINYQWEN_SWIGLU_VARIANT 自注册宏
#include "ref_ops.h"  // 声明 swiglu 函数签名

#if defined(__aarch64__) || defined(_M_ARM64) // 仅在 ARM64 平台编译

#include <arm_neon.h> // ARM NEON SIMD intrinsic 头文件

#include <cmath> // __builtin_expf（标量尾段）

namespace tinyqwen {
  namespace { // 匿名命名空间：内部辅助函数不暴露到外部链接
    // log2(e) = 1/ln(2)，用于将自然指数转换为以 2 为底的指数
    constexpr float kLog2e = 1.4426950408889634f;
    // ln(2)，用于泰勒展开中将 2^f 转换为 e^(f·ln2)
    constexpr float kLn2 = 0.6931471805599453f;

    // =========================================================================
    // vexpq_f32 — NEON 向量化的 exp(x) 近似计算
    // =========================================================================
    // 功能：对 4 个 float 并行计算 exp(x) 的近似值
    // 参数：
    //   x — 包含 4 个 float 的 NEON 寄存器
    // 返回值：包含 4 个 exp(x) 近似值的 NEON 寄存器
    // 算法：
    //   exp(x) = 2^(x · log2(e))
    //          = 2^(n + f)        其中 n = round(x·log2e), f = x·log2e - n
    //          = 2^n · 2^f
    //   - 2^n：通过 IEEE 754 浮点格式直接构造（将 n+127 左移 23 位作为指数域）
    //   - 2^f = e^(f·ln2)：在 f ∈ [-0.5, 0.5] 上用 6 阶泰勒展开逼近
    // 精度：相对误差 ~1e-7，对越界指数钳位到 [-126, 127] 避免 inf/nan
    inline float32x4_t vexpq_f32(float32x4_t x) {
      // ---- 步骤 1：将 x 转换为以 2 为底的指数 y = x * log2(e) ----
      const float32x4_t vlog2e = vdupq_n_f32(kLog2e); // 广播 log2(e) 到 4 lane
      const float32x4_t y = vmulq_f32(x, vlog2e);     // y = x * log2(e)

      // ---- 步骤 2：将 y 分解为整数部分 n 和小数部分 f ----
      // round-to-nearest 技巧：利用 IEEE 754 float 的尾数截断特性
      // magic = 1.5 * 2^23 = 12582912.0f
      // 当 |y| < 2^23 时，(y + magic) 的尾数低位被截断，等效于四舍五入到整数
      const float32x4_t magic = vdupq_n_f32(12582912.0f); // 1.5 * 2^23，round-to-nearest 魔法数
      // n_f = round(y)：加 magic 再减 magic，利用浮点精度丢失实现舍入
      const float32x4_t n_f = vsubq_f32(vaddq_f32(y, magic), magic);
      // f = y - round(y)：小数部分，范围约 [-0.5, 0.5]
      const float32x4_t f = vsubq_f32(y, n_f);

      // ---- 步骤 3：构造 2^n ----
      // IEEE 754 float32 格式：符号(1bit) + 指数(8bit) + 尾数(23bit)
      // 2^n 的表示：指数域 = n + 127（bias），尾数域 = 0
      // vcvtq_s32_f32: 将 float 转为 int32（截断取整）
      int32x4_t n_i = vcvtq_s32_f32(n_f);
      // 钳位到 [-126, 127]：防止指数溢出产生 inf 或 denormal
      n_i = vmaxq_s32(n_i, vdupq_n_s32(-126)); // 下界：float 最小正规指数
      n_i = vminq_s32(n_i, vdupq_n_s32(127));  // 上界：float 最大指数
      // vaddq_s32: n + 127（加上 IEEE 754 bias）
      // vshlq_n_s32: 左移 23 位，放到指数域的位置
      const int32x4_t exp_bits = vshlq_n_s32(vaddq_s32(n_i, vdupq_n_s32(127)), 23);
      // vreinterpretq_f32_s32: 将 int32 位模式重新解释为 float（零开销类型转换）
      const float32x4_t pow2n = vreinterpretq_f32_s32(exp_bits); // 得到 2^n

      // ---- 步骤 4：用 6 阶泰勒展开计算 2^f = e^(f·ln2) ----
      // e^t ≈ 1 + t + t²/2! + t³/3! + t⁴/4! + t⁵/5! + t⁶/6!
      // Horner 法则从内到外求值：p = c0 + t*(c1 + t*(c2 + ...))
      // 这样只需 n 次乘法和 n 次加法，且数值更稳定
      const float32x4_t t = vmulq_f32(f, vdupq_n_f32(kLn2)); // t = f * ln(2)
      // 从最高阶系数开始，逐步 Horner 求值
      float32x4_t p = vdupq_n_f32(1.0f / 720.0f);             // c6 = 1/6! = 1/720
      p = vfmaq_f32(vdupq_n_f32(1.0f / 120.0f), t, p);       // c5 + t*p = 1/5! + t*c6
      p = vfmaq_f32(vdupq_n_f32(1.0f / 24.0f), t, p);        // c4 + t*p = 1/4! + t*(...)
      p = vfmaq_f32(vdupq_n_f32(1.0f / 6.0f), t, p);         // c3 + t*p = 1/3! + t*(...)
      p = vfmaq_f32(vdupq_n_f32(0.5f), t, p);                 // c2 + t*p = 1/2! + t*(...)
      p = vfmaq_f32(vdupq_n_f32(1.0f), t, p);                 // c1 + t*p = 1/1! + t*(...)
      p = vfmaq_f32(vdupq_n_f32(1.0f), t, p);                 // c0 + t*p = 1/0! + t*(...)

      // ---- 步骤 5：合并 2^n * 2^f = exp(x) ----
      return vmulq_f32(pow2n, p); // 最终结果 = 2^n × 泰勒逼近(2^f)
    }

    // =========================================================================
    // swiglu_neon — SwiGLU 融合算子的 NEON 实现
    // =========================================================================
    // 功能：对 gate 数组就地执行 SwiGLU：gate[i] = silu(gate[i]) * up[i]
    // 参数：
    //   gate — gate 投影输出，长度 n；既是输入也是输出（就地修改）
    //   up   — up 投影输出，长度 n（只读）
    //   n    — 数组长度（通常为 inter_size = 4864）
    void swiglu_neon(float *gate, const float *up, int n) {
      // 广播常量 1.0f 到 4 lane，用于计算 SiLU 分母
      const float32x4_t one = vdupq_n_f32(1.0f);
      int i = 0;
      // 向量化主循环：每次处理 4 个元素
      for (; i + 4 <= n; i += 4) {
        // vld1q_f32: 从内存加载 4 个连续 float 到 NEON 寄存器
        const float32x4_t g = vld1q_f32(gate + i);  // 加载 gate[i..i+3]
        const float32x4_t u = vld1q_f32(up + i);    // 加载 up[i..i+3]
        // vnegq_f32: 逐 lane 取负，得到 -gate
        // vexpq_f32: 调用上面定义的向量化 exp，计算 exp(-gate)
        const float32x4_t e = vexpq_f32(vnegq_f32(g)); // exp(-gate[i..i+3])
        // vaddq_f32: 4 lane 并行加法，计算 1 + exp(-gate)
        const float32x4_t denom = vaddq_f32(one, e);   // 分母 = 1 + exp(-gate)
        // vdivq_f32: 4 lane 并行除法，计算 gate / (1 + exp(-gate)) = silu(gate)
        const float32x4_t silu = vdivq_f32(g, denom);  // silu(gate[i..i+3])
        // vmulq_f32: 4 lane 并行乘法，silu * up
        // vst1q_f32: 将 4 个结果写回 gate 数组（就地修改）
        vst1q_f32(gate + i, vmulq_f32(silu, u));       // gate[i..i+3] = silu * up
      }
      // 标量尾段：处理不足 4 个的剩余元素（与 swiglu_ref 同款公式）
      for (; i < n; ++i) {
        // __builtin_expf: GCC/Clang 内置的 float 版 exp，比 std::exp 更快
        const float denom = 1.0f + __builtin_expf(-gate[i]); // 分母 = 1 + exp(-gate[i])
        const float s = gate[i] / denom;                     // silu(gate[i])
        gate[i] = s * up[i];                                 // gate[i] = silu * up[i]
      }
    }
  } // namespace

  // 自注册进 swiglu 注册表，名称为 "neon"。仅在 aarch64 构建中存在。
  // TINYQWEN_SWIGLU_VARIANT 宏会在 dispatch 表中注册此函数指针，
  // 运行时可通过 backend 选择机制自动选用 NEON 优化版本。
  TINYQWEN_SWIGLU_VARIANT(swiglu_neon, "neon");
} // namespace tinyqwen

#endif // defined(__aarch64__) || defined(_M_ARM64)
