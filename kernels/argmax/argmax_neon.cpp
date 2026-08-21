// ============================================================================
// argmax_neon.cpp — Argmax 算子的 NEON SIMD 优化版（aarch64）
// ============================================================================
// 本文件实现 argmax 的 ARM NEON 向量化版本：返回第一个最大值的下标。
//
// 数学定义与 argmax_ref.cpp 完全相同：
//   argmax(x) = 使得 x[i] 最大的最小下标 i
//
// 在 Transformer 中的位置：
//   与 ref 版相同——greedy 解码的最后一步，每 token 对 151936 维 logits 调用。
//
// 为什么是 decode 非 matvec 里的头号优化目标：
//   greedy 每 token 都要对 ~152K 维 logits 扫一遍找最大。标量版是"逐个比较 +
//   维护当前最大"，一条串行依赖链（每步等上一步的比较结果），且对 ~600KB 数据
//   只做 ~2.5 GB/s 的有效吞吐——既没喂饱带宽，也没用 SIMD。
//
// ref 与 neon 版本的关系：
//   - 数值语义逐位一致：max 在浮点下是精确运算（无舍入），
//     "第一个 == max 的位置"正是 ref 用严格 '>' 保留首个平局的语义。
//   - 因此本变体无需容差门禁，单测直接断言下标相等。
//
// 两遍法（经典 SIMD argmax 套路）：
//   pass 1：只求"最大值是多少"——vmaxq_f32 对 4 lane 并行取 max，
//           vmaxvq_f32 横向归约。max 运算没有下标依赖，纯归约，SIMD 满载。
//   pass 2：再扫一遍找"第一个等于最大值的下标"——vceqq_f32 并行比较出掩码。
//           logits 刚被 lm_head 写出、热在 L2，第二遍几乎不花 DRAM 带宽。
// ============================================================================

#include "dispatch.h" // TINYQWEN_ARGMAX_VARIANT 自注册宏
#include "ref_ops.h"  // 声明 argmax 函数签名

#if defined(__aarch64__) || defined(_M_ARM64) // 仅在 ARM64 平台编译

#include <arm_neon.h> // ARM NEON SIMD intrinsic 头文件

#include <cmath> // INFINITY 常量

namespace tinyqwen {
  namespace { // 匿名命名空间：内部函数不暴露到外部链接
    // =========================================================================
    // argmax_neon — Argmax 的 NEON 实现（两遍法）
    // =========================================================================
    // 功能：找出 logits 数组中最大值的下标（平局取第一个）
    // 参数：
    //   logits — 输入数组（词表 logits），长度 n
    //   n      — 数组长度
    // 返回值：最大元素的下标（int）
    int argmax_neon(const float *logits, int n) {
      // 边界情况：空数组返回 0（不应发生，防御性编程）
      if (n <= 0) return 0;

      // ======== pass 1：NEON 归约求最大值 ========
      // 初始化 4-lane 最大值为负无穷（确保任何实际值都比它大）
      float32x4_t vmax = vdupq_n_f32(-INFINITY);
      int i = 0;
      // 向量化循环：每次处理 4 个元素，用 vmaxq_f32 并行取 max
      for (; i + 4 <= n; i += 4) {
        // vld1q_f32: 加载 4 个 logits
        // vmaxq_f32: 4 lane 并行取最大值（每个 lane 独立比较）
        vmax = vmaxq_f32(vmax, vld1q_f32(logits + i));
      }
      // vmaxvq_f32: 将 4 个 lane 横向归约为单个 float（取 4 lane 中的最大值）
      float m = vmaxvq_f32(vmax);
      // 标量尾段：处理不足 4 个的剩余元素
      for (; i < n; ++i) {
        m = logits[i] > m ? logits[i] : m; // 标量 max
      }

      // ======== pass 2：找第一个等于最大值的下标 ========
      // 将最大值 m 广播到 4 lane，用于并行比较
      const float32x4_t vm = vdupq_n_f32(m);
      // alignas(16): 确保 mask 数组 16 字节对齐，满足 vst1q_u32 的对齐要求
      alignas(16) uint32_t mask[4]; // 存储 4 lane 比较结果的掩码数组
      int j = 0;
      // 向量化搜索：每次检查 4 个元素是否等于最大值
      for (; j + 4 <= n; j += 4) {
        // vceqq_f32: 4 lane 并行比较（相等则对应 lane 全 1，否则全 0）
        // 返回 uint32x4_t 类型的掩码
        const uint32x4_t eq = vceqq_f32(vld1q_f32(logits + j), vm);
        // vst1q_u32: 将 4 个掩码值存储到内存
        vst1q_u32(mask, eq);
        // 按顺序检查 4 个 lane 的掩码（lane 0 优先，保证取第一个匹配）
        for (int l = 0; l < 4; ++l) {
          if (mask[l]) return j + l; // 找到第一个等于最大值的位置，立即返回
        }
      }
      // 标量尾段：处理不足 4 个的剩余元素
      for (; j < n; ++j) {
        if (logits[j] == m) return j; // 标量比较，找到即返回
      }
      // 不可达：最大值必然存在于数组中（除非全是 NaN，但 logits 不会出现 NaN）
      return 0;
    }
  } // namespace

  // 自注册进 argmax 注册表，名称为 "neon"。仅在 aarch64 构建中存在。
  // TINYQWEN_ARGMAX_VARIANT 宏会在 dispatch 表中注册此函数指针，
  // 运行时可通过 backend 选择机制自动选用 NEON 优化版本。
  TINYQWEN_ARGMAX_VARIANT(argmax_neon, "neon");
} // namespace tinyqwen

#endif // defined(__aarch64__) || defined(_M_ARM64)
