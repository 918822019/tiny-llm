// ============================================================================
// softmax_ref.cpp — Softmax 算子的参考实现（标量版）
// ============================================================================
// 本文件实现标准 softmax 函数。
//
// 数学定义：
//   softmax(x_i) = exp(x_i) / Σⱼ exp(x_j)
//   将一组任意实数转换为加起来等于 1 的概率分布。
//   分数越大的项，分到的概率越大。
//
// 在 Transformer 中的位置：
//   - Attention 内部：对 Q·K^T/sqrt(d) 的结果做 softmax 得到注意力权重。
//     （但 decode 时的 attention 不调用本函数，它在 attention_decode_ref 里
//      用 online softmax 融合了，避免 O(seq_len) 暂存。）
//   - 采样解码：temperature sampling、top-k/top-p sampling 都需要先对 logits
//     做 softmax 得到概率分布，再按概率随机采样。
//   - 本 kernel 是独立的 reference 实现，供单测和将来采样功能使用。
//
// ref 与 neon 版本的关系：
//   - 目前只有 ref 版（softmax 不是 decode 热点，attention 已用 online softmax）。
//   - 若将来需要 NEON 版，可参照 swiglu_neon 的 vexpq_f32 向量化 exp。
//
// 数值稳定性技巧：
//   直接算 exp(大数) 会上溢为无穷大。解决办法是先把每个数减去最大值 m：
//     softmax(x_i) = exp(x_i - m) / Σⱼ exp(x_j - m)
//   分子分母同时乘了 e^(-m) 抵消，结果不变；但保证 exp 的输入 <= 0，
//   永远不会上溢。这是工业界的标准做法。
//
// 优化要点：
//   1. 三遍扫描：第一遍找 max，第二遍算 exp 并累加分母，第三遍归一化。
//   2. 第二遍用 double 累加分母（Qwen2.5 词表 151936 维，连加误差不可忽视）。
//   3. 第三遍用"乘倒数"代替除法（一次除法 + n 次乘法 vs n 次除法），更快。
// ============================================================================

#include "ref_ops.h" // 声明 softmax_ref 等参考算子的头文件

#include <cmath> // std::exp

namespace tinyqwen {
  // =========================================================================
  // softmax_ref — Softmax 参考实现
  // =========================================================================
  // 功能：对长度为 n 的输入数组计算 softmax，输出概率分布
  // 参数：
  //   x — 输入数组（logits），长度 n
  //   y — 输出数组（概率），长度 n，满足 Σ y[i] = 1
  //   n — 数组长度（Qwen2.5 词表大小为 151936）
  void softmax_ref(const float *x, float *y, int n) {
    // ======== 第 1 遍：找最大值 m，用于数值稳定 ========
    // 初始化最大值为第一个元素
    float m = x[0];
    // 线性扫描找全局最大值
    for (int i = 1; i < n; ++i) {
      if (x[i] > m) {
        m = x[i]; // 更新最大值
      }
    }

    // ======== 第 2 遍：计算 exp(x_i - m) 并累加分母 ========
    // 使用 double 累加分母，减少舍入漂移
    // （Qwen2.5 词表有 151936 这么大，float 连加的相对误差可达 n·ε ≈ 0.01）
    double sum = 0.0;
    for (int i = 0; i < n; ++i) {
      // 减去最大值 m 后再取 exp，保证指数 <= 0，不会上溢
      y[i] = std::exp(x[i] - m);
      // 将未归一化概率累加到分母中
      sum += y[i];
    }

    // ======== 第 3 遍：除以分母归一化 ========
    // 用"乘倒数"技巧：只做一次除法得到 inv，后续全是乘法（比逐元素除法快）
    const float inv = static_cast<float>(1.0 / sum);
    for (int i = 0; i < n; ++i) {
      // y[i] = exp(x[i]-m) / sum = exp(x[i]-m) * (1/sum)
      y[i] = y[i] * inv;
    }
  }
} // namespace tinyqwen
