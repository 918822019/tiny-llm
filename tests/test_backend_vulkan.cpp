#include "test_framework.h"

#include <cstdint>
#include <string>
#include <vector>

#include "backend_cpu.h"
#include "backend_vulkan.h"
#include "ref_ops.h"

using namespace tinyqwen;

namespace {

  std::vector<uint16_t> make_weights(int rows, int cols) {
    std::vector<uint16_t> weights(static_cast<size_t>(rows) * cols);
    for (int row = 0; row < rows; ++row) {
      for (int col = 0; col < cols; ++col) {
        const int centered = ((row * 17 + col * 13) % 29) - 14;
        weights[static_cast<size_t>(row) * cols + col] =
            float_to_half(static_cast<float>(centered) / 17.0f);
      }
    }
    return weights;
  }

  std::vector<float> make_inputs(int cols, int tokens) {
    std::vector<float> inputs(static_cast<size_t>(cols) * tokens);
    for (int token = 0; token < tokens; ++token) {
      for (int col = 0; col < cols; ++col) {
        const int centered = ((token * 11 + col * 7) % 23) - 11;
        inputs[static_cast<size_t>(token) * cols + col] = static_cast<float>(centered) / 19.0f;
      }
    }
    return inputs;
  }

  void expect_close(const std::vector<float> &got, const std::vector<float> &reference,
                    float tolerance) {
    EXPECT_EQ(got.size(), reference.size());
    for (size_t i = 0; i < got.size(); ++i)
      EXPECT_NEAR(got[i], reference[i], tolerance);
  }

} // namespace

TEST(vulkan_backend_f16_matvec_matches_cpu) {
  std::string error;
  auto vulkan = create_vulkan_backend(&error);
  EXPECT_TRUE(vulkan != nullptr);
  if (!vulkan)
    return;
  auto cpu = create_cpu_backend();

  constexpr int rows = 19;
  constexpr int cols = 37;
  auto weights = make_weights(rows, cols);
  auto input = make_inputs(cols, 1);
  std::vector<float> got(rows), reference(rows);
  const WeightTensor tensor{weights.data(), QuantType::kF16, rows, cols, 0};
  vulkan->matvec(tensor, input.data(), got.data(), rows, cols);
  cpu->matvec(tensor, input.data(), reference.data(), rows, cols);
  expect_close(got, reference, 2e-4f);
}

TEST(vulkan_backend_f16_matmul_tail_and_chunk_match_cpu) {
  std::string error;
  auto vulkan = create_vulkan_backend(&error);
  EXPECT_TRUE(vulkan != nullptr);
  if (!vulkan)
    return;
  auto cpu = create_cpu_backend();

  // rows is not divisible by the shader's 8-row tile and tokens exceeds the
  // per-dispatch tile, exercising both boundary paths.
  constexpr int rows = 23;
  constexpr int cols = 41;
  constexpr int tokens = 11;
  auto weights = make_weights(rows, cols);
  auto input = make_inputs(cols, tokens);
  std::vector<float> got(static_cast<size_t>(rows) * tokens);
  std::vector<float> reference(static_cast<size_t>(rows) * tokens);
  const WeightTensor tensor{weights.data(), QuantType::kF16, rows, cols, 0};
  vulkan->matmul(tensor, input.data(), got.data(), rows, cols, tokens);
  cpu->matmul(tensor, input.data(), reference.data(), rows, cols, tokens);
  expect_close(got, reference, 3e-4f);
}
