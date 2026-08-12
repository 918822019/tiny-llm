#include "test_framework.h"

#include <cstring>
#include <vector>

#include "kv_cache.h"

using namespace tinyqwen;

TEST(kv_cache_layout_and_append) {
  const int n_layers = 2, n_kv = 2, max_seq = 4, head_dim = 3;
  KvCache cache(n_layers, n_kv, max_seq, head_dim);
  EXPECT_EQ(cache.seq_len(), 0);
  EXPECT_EQ(cache.memory_bytes(),
            (size_t)2 * n_layers * n_kv * max_seq * head_dim * sizeof(float));

  // Append one position across all layers/heads using the same index math the
  // model uses: (h * max_seq + pos) * head_dim.
  const int pos = 0;
  for (int l = 0; l < n_layers; ++l) {
    for (int h = 0; h < n_kv; ++h) {
      for (int d = 0; d < head_dim; ++d) {
        cache.k(l)[(h * max_seq + pos) * head_dim + d] = 100.0f + l * 10 + h + d * 0.1f;
        cache.v(l)[(h * max_seq + pos) * head_dim + d] = -(100.0f + l * 10 + h + d * 0.1f);
      }
    }
  }
  cache.advance(1);
  EXPECT_EQ(cache.seq_len(), 1);

  for (int l = 0; l < n_layers; ++l) {
    for (int h = 0; h < n_kv; ++h) {
      EXPECT_NEAR(cache.k(l)[(h * max_seq) * head_dim + 1], 100.0f + l * 10 + h + 0.1f, 1e-6);
      EXPECT_NEAR(cache.v(l)[(h * max_seq) * head_dim + 2], -(100.0f + l * 10 + h + 0.2f),
                  1e-6);
    }
  }

  cache.reset();
  EXPECT_EQ(cache.seq_len(), 0);
}

TEST(kv_cache_k_v_independent) {
  KvCache cache(1, 1, 2, 2);
  cache.k(0)[0] = 1.0f;
  cache.v(0)[0] = 2.0f;
  EXPECT_NEAR(cache.k(0)[0], 1.0, 0.0);
  EXPECT_NEAR(cache.v(0)[0], 2.0, 0.0);
  EXPECT_TRUE(cache.k(0) != cache.v(0));
}
