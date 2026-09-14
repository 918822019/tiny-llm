#include "test_framework.h"

#include <string>
#include <vector>

#include "speculative_decoder.h"

TEST(speculative_decision_accepts_all_and_returns_bonus) {
    const std::vector<int> draft = {10, 11, 12};
    const std::vector<int> target_after = {10, 11, 12, 99};
    tinyqwen::SpeculativeDecision d;
    std::string err;
    EXPECT_TRUE(tinyqwen::decide_speculative(draft, target_after, &d, &err));
    EXPECT_EQ(d.accepted, 3);
    EXPECT_TRUE(d.all_accepted);
    EXPECT_EQ(d.next_pending, 99);
}

TEST(speculative_decision_stops_at_first_rejection) {
    const std::vector<int> draft = {10, 11, 12, 13};
    const std::vector<int> target_after = {10, 11, 77, 13, 99};
    tinyqwen::SpeculativeDecision d;
    std::string err;
    EXPECT_TRUE(tinyqwen::decide_speculative(draft, target_after, &d, &err));
    EXPECT_EQ(d.accepted, 2);
    EXPECT_TRUE(!d.all_accepted);
    EXPECT_EQ(d.next_pending, 77);
}

TEST(speculative_decision_handles_first_token_rejection) {
    const std::vector<int> draft = {10, 11};
    const std::vector<int> target_after = {42, 11, 99};
    tinyqwen::SpeculativeDecision d;
    std::string err;
    EXPECT_TRUE(tinyqwen::decide_speculative(draft, target_after, &d, &err));
    EXPECT_EQ(d.accepted, 0);
    EXPECT_EQ(d.next_pending, 42);
}

TEST(speculative_decision_rejects_bad_prediction_shape) {
    const std::vector<int> draft = {10, 11};
    const std::vector<int> target_after = {10, 11};
    tinyqwen::SpeculativeDecision d;
    std::string err;
    EXPECT_TRUE(!tinyqwen::decide_speculative(draft, target_after, &d, &err));
    EXPECT_TRUE(!err.empty());
}
