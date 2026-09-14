#pragma once

#include <string>
#include <vector>

#include "qwen_model.h"
#include "dflash_model.h"
#include "eagle3_model.h"

namespace tinyqwen {
    struct MetalPrefillEngine;

    struct SpeculativeConfig {
        int max_new_tokens = 16;
        int draft_tokens = 4;
        int eos_token_id = -1;
    };

    struct SpeculativeStats {
        int blocks = 0;
        int target_verify_calls = 0;
        int target_input_tokens = 0;
        int draft_forward_calls = 0;
        int draft_proposed = 0;
        int draft_accepted = 0;
        int corrections = 0;
        int bonus_tokens = 0;
        int rollbacks = 0;
        int baseline_tail_steps = 0;
        double prefill_ms = 0.0;
        double decode_ms = 0.0;
        double draft_ms = 0.0;
        double target_verify_ms = 0.0;

        double acceptance_rate() const {
            return draft_proposed > 0
                       ? static_cast<double>(draft_accepted) / draft_proposed
                       : 0.0;
        }
    };

    struct SpeculativeDecision {
        int accepted = 0;
        bool all_accepted = false;
        int next_pending = -1;
    };

    // target_after 的长度必须是 draft.size()+1：target_after[i] 预测 draft[i]，
    // 最后一项在全部接受时成为 bonus token。
    bool decide_speculative(const std::vector<int> &draft,
                            const std::vector<int> &target_after,
                            SpeculativeDecision *decision, std::string *err);

    struct SpeculativeResult {
        std::vector<int> generated_ids;
        SpeculativeStats stats;
        bool hit_eos = false;
    };

    // 精确 greedy 投机解码。target_metal 可为空；非空时目标验证走 Metal 的
    // continuation pass，草稿模型仍走 CPU。
    bool speculative_generate(QwenModel &target, QwenModel &draft,
                              MetalPrefillEngine *target_metal,
                              const std::vector<int> &prompt,
                              const SpeculativeConfig &config,
                              SpeculativeResult *result, std::string *err);

    // DFlash/DFlare greedy chain decoding. Unlike an autoregressive draft LM,
    // the drafter consumes selected target residual streams and proposes a
    // whole mask block with an optional Markov dependency between positions.
    bool dflash_speculative_generate(QwenModel &target, DFlashModel &draft,
                                     const std::vector<int> &prompt,
                                     const SpeculativeConfig &config,
                                     SpeculativeResult *result, std::string *err);

    // EAGLE3 greedy chain decoding. For this path draft_tokens is the total
    // verification width (pending root + proposals), matching SGLang's
    // --speculative-num-draft-tokens convention. Thus width 4 runs 3 recurrent
    // proposal steps and can additionally yield the target bonus token.
    bool eagle3_speculative_generate(QwenModel &target, Eagle3Model &draft,
                                     const std::vector<int> &prompt,
                                     const SpeculativeConfig &config,
                                     SpeculativeResult *result, std::string *err);
} // namespace tinyqwen
