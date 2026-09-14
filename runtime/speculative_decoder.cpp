#include "speculative_decoder.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>

#include "metal_prefill.h"

namespace tinyqwen {
namespace {
    using Clock = std::chrono::steady_clock;

    double elapsed_ms(Clock::time_point begin, Clock::time_point end) {
        return std::chrono::duration<double, std::milli>(end - begin).count();
    }

    bool target_forward(QwenModel &target, MetalPrefillEngine *metal,
                        const int *tokens, int n, std::vector<int> *next,
                        std::string *err) {
        if (!next || n <= 0) {
            if (err) *err = "target verify needs at least one token";
            return false;
        }
        if (!metal) {
            const int last = target.forward_verify(tokens, n, next);
            if (last < 0 || static_cast<int>(next->size()) != n) {
                if (err) *err = "CPU target verify did not return every position";
                return false;
            }
            return true;
        }

        next->resize(n);
        const int last = metal_prefill_run(metal, tokens, n, nullptr, true,
                                           &target.kv_cache(), &target.gdn_state(), err,
                                           next->data());
        if (last < 0) return false;
        target.sync_external_state();
        return true;
    }

    bool restore_prefix(QwenModel &model, MetalPrefillEngine *metal,
                        const QwenModel::StateCheckpoint &checkpoint,
                        const std::vector<int> &verified_inputs, int keep,
                        std::string *err) {
        if (keep < 0 || keep > static_cast<int>(verified_inputs.size())) {
            if (err) *err = "invalid speculative prefix length";
            return false;
        }

        if (!model.has_recurrent_state()) {
            const int wanted = checkpoint.seq_len + keep;
            if (!model.truncate(wanted, err)) return false;
            if (metal && !metal_prefill_rewind(metal, wanted, nullptr, err)) return false;
            return true;
        }

        // GDN 的最终递归状态不能像 KV 一样只改长度。先回到 checkpoint，
        // 再只重放 pending + 已接受草稿；拒绝是低频路径，优先保证正确性。
        if (!model.restore(checkpoint, err)) return false;
        if (metal && !metal_prefill_rewind(metal, checkpoint.seq_len,
                                           &model.gdn_state(), err)) {
            return false;
        }
        if (keep == 0) return true;
        std::vector<int> replay_next;
        return target_forward(model, metal, verified_inputs.data(), keep,
                              &replay_next, err);
    }

    bool eagle3_target_verify(QwenModel &target,
                              const std::vector<int> &inputs,
                              const std::vector<int> &capture_layer_ids,
                              size_t capture_row, bool batch,
                              std::vector<int> *target_after,
                              std::vector<float> *verified_hidden,
                              int *forward_calls, std::string *err) {
        if (!target_after || !verified_hidden || !forward_calls || inputs.empty()) {
            if (err) *err = "invalid EAGLE3 target verification arguments";
            return false;
        }
        target_after->clear();
        verified_hidden->clear();
        *forward_calls = 0;

        if (batch) {
            if (target.forward_verify_capture(inputs.data(),
                                              static_cast<int>(inputs.size()),
                                              capture_layer_ids, target_after,
                                              verified_hidden) < 0) {
                if (err) *err = "EAGLE3 batched target verification/capture failed";
                return false;
            }
            *forward_calls = 1;
        } else {
            target_after->reserve(inputs.size());
            verified_hidden->reserve(inputs.size() * capture_row);
            for (int token : inputs) {
                std::vector<int> step_after;
                std::vector<float> step_hidden;
                if (target.forward_verify_capture(&token, 1, capture_layer_ids,
                                                  &step_after, &step_hidden) < 0 ||
                    step_after.size() != 1 || step_hidden.size() != capture_row) {
                    if (err) *err = "EAGLE3 sequential target verification/capture failed";
                    return false;
                }
                target_after->push_back(step_after[0]);
                verified_hidden->insert(verified_hidden->end(),
                                        step_hidden.begin(), step_hidden.end());
                ++*forward_calls;
            }
        }

        if (target_after->size() != inputs.size() ||
            verified_hidden->size() != inputs.size() * capture_row) {
            if (err) *err = "EAGLE3 target verification returned an invalid shape";
            return false;
        }
        return true;
    }
} // namespace

bool decide_speculative(const std::vector<int> &draft,
                        const std::vector<int> &target_after,
                        SpeculativeDecision *decision, std::string *err) {
    if (!decision) {
        if (err) *err = "null speculative decision output";
        return false;
    }
    if (draft.empty() || target_after.size() != draft.size() + 1) {
        if (err) *err = "target predictions must contain draft.size()+1 entries";
        return false;
    }

    int accepted = 0;
    while (accepted < static_cast<int>(draft.size()) &&
           draft[accepted] == target_after[accepted]) {
        ++accepted;
    }
    decision->accepted = accepted;
    decision->all_accepted = accepted == static_cast<int>(draft.size());
    decision->next_pending = target_after[accepted];
    return true;
}

bool speculative_generate(QwenModel &target, QwenModel &draft,
                          MetalPrefillEngine *target_metal,
                          const std::vector<int> &prompt,
                          const SpeculativeConfig &config,
                          SpeculativeResult *result, std::string *err) {
    if (!result) {
        if (err) *err = "null speculative result output";
        return false;
    }
    *result = SpeculativeResult{};
    if (prompt.empty()) {
        if (err) *err = "speculative decoding needs a non-empty prompt";
        return false;
    }
    if (config.max_new_tokens <= 0 || config.draft_tokens <= 0) {
        if (err) *err = "max_new_tokens and draft_tokens must be positive";
        return false;
    }
    if (target.config().vocab_size != draft.config().vocab_size) {
        if (err) *err = "target and draft vocab sizes differ";
        return false;
    }
    const int total_needed = static_cast<int>(prompt.size()) + config.max_new_tokens;
    if (total_needed > target.kv_cache().max_seq_len() ||
        total_needed > draft.kv_cache().max_seq_len()) {
        if (err) *err = "prompt + generation exceeds target or draft KV capacity";
        return false;
    }

    target.reset();
    draft.reset();
    target.set_prompt_len(static_cast<int>(prompt.size()));
    draft.set_prompt_len(static_cast<int>(prompt.size()));
    if (target_metal) metal_prefill_reset_kv(target_metal);

    const Clock::time_point prefill_begin = Clock::now();
    int pending = -1;
    if (target_metal) {
        pending = metal_prefill_run(target_metal, prompt.data(),
                                    static_cast<int>(prompt.size()), nullptr, false,
                                    &target.kv_cache(), &target.gdn_state(), err);
        if (pending < 0) return false;
        target.sync_external_state();
    } else {
        pending = target.forward_prefill(prompt.data(), static_cast<int>(prompt.size()));
    }
    if (pending < 0) {
        if (err) *err = "target prefill failed";
        return false;
    }
    if (draft.forward_prefill(prompt.data(), static_cast<int>(prompt.size())) < 0) {
        if (err) *err = "draft prefill failed";
        return false;
    }
    result->stats.prefill_ms = elapsed_ms(prefill_begin, Clock::now());

    const Clock::time_point decode_begin = Clock::now();
    while (static_cast<int>(result->generated_ids.size()) < config.max_new_tokens) {
        // pending 是目标模型已经选出、但尚未进入两边 cache 的 token。
        result->generated_ids.push_back(pending);
        if (config.eos_token_id >= 0 && pending == config.eos_token_id) {
            result->hit_eos = true;
            break;
        }

        const int remaining = config.max_new_tokens -
                              static_cast<int>(result->generated_ids.size());
        if (remaining <= 0) break;

        // 只剩一个输出位时直接做一次目标 step，避免为了一个 token 构造
        // draft+bonus 两个候选。两边都消费刚输出的 pending，保持状态对齐。
        if (remaining == 1) {
            std::vector<int> after;
            const Clock::time_point target_begin = Clock::now();
            if (!target_forward(target, target_metal, &pending, 1, &after, err))
                return false;
            result->stats.target_verify_ms += elapsed_ms(target_begin, Clock::now());
            const Clock::time_point draft_begin = Clock::now();
            draft.forward_token(pending);
            result->stats.draft_ms += elapsed_ms(draft_begin, Clock::now());
            ++result->stats.target_verify_calls;
            ++result->stats.target_input_tokens;
            ++result->stats.draft_forward_calls;
            ++result->stats.baseline_tail_steps;
            pending = after[0];
            continue;
        }

        const int want = std::min(config.draft_tokens, remaining - 1);
        const QwenModel::StateCheckpoint draft_checkpoint = draft.checkpoint();

        // 草稿状态起初也排除 pending。先消费 pending 得到第一个提案；随后每
        // 消费一个提案得到下一个。末个提案暂不消费，等验证结果决定保留与否。
        std::vector<int> proposals;
        proposals.reserve(want);
        const Clock::time_point draft_begin = Clock::now();
        int proposal = draft.forward_token(pending);
        ++result->stats.draft_forward_calls;
        for (int i = 0; i < want; ++i) {
            proposals.push_back(proposal);
            if (config.eos_token_id >= 0 && proposal == config.eos_token_id) break;
            if (i + 1 < want) {
                proposal = draft.forward_token(proposal);
                ++result->stats.draft_forward_calls;
            }
        }
        result->stats.draft_ms += elapsed_ms(draft_begin, Clock::now());

        std::vector<int> verified_inputs;
        verified_inputs.reserve(proposals.size() + 1);
        verified_inputs.push_back(pending);
        verified_inputs.insert(verified_inputs.end(), proposals.begin(), proposals.end());

        const QwenModel::StateCheckpoint target_checkpoint = target.checkpoint();
        std::vector<int> target_after;
        const Clock::time_point target_begin = Clock::now();
        if (!target_forward(target, target_metal, verified_inputs.data(),
                            static_cast<int>(verified_inputs.size()),
                            &target_after, err)) {
            return false;
        }
        result->stats.target_verify_ms += elapsed_ms(target_begin, Clock::now());
        ++result->stats.blocks;
        ++result->stats.target_verify_calls;
        result->stats.target_input_tokens += static_cast<int>(verified_inputs.size());
        result->stats.draft_proposed += static_cast<int>(proposals.size());

        SpeculativeDecision decision;
        if (!decide_speculative(proposals, target_after, &decision, err)) return false;
        result->stats.draft_accepted += decision.accepted;

        const int keep = 1 + decision.accepted; // pending + accepted drafts
        if (!decision.all_accepted) {
            const bool target_replays = target.has_recurrent_state();
            const bool draft_replays = draft.has_recurrent_state();
            const Clock::time_point target_restore_begin = Clock::now();
            if (!restore_prefix(target, target_metal, target_checkpoint,
                                verified_inputs, keep, err)) {
                return false;
            }
            result->stats.target_verify_ms +=
                elapsed_ms(target_restore_begin, Clock::now());
            const Clock::time_point draft_restore_begin = Clock::now();
            if (!restore_prefix(draft, nullptr, draft_checkpoint,
                                verified_inputs, keep, err)) {
                return false;
            }
            result->stats.draft_ms += elapsed_ms(draft_restore_begin, Clock::now());
            if (target_replays) {
                ++result->stats.target_verify_calls;
                result->stats.target_input_tokens += keep;
            }
            if (draft_replays) result->stats.draft_forward_calls += keep;
            ++result->stats.corrections;
            ++result->stats.rollbacks;
        } else {
            // 目标验证后已消费全部输入；草稿还差最后一个提案没进 cache。
            const Clock::time_point draft_bonus_begin = Clock::now();
            draft.forward_token(proposals.back());
            result->stats.draft_ms += elapsed_ms(draft_bonus_begin, Clock::now());
            ++result->stats.draft_forward_calls;
            ++result->stats.bonus_tokens;
        }

        for (int i = 0; i < decision.accepted; ++i) {
            result->generated_ids.push_back(proposals[i]);
            if (config.eos_token_id >= 0 && proposals[i] == config.eos_token_id) {
                result->hit_eos = true;
                result->stats.decode_ms = elapsed_ms(decode_begin, Clock::now());
                return true;
            }
        }
        pending = decision.next_pending;
    }

    result->stats.decode_ms = elapsed_ms(decode_begin, Clock::now());
    return true;
}

bool dflash_speculative_generate(QwenModel &target, DFlashModel &draft,
                                 const std::vector<int> &prompt,
                                 const SpeculativeConfig &config,
                                 SpeculativeResult *result, std::string *err) {
    if (!result || prompt.empty() || config.max_new_tokens <= 0) {
        if (err) *err = "invalid DFlash generation arguments";
        return false;
    }
    *result = SpeculativeResult{};
    const int block = std::min(config.draft_tokens, draft.block_size());
    if (block < 2) {
        if (err) *err = "DFlash requires at least two verification tokens";
        return false;
    }
    if (static_cast<int>(prompt.size()) + config.max_new_tokens >
        target.kv_cache().max_seq_len()) {
        if (err) *err = "prompt + generation exceeds target KV capacity";
        return false;
    }

    target.reset();
    draft.reset();
    target.set_prompt_len(static_cast<int>(prompt.size()));
    const Clock::time_point prefill_begin = Clock::now();
    std::vector<float> target_hidden;
    int pending = target.forward_prefill(prompt.data(), static_cast<int>(prompt.size()),
                                         nullptr, 0, nullptr,
                                         &draft.target_layer_ids(), &target_hidden);
    if (pending < 0) {
        if (err) *err = "DFlash target prefill/capture failed";
        return false;
    }
    if (draft.using_vulkan() && !draft.initialize_vulkan_target(err)) {
        if (err && err->empty()) *err = "DFlash Vulkan target KV import failed";
        return false;
    }
    result->stats.prefill_ms = elapsed_ms(prefill_begin, Clock::now());
    int ctx_tokens = static_cast<int>(prompt.size());

    const Clock::time_point decode_begin = Clock::now();
    while (static_cast<int>(result->generated_ids.size()) < config.max_new_tokens) {
        result->generated_ids.push_back(pending);
        if (config.eos_token_id >= 0 && pending == config.eos_token_id) {
            result->hit_eos = true;
            break;
        }
        const int remaining = config.max_new_tokens -
                              static_cast<int>(result->generated_ids.size());
        if (remaining <= 0) break;
        if (remaining == 1) {
            std::vector<int> after;
            const Clock::time_point target_begin = Clock::now();
            bool target_ok = false;
            if (draft.using_vulkan()) {
                target_ok = draft.verify_vulkan_target(&pending, 1, &after, nullptr, err);
            } else {
                target_ok = target.forward_verify(&pending, 1, &after) >= 0 && after.size() == 1;
            }
            if (!target_ok || after.size() != 1) {
                if (err && err->empty()) *err = "DFlash target tail step failed";
                return false;
            }
            result->stats.target_verify_ms += elapsed_ms(target_begin, Clock::now());
            pending = after[0];
            ++result->stats.target_verify_calls;
            ++result->stats.target_input_tokens;
            ++result->stats.baseline_tail_steps;
            continue;
        }

        const int verify_size = std::min(block, remaining);
        std::vector<int> proposals;
        const Clock::time_point draft_begin = Clock::now();
        if (!draft.propose(target_hidden.data(), ctx_tokens, pending, verify_size,
                           &proposals, err)) return false;
        result->stats.draft_ms += elapsed_ms(draft_begin, Clock::now());
        ++result->stats.draft_forward_calls;
        result->stats.draft_proposed += static_cast<int>(proposals.size());

        std::vector<int> inputs;
        inputs.reserve(verify_size);
        inputs.push_back(pending);
        inputs.insert(inputs.end(), proposals.begin(), proposals.end());
        const int checkpoint_seq = draft.using_vulkan()
            ? draft.vulkan_target_seq_len()
            : target.checkpoint().seq_len;
        std::vector<int> target_after;
        std::vector<float> verified_hidden;
        const Clock::time_point target_begin = Clock::now();
        bool target_ok = false;
        if (draft.using_vulkan()) {
            target_ok = draft.verify_vulkan_target(inputs.data(), static_cast<int>(inputs.size()),
                                                   &target_after, &verified_hidden, err);
        } else {
            target_ok = target.forward_verify_capture(inputs.data(),
                                                      static_cast<int>(inputs.size()),
                                                      draft.target_layer_ids(), &target_after,
                                                      &verified_hidden) >= 0;
        }
        if (!target_ok) {
            if (err && err->empty()) *err = "DFlash target verification/capture failed";
            return false;
        }
        result->stats.target_verify_ms += elapsed_ms(target_begin, Clock::now());
        ++result->stats.blocks;
        ++result->stats.target_verify_calls;
        result->stats.target_input_tokens += static_cast<int>(inputs.size());

        SpeculativeDecision decision;
        if (!decide_speculative(proposals, target_after, &decision, err)) return false;
        if (std::getenv("TINYQWEN_DFLASH_TRACE")) {
            std::fprintf(stderr, "[dflash-trace] anchor=%d draft:", pending);
            for (int id : proposals) std::fprintf(stderr, " %d", id);
            std::fprintf(stderr, " target:");
            for (int id : target_after) std::fprintf(stderr, " %d", id);
            std::fprintf(stderr, " accepted=%d\n", decision.accepted);
        }
        result->stats.draft_accepted += decision.accepted;
        const int keep = 1 + decision.accepted;
        if (draft.using_vulkan()) {
            if (!draft.truncate_vulkan_target(checkpoint_seq + keep, err)) return false;
        } else if (!target.truncate(checkpoint_seq + keep, err)) {
            return false;
        }
        if (decision.all_accepted) ++result->stats.bonus_tokens;
        else {
            ++result->stats.corrections;
            ++result->stats.rollbacks;
        }

        const int H = static_cast<int>(target.config().hidden_size);
        const int K = static_cast<int>(draft.target_layer_ids().size());
        target_hidden.assign(verified_hidden.begin(),
                             verified_hidden.begin() + static_cast<size_t>(keep) * K * H);
        ctx_tokens = keep;
        for (int i = 0; i < decision.accepted; ++i) {
            result->generated_ids.push_back(proposals[i]);
            if (config.eos_token_id >= 0 && proposals[i] == config.eos_token_id) {
                result->hit_eos = true;
                result->stats.decode_ms = elapsed_ms(decode_begin, Clock::now());
                return true;
            }
        }
        pending = decision.next_pending;
    }
    result->stats.decode_ms = elapsed_ms(decode_begin, Clock::now());
    return true;
}

bool eagle3_speculative_generate(QwenModel &target, Eagle3Model &draft,
                                 const std::vector<int> &prompt,
                                 const SpeculativeConfig &config,
                                 SpeculativeResult *result, std::string *err) {
    if (!result || prompt.empty() || config.max_new_tokens <= 0 ||
        config.draft_tokens < 2) {
        if (err) *err = "invalid EAGLE3 generation arguments";
        return false;
    }
    *result = SpeculativeResult{};
    const int total_needed = static_cast<int>(prompt.size()) + config.max_new_tokens;
    if (total_needed > target.kv_cache().max_seq_len() ||
        total_needed > draft.max_seq_len()) {
        if (err) *err = "prompt + generation exceeds target or EAGLE3 KV capacity";
        return false;
    }

    target.reset();
    draft.reset();
    target.set_prompt_len(static_cast<int>(prompt.size()));

    const Clock::time_point prefill_begin = Clock::now();
    std::vector<float> target_hidden;
    int pending = target.forward_prefill(prompt.data(), static_cast<int>(prompt.size()),
                                         nullptr, 0, nullptr,
                                         &draft.target_layer_ids(), &target_hidden);
    if (pending < 0) {
        if (err) *err = "EAGLE3 target prefill/capture failed";
        return false;
    }
    const size_t capture_row = draft.target_layer_ids().size() *
                               static_cast<size_t>(target.config().hidden_size);
    if (target_hidden.size() != prompt.size() * capture_row) {
        if (err) *err = "EAGLE3 target prefill returned an invalid capture shape";
        return false;
    }

    // Batched verification routes the tied lm_head through backend matmul.
    // Materialize persistent GPU storage before decode timing; otherwise its
    // first-use allocation/copy is charged to the first speculative block and
    // makes short-run throughput depend on allocator/page-cache noise.
    if (config.batch_target_verify && !target.prepare_lm_head(err)) {
        if (err && err->empty()) *err = "EAGLE3 target lm_head preparation failed";
        return false;
    }

    // EAGLE3 was trained on next-token-shifted embeddings: H(prompt[i]) pairs
    // with embed(prompt[i+1]), and the last prompt state pairs with the target's
    // already selected pending token. This also seeds the first proposal.
    std::vector<int> shifted_tokens;
    shifted_tokens.reserve(prompt.size());
    shifted_tokens.insert(shifted_tokens.end(), prompt.begin() + 1, prompt.end());
    shifted_tokens.push_back(pending);
    int prepared_proposal = -1;
    if (!draft.extend_target(target_hidden.data(), static_cast<int>(prompt.size()),
                             shifted_tokens.data(), &prepared_proposal, err)) {
        return false;
    }
    result->stats.prefill_ms = elapsed_ms(prefill_begin, Clock::now());

    const int block = config.draft_tokens;
    const Clock::time_point decode_begin = Clock::now();
    while (static_cast<int>(result->generated_ids.size()) < config.max_new_tokens) {
        result->generated_ids.push_back(pending);
        if (config.eos_token_id >= 0 && pending == config.eos_token_id) {
            result->hit_eos = true;
            break;
        }

        const int remaining = config.max_new_tokens -
                              static_cast<int>(result->generated_ids.size());
        if (remaining <= 0) break;
        if (remaining == 1) {
            std::vector<int> after;
            const Clock::time_point target_begin = Clock::now();
            if (target.forward_verify(&pending, 1, &after) < 0 || after.size() != 1) {
                if (err) *err = "EAGLE3 target tail step failed";
                return false;
            }
            result->stats.target_verify_ms += elapsed_ms(target_begin, Clock::now());
            ++result->stats.target_verify_calls;
            ++result->stats.target_input_tokens;
            ++result->stats.baseline_tail_steps;
            pending = after[0];
            continue;
        }

        const int verify_size = std::min(block, remaining);
        const int want = verify_size - 1;
        const int draft_checkpoint = draft.seq_len();
        std::vector<int> proposals;
        const Clock::time_point draft_begin = Clock::now();
        if (!draft.propose(want, &proposals, err)) return false;
        result->stats.draft_ms += elapsed_ms(draft_begin, Clock::now());
        result->stats.draft_forward_calls += std::max(0, want - 1);
        result->stats.draft_proposed += static_cast<int>(proposals.size());

        std::vector<int> inputs;
        inputs.reserve(verify_size);
        inputs.push_back(pending);
        inputs.insert(inputs.end(), proposals.begin(), proposals.end());
        const QwenModel::StateCheckpoint target_checkpoint = target.checkpoint();
        std::vector<int> target_after;
        std::vector<float> verified_hidden;
        int target_forward_calls = 0;
        const Clock::time_point target_begin = Clock::now();
        if (!eagle3_target_verify(target, inputs, draft.target_layer_ids(), capture_row,
                                  config.batch_target_verify, &target_after,
                                  &verified_hidden, &target_forward_calls, err)) {
            return false;
        }
        result->stats.target_verify_ms += elapsed_ms(target_begin, Clock::now());
        ++result->stats.blocks;
        result->stats.target_verify_calls += target_forward_calls;
        result->stats.target_input_tokens += static_cast<int>(inputs.size());

        SpeculativeDecision decision;
        if (!decide_speculative(proposals, target_after, &decision, err)) return false;
        if (std::getenv("TINYQWEN_EAGLE3_TRACE")) {
            std::fprintf(stderr, "[eagle3-trace] root=%d draft:", pending);
            for (int id : proposals) std::fprintf(stderr, " %d", id);
            std::fprintf(stderr, " target:");
            for (int id : target_after) std::fprintf(stderr, " %d", id);
            std::fprintf(stderr, " accepted=%d\n", decision.accepted);
        }
        result->stats.draft_accepted += decision.accepted;
        const int keep = 1 + decision.accepted;
        if (!target.truncate(target_checkpoint.seq_len + keep, err)) return false;

        // Recurrent draft rows derived from unverified proposals cannot simply
        // be retained. Rebuild the accepted prefix from the target's captured
        // states, paired with accepted proposals and finally the correction or
        // bonus token. The final row prepares the next block's first proposal.
        if (!draft.truncate(draft_checkpoint, err)) return false;
        std::vector<int> refresh_tokens;
        refresh_tokens.reserve(keep);
        refresh_tokens.insert(refresh_tokens.end(), proposals.begin(),
                              proposals.begin() + decision.accepted);
        refresh_tokens.push_back(decision.next_pending);
        const Clock::time_point refresh_begin = Clock::now();
        if (!draft.extend_target(verified_hidden.data(), keep, refresh_tokens.data(),
                                 &prepared_proposal, err)) {
            return false;
        }
        result->stats.draft_ms += elapsed_ms(refresh_begin, Clock::now());
        result->stats.draft_forward_calls += keep;

        if (decision.all_accepted) {
            ++result->stats.bonus_tokens;
        } else {
            ++result->stats.corrections;
            ++result->stats.rollbacks;
        }
        for (int i = 0; i < decision.accepted; ++i) {
            result->generated_ids.push_back(proposals[i]);
            if (config.eos_token_id >= 0 && proposals[i] == config.eos_token_id) {
                result->hit_eos = true;
                result->stats.decode_ms = elapsed_ms(decode_begin, Clock::now());
                return true;
            }
        }
        pending = decision.next_pending;
    }

    result->stats.decode_ms = elapsed_ms(decode_begin, Clock::now());
    return true;
}
} // namespace tinyqwen
