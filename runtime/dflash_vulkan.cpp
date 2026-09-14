#include "dflash_vulkan.h"

#include <vulkan/vulkan.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "dflash_model.h"
#include "ref_ops.h"
#include "vulkan_dflash_argmax_stage1_spv.h"
#include "vulkan_dflash_argmax_stage2_spv.h"
#include "vulkan_dflash_attention_spv.h"
#include "vulkan_dflash_elementwise_spv.h"
#include "vulkan_dflash_markov_add_spv.h"
#include "vulkan_dflash_rmsnorm_spv.h"
#include "vulkan_dflash_rope_spv.h"
#include "vulkan_matmul_f16_packed_spv.h"

#if defined(__clang__)
// See backend_vulkan.cpp: all omitted Vulkan aggregate fields are deliberately
// value-initialized to zero after sType.
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wmissing-field-initializers"
#endif

namespace tinyqwen {
  namespace {

    constexpr uint32_t kElementFuse = 0;
    constexpr uint32_t kElementResidualAdd = 1;
    constexpr uint32_t kElementSwiGlu = 2;
    constexpr uint32_t kElementStoreKv = 3;
    constexpr uint32_t kElementThreads = 256;
    constexpr uint32_t kMatmulRowsPerGroup = 8;
    constexpr uint32_t kMatmulMaxTokens = 8;
    constexpr uint32_t kArgmaxThreads = 256;
    constexpr uint32_t kArgmaxValuesPerLane = 4;

    const char *vk_result_name(VkResult result) {
      switch (result) {
      case VK_SUCCESS:
        return "VK_SUCCESS";
      case VK_NOT_READY:
        return "VK_NOT_READY";
      case VK_TIMEOUT:
        return "VK_TIMEOUT";
      case VK_ERROR_OUT_OF_HOST_MEMORY:
        return "VK_ERROR_OUT_OF_HOST_MEMORY";
      case VK_ERROR_OUT_OF_DEVICE_MEMORY:
        return "VK_ERROR_OUT_OF_DEVICE_MEMORY";
      case VK_ERROR_INITIALIZATION_FAILED:
        return "VK_ERROR_INITIALIZATION_FAILED";
      case VK_ERROR_DEVICE_LOST:
        return "VK_ERROR_DEVICE_LOST";
      case VK_ERROR_MEMORY_MAP_FAILED:
        return "VK_ERROR_MEMORY_MAP_FAILED";
      case VK_ERROR_FEATURE_NOT_PRESENT:
        return "VK_ERROR_FEATURE_NOT_PRESENT";
      case VK_ERROR_INCOMPATIBLE_DRIVER:
        return "VK_ERROR_INCOMPATIBLE_DRIVER";
      case VK_ERROR_TOO_MANY_OBJECTS:
        return "VK_ERROR_TOO_MANY_OBJECTS";
      default:
        return "VK_ERROR_UNKNOWN";
      }
    }

    bool vk_ok(VkResult result, const char *operation, std::string *err) {
      if (result == VK_SUCCESS)
        return true;
      if (err) {
        *err = std::string(operation) + " failed: " + vk_result_name(result) + " (" +
               std::to_string(static_cast<int>(result)) + ")";
      }
      return false;
    }

    uint32_t ceil_div_u32(uint64_t value, uint32_t divisor) {
      return static_cast<uint32_t>((value + divisor - 1) / divisor);
    }

    struct Buffer {
      VkBuffer buffer = VK_NULL_HANDLE;
      VkDeviceMemory memory = VK_NULL_HANDLE;
      VkDeviceSize size = 0;
      void *mapped = nullptr;
      bool coherent = false;
    };

    struct Binding {
      Buffer *buffer = nullptr;
      VkDeviceSize offset = 0;
      VkDeviceSize range = VK_WHOLE_SIZE;
    };

    struct Pipeline {
      VkDescriptorSetLayout descriptor_layout = VK_NULL_HANDLE;
      VkPipelineLayout layout = VK_NULL_HANDLE;
      VkPipeline pipeline = VK_NULL_HANDLE;
      uint32_t bindings = 0;
      uint32_t push_size = 0;
    };

    struct MatmulPush {
      uint32_t rows;
      uint32_t cols;
      uint32_t tokens;
    };

    struct ElementPush {
      uint32_t op;
      uint32_t p0;
      uint32_t p1;
      uint32_t p2;
      uint32_t p3;
      uint32_t p4;
      uint32_t p5;
      uint32_t p6;
    };

    struct RmsPush {
      uint32_t vectors;
      uint32_t vector_length;
      uint32_t input_stride;
      uint32_t output_stride;
      uint32_t input_offset;
      uint32_t output_offset;
      float epsilon;
      uint32_t pad0;
    };

    struct RopePush {
      uint32_t heads;
      uint32_t head_dim;
      uint32_t tokens;
      uint32_t start_position;
      float theta;
      uint32_t pad0;
      uint32_t pad1;
      uint32_t pad2;
    };

    struct AttentionPush {
      uint32_t layer;
      uint32_t max_seq;
      uint32_t kv_dim;
      uint32_t total_context;
      uint32_t noise_tokens;
      uint32_t query_heads;
      uint32_t kv_heads;
      uint32_t head_dim;
      float attention_scale;
      uint32_t pad0;
      uint32_t pad1;
      uint32_t pad2;
    };

    struct MarkovPush {
      uint32_t vocab;
      uint32_t rank;
      uint32_t position;
      int32_t anchor;
    };

    struct ArgmaxPush {
      uint32_t vocab;
      uint32_t position;
      uint32_t values_per_lane;
      uint32_t pad0;
    };

    struct FinalArgmaxPush {
      uint32_t groups;
      uint32_t position;
      uint32_t pad0;
      uint32_t pad1;
    };

    struct HostLayer {
      const float *input_norm = nullptr;
      const uint16_t *q_proj = nullptr;
      const uint16_t *k_proj = nullptr;
      const uint16_t *v_proj = nullptr;
      const uint16_t *k_proj_ctx = nullptr;
      const uint16_t *v_proj_ctx = nullptr;
      const float *q_norm = nullptr;
      const float *k_norm = nullptr;
      const uint16_t *o_proj = nullptr;
      const float *post_norm = nullptr;
      const uint16_t *gate = nullptr;
      const uint16_t *up = nullptr;
      const uint16_t *down = nullptr;
      std::vector<float> fusion_alpha;
    };

    struct HostConfig {
      QwenModel *target = nullptr;
      int layers = 0;
      int hidden = 0;
      int intermediate = 0;
      int query_dim = 0;
      int kv_dim = 0;
      int query_heads = 0;
      int kv_heads = 0;
      int head_dim = 0;
      int vocab = 0;
      int selected_layers = 0;
      int block_size = 0;
      int max_seq = 0;
      int markov_rank = 0;
      int mask_token = -1;
      float rms_epsilon = 0.0f;
      float rope_theta = 0.0f;
      const uint16_t *block_position = nullptr;
      const float *final_norm = nullptr;
      const uint16_t *markov_w1 = nullptr;
      const uint16_t *markov_w2 = nullptr;
      const uint16_t *lm_head = nullptr;
      std::vector<HostLayer> layer;
    };

    struct LayerGpu {
      Buffer input_norm;
      Buffer q_proj;
      Buffer k_proj;
      Buffer v_proj;
      Buffer k_proj_ctx;
      Buffer v_proj_ctx;
      Buffer q_norm;
      Buffer k_norm;
      Buffer o_proj;
      Buffer post_norm;
      Buffer gate;
      Buffer up;
      Buffer down;
      Buffer fusion_alpha;
    };

  } // namespace

  struct DFlashVulkanEngine::Impl {
    explicit Impl(HostConfig config) : cfg(std::move(config)) {}
    ~Impl() {
      shutdown();
    }

    bool init(std::string *err) {
      if (!init_device(err) || !init_pipelines(err) || !init_commands(err))
        return false;
      if (!upload_weights(err) || !allocate_fixed_buffers(err))
        return false;
      trace = std::getenv("TINYQWEN_VULKAN_TRACE") != nullptr;
      std::fprintf(stderr,
                   "[vulkan] DFlash weights resident: %.1f MB, cache %.1f MB, "
                   "one submit/block\n",
                   static_cast<double>(weight_bytes) / 1048576.0,
                   static_cast<double>(key_cache.size + value_cache.size) / 1048576.0);
      return true;
    }

    bool propose(const float *target_hidden_host, int ctx_tokens, int confirmed_seq, int anchor,
                 int block_tokens, std::vector<int> *out, std::string *err) {
      if (!target_hidden_host || !out || ctx_tokens <= 0 || confirmed_seq < 0 || block_tokens < 2 ||
          block_tokens > cfg.block_size ||
          confirmed_seq + ctx_tokens + block_tokens > cfg.max_seq) {
        if (err)
          *err = "invalid Vulkan DFlash proposal dimensions";
        return false;
      }
      if (!ensure_context_buffers(ctx_tokens, err))
        return false;

      const size_t target_hidden_count =
          static_cast<size_t>(ctx_tokens) * cfg.selected_layers * cfg.hidden;
      std::memcpy(target_hidden.mapped, target_hidden_host, target_hidden_count * sizeof(float));
      if (!flush(target_hidden, err))
        return false;

      std::vector<float> initial_hidden(static_cast<size_t>(block_tokens) * cfg.hidden);
      for (int token = 0; token < block_tokens; ++token) {
        float *dst = initial_hidden.data() + static_cast<size_t>(token) * cfg.hidden;
        if (!cfg.target->copy_embedding(token == 0 ? anchor : cfg.mask_token, dst)) {
          if (err)
            *err = "failed to read target embedding for Vulkan DFlash";
          return false;
        }
        const uint16_t *position = cfg.block_position + static_cast<size_t>(token) * cfg.hidden;
        for (int dim = 0; dim < cfg.hidden; ++dim)
          dst[dim] += half_to_float(position[dim]);
      }
      std::memcpy(hidden.mapped, initial_hidden.data(), initial_hidden.size() * sizeof(float));
      if (!flush(hidden, err))
        return false;

      if (!vk_ok(vkResetDescriptorPool(device, descriptor_pool, 0), "vkResetDescriptorPool", err) ||
          !vk_ok(vkResetCommandBuffer(command_buffer, 0), "vkResetCommandBuffer", err)) {
        return false;
      }
      VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
      begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
      if (!vk_ok(vkBeginCommandBuffer(command_buffer, &begin), "vkBeginCommandBuffer", err)) {
        return false;
      }

      // Make mapped host writes visible to the first compute consumers.
      VkMemoryBarrier host_barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
      host_barrier.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
      host_barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
      vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_HOST_BIT,
                           VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &host_barrier, 0, nullptr, 0,
                           nullptr);

      const int start = confirmed_seq + ctx_tokens;
      for (int layer_index = 0; layer_index < cfg.layers; ++layer_index) {
        LayerGpu &weights = gpu_layers[static_cast<size_t>(layer_index)];

        ElementPush fuse{kElementFuse,
                         static_cast<uint32_t>(ctx_tokens),
                         static_cast<uint32_t>(cfg.selected_layers),
                         static_cast<uint32_t>(cfg.hidden),
                         0,
                         0,
                         0,
                         0};
        if (!record(elementwise_pipeline,
                    {{&target_hidden}, {&weights.fusion_alpha}, {&fused}, {&dummy}}, &fuse,
                    sizeof(fuse),
                    ceil_div_u32(static_cast<uint64_t>(ctx_tokens) * cfg.hidden, kElementThreads),
                    err))
          return false;

        for (int token_base = 0; token_base < ctx_tokens;
             token_base += static_cast<int>(kMatmulMaxTokens)) {
          const int chunk = std::min(static_cast<int>(kMatmulMaxTokens), ctx_tokens - token_base);
          const VkDeviceSize input_offset =
              static_cast<VkDeviceSize>(token_base) * cfg.hidden * sizeof(float);
          const VkDeviceSize output_offset =
              static_cast<VkDeviceSize>(token_base) * cfg.kv_dim * sizeof(float);
          if (!record_matmul(weights.k_proj_ctx, fused, key_context, cfg.kv_dim, cfg.hidden, chunk,
                             input_offset, output_offset, err) ||
              !record_matmul(weights.v_proj_ctx, fused, value_context, cfg.kv_dim, cfg.hidden,
                             chunk, input_offset, output_offset, err))
            return false;
        }

        RmsPush context_norm{static_cast<uint32_t>(ctx_tokens * cfg.kv_heads),
                             static_cast<uint32_t>(cfg.head_dim),
                             static_cast<uint32_t>(cfg.head_dim),
                             static_cast<uint32_t>(cfg.head_dim),
                             0,
                             0,
                             cfg.rms_epsilon,
                             0};
        if (!record(rmsnorm_pipeline, {{&key_context}, {&weights.k_norm}, {&key_context}},
                    &context_norm, sizeof(context_norm),
                    static_cast<uint32_t>(ctx_tokens * cfg.kv_heads), err))
          return false;

        RopePush context_rope{static_cast<uint32_t>(cfg.kv_heads),
                              static_cast<uint32_t>(cfg.head_dim),
                              static_cast<uint32_t>(ctx_tokens),
                              static_cast<uint32_t>(confirmed_seq),
                              cfg.rope_theta,
                              0,
                              0,
                              0};
        if (!record(rope_pipeline, {{&key_context}}, &context_rope, sizeof(context_rope),
                    ceil_div_u32(
                        static_cast<uint64_t>(ctx_tokens) * cfg.kv_heads * (cfg.head_dim / 2), 256),
                    err))
          return false;

        ElementPush store{kElementStoreKv,
                          static_cast<uint32_t>(ctx_tokens),
                          static_cast<uint32_t>(cfg.kv_dim),
                          static_cast<uint32_t>(layer_index),
                          static_cast<uint32_t>(cfg.max_seq),
                          static_cast<uint32_t>(confirmed_seq),
                          0,
                          0};
        if (!record(elementwise_pipeline,
                    {{&key_context}, {&value_context}, {&key_cache}, {&value_cache}}, &store,
                    sizeof(store),
                    ceil_div_u32(static_cast<uint64_t>(ctx_tokens) * cfg.kv_dim, kElementThreads),
                    err))
          return false;

        RmsPush input_norm{static_cast<uint32_t>(block_tokens),
                           static_cast<uint32_t>(cfg.hidden),
                           static_cast<uint32_t>(cfg.hidden),
                           static_cast<uint32_t>(cfg.hidden),
                           0,
                           0,
                           cfg.rms_epsilon,
                           0};
        if (!record(rmsnorm_pipeline, {{&hidden}, {&weights.input_norm}, {&normed}}, &input_norm,
                    sizeof(input_norm), static_cast<uint32_t>(block_tokens), err) ||
            !record_matmul(weights.q_proj, normed, query, cfg.query_dim, cfg.hidden, block_tokens,
                           0, 0, err) ||
            !record_matmul(weights.k_proj, normed, noise_key, cfg.kv_dim, cfg.hidden, block_tokens,
                           0, 0, err) ||
            !record_matmul(weights.v_proj, normed, noise_value, cfg.kv_dim, cfg.hidden,
                           block_tokens, 0, 0, err))
          return false;

        RmsPush query_norm{static_cast<uint32_t>(block_tokens * cfg.query_heads),
                           static_cast<uint32_t>(cfg.head_dim),
                           static_cast<uint32_t>(cfg.head_dim),
                           static_cast<uint32_t>(cfg.head_dim),
                           0,
                           0,
                           cfg.rms_epsilon,
                           0};
        RmsPush key_norm{static_cast<uint32_t>(block_tokens * cfg.kv_heads),
                         static_cast<uint32_t>(cfg.head_dim),
                         static_cast<uint32_t>(cfg.head_dim),
                         static_cast<uint32_t>(cfg.head_dim),
                         0,
                         0,
                         cfg.rms_epsilon,
                         0};
        if (!record(rmsnorm_pipeline, {{&query}, {&weights.q_norm}, {&query}}, &query_norm,
                    sizeof(query_norm), static_cast<uint32_t>(block_tokens * cfg.query_heads),
                    err) ||
            !record(rmsnorm_pipeline, {{&noise_key}, {&weights.k_norm}, {&noise_key}}, &key_norm,
                    sizeof(key_norm), static_cast<uint32_t>(block_tokens * cfg.kv_heads), err))
          return false;

        RopePush query_rope{static_cast<uint32_t>(cfg.query_heads),
                            static_cast<uint32_t>(cfg.head_dim),
                            static_cast<uint32_t>(block_tokens),
                            static_cast<uint32_t>(start),
                            cfg.rope_theta,
                            0,
                            0,
                            0};
        RopePush key_rope{static_cast<uint32_t>(cfg.kv_heads),
                          static_cast<uint32_t>(cfg.head_dim),
                          static_cast<uint32_t>(block_tokens),
                          static_cast<uint32_t>(start),
                          cfg.rope_theta,
                          0,
                          0,
                          0};
        if (!record(rope_pipeline, {{&query}}, &query_rope, sizeof(query_rope),
                    ceil_div_u32(static_cast<uint64_t>(block_tokens) * cfg.query_heads *
                                     (cfg.head_dim / 2),
                                 256),
                    err) ||
            !record(rope_pipeline, {{&noise_key}}, &key_rope, sizeof(key_rope),
                    ceil_div_u32(static_cast<uint64_t>(block_tokens) * cfg.kv_heads *
                                     (cfg.head_dim / 2),
                                 256),
                    err))
          return false;

        AttentionPush attention_params{static_cast<uint32_t>(layer_index),
                                       static_cast<uint32_t>(cfg.max_seq),
                                       static_cast<uint32_t>(cfg.kv_dim),
                                       static_cast<uint32_t>(start),
                                       static_cast<uint32_t>(block_tokens),
                                       static_cast<uint32_t>(cfg.query_heads),
                                       static_cast<uint32_t>(cfg.kv_heads),
                                       static_cast<uint32_t>(cfg.head_dim),
                                       1.0f / std::sqrt(static_cast<float>(cfg.head_dim)),
                                       0,
                                       0,
                                       0};
        if (!record(attention_pipeline,
                    {{&query},
                     {&noise_key},
                     {&noise_value},
                     {&key_cache},
                     {&value_cache},
                     {&attention}},
                    &attention_params, sizeof(attention_params),
                    static_cast<uint32_t>(block_tokens * cfg.query_heads), err) ||
            !record_matmul(weights.o_proj, attention, projected, cfg.hidden, cfg.query_dim,
                           block_tokens, 0, 0, err))
          return false;

        ElementPush add_attention{kElementResidualAdd,
                                  static_cast<uint32_t>(block_tokens * cfg.hidden),
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0};
        if (!record(elementwise_pipeline, {{&hidden}, {&projected}, {&dummy}, {&dummy}},
                    &add_attention, sizeof(add_attention),
                    ceil_div_u32(static_cast<uint64_t>(block_tokens) * cfg.hidden, kElementThreads),
                    err))
          return false;

        RmsPush post_norm{static_cast<uint32_t>(block_tokens),
                          static_cast<uint32_t>(cfg.hidden),
                          static_cast<uint32_t>(cfg.hidden),
                          static_cast<uint32_t>(cfg.hidden),
                          0,
                          0,
                          cfg.rms_epsilon,
                          0};
        if (!record(rmsnorm_pipeline, {{&hidden}, {&weights.post_norm}, {&normed}}, &post_norm,
                    sizeof(post_norm), static_cast<uint32_t>(block_tokens), err) ||
            !record_matmul(weights.gate, normed, gate, cfg.intermediate, cfg.hidden, block_tokens,
                           0, 0, err) ||
            !record_matmul(weights.up, normed, up, cfg.intermediate, cfg.hidden, block_tokens, 0, 0,
                           err))
          return false;

        ElementPush swiglu{kElementSwiGlu,
                           static_cast<uint32_t>(block_tokens * cfg.intermediate),
                           0,
                           0,
                           0,
                           0,
                           0,
                           0};
        if (!record(elementwise_pipeline, {{&gate}, {&up}, {&dummy}, {&dummy}}, &swiglu,
                    sizeof(swiglu),
                    ceil_div_u32(static_cast<uint64_t>(block_tokens) * cfg.intermediate,
                                 kElementThreads),
                    err) ||
            !record_matmul(weights.down, gate, feed_forward, cfg.hidden, cfg.intermediate,
                           block_tokens, 0, 0, err))
          return false;

        ElementPush add_ffn{kElementResidualAdd,
                            static_cast<uint32_t>(block_tokens * cfg.hidden),
                            0,
                            0,
                            0,
                            0,
                            0,
                            0};
        if (!record(elementwise_pipeline, {{&hidden}, {&feed_forward}, {&dummy}, {&dummy}},
                    &add_ffn, sizeof(add_ffn),
                    ceil_div_u32(static_cast<uint64_t>(block_tokens) * cfg.hidden, kElementThreads),
                    err))
          return false;
      }

      const int proposal_count = block_tokens - 1;
      RmsPush final_norm_params{static_cast<uint32_t>(proposal_count),
                                static_cast<uint32_t>(cfg.hidden),
                                static_cast<uint32_t>(cfg.hidden),
                                static_cast<uint32_t>(cfg.hidden),
                                static_cast<uint32_t>(cfg.hidden),
                                0,
                                cfg.rms_epsilon,
                                0};
      if (!record(rmsnorm_pipeline, {{&hidden}, {&final_norm}, {&normed}}, &final_norm_params,
                  sizeof(final_norm_params), static_cast<uint32_t>(proposal_count), err) ||
          !record_matmul(lm_head, normed, logits, cfg.vocab, cfg.hidden, proposal_count, 0, 0, err))
        return false;

      const uint32_t argmax_groups = ceil_div_u32(cfg.vocab, kArgmaxThreads * kArgmaxValuesPerLane);
      for (int position = 0; position < proposal_count; ++position) {
        MarkovPush markov_params{static_cast<uint32_t>(cfg.vocab),
                                 static_cast<uint32_t>(cfg.markov_rank),
                                 static_cast<uint32_t>(position), anchor};
        if (!record(markov_pipeline, {{&logits}, {&markov_w1}, {&markov_w2}, {&proposal_ids}},
                    &markov_params, sizeof(markov_params), ceil_div_u32(cfg.vocab, 256), err))
          return false;

        ArgmaxPush first{static_cast<uint32_t>(cfg.vocab), static_cast<uint32_t>(position),
                         kArgmaxValuesPerLane, 0};
        if (!record(argmax_stage1_pipeline, {{&logits}, {&argmax_values}, {&argmax_indices}},
                    &first, sizeof(first), argmax_groups, err))
          return false;
        FinalArgmaxPush final{argmax_groups, static_cast<uint32_t>(position), 0, 0};
        if (!record(argmax_stage2_pipeline, {{&argmax_values}, {&argmax_indices}, {&proposal_ids}},
                    &final, sizeof(final), 1, err))
          return false;
      }

      if (!vk_ok(vkEndCommandBuffer(command_buffer), "vkEndCommandBuffer", err) ||
          !vk_ok(vkResetFences(device, 1, &fence), "vkResetFences", err)) {
        return false;
      }
      VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
      submit.commandBufferCount = 1;
      submit.pCommandBuffers = &command_buffer;
      const auto begin_time = std::chrono::steady_clock::now();
      if (!vk_ok(vkQueueSubmit(queue, 1, &submit, fence), "vkQueueSubmit", err) ||
          !vk_ok(vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX), "vkWaitForFences", err)) {
        return false;
      }
      const auto end_time = std::chrono::steady_clock::now();
      const double elapsed =
          std::chrono::duration<double, std::milli>(end_time - begin_time).count();
      queue_wait_ms += elapsed;
      ++blocks;

      if (!invalidate(proposal_ids, err))
        return false;
      const auto *ids = static_cast<const int32_t *>(proposal_ids.mapped);
      out->assign(ids, ids + proposal_count);
      if (trace) {
        std::fprintf(
            stderr,
            "[vulkan-dflash] ctx=%d block=%d commands=%llu queue_wait=%.3f ms ids:", ctx_tokens,
            block_tokens, static_cast<unsigned long long>(commands_this_block), elapsed);
        for (int id : *out)
          std::fprintf(stderr, " %d", id);
        std::fprintf(stderr, "\n");
      }
      commands += commands_this_block;
      commands_this_block = 0;
      return true;
    }

    bool init_device(std::string *err) {
      uint32_t loader_version = VK_API_VERSION_1_0;
      auto enumerate_version = reinterpret_cast<PFN_vkEnumerateInstanceVersion>(
          vkGetInstanceProcAddr(VK_NULL_HANDLE, "vkEnumerateInstanceVersion"));
      if (enumerate_version)
        enumerate_version(&loader_version);
      if (loader_version < VK_API_VERSION_1_2) {
        if (err)
          *err = "Vulkan 1.2 loader is required";
        return false;
      }

      VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
      app.pApplicationName = "tinyqwen-dflash";
      app.apiVersion = VK_API_VERSION_1_2;
      VkInstanceCreateInfo instance_info{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
      instance_info.pApplicationInfo = &app;
      if (!vk_ok(vkCreateInstance(&instance_info, nullptr, &instance), "vkCreateInstance", err))
        return false;

      uint32_t count = 0;
      if (!vk_ok(vkEnumeratePhysicalDevices(instance, &count, nullptr),
                 "vkEnumeratePhysicalDevices(count)", err) ||
          count == 0) {
        if (count == 0 && err)
          *err = "no Vulkan physical device found";
        return false;
      }
      std::vector<VkPhysicalDevice> candidates(count);
      if (!vk_ok(vkEnumeratePhysicalDevices(instance, &count, candidates.data()),
                 "vkEnumeratePhysicalDevices", err))
        return false;

      for (VkPhysicalDevice candidate : candidates) {
        VkPhysicalDeviceSubgroupProperties subgroup{
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES};
        VkPhysicalDeviceProperties2 props2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
        props2.pNext = &subgroup;
        vkGetPhysicalDeviceProperties2(candidate, &props2);
        const VkPhysicalDeviceProperties &props = props2.properties;
        if (props.apiVersion < VK_API_VERSION_1_2 ||
            props.limits.maxComputeWorkGroupInvocations < 256 ||
            props.limits.maxComputeWorkGroupSize[0] < 256 ||
            props.limits.maxPushConstantsSize < sizeof(AttentionPush) ||
            subgroup.subgroupSize % 16 != 0 ||
            (subgroup.supportedStages & VK_SHADER_STAGE_COMPUTE_BIT) == 0 ||
            (subgroup.supportedOperations & VK_SUBGROUP_FEATURE_CLUSTERED_BIT) == 0)
          continue;

        VkPhysicalDeviceVulkan12Features f12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
        VkPhysicalDeviceVulkan11Features f11{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES};
        f11.pNext = &f12;
        VkPhysicalDeviceFeatures2 features{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
        features.pNext = &f11;
        vkGetPhysicalDeviceFeatures2(candidate, &features);
        if (!f11.storageBuffer16BitAccess || !f12.shaderFloat16)
          continue;

        uint32_t queue_count = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(candidate, &queue_count, nullptr);
        std::vector<VkQueueFamilyProperties> queue_props(queue_count);
        vkGetPhysicalDeviceQueueFamilyProperties(candidate, &queue_count, queue_props.data());
        uint32_t family = std::numeric_limits<uint32_t>::max();
        for (uint32_t i = 0; i < queue_count; ++i) {
          if (queue_props[i].queueCount && (queue_props[i].queueFlags & VK_QUEUE_COMPUTE_BIT)) {
            family = i;
            break;
          }
        }
        if (family == std::numeric_limits<uint32_t>::max())
          continue;
        physical = candidate;
        properties = props;
        queue_family = family;
        break;
      }
      if (physical == VK_NULL_HANDLE) {
        if (err)
          *err = "no FP16-capable Vulkan compute device found";
        return false;
      }
      vkGetPhysicalDeviceMemoryProperties(physical, &memory_properties);
      device_name_value = properties.deviceName;

      const float priority = 1.0f;
      VkDeviceQueueCreateInfo queue_info{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
      queue_info.queueFamilyIndex = queue_family;
      queue_info.queueCount = 1;
      queue_info.pQueuePriorities = &priority;
      VkPhysicalDeviceVulkan12Features enabled12{
          VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
      enabled12.shaderFloat16 = VK_TRUE;
      VkPhysicalDeviceVulkan11Features enabled11{
          VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES};
      enabled11.storageBuffer16BitAccess = VK_TRUE;
      enabled11.pNext = &enabled12;
      VkDeviceCreateInfo device_info{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
      device_info.pNext = &enabled11;
      device_info.queueCreateInfoCount = 1;
      device_info.pQueueCreateInfos = &queue_info;
      if (!vk_ok(vkCreateDevice(physical, &device_info, nullptr, &device), "vkCreateDevice", err))
        return false;
      vkGetDeviceQueue(device, queue_family, 0, &queue);
      return true;
    }

    bool init_pipelines(std::string *err) {
      return create_pipeline(kVulkanMatmulF16PackedSpv, kVulkanMatmulF16PackedSpvSize, 3,
                             sizeof(MatmulPush), &matmul_pipeline, err) &&
             create_pipeline(kDFlashElementwiseSpv, kDFlashElementwiseSpvSize, 4,
                             sizeof(ElementPush), &elementwise_pipeline, err) &&
             create_pipeline(kDFlashRmsnormSpv, kDFlashRmsnormSpvSize, 3, sizeof(RmsPush),
                             &rmsnorm_pipeline, err) &&
             create_pipeline(kDFlashRopeSpv, kDFlashRopeSpvSize, 1, sizeof(RopePush),
                             &rope_pipeline, err) &&
             create_pipeline(kDFlashAttentionSpv, kDFlashAttentionSpvSize, 6, sizeof(AttentionPush),
                             &attention_pipeline, err) &&
             create_pipeline(kDFlashMarkovAddSpv, kDFlashMarkovAddSpvSize, 4, sizeof(MarkovPush),
                             &markov_pipeline, err) &&
             create_pipeline(kDFlashArgmaxStage1Spv, kDFlashArgmaxStage1SpvSize, 3,
                             sizeof(ArgmaxPush), &argmax_stage1_pipeline, err) &&
             create_pipeline(kDFlashArgmaxStage2Spv, kDFlashArgmaxStage2SpvSize, 3,
                             sizeof(FinalArgmaxPush), &argmax_stage2_pipeline, err);
    }

    bool create_pipeline(const unsigned char *bytes, size_t byte_count, uint32_t binding_count,
                         uint32_t push_size, Pipeline *out, std::string *err) {
      if (!out || (byte_count & 3u) != 0) {
        if (err)
          *err = "invalid embedded Vulkan shader";
        return false;
      }
      std::vector<VkDescriptorSetLayoutBinding> bindings(binding_count);
      for (uint32_t i = 0; i < binding_count; ++i) {
        bindings[i].binding = i;
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
      }
      VkDescriptorSetLayoutCreateInfo descriptor_info{
          VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
      descriptor_info.bindingCount = binding_count;
      descriptor_info.pBindings = bindings.data();
      if (!vk_ok(vkCreateDescriptorSetLayout(device, &descriptor_info, nullptr,
                                             &out->descriptor_layout),
                 "vkCreateDescriptorSetLayout", err))
        return false;

      VkPushConstantRange push{};
      push.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
      push.size = push_size;
      VkPipelineLayoutCreateInfo layout_info{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
      layout_info.setLayoutCount = 1;
      layout_info.pSetLayouts = &out->descriptor_layout;
      layout_info.pushConstantRangeCount = 1;
      layout_info.pPushConstantRanges = &push;
      if (!vk_ok(vkCreatePipelineLayout(device, &layout_info, nullptr, &out->layout),
                 "vkCreatePipelineLayout", err))
        return false;

      VkShaderModuleCreateInfo shader_info{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
      shader_info.codeSize = byte_count;
      shader_info.pCode = reinterpret_cast<const uint32_t *>(bytes);
      VkShaderModule shader = VK_NULL_HANDLE;
      if (!vk_ok(vkCreateShaderModule(device, &shader_info, nullptr, &shader),
                 "vkCreateShaderModule", err))
        return false;
      VkPipelineShaderStageCreateInfo stage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
      stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
      stage.module = shader;
      stage.pName = "main";
      VkComputePipelineCreateInfo pipeline_info{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
      pipeline_info.stage = stage;
      pipeline_info.layout = out->layout;
      const VkResult result = vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipeline_info,
                                                       nullptr, &out->pipeline);
      vkDestroyShaderModule(device, shader, nullptr);
      if (!vk_ok(result, "vkCreateComputePipelines", err))
        return false;
      out->bindings = binding_count;
      out->push_size = push_size;
      return true;
    }

    bool init_commands(std::string *err) {
      VkDescriptorPoolSize pool_size{};
      pool_size.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      pool_size.descriptorCount = 2048;
      VkDescriptorPoolCreateInfo descriptor_info{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
      descriptor_info.maxSets = 256;
      descriptor_info.poolSizeCount = 1;
      descriptor_info.pPoolSizes = &pool_size;
      if (!vk_ok(vkCreateDescriptorPool(device, &descriptor_info, nullptr, &descriptor_pool),
                 "vkCreateDescriptorPool", err))
        return false;

      VkCommandPoolCreateInfo command_info{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
      command_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
      command_info.queueFamilyIndex = queue_family;
      if (!vk_ok(vkCreateCommandPool(device, &command_info, nullptr, &command_pool),
                 "vkCreateCommandPool", err))
        return false;
      VkCommandBufferAllocateInfo allocate_info{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
      allocate_info.commandPool = command_pool;
      allocate_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
      allocate_info.commandBufferCount = 1;
      if (!vk_ok(vkAllocateCommandBuffers(device, &allocate_info, &command_buffer),
                 "vkAllocateCommandBuffers", err))
        return false;
      VkFenceCreateInfo fence_info{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
      return vk_ok(vkCreateFence(device, &fence_info, nullptr, &fence), "vkCreateFence", err);
    }

    uint32_t find_memory_type(uint32_t bits, VkMemoryPropertyFlags required,
                              VkMemoryPropertyFlags preferred) const {
      for (uint32_t i = 0; i < memory_properties.memoryTypeCount; ++i) {
        const VkMemoryPropertyFlags flags = memory_properties.memoryTypes[i].propertyFlags;
        if ((bits & (1u << i)) && (flags & required) == required &&
            (flags & preferred) == preferred)
          return i;
      }
      for (uint32_t i = 0; i < memory_properties.memoryTypeCount; ++i) {
        const VkMemoryPropertyFlags flags = memory_properties.memoryTypes[i].propertyFlags;
        if ((bits & (1u << i)) && (flags & required) == required)
          return i;
      }
      return std::numeric_limits<uint32_t>::max();
    }

    bool create_buffer(VkDeviceSize size, const void *initial, Buffer *out, std::string *err) {
      if (!out || size == 0 || size > properties.limits.maxStorageBufferRange) {
        if (err)
          *err = "invalid or oversized Vulkan DFlash buffer";
        return false;
      }
      Buffer result;
      result.size = size;
      VkBufferCreateInfo info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
      info.size = size;
      info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
      info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
      if (!vk_ok(vkCreateBuffer(device, &info, nullptr, &result.buffer), "vkCreateBuffer", err))
        return false;
      VkMemoryRequirements requirements{};
      vkGetBufferMemoryRequirements(device, result.buffer, &requirements);
      const uint32_t type = find_memory_type(
          requirements.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
          VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
      if (type == std::numeric_limits<uint32_t>::max()) {
        if (err)
          *err = "no host-visible memory for Vulkan DFlash buffer";
        vkDestroyBuffer(device, result.buffer, nullptr);
        return false;
      }
      VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
      allocation.allocationSize = requirements.size;
      allocation.memoryTypeIndex = type;
      if (!vk_ok(vkAllocateMemory(device, &allocation, nullptr, &result.memory), "vkAllocateMemory",
                 err)) {
        vkDestroyBuffer(device, result.buffer, nullptr);
        return false;
      }
      if (!vk_ok(vkBindBufferMemory(device, result.buffer, result.memory, 0), "vkBindBufferMemory",
                 err) ||
          !vk_ok(vkMapMemory(device, result.memory, 0, VK_WHOLE_SIZE, 0, &result.mapped),
                 "vkMapMemory", err)) {
        if (result.memory != VK_NULL_HANDLE)
          vkFreeMemory(device, result.memory, nullptr);
        vkDestroyBuffer(device, result.buffer, nullptr);
        return false;
      }
      result.coherent = (memory_properties.memoryTypes[type].propertyFlags &
                         VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0;
      if (initial) {
        std::memcpy(result.mapped, initial, static_cast<size_t>(size));
        if (!flush(result, err)) {
          destroy_buffer(&result);
          return false;
        }
      }
      *out = result;
      return true;
    }

    bool upload_f16(const uint16_t *data, size_t count, Buffer *out, std::string *err) {
      if (!data || !create_buffer(count * sizeof(uint16_t), data, out, err))
        return false;
      weight_bytes += count * sizeof(uint16_t);
      return true;
    }

    bool upload_f32(const float *data, size_t count, Buffer *out, std::string *err) {
      if (!data || !create_buffer(count * sizeof(float), data, out, err))
        return false;
      weight_bytes += count * sizeof(float);
      return true;
    }

    bool upload_weights(std::string *err) {
      gpu_layers.resize(static_cast<size_t>(cfg.layers));
      for (int i = 0; i < cfg.layers; ++i) {
        const HostLayer &host = cfg.layer[static_cast<size_t>(i)];
        LayerGpu &gpu = gpu_layers[static_cast<size_t>(i)];
        if (!upload_f32(host.input_norm, cfg.hidden, &gpu.input_norm, err) ||
            !upload_f16(host.q_proj, static_cast<size_t>(cfg.query_dim) * cfg.hidden, &gpu.q_proj,
                        err) ||
            !upload_f16(host.k_proj, static_cast<size_t>(cfg.kv_dim) * cfg.hidden, &gpu.k_proj,
                        err) ||
            !upload_f16(host.v_proj, static_cast<size_t>(cfg.kv_dim) * cfg.hidden, &gpu.v_proj,
                        err) ||
            !upload_f16(host.k_proj_ctx, static_cast<size_t>(cfg.kv_dim) * cfg.hidden,
                        &gpu.k_proj_ctx, err) ||
            !upload_f16(host.v_proj_ctx, static_cast<size_t>(cfg.kv_dim) * cfg.hidden,
                        &gpu.v_proj_ctx, err) ||
            !upload_f32(host.q_norm, cfg.head_dim, &gpu.q_norm, err) ||
            !upload_f32(host.k_norm, cfg.head_dim, &gpu.k_norm, err) ||
            !upload_f16(host.o_proj, static_cast<size_t>(cfg.hidden) * cfg.query_dim, &gpu.o_proj,
                        err) ||
            !upload_f32(host.post_norm, cfg.hidden, &gpu.post_norm, err) ||
            !upload_f16(host.gate, static_cast<size_t>(cfg.intermediate) * cfg.hidden, &gpu.gate,
                        err) ||
            !upload_f16(host.up, static_cast<size_t>(cfg.intermediate) * cfg.hidden, &gpu.up,
                        err) ||
            !upload_f16(host.down, static_cast<size_t>(cfg.hidden) * cfg.intermediate, &gpu.down,
                        err) ||
            !upload_f32(host.fusion_alpha.data(), host.fusion_alpha.size(), &gpu.fusion_alpha, err))
          return false;
      }
      return upload_f32(cfg.final_norm, cfg.hidden, &final_norm, err) &&
             upload_f16(cfg.lm_head, static_cast<size_t>(cfg.vocab) * cfg.hidden, &lm_head, err) &&
             upload_f16(cfg.markov_w1, static_cast<size_t>(cfg.vocab) * cfg.markov_rank, &markov_w1,
                        err) &&
             upload_f16(cfg.markov_w2, static_cast<size_t>(cfg.vocab) * cfg.markov_rank, &markov_w2,
                        err);
    }

    bool allocate_fixed_buffers(std::string *err) {
      const size_t block = static_cast<size_t>(cfg.block_size);
      const size_t proposals = block - 1;
      const uint32_t argmax_groups = ceil_div_u32(cfg.vocab, kArgmaxThreads * kArgmaxValuesPerLane);
      const size_t cache_values = static_cast<size_t>(cfg.layers) * cfg.max_seq * cfg.kv_dim;
      return create_buffer(block * cfg.hidden * sizeof(float), nullptr, &hidden, err) &&
             create_buffer(block * cfg.hidden * sizeof(float), nullptr, &normed, err) &&
             create_buffer(block * cfg.query_dim * sizeof(float), nullptr, &query, err) &&
             create_buffer(block * cfg.kv_dim * sizeof(float), nullptr, &noise_key, err) &&
             create_buffer(block * cfg.kv_dim * sizeof(float), nullptr, &noise_value, err) &&
             create_buffer(block * cfg.query_dim * sizeof(float), nullptr, &attention, err) &&
             create_buffer(block * cfg.hidden * sizeof(float), nullptr, &projected, err) &&
             create_buffer(block * cfg.intermediate * sizeof(float), nullptr, &gate, err) &&
             create_buffer(block * cfg.intermediate * sizeof(float), nullptr, &up, err) &&
             create_buffer(block * cfg.hidden * sizeof(float), nullptr, &feed_forward, err) &&
             create_buffer(proposals * cfg.vocab * sizeof(float), nullptr, &logits, err) &&
             create_buffer(proposals * sizeof(int32_t), nullptr, &proposal_ids, err) &&
             create_buffer(argmax_groups * sizeof(float), nullptr, &argmax_values, err) &&
             create_buffer(argmax_groups * sizeof(uint32_t), nullptr, &argmax_indices, err) &&
             create_buffer(cache_values * sizeof(float), nullptr, &key_cache, err) &&
             create_buffer(cache_values * sizeof(float), nullptr, &value_cache, err) &&
             create_buffer(sizeof(float), nullptr, &dummy, err);
    }

    bool ensure_context_buffers(int ctx_tokens, std::string *err) {
      const VkDeviceSize target_bytes =
          static_cast<VkDeviceSize>(ctx_tokens) * cfg.selected_layers * cfg.hidden * sizeof(float);
      const VkDeviceSize fused_bytes =
          static_cast<VkDeviceSize>(ctx_tokens) * cfg.hidden * sizeof(float);
      const VkDeviceSize kv_bytes =
          static_cast<VkDeviceSize>(ctx_tokens) * cfg.kv_dim * sizeof(float);
      return ensure_buffer(target_bytes, &target_hidden, err) &&
             ensure_buffer(fused_bytes, &fused, err) &&
             ensure_buffer(kv_bytes, &key_context, err) &&
             ensure_buffer(kv_bytes, &value_context, err);
    }

    bool ensure_buffer(VkDeviceSize size, Buffer *buffer, std::string *err) {
      if (buffer->size >= size)
        return true;
      if (buffer->buffer != VK_NULL_HANDLE)
        destroy_buffer(buffer);
      return create_buffer(size, nullptr, buffer, err);
    }

    bool flush(const Buffer &buffer, std::string *err) const {
      if (buffer.coherent)
        return true;
      VkMappedMemoryRange range{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE};
      range.memory = buffer.memory;
      range.size = VK_WHOLE_SIZE;
      return vk_ok(vkFlushMappedMemoryRanges(device, 1, &range), "vkFlushMappedMemoryRanges", err);
    }

    bool invalidate(const Buffer &buffer, std::string *err) const {
      if (buffer.coherent)
        return true;
      VkMappedMemoryRange range{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE};
      range.memory = buffer.memory;
      range.size = VK_WHOLE_SIZE;
      return vk_ok(vkInvalidateMappedMemoryRanges(device, 1, &range),
                   "vkInvalidateMappedMemoryRanges", err);
    }

    bool record_matmul(Buffer &weight, Buffer &input, Buffer &output, int rows, int cols,
                       int tokens, VkDeviceSize input_offset, VkDeviceSize output_offset,
                       std::string *err) {
      const VkDeviceSize weight_range = static_cast<VkDeviceSize>(rows) * cols * sizeof(uint16_t);
      const VkDeviceSize input_range = static_cast<VkDeviceSize>(cols) * tokens * sizeof(float);
      const VkDeviceSize output_range = static_cast<VkDeviceSize>(rows) * tokens * sizeof(float);
      MatmulPush params{static_cast<uint32_t>(rows), static_cast<uint32_t>(cols),
                        static_cast<uint32_t>(tokens)};
      return record(matmul_pipeline,
                    {{&weight, 0, weight_range},
                     {&input, input_offset, input_range},
                     {&output, output_offset, output_range}},
                    &params, sizeof(params), ceil_div_u32(rows, kMatmulRowsPerGroup), err);
    }

    bool record(Pipeline &pipeline, const std::vector<Binding> &bindings, const void *push,
                uint32_t push_size, uint32_t groups, std::string *err) {
      if (bindings.size() != pipeline.bindings || push_size != pipeline.push_size || groups == 0) {
        if (err)
          *err = "invalid Vulkan DFlash dispatch description";
        return false;
      }
      VkDescriptorSet set = VK_NULL_HANDLE;
      VkDescriptorSetAllocateInfo allocate_info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
      allocate_info.descriptorPool = descriptor_pool;
      allocate_info.descriptorSetCount = 1;
      allocate_info.pSetLayouts = &pipeline.descriptor_layout;
      if (!vk_ok(vkAllocateDescriptorSets(device, &allocate_info, &set), "vkAllocateDescriptorSets",
                 err))
        return false;

      std::vector<VkDescriptorBufferInfo> infos(bindings.size());
      std::vector<VkWriteDescriptorSet> writes(bindings.size());
      for (size_t i = 0; i < bindings.size(); ++i) {
        const Binding &binding = bindings[i];
        if (!binding.buffer || binding.buffer->buffer == VK_NULL_HANDLE ||
            binding.offset >= binding.buffer->size) {
          if (err)
            *err = "invalid Vulkan DFlash buffer binding";
          return false;
        }
        const VkDeviceSize range =
            binding.range == VK_WHOLE_SIZE ? binding.buffer->size - binding.offset : binding.range;
        if (range == 0 || binding.offset + range > binding.buffer->size ||
            range > properties.limits.maxStorageBufferRange ||
            (binding.offset % properties.limits.minStorageBufferOffsetAlignment) != 0) {
          if (err)
            *err = "invalid Vulkan DFlash buffer slice";
          return false;
        }
        infos[i] = {binding.buffer->buffer, binding.offset, range};
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = set;
        writes[i].dstBinding = static_cast<uint32_t>(i);
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &infos[i];
      }
      vkUpdateDescriptorSets(device, static_cast<uint32_t>(writes.size()), writes.data(), 0,
                             nullptr);
      vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.pipeline);
      vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.layout, 0, 1,
                              &set, 0, nullptr);
      vkCmdPushConstants(command_buffer, pipeline.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, push_size,
                         push);
      vkCmdDispatch(command_buffer, groups, 1, 1);

      VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
      barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
      barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
      vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                           VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0, nullptr, 0,
                           nullptr);
      ++commands_this_block;
      return true;
    }

    void destroy_buffer(Buffer *buffer) {
      if (!buffer || device == VK_NULL_HANDLE)
        return;
      if (buffer->mapped && buffer->memory != VK_NULL_HANDLE)
        vkUnmapMemory(device, buffer->memory);
      if (buffer->buffer != VK_NULL_HANDLE)
        vkDestroyBuffer(device, buffer->buffer, nullptr);
      if (buffer->memory != VK_NULL_HANDLE)
        vkFreeMemory(device, buffer->memory, nullptr);
      *buffer = Buffer{};
    }

    void destroy_pipeline(Pipeline *pipeline) {
      if (!pipeline || device == VK_NULL_HANDLE)
        return;
      if (pipeline->pipeline != VK_NULL_HANDLE)
        vkDestroyPipeline(device, pipeline->pipeline, nullptr);
      if (pipeline->layout != VK_NULL_HANDLE)
        vkDestroyPipelineLayout(device, pipeline->layout, nullptr);
      if (pipeline->descriptor_layout != VK_NULL_HANDLE)
        vkDestroyDescriptorSetLayout(device, pipeline->descriptor_layout, nullptr);
      *pipeline = Pipeline{};
    }

    void destroy_layer(LayerGpu *layer) {
      destroy_buffer(&layer->input_norm);
      destroy_buffer(&layer->q_proj);
      destroy_buffer(&layer->k_proj);
      destroy_buffer(&layer->v_proj);
      destroy_buffer(&layer->k_proj_ctx);
      destroy_buffer(&layer->v_proj_ctx);
      destroy_buffer(&layer->q_norm);
      destroy_buffer(&layer->k_norm);
      destroy_buffer(&layer->o_proj);
      destroy_buffer(&layer->post_norm);
      destroy_buffer(&layer->gate);
      destroy_buffer(&layer->up);
      destroy_buffer(&layer->down);
      destroy_buffer(&layer->fusion_alpha);
    }

    void shutdown() {
      if (device != VK_NULL_HANDLE)
        vkDeviceWaitIdle(device);
      if (device != VK_NULL_HANDLE && (blocks || weight_bytes)) {
        std::fprintf(stderr,
                     "[vulkan-dflash] summary: blocks=%llu commands=%llu "
                     "queue_wait=%.2f ms weights=%.1f MB\n",
                     static_cast<unsigned long long>(blocks),
                     static_cast<unsigned long long>(commands), queue_wait_ms,
                     static_cast<double>(weight_bytes) / 1048576.0);
      }
      for (LayerGpu &layer : gpu_layers)
        destroy_layer(&layer);
      gpu_layers.clear();
      destroy_buffer(&final_norm);
      destroy_buffer(&lm_head);
      destroy_buffer(&markov_w1);
      destroy_buffer(&markov_w2);
      destroy_buffer(&target_hidden);
      destroy_buffer(&hidden);
      destroy_buffer(&fused);
      destroy_buffer(&key_context);
      destroy_buffer(&value_context);
      destroy_buffer(&normed);
      destroy_buffer(&query);
      destroy_buffer(&noise_key);
      destroy_buffer(&noise_value);
      destroy_buffer(&attention);
      destroy_buffer(&projected);
      destroy_buffer(&gate);
      destroy_buffer(&up);
      destroy_buffer(&feed_forward);
      destroy_buffer(&logits);
      destroy_buffer(&proposal_ids);
      destroy_buffer(&argmax_values);
      destroy_buffer(&argmax_indices);
      destroy_buffer(&key_cache);
      destroy_buffer(&value_cache);
      destroy_buffer(&dummy);

      destroy_pipeline(&matmul_pipeline);
      destroy_pipeline(&elementwise_pipeline);
      destroy_pipeline(&rmsnorm_pipeline);
      destroy_pipeline(&rope_pipeline);
      destroy_pipeline(&attention_pipeline);
      destroy_pipeline(&markov_pipeline);
      destroy_pipeline(&argmax_stage1_pipeline);
      destroy_pipeline(&argmax_stage2_pipeline);

      if (device != VK_NULL_HANDLE && fence != VK_NULL_HANDLE)
        vkDestroyFence(device, fence, nullptr);
      if (device != VK_NULL_HANDLE && command_pool != VK_NULL_HANDLE)
        vkDestroyCommandPool(device, command_pool, nullptr);
      if (device != VK_NULL_HANDLE && descriptor_pool != VK_NULL_HANDLE)
        vkDestroyDescriptorPool(device, descriptor_pool, nullptr);
      if (device != VK_NULL_HANDLE)
        vkDestroyDevice(device, nullptr);
      if (instance != VK_NULL_HANDLE)
        vkDestroyInstance(instance, nullptr);
      device = VK_NULL_HANDLE;
      instance = VK_NULL_HANDLE;
    }

    HostConfig cfg;
    std::string device_name_value;
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice physical = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue queue = VK_NULL_HANDLE;
    uint32_t queue_family = 0;
    VkPhysicalDeviceProperties properties{};
    VkPhysicalDeviceMemoryProperties memory_properties{};
    VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
    VkCommandPool command_pool = VK_NULL_HANDLE;
    VkCommandBuffer command_buffer = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;

    Pipeline matmul_pipeline;
    Pipeline elementwise_pipeline;
    Pipeline rmsnorm_pipeline;
    Pipeline rope_pipeline;
    Pipeline attention_pipeline;
    Pipeline markov_pipeline;
    Pipeline argmax_stage1_pipeline;
    Pipeline argmax_stage2_pipeline;

    std::vector<LayerGpu> gpu_layers;
    Buffer final_norm;
    Buffer lm_head;
    Buffer markov_w1;
    Buffer markov_w2;
    Buffer target_hidden;
    Buffer hidden;
    Buffer fused;
    Buffer key_context;
    Buffer value_context;
    Buffer normed;
    Buffer query;
    Buffer noise_key;
    Buffer noise_value;
    Buffer attention;
    Buffer projected;
    Buffer gate;
    Buffer up;
    Buffer feed_forward;
    Buffer logits;
    Buffer proposal_ids;
    Buffer argmax_values;
    Buffer argmax_indices;
    Buffer key_cache;
    Buffer value_cache;
    Buffer dummy;

    bool trace = false;
    uint64_t weight_bytes = 0;
    uint64_t blocks = 0;
    uint64_t commands = 0;
    uint64_t commands_this_block = 0;
    double queue_wait_ms = 0.0;
  };

  std::unique_ptr<DFlashVulkanEngine> DFlashVulkanEngine::create(const DFlashModel &model,
                                                                 std::string *err) {
    HostConfig cfg;
    cfg.target = model.target_;
    cfg.layers = static_cast<int>(model.cfg_.n_layers);
    cfg.hidden = static_cast<int>(model.cfg_.hidden_size);
    cfg.intermediate = static_cast<int>(model.cfg_.intermediate_size);
    cfg.query_heads = static_cast<int>(model.cfg_.n_heads);
    cfg.kv_heads = static_cast<int>(model.cfg_.n_kv_heads);
    cfg.head_dim = static_cast<int>(model.cfg_.head_dim);
    cfg.query_dim = cfg.query_heads * cfg.head_dim;
    cfg.kv_dim = cfg.kv_heads * cfg.head_dim;
    cfg.vocab = static_cast<int>(model.cfg_.vocab_size);
    cfg.selected_layers = static_cast<int>(model.target_layer_ids_.size());
    cfg.block_size = model.block_size_;
    cfg.max_seq = model.max_seq_len_;
    cfg.markov_rank = model.markov_rank_;
    cfg.mask_token = model.mask_token_id_;
    cfg.rms_epsilon = model.cfg_.rms_norm_eps;
    cfg.rope_theta = model.cfg_.rope_theta;
    cfg.block_position = model.block_pos_;
    cfg.final_norm = model.final_norm_;
    cfg.markov_w1 = model.markov_w1_;
    cfg.markov_w2 = model.markov_w2_;

    WeightTensor target_lm_head{};
    if (!cfg.target || !cfg.target->lm_head_weight(&target_lm_head) ||
        target_lm_head.quant_type != QuantType::kF16 || target_lm_head.rows != cfg.vocab ||
        target_lm_head.cols != cfg.hidden) {
      if (err)
        *err = "Vulkan DFlash currently requires the target lm_head in FP16";
      return nullptr;
    }
    cfg.lm_head = static_cast<const uint16_t *>(target_lm_head.data);
    if (cfg.layers <= 0 || cfg.layers != static_cast<int>(model.layers_.size()) ||
        cfg.hidden <= 0 || cfg.intermediate <= 0 || cfg.head_dim <= 0 || cfg.head_dim > 128 ||
        cfg.query_heads % cfg.kv_heads != 0 || cfg.block_size < 2 ||
        cfg.block_size > static_cast<int>(kMatmulMaxTokens) || cfg.selected_layers <= 0 ||
        cfg.markov_rank <= 0 || (cfg.hidden & 1) || (cfg.intermediate & 1) || (cfg.query_dim & 1) ||
        (cfg.kv_dim & 1) || (cfg.markov_rank & 1)) {
      if (err)
        *err = "unsupported Vulkan DFlash model configuration";
      return nullptr;
    }

    cfg.layer.resize(static_cast<size_t>(cfg.layers));
    for (int i = 0; i < cfg.layers; ++i) {
      const DFlashModel::Layer &source = model.layers_[static_cast<size_t>(i)];
      HostLayer &dest = cfg.layer[static_cast<size_t>(i)];
      dest.input_norm = source.input_norm;
      dest.q_proj = source.q_proj;
      dest.k_proj = source.k_proj;
      dest.v_proj = source.v_proj;
      dest.k_proj_ctx = source.k_proj_ctx;
      dest.v_proj_ctx = source.v_proj_ctx;
      dest.q_norm = source.q_norm;
      dest.k_norm = source.k_norm;
      dest.o_proj = source.o_proj;
      dest.post_norm = source.post_norm;
      dest.gate = source.gate;
      dest.up = source.up;
      dest.down = source.down;
      dest.fusion_alpha.resize(static_cast<size_t>(cfg.selected_layers));
      float maximum = -INFINITY;
      for (int k = 0; k < cfg.selected_layers; ++k) {
        maximum = std::max(
            maximum,
            half_to_float(model.fusion_logits_[static_cast<size_t>(i) * cfg.selected_layers + k]));
      }
      float sum = 0.0f;
      for (int k = 0; k < cfg.selected_layers; ++k) {
        const float value = std::exp(
            half_to_float(model.fusion_logits_[static_cast<size_t>(i) * cfg.selected_layers + k]) -
            maximum);
        dest.fusion_alpha[static_cast<size_t>(k)] = value;
        sum += value;
      }
      for (float &value : dest.fusion_alpha)
        value /= sum;
    }

    auto impl = std::make_unique<Impl>(std::move(cfg));
    if (!impl->init(err))
      return nullptr;
    return std::unique_ptr<DFlashVulkanEngine>(new DFlashVulkanEngine(std::move(impl)));
  }

  DFlashVulkanEngine::DFlashVulkanEngine(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}

  DFlashVulkanEngine::~DFlashVulkanEngine() = default;

  bool DFlashVulkanEngine::propose(const float *target_hidden, int ctx_tokens, int confirmed_seq,
                                   int anchor, int block_tokens, std::vector<int> *proposals,
                                   std::string *err) {
    return impl_ && impl_->propose(target_hidden, ctx_tokens, confirmed_seq, anchor, block_tokens,
                                   proposals, err);
  }

  const std::string &DFlashVulkanEngine::device_name() const {
    static const std::string empty;
    return impl_ ? impl_->device_name_value : empty;
  }

} // namespace tinyqwen

#if defined(__clang__)
#pragma clang diagnostic pop
#endif
