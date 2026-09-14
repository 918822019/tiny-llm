#include "backend_vulkan.h"

#include <vulkan/vulkan.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "backend_cpu.h"
#include "vulkan_matmul_f16_spv.h"

#if defined(__clang__)
// Vulkan's canonical C++ aggregate idiom sets sType and value-initializes every
// remaining field.  Clang diagnoses those intentional zero fields under
// -Wextra, which would otherwise drown the Android build in non-actionable
// warnings.
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wmissing-field-initializers"
#endif

namespace tinyqwen {
  namespace {

    constexpr uint32_t kRowsPerGroup = 8;
    constexpr int kMaxTokensPerDispatch = 8;

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
      default:
        return "VK_ERROR_UNKNOWN";
      }
    }

    bool vk_ok(VkResult result, const char *what, std::string *err) {
      if (result == VK_SUCCESS)
        return true;
      if (err) {
        *err = std::string(what) + " failed: " + vk_result_name(result) + " (" +
               std::to_string(static_cast<int>(result)) + ")";
      }
      return false;
    }

    struct Buffer {
      VkBuffer buffer = VK_NULL_HANDLE;
      VkDeviceMemory memory = VK_NULL_HANDLE;
      VkDeviceSize size = 0;
      void *mapped = nullptr;
      bool coherent = false;
    };

    struct MatmulShape {
      uint32_t rows;
      uint32_t cols;
      uint32_t tokens;
    };

  } // namespace

  // VulkanBackend intentionally inherits CPUBackend.  FP16 matrix products use
  // Vulkan; all scalar/stateful operators keep the mature CPU implementation.
  // This provides a correctness-first bridge before the whole DFlash command
  // stream is fused into a fully GPU-resident engine.
  class VulkanBackend final : public CPUBackend {
  public:
    VulkanBackend() = default;
    ~VulkanBackend() override {
      shutdown();
    }

    bool init(std::string *err) {
      uint32_t loader_version = VK_API_VERSION_1_0;
      auto enumerate_version = reinterpret_cast<PFN_vkEnumerateInstanceVersion>(
          vkGetInstanceProcAddr(VK_NULL_HANDLE, "vkEnumerateInstanceVersion"));
      if (enumerate_version)
        enumerate_version(&loader_version);
      if (loader_version < VK_API_VERSION_1_2) {
        if (err)
          *err = "Vulkan 1.2 loader is required for FP16 storage/compute";
        return false;
      }

      VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
      app.pApplicationName = "tinyqwen";
      app.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
      app.pEngineName = "tinyqwen-vulkan";
      app.engineVersion = VK_MAKE_VERSION(1, 0, 0);
      app.apiVersion = VK_API_VERSION_1_2;

      VkInstanceCreateInfo instance_info{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
      instance_info.pApplicationInfo = &app;
      if (!vk_ok(vkCreateInstance(&instance_info, nullptr, &instance_), "vkCreateInstance", err)) {
        return false;
      }

      uint32_t physical_count = 0;
      if (!vk_ok(vkEnumeratePhysicalDevices(instance_, &physical_count, nullptr),
                 "vkEnumeratePhysicalDevices(count)", err) ||
          physical_count == 0) {
        if (physical_count == 0 && err)
          *err = "no Vulkan physical device found";
        return false;
      }
      std::vector<VkPhysicalDevice> physicals(physical_count);
      if (!vk_ok(vkEnumeratePhysicalDevices(instance_, &physical_count, physicals.data()),
                 "vkEnumeratePhysicalDevices", err)) {
        return false;
      }

      for (VkPhysicalDevice candidate : physicals) {
        VkPhysicalDeviceProperties props{};
        vkGetPhysicalDeviceProperties(candidate, &props);
        if (props.apiVersion < VK_API_VERSION_1_2 ||
            props.limits.maxComputeWorkGroupInvocations < 128 ||
            props.limits.maxComputeWorkGroupSize[0] < 128) {
          continue;
        }

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
        std::vector<VkQueueFamilyProperties> queues(queue_count);
        vkGetPhysicalDeviceQueueFamilyProperties(candidate, &queue_count, queues.data());
        uint32_t queue_family = std::numeric_limits<uint32_t>::max();
        for (uint32_t i = 0; i < queue_count; ++i) {
          if (queues[i].queueCount > 0 && (queues[i].queueFlags & VK_QUEUE_COMPUTE_BIT) != 0) {
            queue_family = i;
            break;
          }
        }
        if (queue_family == std::numeric_limits<uint32_t>::max())
          continue;

        physical_ = candidate;
        properties_ = props;
        queue_family_ = queue_family;
        break;
      }
      if (physical_ == VK_NULL_HANDLE) {
        if (err) {
          *err = "no Vulkan 1.2 device with shaderFloat16 and "
                 "storageBuffer16BitAccess was found";
        }
        return false;
      }

      vkGetPhysicalDeviceMemoryProperties(physical_, &memory_properties_);

      const float priority = 1.0f;
      VkDeviceQueueCreateInfo queue_info{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
      queue_info.queueFamilyIndex = queue_family_;
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
      if (!vk_ok(vkCreateDevice(physical_, &device_info, nullptr, &device_), "vkCreateDevice",
                 err)) {
        return false;
      }
      vkGetDeviceQueue(device_, queue_family_, 0, &queue_);

      if (!create_pipeline(err) || !create_command_objects(err))
        return false;

      trace_ = std::getenv("TINYQWEN_VULKAN_TRACE") != nullptr;
      std::fprintf(stderr,
                   "[vulkan] device: %s, API %u.%u.%u, FP16 storage/compute, "
                   "maxStorageBuffer=%.0f MB\n",
                   properties_.deviceName, VK_API_VERSION_MAJOR(properties_.apiVersion),
                   VK_API_VERSION_MINOR(properties_.apiVersion),
                   VK_API_VERSION_PATCH(properties_.apiVersion),
                   static_cast<double>(properties_.limits.maxStorageBufferRange) / 1048576.0);
      return true;
    }

    void matvec(const WeightTensor &w, const float *x, float *y, int out_dim, int in_dim) override {
      matmul(w, x, y, out_dim, in_dim, 1);
    }

    void matvec_pair(const WeightTensor &w1, const WeightTensor &w2, const float *x, float *y1,
                     float *y2, int out_dim, int in_dim) override {
      if (w1.quant_type == QuantType::kF16 && w2.quant_type == QuantType::kF16) {
        matmul(w1, x, y1, out_dim, in_dim, 1);
        matmul(w2, x, y2, out_dim, in_dim, 1);
        return;
      }
      CPUBackend::matvec_pair(w1, w2, x, y1, y2, out_dim, in_dim);
    }

    void matvec_qkv(const WeightTensor &wq, const WeightTensor &wk, const WeightTensor &wv,
                    const float *x, float *yq, float *yk, float *yv, int q_dim, int kv_dim,
                    int in_dim) override {
      if (wq.quant_type == QuantType::kF16 && wk.quant_type == QuantType::kF16 &&
          wv.quant_type == QuantType::kF16) {
        matmul(wq, x, yq, q_dim, in_dim, 1);
        matmul(wk, x, yk, kv_dim, in_dim, 1);
        matmul(wv, x, yv, kv_dim, in_dim, 1);
        return;
      }
      CPUBackend::matvec_qkv(wq, wk, wv, x, yq, yk, yv, q_dim, kv_dim, in_dim);
    }

    void matmul(const WeightTensor &w, const float *x, float *y, int rows, int cols,
                int tokens) override {
      if (w.quant_type != QuantType::kF16 || disabled_ || !x || !y || !w.data || rows <= 0 ||
          cols <= 0 || tokens <= 0) {
        CPUBackend::matmul(w, x, y, rows, cols, tokens);
        return;
      }

      std::lock_guard<std::mutex> lock(mutex_);
      std::string error;
      bool ok = true;
      for (int token_base = 0; token_base < tokens; token_base += kMaxTokensPerDispatch) {
        const int chunk = std::min(kMaxTokensPerDispatch, tokens - token_base);
        ok = run_matmul_f16(static_cast<const uint16_t *>(w.data),
                            x + static_cast<size_t>(token_base) * cols,
                            y + static_cast<size_t>(token_base) * rows, rows, cols, chunk, &error);
        if (!ok)
          break;
      }
      if (ok)
        return;

      disabled_ = true;
      ++fallbacks_;
      std::fprintf(stderr,
                   "[vulkan] GPU matmul failed (%s); disabling Vulkan and "
                   "recomputing on CPU\n",
                   error.c_str());
      CPUBackend::matmul(w, x, y, rows, cols, tokens);
    }

    bool prepare_weight(const WeightTensor &w, std::string *err) override {
      if (w.quant_type != QuantType::kF16 || disabled_)
        return true;
      if (!w.data || w.rows <= 0 || w.cols <= 0) {
        if (err)
          *err = "invalid Vulkan weight preparation request";
        return false;
      }
      const uint64_t elements = static_cast<uint64_t>(w.rows) * w.cols;
      if (elements > std::numeric_limits<VkDeviceSize>::max() / sizeof(uint16_t)) {
        if (err)
          *err = "Vulkan prepared weight size overflow";
        return false;
      }
      const VkDeviceSize bytes = elements * sizeof(uint16_t);
      if (bytes > properties_.limits.maxStorageBufferRange) {
        if (err)
          *err = "Vulkan prepared weight exceeds maxStorageBufferRange";
        return false;
      }
      std::lock_guard<std::mutex> lock(mutex_);
      return weight_buffer(static_cast<const uint16_t *>(w.data), bytes, err) != nullptr;
    }

  private:
    bool create_pipeline(std::string *err) {
      VkDescriptorSetLayoutBinding bindings[3]{};
      for (uint32_t i = 0; i < 3; ++i) {
        bindings[i].binding = i;
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
      }
      VkDescriptorSetLayoutCreateInfo set_info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
      set_info.bindingCount = 3;
      set_info.pBindings = bindings;
      if (!vk_ok(vkCreateDescriptorSetLayout(device_, &set_info, nullptr, &descriptor_layout_),
                 "vkCreateDescriptorSetLayout", err)) {
        return false;
      }

      VkPushConstantRange push{};
      push.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
      push.offset = 0;
      push.size = sizeof(MatmulShape);
      VkPipelineLayoutCreateInfo layout_info{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
      layout_info.setLayoutCount = 1;
      layout_info.pSetLayouts = &descriptor_layout_;
      layout_info.pushConstantRangeCount = 1;
      layout_info.pPushConstantRanges = &push;
      if (!vk_ok(vkCreatePipelineLayout(device_, &layout_info, nullptr, &pipeline_layout_),
                 "vkCreatePipelineLayout", err)) {
        return false;
      }

      if ((kVulkanMatmulF16SpvSize & 3u) != 0) {
        if (err)
          *err = "embedded matmul SPIR-V is not word aligned";
        return false;
      }
      VkShaderModuleCreateInfo shader_info{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
      shader_info.codeSize = kVulkanMatmulF16SpvSize;
      shader_info.pCode = reinterpret_cast<const uint32_t *>(kVulkanMatmulF16Spv);
      VkShaderModule shader = VK_NULL_HANDLE;
      if (!vk_ok(vkCreateShaderModule(device_, &shader_info, nullptr, &shader),
                 "vkCreateShaderModule", err)) {
        return false;
      }

      VkPipelineShaderStageCreateInfo stage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
      stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
      stage.module = shader;
      stage.pName = "main";
      VkComputePipelineCreateInfo pipeline_info{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
      pipeline_info.stage = stage;
      pipeline_info.layout = pipeline_layout_;
      const VkResult pipeline_result =
          vkCreateComputePipelines(device_, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &pipeline_);
      vkDestroyShaderModule(device_, shader, nullptr);
      if (!vk_ok(pipeline_result, "vkCreateComputePipelines", err))
        return false;

      VkDescriptorPoolSize pool_size{};
      pool_size.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      pool_size.descriptorCount = 3;
      VkDescriptorPoolCreateInfo pool_info{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
      pool_info.maxSets = 1;
      pool_info.poolSizeCount = 1;
      pool_info.pPoolSizes = &pool_size;
      if (!vk_ok(vkCreateDescriptorPool(device_, &pool_info, nullptr, &descriptor_pool_),
                 "vkCreateDescriptorPool", err)) {
        return false;
      }
      VkDescriptorSetAllocateInfo alloc_info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
      alloc_info.descriptorPool = descriptor_pool_;
      alloc_info.descriptorSetCount = 1;
      alloc_info.pSetLayouts = &descriptor_layout_;
      return vk_ok(vkAllocateDescriptorSets(device_, &alloc_info, &descriptor_set_),
                   "vkAllocateDescriptorSets", err);
    }

    bool create_command_objects(std::string *err) {
      VkCommandPoolCreateInfo pool_info{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
      pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
      pool_info.queueFamilyIndex = queue_family_;
      if (!vk_ok(vkCreateCommandPool(device_, &pool_info, nullptr, &command_pool_),
                 "vkCreateCommandPool", err)) {
        return false;
      }
      VkCommandBufferAllocateInfo alloc_info{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
      alloc_info.commandPool = command_pool_;
      alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
      alloc_info.commandBufferCount = 1;
      if (!vk_ok(vkAllocateCommandBuffers(device_, &alloc_info, &command_buffer_),
                 "vkAllocateCommandBuffers", err)) {
        return false;
      }
      VkFenceCreateInfo fence_info{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
      return vk_ok(vkCreateFence(device_, &fence_info, nullptr, &fence_), "vkCreateFence", err);
    }

    uint32_t find_memory_type(uint32_t bits, VkMemoryPropertyFlags required,
                              VkMemoryPropertyFlags preferred) const {
      for (uint32_t i = 0; i < memory_properties_.memoryTypeCount; ++i) {
        const VkMemoryPropertyFlags flags = memory_properties_.memoryTypes[i].propertyFlags;
        if ((bits & (1u << i)) && (flags & required) == required &&
            (flags & preferred) == preferred) {
          return i;
        }
      }
      for (uint32_t i = 0; i < memory_properties_.memoryTypeCount; ++i) {
        const VkMemoryPropertyFlags flags = memory_properties_.memoryTypes[i].propertyFlags;
        if ((bits & (1u << i)) && (flags & required) == required)
          return i;
      }
      return std::numeric_limits<uint32_t>::max();
    }

    bool create_mapped_buffer(VkDeviceSize size, const void *initial, Buffer *out,
                              std::string *err) {
      if (!out || size == 0) {
        if (err)
          *err = "invalid Vulkan buffer size";
        return false;
      }
      Buffer result;
      result.size = size;

      VkBufferCreateInfo buffer_info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
      buffer_info.size = size;
      buffer_info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
      buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
      if (!vk_ok(vkCreateBuffer(device_, &buffer_info, nullptr, &result.buffer), "vkCreateBuffer",
                 err)) {
        return false;
      }

      VkMemoryRequirements req{};
      vkGetBufferMemoryRequirements(device_, result.buffer, &req);
      const VkMemoryPropertyFlags required = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
      const VkMemoryPropertyFlags preferred =
          VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
      const uint32_t memory_type = find_memory_type(req.memoryTypeBits, required, preferred);
      if (memory_type == std::numeric_limits<uint32_t>::max()) {
        if (err)
          *err = "no host-visible Vulkan memory type for storage buffer";
        vkDestroyBuffer(device_, result.buffer, nullptr);
        return false;
      }

      VkMemoryAllocateInfo memory_info{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
      memory_info.allocationSize = req.size;
      memory_info.memoryTypeIndex = memory_type;
      if (!vk_ok(vkAllocateMemory(device_, &memory_info, nullptr, &result.memory),
                 "vkAllocateMemory", err)) {
        vkDestroyBuffer(device_, result.buffer, nullptr);
        return false;
      }
      if (!vk_ok(vkBindBufferMemory(device_, result.buffer, result.memory, 0), "vkBindBufferMemory",
                 err)) {
        vkFreeMemory(device_, result.memory, nullptr);
        vkDestroyBuffer(device_, result.buffer, nullptr);
        return false;
      }
      if (!vk_ok(vkMapMemory(device_, result.memory, 0, VK_WHOLE_SIZE, 0, &result.mapped),
                 "vkMapMemory", err)) {
        vkFreeMemory(device_, result.memory, nullptr);
        vkDestroyBuffer(device_, result.buffer, nullptr);
        return false;
      }
      const VkMemoryPropertyFlags memory_flags =
          memory_properties_.memoryTypes[memory_type].propertyFlags;
      result.coherent = (memory_flags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0;
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

    bool flush(const Buffer &buffer, std::string *err) const {
      if (buffer.coherent)
        return true;
      VkMappedMemoryRange range{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE};
      range.memory = buffer.memory;
      range.offset = 0;
      range.size = VK_WHOLE_SIZE;
      return vk_ok(vkFlushMappedMemoryRanges(device_, 1, &range), "vkFlushMappedMemoryRanges", err);
    }

    bool invalidate(const Buffer &buffer, std::string *err) const {
      if (buffer.coherent)
        return true;
      VkMappedMemoryRange range{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE};
      range.memory = buffer.memory;
      range.offset = 0;
      range.size = VK_WHOLE_SIZE;
      return vk_ok(vkInvalidateMappedMemoryRanges(device_, 1, &range),
                   "vkInvalidateMappedMemoryRanges", err);
    }

    void destroy_buffer(Buffer *buffer) {
      if (!buffer || device_ == VK_NULL_HANDLE)
        return;
      if (buffer->mapped && buffer->memory != VK_NULL_HANDLE)
        vkUnmapMemory(device_, buffer->memory);
      if (buffer->buffer != VK_NULL_HANDLE)
        vkDestroyBuffer(device_, buffer->buffer, nullptr);
      if (buffer->memory != VK_NULL_HANDLE)
        vkFreeMemory(device_, buffer->memory, nullptr);
      *buffer = Buffer{};
    }

    bool ensure_workspace(VkDeviceSize input_size, VkDeviceSize output_size, std::string *err) {
      if (input_.size < input_size) {
        if (input_.buffer != VK_NULL_HANDLE) {
          vkQueueWaitIdle(queue_);
          destroy_buffer(&input_);
        }
        if (!create_mapped_buffer(input_size, nullptr, &input_, err))
          return false;
      }
      if (output_.size < output_size) {
        if (output_.buffer != VK_NULL_HANDLE) {
          vkQueueWaitIdle(queue_);
          destroy_buffer(&output_);
        }
        if (!create_mapped_buffer(output_size, nullptr, &output_, err))
          return false;
      }
      return true;
    }

    Buffer *weight_buffer(const uint16_t *weight, VkDeviceSize size, std::string *err) {
      auto found = weights_.find(weight);
      if (found != weights_.end()) {
        if (found->second->size != size) {
          if (err)
            *err = "one host weight pointer was reused with a different shape";
          return nullptr;
        }
        return found->second.get();
      }
      auto uploaded = std::make_unique<Buffer>();
      if (!create_mapped_buffer(size, weight, uploaded.get(), err))
        return nullptr;
      uploaded_weight_bytes_ += static_cast<uint64_t>(size);
      if (trace_) {
        std::fprintf(stderr, "[vulkan] upload weight %.2f MB\n",
                     static_cast<double>(size) / 1048576.0);
      }
      Buffer *raw = uploaded.get();
      weights_.emplace(weight, std::move(uploaded));
      return raw;
    }

    bool run_matmul_f16(const uint16_t *weight, const float *x, float *y, int rows, int cols,
                        int tokens, std::string *err) {
      if (tokens < 1 || tokens > kMaxTokensPerDispatch) {
        if (err)
          *err = "Vulkan matmul token tile is outside [1,8]";
        return false;
      }
      const uint64_t weight_elems = static_cast<uint64_t>(rows) * cols;
      const uint64_t input_elems = static_cast<uint64_t>(cols) * tokens;
      const uint64_t output_elems = static_cast<uint64_t>(rows) * tokens;
      if (weight_elems > std::numeric_limits<VkDeviceSize>::max() / sizeof(uint16_t) ||
          input_elems > std::numeric_limits<VkDeviceSize>::max() / sizeof(float) ||
          output_elems > std::numeric_limits<VkDeviceSize>::max() / sizeof(float)) {
        if (err)
          *err = "Vulkan matmul size overflow";
        return false;
      }
      const VkDeviceSize weight_size = weight_elems * sizeof(uint16_t);
      const VkDeviceSize input_size = input_elems * sizeof(float);
      const VkDeviceSize output_size = output_elems * sizeof(float);
      if (weight_size > properties_.limits.maxStorageBufferRange ||
          input_size > properties_.limits.maxStorageBufferRange ||
          output_size > properties_.limits.maxStorageBufferRange) {
        if (err)
          *err = "Vulkan matmul buffer exceeds maxStorageBufferRange";
        return false;
      }

      Buffer *weight_gpu = weight_buffer(weight, weight_size, err);
      if (!weight_gpu || !ensure_workspace(input_size, output_size, err))
        return false;
      std::memcpy(input_.mapped, x, static_cast<size_t>(input_size));
      if (!flush(input_, err))
        return false;

      VkDescriptorBufferInfo buffer_infos[3]{};
      buffer_infos[0] = {weight_gpu->buffer, 0, weight_size};
      buffer_infos[1] = {input_.buffer, 0, input_size};
      buffer_infos[2] = {output_.buffer, 0, output_size};
      VkWriteDescriptorSet writes[3]{};
      for (uint32_t i = 0; i < 3; ++i) {
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = descriptor_set_;
        writes[i].dstBinding = i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &buffer_infos[i];
      }
      vkUpdateDescriptorSets(device_, 3, writes, 0, nullptr);

      if (!vk_ok(vkResetCommandBuffer(command_buffer_, 0), "vkResetCommandBuffer", err)) {
        return false;
      }
      VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
      begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
      if (!vk_ok(vkBeginCommandBuffer(command_buffer_, &begin), "vkBeginCommandBuffer", err)) {
        return false;
      }
      vkCmdBindPipeline(command_buffer_, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_);
      vkCmdBindDescriptorSets(command_buffer_, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout_, 0,
                              1, &descriptor_set_, 0, nullptr);
      const MatmulShape shape{static_cast<uint32_t>(rows), static_cast<uint32_t>(cols),
                              static_cast<uint32_t>(tokens)};
      vkCmdPushConstants(command_buffer_, pipeline_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                         sizeof(shape), &shape);
      vkCmdDispatch(command_buffer_,
                    (static_cast<uint32_t>(rows) + kRowsPerGroup - 1) / kRowsPerGroup, 1, 1);
      if (!vk_ok(vkEndCommandBuffer(command_buffer_), "vkEndCommandBuffer", err))
        return false;

      if (!vk_ok(vkResetFences(device_, 1, &fence_), "vkResetFences", err))
        return false;
      VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
      submit.commandBufferCount = 1;
      submit.pCommandBuffers = &command_buffer_;
      const auto begin_time = std::chrono::steady_clock::now();
      if (!vk_ok(vkQueueSubmit(queue_, 1, &submit, fence_), "vkQueueSubmit", err) ||
          !vk_ok(vkWaitForFences(device_, 1, &fence_, VK_TRUE, UINT64_MAX), "vkWaitForFences",
                 err)) {
        return false;
      }
      const auto end_time = std::chrono::steady_clock::now();
      gpu_wait_ms_ += std::chrono::duration<double, std::milli>(end_time - begin_time).count();
      ++dispatches_;
      if (!invalidate(output_, err))
        return false;
      std::memcpy(y, output_.mapped, static_cast<size_t>(output_size));
      return true;
    }

    void shutdown() {
      if (device_ != VK_NULL_HANDLE)
        vkDeviceWaitIdle(device_);
      for (auto &entry : weights_)
        destroy_buffer(entry.second.get());
      weights_.clear();
      destroy_buffer(&input_);
      destroy_buffer(&output_);

      if (device_ != VK_NULL_HANDLE && (dispatches_ > 0 || uploaded_weight_bytes_ > 0)) {
        std::fprintf(stderr,
                     "[vulkan] summary: dispatches=%llu, weights=%.1f MB, "
                     "queue_wait=%.2f ms, fallbacks=%llu\n",
                     static_cast<unsigned long long>(dispatches_),
                     static_cast<double>(uploaded_weight_bytes_) / 1048576.0, gpu_wait_ms_,
                     static_cast<unsigned long long>(fallbacks_));
      }
      if (device_ != VK_NULL_HANDLE && fence_ != VK_NULL_HANDLE)
        vkDestroyFence(device_, fence_, nullptr);
      if (device_ != VK_NULL_HANDLE && command_pool_ != VK_NULL_HANDLE)
        vkDestroyCommandPool(device_, command_pool_, nullptr);
      if (device_ != VK_NULL_HANDLE && descriptor_pool_ != VK_NULL_HANDLE)
        vkDestroyDescriptorPool(device_, descriptor_pool_, nullptr);
      if (device_ != VK_NULL_HANDLE && pipeline_ != VK_NULL_HANDLE)
        vkDestroyPipeline(device_, pipeline_, nullptr);
      if (device_ != VK_NULL_HANDLE && pipeline_layout_ != VK_NULL_HANDLE)
        vkDestroyPipelineLayout(device_, pipeline_layout_, nullptr);
      if (device_ != VK_NULL_HANDLE && descriptor_layout_ != VK_NULL_HANDLE)
        vkDestroyDescriptorSetLayout(device_, descriptor_layout_, nullptr);
      if (device_ != VK_NULL_HANDLE)
        vkDestroyDevice(device_, nullptr);
      if (instance_ != VK_NULL_HANDLE)
        vkDestroyInstance(instance_, nullptr);

      instance_ = VK_NULL_HANDLE;
      physical_ = VK_NULL_HANDLE;
      device_ = VK_NULL_HANDLE;
      queue_ = VK_NULL_HANDLE;
    }

    VkInstance instance_ = VK_NULL_HANDLE;
    VkPhysicalDevice physical_ = VK_NULL_HANDLE;
    VkDevice device_ = VK_NULL_HANDLE;
    VkQueue queue_ = VK_NULL_HANDLE;
    uint32_t queue_family_ = 0;
    VkPhysicalDeviceProperties properties_{};
    VkPhysicalDeviceMemoryProperties memory_properties_{};

    VkDescriptorSetLayout descriptor_layout_ = VK_NULL_HANDLE;
    VkPipelineLayout pipeline_layout_ = VK_NULL_HANDLE;
    VkPipeline pipeline_ = VK_NULL_HANDLE;
    VkDescriptorPool descriptor_pool_ = VK_NULL_HANDLE;
    VkDescriptorSet descriptor_set_ = VK_NULL_HANDLE;
    VkCommandPool command_pool_ = VK_NULL_HANDLE;
    VkCommandBuffer command_buffer_ = VK_NULL_HANDLE;
    VkFence fence_ = VK_NULL_HANDLE;

    std::unordered_map<const void *, std::unique_ptr<Buffer>> weights_;
    Buffer input_;
    Buffer output_;
    std::mutex mutex_;
    bool trace_ = false;
    bool disabled_ = false;
    uint64_t dispatches_ = 0;
    uint64_t uploaded_weight_bytes_ = 0;
    uint64_t fallbacks_ = 0;
    double gpu_wait_ms_ = 0.0;
  };

  bool vulkan_backend_available() {
    return true;
  }

  std::unique_ptr<IBackend> create_vulkan_backend(std::string *err) {
    auto backend = std::make_unique<VulkanBackend>();
    if (!backend->init(err))
      return nullptr;
    return backend;
  }

} // namespace tinyqwen

#if defined(__clang__)
#pragma clang diagnostic pop
#endif
