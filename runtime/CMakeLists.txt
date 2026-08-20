# runtime：loader / tensor view / kv cache / profiler / qwen forward
#
# 这里是 OBJECT 库而不是 STATIC：kernel 变体靠静态注册器接入 dispatch，
# 静态库归档是"按需提取"成员——没人引用的 .o（比如注册器所在的变体文件）
# 不会被链入，注册就丢了。OBJECT 库配合下方可执行的"显式直链"
# （tinyqwen / tinyqwen_tests 各自直链 runtime + kernels），把所有 .o
# 直接放上链接行，注册器必然生效。
add_library(tinyqwen_runtime OBJECT
        tensor.cpp
        model_loader.cpp
        kv_cache.cpp
        gdn_state.cpp
        profiler.cpp
        config.cpp
        backend_cpu.cpp
        qwen_model.cpp
        qwen_forward_token.cpp
        qwen_forward_prefill.cpp
        ../quantization/quant.cpp
)

# CUDA 后端（仅 CUDA 构建编译）
if (TINYQWEN_HAS_CUDA)
    target_sources(tinyqwen_runtime PRIVATE backend_cuda.cpp)
endif ()
target_include_directories(tinyqwen_runtime PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(tinyqwen_runtime PUBLIC tinyqwen_kernels)

# CLI 可执行
# 显式链两个 OBJECT 库：CMake 3.21 之前，OBJECT 库的 .o 不会沿依赖传递到
# 最终目标，只有"直接链给可执行"才把 .o 真正放上链接行（注册器依赖这一点）。
add_executable(tinyqwen main.cpp)
target_link_libraries(tinyqwen PRIVATE tinyqwen_runtime tinyqwen_kernels)
