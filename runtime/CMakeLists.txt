add_library(tinyqwen_runtime STATIC
  tensor.cpp
  model_loader.cpp
  kv_cache.cpp
  profiler.cpp
  qwen_model.cpp
)
target_include_directories(tinyqwen_runtime PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(tinyqwen_runtime PUBLIC tinyqwen_kernels)

add_executable(tinyqwen main.cpp)
target_link_libraries(tinyqwen PRIVATE tinyqwen_runtime)
