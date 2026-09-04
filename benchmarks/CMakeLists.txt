# ============================================================================
# benchmarks/CMakeLists.txt — 算子级微基准构建配置
# ============================================================================
# bench_kernels：枚举 dispatch 各注册表的全部 matvec 实现，逐个校验（对 ref）
# 并计时，输出对比表/CSV/JSON——算子优化的标准评测管线。
# 只依赖 kernels（dispatch + ref_ops），不依赖 runtime。
# ============================================================================

add_executable(bench_kernels bench_kernels.cpp)

# 与 tests 同理：kernels 是 OBJECT 库，.o 直接链入可执行，
# 各变体的静态自注册器（文件末尾 static Registrar）才能被链接器保留。
target_link_libraries(bench_kernels PRIVATE tinyqwen_kernels)

# ============================================================================
# test_metal_continuation：Metal prefill 的 KV 接续等价性测试。
# 纯 C++17（只用 metal_prefill.h 的不透明句柄接口），所以各平台都能编译；
# 非 Apple 平台运行期由 metal_prefill_available() 报"本构建没有引擎"并退出 2。
# 需要链 tinyqwen_runtime（ModelFile 加载 + metal_prefill_*）。
# ============================================================================
add_executable(test_metal_continuation test_metal_continuation.cpp)
target_link_libraries(test_metal_continuation PRIVATE tinyqwen_runtime tinyqwen_kernels)

# ============================================================================
# bench_metal_gemm（仅 Apple）：MPS vs 自写 tiled GEMM 微基准。
# 与 runtime/metal_prefill.mm 同样的门控理由 —— .mm 是 ObjC++，需要 OBJCXX
# （顶层 CMakeLists.txt 已在第一个 add_subdirectory 前 enable_language(OBJCXX)）。
# 非 Apple 平台整段跳过，不影响主线。
# ============================================================================
if (APPLE)
    add_executable(bench_metal_gemm bench_metal_gemm.mm)
    target_link_libraries(bench_metal_gemm PRIVATE "-framework Metal"
                                                   "-framework MetalPerformanceShaders"
                                                   "-framework Foundation")
    # ObjC++ 的 ARC：与 runtime/CMakeLists.txt 对 metal_prefill.mm 的处理一致
    set_target_properties(bench_metal_gemm PROPERTIES COMPILE_FLAGS "-fobjc-arc")
endif ()
