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
