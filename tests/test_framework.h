// ============================================================================
// test_framework.h — 最小单文件测试框架
// ============================================================================
// 本文件实现了一个轻量级的 C++ 单元测试框架，不依赖任何第三方库。
// 设计目标：一个头文件搞定全部测试基础设施，适合嵌入式/端侧项目。
//
// 核心机制：
//   1. TEST(name) 宏：利用 C++ 静态初始化自注册机制，每个 TEST 用例在
//      程序启动时自动注册到全局 registry，无需手动维护测试列表。
//   2. 断言宏（EXPECT_TRUE/EQ/NEAR）：失败时抛出 Failure 异常，携带
//      文件名、行号和可读的错误信息。run_all() 捕获异常并报告。
//   3. display<T>() 模板：利用 if constexpr 将不同类型的值转为可读字符串，
//      用于 EXPECT_EQ 失败时的诊断输出。
//
// 使用方式：
//   - 测试文件中 #include "test_framework.h"，用 TEST(name) { ... } 定义用例
//   - test_main.cpp 中调用 tinytest::run_all() 统一运行全部用例
//   - 返回值 = 失败的用例数（0 = 全通过）
// ============================================================================

#pragma once  // 防止头文件被重复包含

#include <cmath>        // std::fabs（EXPECT_NEAR 需要）
#include <cstdio>       // std::printf（测试结果输出）
#include <functional>   // std::function（存储测试函数的回调）
#include <string>       // std::string（测试名称、错误消息）
#include <type_traits>  // std::is_same, std::is_arithmetic（display 模板特化）
#include <vector>       // std::vector（测试用例注册表）

namespace tinytest {

    // =========================================================================
    // display<T>() — 将断言中的值转为可读字符串
    // =========================================================================
    // 利用 C++17 if constexpr 在编译期根据类型选择转换策略：
    //   - std::string → 原样返回
    //   - 算术类型（int/float/double/...）→ std::to_string
    //     注意 +v 技巧：char/short 没有 to_string 重载，+v 触发整数提升
    //   - 其他（const char* 等）→ 隐式构造 std::string
    template<typename T>
    inline std::string display(const T &v) {
        if constexpr (std::is_same<T, std::string>::value) {
            // std::string 类型直接返回
            return v;
        } else if constexpr (std::is_arithmetic<T>::value) {
            // 算术类型：+v 把 char/short 提升为 int，避免 to_string 无对应重载
            return std::to_string(+v);
        } else {
            // const char* 等其他类型：通过构造函数转换
            return std::string(v);
        }
    }

    // =========================================================================
    // struct Failure — 断言失败异常
    // =========================================================================
    // 当 EXPECT_* 宏检测到条件不满足时，抛出此异常。
    // msg 字段包含完整的诊断信息：文件名:行号: 断言表达式 + 实际值。
    struct Failure {
        std::string msg;  // 格式化的错误消息
    };

    // =========================================================================
    // struct Case — 单个测试用例的描述
    // =========================================================================
    struct Case {
        std::string name;           // 测试名称（由 TEST 宏的 #name 字符串化得到）
        std::function<void()> fn;   // 测试函数体（无参无返回值的 callable）
    };

    // =========================================================================
    // registry() — 全局测试用例注册表
    // =========================================================================
    // 使用 Meyers' Singleton 模式（函数内 static 变量），确保：
    //   - 线程安全的懒初始化（C++11 起保证）
    //   - 在所有静态对象的构造期间都可用（比全局 static 更可靠）
    // 返回：指向全局 vector<Case> 的引用
    inline std::vector<Case> &registry() {
        static std::vector<Case> cases;  // 首次调用时构造，之后复用
        return cases;
    }

    // =========================================================================
    // struct Registrar — 测试自注册的桥梁
    // =========================================================================
    // 原理：TEST 宏在文件作用域创建一个 static Registrar 对象。
    // C++ 标准保证：同一翻译单元内的 static 对象按声明顺序构造。
    // Registrar 的构造函数将测试函数推入全局 registry，实现"声明即注册"。
    struct Registrar {
        // 构造函数：将测试用例注册到全局 registry
        // name — 测试名称字符串
        // fn   — 测试函数体
        Registrar(const std::string &name, std::function<void()> fn) {
            registry().push_back(Case{name, std::move(fn)});
        }
    };

    // =========================================================================
    // run_all() — 运行所有已注册的测试用例
    // =========================================================================
    // 遍历 registry 中的每个 Case，依次执行其 fn：
    //   - 正常完成 → 打印 [ OK ]
    //   - 抛出 Failure → 打印 [FAIL] + 诊断信息，计数器 +1
    //   - 抛出 std::exception → 打印 [FAIL] + exception.what()，计数器 +1
    // 最后打印汇总（总数 + 失败数），返回失败数作为进程退出码。
    inline int run_all() {
        int failed = 0;  // 失败计数器
        for (const Case &c: registry()) {
            try {
                c.fn();  // 执行测试函数
                std::printf("[ OK ] %s\n", c.name.c_str());
            } catch (const Failure &f) {
                // 捕获断言失败异常，打印详细诊断
                std::printf("[FAIL] %s\n       %s\n", c.name.c_str(), f.msg.c_str());
                ++failed;
            } catch (const std::exception &e) {
                // 捕获未预期的标准异常（如 std::bad_alloc）
                std::printf("[FAIL] %s\n       exception: %s\n", c.name.c_str(), e.what());
                ++failed;
            }
        }
        // 打印汇总行
        std::printf("%zu tests, %d failed\n", registry().size(), failed);
        return failed;  // 返回失败数，0 表示全部通过
    }

} // namespace tinytest

// =============================================================================
// 宏定义区
// =============================================================================

// ---------------------------------------------------------------------------
// TEST(name) — 定义一个测试用例
// ---------------------------------------------------------------------------
// 展开后生成三样东西：
//   1. static void test_fn_##name();        — 测试函数的前向声明
//   2. static tinytest::Registrar test_reg_##name(...);  — 静态注册器对象
//      在程序启动时自动构造，将 test_fn 注册到全局 registry
//   3. static void test_fn_##name()         — 测试函数的定义起始
//      用户在大括号 {} 内编写测试逻辑
//
// ## 是 C 预处理器的 token pasting 运算符，将 "test_fn_" 和 name 拼接。
// #name 是字符串化运算符，将 name 转为字符串字面量。
// static 限制符号可见性到当前翻译单元，避免不同 .cpp 中同名测试冲突。
#define TEST(name)                                                     \
  static void test_fn_## name();                                        \
  static tinytest::Registrar test_reg_## name(#name, test_fn_## name);   \
  static void test_fn_## name()

// ---------------------------------------------------------------------------
// TQ_FAIL(msg) — 无条件抛出断言失败异常
// ---------------------------------------------------------------------------
// 自动附加 __FILE__ 和 __LINE__，精确定位失败位置。
// 其他 EXPECT_* 宏内部都通过 TQ_FAIL 报告错误。
#define TQ_FAIL(msg) throw tinytest::Failure{std::string(__FILE__) + ":" + \
                                             std::to_string(__LINE__) + ": " + (msg)}

// ---------------------------------------------------------------------------
// EXPECT_TRUE(cond) — 断言条件为真
// ---------------------------------------------------------------------------
// do { ... } while(0) 是经典的"安全宏"惯用法：
//   - 保证宏在任何语句上下文中语法正确（if/else/循环体等）
//   - 不会引入多余的分支或悬空 else 问题
// #cond 将条件表达式字符串化，失败时显示原始代码
#define EXPECT_TRUE(cond)                       \
  do {                                          \
    if (!(cond)) TQ_FAIL("EXPECT_TRUE(" #cond ") failed"); \
  } while (0)

// ---------------------------------------------------------------------------
// EXPECT_EQ(a, b) — 断言两个值相等
// ---------------------------------------------------------------------------
// 技巧：用 auto va/vb 各求值一次，避免 a/b 有副作用时被多次求值。
// 失败时用 display() 将双方值转为字符串，方便诊断。
#define EXPECT_EQ(a, b)                                            \
  do {                                                             \
    auto va = (a);                                                 \
    auto vb = (b);                                                 \
    if (!(va == vb)) {                                             \
      TQ_FAIL("EXPECT_EQ(" #a ", " #b ") failed: " + tinytest::display(va) + \
              " vs " + tinytest::display(vb));                     \
    }                                                              \
  } while (0)

// ---------------------------------------------------------------------------
// EXPECT_NEAR(a, b, tol) — 断言两个浮点值在容差范围内
// ---------------------------------------------------------------------------
// 将双方都提升为 double 再比较，避免 float 精度问题。
// 使用 std::fabs 计算绝对差值，与 tol 比较。
// 失败时打印双方值和容差，便于判断是否需要调整 tol。
#define EXPECT_NEAR(a, b, tol)                                              \
  do {                                                                      \
    double va = static_cast<double>(a);                                     \
    double vb = static_cast<double>(b);                                     \
    if (std::fabs(va - vb) > (tol)) {                                       \
      TQ_FAIL("EXPECT_NEAR(" #a ", " #b ") failed: " + std::to_string(va) + \
              " vs " + std::to_string(vb) + " (tol " + std::to_string(tol) + ")"); \
    }                                                                       \
  } while (0)
