#pragma once

// 最小单文件测试框架：不引第三方依赖。
// 测试用 TEST(name) 自注册；test_main.cpp 统一运行全部用例。

#include <cmath>
#include <cstdio>
#include <functional>
#include <string>
#include <type_traits>
#include <vector>

namespace tinytest {
    // 把断言里的值转成可读字符串：数字走 to_string，字符串原样返回。
    template<typename T>
    inline std::string display(const T &v) {
        if constexpr (std::is_same<T, std::string>::value) {
            return v;
        } else if constexpr (std::is_arithmetic<T>::value) {
            return std::to_string(+v); // +v 把 char/short 提升，避免 to_string 无重载
        } else {
            return std::string(v); // const char* 等
        }
    }

    struct Failure {
        std::string msg;
    };

    struct Case {
        std::string name;
        std::function<void()> fn;
    };

    inline std::vector<Case> &registry() {
        static std::vector<Case> cases;
        return cases;
    }

    struct Registrar {
        Registrar(const std::string &name, std::function<void()> fn) {
            registry().push_back(Case{name, std::move(fn)});
        }
    };

    inline int run_all() {
        int failed = 0;
        for (const Case &c: registry()) {
            try {
                c.fn();
                std::printf("[ OK ] %s\n", c.name.c_str());
            } catch (const Failure &f) {
                std::printf("[FAIL] %s\n       %s\n", c.name.c_str(), f.msg.c_str());
                ++failed;
            } catch (const std::exception &e) {
                std::printf("[FAIL] %s\n       exception: %s\n", c.name.c_str(), e.what());
                ++failed;
            }
        }
        std::printf("%zu tests, %d failed\n", registry().size(), failed);
        return failed;
    }
} // namespace tinytest

#define TEST(name)                                                     \
  static void test_fn_##name();                                        \
  static tinytest::Registrar test_reg_##name(#name, test_fn_##name);   \
  static void test_fn_##name()

#define TQ_FAIL(msg) throw tinytest::Failure{std::string(__FILE__) + ":" + \
                                             std::to_string(__LINE__) + ": " + (msg)}

#define EXPECT_TRUE(cond)                       \
  do {                                          \
    if (!(cond)) TQ_FAIL("EXPECT_TRUE(" #cond ") failed"); \
  } while (0)

#define EXPECT_EQ(a, b)                                            \
  do {                                                             \
    auto va = (a);                                                 \
    auto vb = (b);                                                 \
    if (!(va == vb)) {                                             \
      TQ_FAIL("EXPECT_EQ(" #a ", " #b ") failed: " + tinytest::display(va) + \
              " vs " + tinytest::display(vb));                     \
    }                                                              \
  } while (0)

#define EXPECT_NEAR(a, b, tol)                                              \
  do {                                                                      \
    double va = static_cast<double>(a);                                     \
    double vb = static_cast<double>(b);                                     \
    if (std::fabs(va - vb) > (tol)) {                                       \
      TQ_FAIL("EXPECT_NEAR(" #a ", " #b ") failed: " + std::to_string(va) + \
              " vs " + std::to_string(vb) + " (tol " + std::to_string(tol) + ")"); \
    }                                                                       \
  } while (0)
