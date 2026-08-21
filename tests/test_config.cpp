// =============================================================================
// test_config.cpp — 配置解析器（Config 类）单元测试
// =============================================================================
// 本文件测试 tinyqwen 配置解析器（Config 类）的核心功能。
// Config 类用于解析 key = value 格式的配置文件，支持：
//   - 注释行（以 # 开头，整行忽略）
//   - 行内注释（# 之后的内容被忽略）
//   - 键值对（key = value，key 和 value 都会做 trim 处理）
//   - 空值（key = 后面没有值，则 value 为空字符串）
//   - 格式错误检测（缺少 = 的行应当报错）
//   - 文件不存在时的错误处理
//
// 测试覆盖：
//   1. config_basic_kv       — 基本键值对解析，包括注释、前后空格、空值
//   2. config_rejects_malformed_line — 格式错误的行应被拒绝
//   3. config_missing_file_fails     — 不存在的文件加载失败
// =============================================================================

#include "test_framework.h"

#include <cstdio>
#include <fstream>
#include <string>

#include "config.h"

using namespace tinyqwen;

namespace {
    // =========================================================================
    // write_tmp() — 辅助函数：将文本内容写入临时文件
    // =========================================================================
    // 用于在测试中动态创建配置文件，测试完成后由调用方负责删除。
    //
    // 参数：
    //   name    - 临时文件名（仅文件名，不含路径）
    //   content - 要写入文件的内容
    // 返回：
    //   临时文件的完整路径（与 name 相同，因为写入当前工作目录）
    std::string write_tmp(const std::string &name, const std::string &content) {
        std::ofstream f(name);  // 创建/打开文件
        EXPECT_TRUE(f.good());  // 确保文件打开成功
        f << content;           // 写入内容
        f.close();              // 关闭文件（立即刷新到磁盘）
        return name;            // 返回路径供后续 Config::load() 使用
    }
} // namespace

// =============================================================================
// config_basic_kv — 基本键值对解析测试
// =============================================================================
// 验证 Config 类能够正确解析包含以下格式的配置文件：
//   - 注释行（# 开头，整行忽略）
//   - 空行（忽略）
//   - 正常键值对（key = value）
//   - 键值前后有空格和行内注释（key = value  # comment）
//   - 值为空（key = 后面没有值）
// 验证项：
//   - load() 返回 true，无错误信息
//   - get() 返回正确的值
//   - 不存在的键返回默认值
//   - has() 正确判断键是否存在
TEST (config_basic_kv) {
    // 构造一个包含各种配置格式的临时文件
    const std::string path = write_tmp(
        "tinyqwen_test.conf",
        "# 注释行\n"              // 该行应被完全忽略
        "\n"                      // 空行应被忽略
        "matvec_impl = ref\n"     // 正常键值对
        "  num_threads = 4  # 行内注释\n"  // 键和值前后有空格，行末有注释
        "empty_value =\n");       // 值为空

    Config cfg;
    std::string err;
    // 加载配置文件，应成功
    EXPECT_TRUE(cfg.load(path, &err));
    EXPECT_TRUE(err.empty());  // 错误信息应为空

    // 验证解析出的键值对数量（3 个有效键值对）
    EXPECT_EQ(cfg.size(), (size_t) 3);

    // 验证具体键值对的值
    EXPECT_EQ(cfg.get("matvec_impl", "?"), std::string("ref"));
    // 注释被剥掉、值被 trim（前后空格应被去除）
    EXPECT_EQ(cfg.get("num_threads", "?"), std::string("4"));
    // 空值就是空串（不是 fallback 值）
    EXPECT_EQ(cfg.get("empty_value", "fallback"), std::string(""));
    // 不存在的键应返回默认值
    EXPECT_EQ(cfg.get("missing_key", "fallback"), std::string("fallback"));

    // 验证 has() 方法
    EXPECT_TRUE(cfg.has("matvec_impl"));   // 存在的键
    EXPECT_TRUE(!cfg.has("missing_key"));  // 不存在的键

    // 清理临时文件
    std::remove(path.c_str());
}

// =============================================================================
// config_rejects_malformed_line — 格式错误行拒绝测试
// =============================================================================
// 验证 Config 类对缺少等号 '=' 的行能够正确报错。
// 错误信息中应包含 "expected 'key = value'" 相关的提示。
TEST (config_rejects_malformed_line) {
    const std::string path = write_tmp("tinyqwen_test_bad.conf", "good = 1\nno_equals_here\n");
    Config cfg;
    std::string err;
    EXPECT_TRUE(!cfg.load(path, &err));  // 加载应失败
    // 错误信息应包含格式提示
    EXPECT_TRUE(err.find("expected 'key = value'") != std::string::npos);
    std::remove(path.c_str());
}

// =============================================================================
// config_missing_file_fails — 文件不存在错误测试
// =============================================================================
// 验证 Config::load() 在文件不存在时返回 false 并给出非空错误信息。
TEST (config_missing_file_fails) {
    Config cfg;
    std::string err;
    EXPECT_TRUE(!cfg.load("does_not_exist.conf", &err));  // 加载应失败
    EXPECT_TRUE(!err.empty());  // 错误信息非空
}