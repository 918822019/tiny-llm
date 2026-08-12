#include "test_framework.h"

#include <cstdio>
#include <fstream>
#include <string>

#include "config.h"

using namespace tinyqwen;

namespace {
    // 把一段文本写成临时配置文件，返回路径。
    std::string write_tmp(const std::string &name, const std::string &content) {
        std::ofstream f(name);
        EXPECT_TRUE(f.good());
        f << content;
        f.close();
        return name;
    }
} // namespace

TEST (config_basic_kv) {
    const std::string path = write_tmp(
        "tinyqwen_test.conf",
        "# 注释行\n"
        "\n"
        "matvec_impl = ref\n"
        "  num_threads = 4  # 行内注释\n"
        "empty_value =\n");

    Config cfg;
    std::string err;
    EXPECT_TRUE(cfg.load(path, &err));
    EXPECT_TRUE(err.empty());
    EXPECT_EQ(cfg.size(), (size_t) 3);
    EXPECT_EQ(cfg.get("matvec_impl", "?"), std::string("ref"));
    EXPECT_EQ(cfg.get("num_threads", "?"), std::string("4")); // 注释被剥掉、值被 trim
    EXPECT_EQ(cfg.get("empty_value", "fallback"), std::string("")); // 空值就是空串
    EXPECT_EQ(cfg.get("missing_key", "fallback"), std::string("fallback"));
    EXPECT_TRUE(cfg.has("matvec_impl"));
    EXPECT_TRUE(!cfg.has("missing_key"));
    std::remove(path.c_str());
}

TEST (config_rejects_malformed_line) {
    const std::string path = write_tmp("tinyqwen_test_bad.conf", "good = 1\nno_equals_here\n");
    Config cfg;
    std::string err;
    EXPECT_TRUE(!cfg.load(path, &err));
    EXPECT_TRUE(err.find("expected 'key = value'") != std::string::npos);
    std::remove(path.c_str());
}

TEST (config_missing_file_fails) {
    Config cfg;
    std::string err;
    EXPECT_TRUE(!cfg.load("does_not_exist.conf", &err));
    EXPECT_TRUE(!err.empty());
}
