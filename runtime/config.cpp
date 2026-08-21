// ============================================================================
// config.cpp — 配置文件解析器
// ============================================================================
// 本文件实现 Config 类，用于从文本文件中读取 key=value 格式的配置项。
// 配置文件格式极为简单，不引入第三方库依赖：
//   - 每行一个 key = value 对
//   - # 开头为行内注释，从 # 到行尾被忽略
//   - 空行和纯注释行被跳过
//   - 键和值两端的空白字符会被自动去除
// 典型配置文件内容示例：
//   matvec_impl = neon_mt_kv_nt
//   ops_impl = neon
//   fuse_gate_up = true  # 启用 gate/up 投影融合
// ============================================================================

#include "config.h"

#include <cstdio>    // 标准输入输出
#include <fstream>   // 文件流（std::ifstream）

namespace tinyqwen {
    namespace {
        // =====================================================================
        // trim() — 去除字符串首尾空白字符
        // =====================================================================
        // 参数：
        //   s — 原始字符串
        // 返回值：去除首尾空白（空格、制表符、回车、换行）后的字符串
        // 说明：使用 find_first_not_of 和 find_last_not_of 定位非空白字符边界，
        //       然后取子串。全空白字符串返回空串。
        std::string trim(const std::string &s) {
            // 找到第一个非空白字符的位置
            size_t b = s.find_first_not_of(" \t\r\n");
            if (b == std::string::npos) return ""; // 全是空白字符，返回空串
            // 找到最后一个非空白字符的位置
            size_t e = s.find_last_not_of(" \t\r\n");
            return s.substr(b, e - b + 1); // 截取 [b, e] 区间
        }
    } // namespace

    // =========================================================================
    // Config::load() — 加载配置文件
    // =========================================================================
    // 参数：
    //   path — 配置文件路径
    //   err  — 输出参数，出错时写入错误信息
    // 返回值：成功返回 true，失败返回 false
    // 说明：逐行读取文件，解析 key=value 对，存入内部 map。
    //       重复的 key 会被后出现的值覆盖（后值优先）。
    bool Config::load(const std::string &path, std::string *err) {
        // 打开文件流
        std::ifstream f(path);
        if (!f) {
            if (err) *err = "cannot open config file: " + path;
            return false;
        }

        kv_.clear(); // 清空已有配置项
        std::string raw;
        int lineno = 0; // 行号计数器，用于错误定位
        // 逐行读取
        while (std::getline(f, raw)) {
            ++lineno;

            // 去掉行内注释：从第一个 '#' 截断
            size_t hash = raw.find('#');
            if (hash != std::string::npos) raw = raw.substr(0, hash);

            // 去除首尾空白
            std::string line = trim(raw);
            if (line.empty()) continue; // 空行 / 纯注释行：跳过

            // 查找等号分隔符
            size_t eq = line.find('=');
            if (eq == std::string::npos) {
                // 没有等号：格式错误
                if (err) *err = path + ":" + std::to_string(lineno) + ": expected 'key = value'";
                return false;
            }
            // 提取 key 和 value，分别去除空白
            std::string key = trim(line.substr(0, eq));
            std::string value = trim(line.substr(eq + 1));
            if (key.empty()) {
                if (err) *err = path + ":" + std::to_string(lineno) + ": empty key";
                return false;
            }
            kv_[key] = value; // 存入 map（重复 key 时后值覆盖前值）
        }
        return true;
    }

    // =========================================================================
    // Config::get() — 获取配置项的值
    // =========================================================================
    // 参数：
    //   key           — 配置项名称
    //   default_value — 当 key 不存在时返回的默认值
    // 返回值：配置项的值（如果存在），否则返回 default_value
    std::string Config::get(const std::string &key, const std::string &default_value) const {
        auto it = kv_.find(key);
        return it == kv_.end() ? default_value : it->second;
    }

    // =========================================================================
    // Config::has() — 检查配置项是否存在
    // =========================================================================
    // 参数：
    //   key — 配置项名称
    // 返回值：存在返回 true，否则返回 false
    bool Config::has(const std::string &key) const { return kv_.count(key) != 0; }
} // namespace tinyqwen