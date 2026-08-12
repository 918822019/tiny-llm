#pragma once

// 极简配置加载器：读 `key = value` 形式的纯文本配置，`#` 开头是注释。
// 刻意不用 YAML/JSON 库（零依赖）——配置项很少，手写解析最简单可控。
//
// 优先级约定：**CLI 开关 > 配置文件 > 默认值**。
// 这样 benchmark 时可以随时用命令行覆盖配置里的值。

#include <cstddef>
#include <string>
#include <unordered_map>

namespace tinyqwen {
    class Config {
    public:
        // 读取配置文件。失败返回 false 并把原因写进 *err（fail fast）。
        bool load(const std::string &path, std::string *err);

        // 取某个 key；不存在时返回 default_value。
        std::string get(const std::string &key, const std::string &default_value) const;

        bool has(const std::string &key) const;

        size_t size() const { return kv_.size(); }

    private:
        std::unordered_map<std::string, std::string> kv_;
    };
} // namespace tinyqwen
