#include "config.h"

#include <cstdio>
#include <fstream>

namespace tinyqwen {
    namespace {
        // 去掉首尾空白（空格 / \t / \r / \n）。
        std::string trim(const std::string &s) {
            size_t b = s.find_first_not_of(" \t\r\n");
            if (b == std::string::npos) return "";
            size_t e = s.find_last_not_of(" \t\r\n");
            return s.substr(b, e - b + 1);
        }
    } // namespace

    bool Config::load(const std::string &path, std::string *err) {
        std::ifstream f(path);
        if (!f) {
            if (err) *err = "cannot open config file: " + path;
            return false;
        }

        kv_.clear();
        std::string raw;
        int lineno = 0;
        while (std::getline(f, raw)) {
            ++lineno;

            // 去掉行内注释：从第一个 '#' 截断。
            size_t hash = raw.find('#');
            if (hash != std::string::npos) raw = raw.substr(0, hash);

            std::string line = trim(raw);
            if (line.empty()) continue; // 空行 / 纯注释

            size_t eq = line.find('=');
            if (eq == std::string::npos) {
                if (err) *err = path + ":" + std::to_string(lineno) + ": expected 'key = value'";
                return false;
            }
            std::string key = trim(line.substr(0, eq));
            std::string value = trim(line.substr(eq + 1));
            if (key.empty()) {
                if (err) *err = path + ":" + std::to_string(lineno) + ": empty key";
                return false;
            }
            kv_[key] = value;
        }
        return true;
    }

    std::string Config::get(const std::string &key, const std::string &default_value) const {
        auto it = kv_.find(key);
        return it == kv_.end() ? default_value : it->second;
    }

    bool Config::has(const std::string &key) const { return kv_.count(key) != 0; }
} // namespace tinyqwen
