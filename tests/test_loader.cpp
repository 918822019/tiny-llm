#include "test_framework.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "model_loader.h"

using namespace tinyqwen;

namespace {
    struct T {
        std::string name;
        std::vector<uint64_t> shape;
        std::vector<float> data;
    };

    uint64_t align_up(uint64_t x) { return (x + kAlignment - 1) / kAlignment * kAlignment; }

    // 按 exporter 同样的规则写一个合法的 .tqwen 文件，供 loader 测试使用。
    void write_file(const std::string &path, const std::vector<T> &tensors,
                    bool corrupt_magic = false) {
        const uint64_t table_end = sizeof(TinyHeader) + tensors.size() * sizeof(TensorEntry);
        const uint64_t data_offset = align_up(table_end);
        std::vector<uint64_t> offsets;
        uint64_t off = data_offset;
        for (const T &t: tensors) {
            offsets.push_back(off);
            off = align_up(off + t.data.size() * 4);
        }
        const uint64_t total = off;

        FILE *f = std::fopen(path.c_str(), "wb");
        EXPECT_TRUE(f != nullptr);

        TinyHeader h{};
        std::memcpy(h.magic, kMagic, 8);
        if (corrupt_magic) h.magic[0] = 'X';
        h.version = kFormatVersion;
        h.dtype = 0;
        h.n_layers = 2;
        h.hidden_size = 4;
        h.intermediate_size = 8;
        h.n_heads = 2;
        h.n_kv_heads = 1;
        h.head_dim = 2;
        h.vocab_size = 16;
        h.max_seq_len = 32;
        h.tied_embeddings = 1;
        h.rms_norm_eps = 1e-6f;
        h.rope_theta = 1e6f;
        h.tensor_count = tensors.size();
        h.tensor_table_offset = sizeof(TinyHeader);
        h.data_offset = data_offset;
        h.total_bytes = total;
        std::fwrite(&h, 1, sizeof(h), f);

        for (size_t i = 0; i < tensors.size(); ++i) {
            TensorEntry e{};
            std::snprintf(e.name, sizeof(e.name), "%s", tensors[i].name.c_str());
            e.dtype = 0;
            e.ndim = static_cast<uint32_t>(tensors[i].shape.size());
            for (int d = 0; d < 4; ++d) {
                e.shape[d] = d < static_cast<int>(e.ndim) ? tensors[i].shape[d] : 0;
            }
            e.offset = offsets[i];
            e.nbytes = tensors[i].data.size() * 4;
            std::fwrite(&e, 1, sizeof(e), f);
        }

        for (size_t i = 0; i < tensors.size(); ++i) {
            std::fseek(f, static_cast<long>(offsets[i]), SEEK_SET);
            std::fwrite(tensors[i].data.data(), 4, tensors[i].data.size(), f);
        }
        // 保证文件长度恰为 total（中间的 padding 字节保持为 0）。
        std::fseek(f, static_cast<long>(total) - 1, SEEK_SET);
        const char z = 0;
        std::fwrite(&z, 1, 1, f);
        std::fclose(f);
    }

    std::vector<T> sample_tensors() {
        return {
            {"a.weight", {2, 3}, {0, 1, 2, 3, 4, 5}},
            {"b", {4}, {-1, 0.5f, 2.25f, 7}},
        };
    }
} // namespace

TEST (loader_round_trip) {
    const std::string path = "tinyqwen_test_roundtrip.tqwen";
    write_file(path, sample_tensors());

    ModelFile file;
    std::string err;
    EXPECT_TRUE(file.load(path, &err));
    EXPECT_TRUE(err.empty());

    const ModelConfig &c = file.config();
    EXPECT_EQ(c.n_layers, 2u);
    EXPECT_EQ(c.hidden_size, 4u);
    EXPECT_EQ(c.n_heads, 2u);
    EXPECT_EQ(c.n_kv_heads, 1u);
    EXPECT_TRUE(c.tied_embeddings);
    EXPECT_NEAR(c.rms_norm_eps, 1e-6, 1e-9);
    EXPECT_NEAR(c.rope_theta, 1e6, 1e-3);

    EXPECT_EQ(file.tensor_count(), (size_t) 2);

    const TensorView *a = file.get("a.weight");
    EXPECT_TRUE(a != nullptr);
    EXPECT_EQ(a->ndim, 2);
    EXPECT_EQ(a->shape[0], (uint64_t) 2);
    EXPECT_EQ(a->shape[1], (uint64_t) 3);
    EXPECT_EQ(a->numel(), (uint64_t) 6);
    // 文件内偏移保证 64B 对齐；指针是否对齐取决于分配基址，
    // 所以这里相对文件 base 检查。
    EXPECT_TRUE(static_cast<uintptr_t>(a->data - file.base()) % kAlignment == 0);
    for (int i = 0; i < 6; ++i) EXPECT_NEAR(a->f32()[i], (float) i, 0.0);

    const TensorView *b = file.get("b");
    EXPECT_TRUE(b != nullptr);
    EXPECT_NEAR(b->f32()[3], 7.0, 0.0);

    EXPECT_TRUE(file.get("missing") == nullptr);
    std::remove(path.c_str());
}

TEST (loader_rejects_bad_magic) {
    const std::string path = "tinyqwen_test_badmagic.tqwen";
    write_file(path, sample_tensors(), /*corrupt_magic=*/true);

    ModelFile file;
    std::string err;
    EXPECT_TRUE(!file.load(path, &err));
    EXPECT_TRUE(err.find("magic") != std::string::npos);
    std::remove(path.c_str());
}

TEST (loader_rejects_truncated_file) {
    const std::string good = "tinyqwen_test_good.tqwen";
    const std::string bad = "tinyqwen_test_trunc.tqwen";
    write_file(good, sample_tensors());

    // 复制时砍掉最后 64 字节，模拟截断文件。
    FILE *in = std::fopen(good.c_str(), "rb");
    std::fseek(in, 0, SEEK_END);
    long size = std::ftell(in);
    std::fseek(in, 0, SEEK_SET);
    std::vector<char> buf(static_cast<size_t>(size) - 64);
    size_t got = std::fread(buf.data(), 1, buf.size(), in);
    std::fclose(in);
    EXPECT_EQ(got, buf.size());
    FILE *out = std::fopen(bad.c_str(), "wb");
    std::fwrite(buf.data(), 1, buf.size(), out);
    std::fclose(out);

    ModelFile file;
    std::string err;
    EXPECT_TRUE(!file.load(bad, &err));
    EXPECT_TRUE(!err.empty());
    std::remove(good.c_str());
    std::remove(bad.c_str());
}

TEST (loader_rejects_duplicate_names) {
    const std::string path = "tinyqwen_test_dup.tqwen";
    std::vector<T> tensors = sample_tensors();
    tensors[1].name = "a.weight"; // 制造重名
    write_file(path, tensors);

    ModelFile file;
    std::string err;
    EXPECT_TRUE(!file.load(path, &err));
    EXPECT_TRUE(err.find("duplicate") != std::string::npos);
    std::remove(path.c_str());
}
