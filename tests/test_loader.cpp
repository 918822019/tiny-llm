#include "test_framework.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "model_loader.h"
#include "ref_ops.h" // float_to_half（构造 f16 测试文件用，与 exporter 同款量化）

using namespace tinyqwen;

namespace {
    struct T {
        std::string name;
        std::vector<uint64_t> shape;
        std::vector<float> data;
    };

    uint64_t align_up(uint64_t x) { return (x + kAlignment - 1) / kAlignment * kAlignment; }

    // 按 exporter 同样的规则写一个合法的 .tqwen 文件，供 loader 测试使用。
    // dtype：0=f32（T::data 按 float 原样写）；1=f16（T::data 的每个 float
    // 先经 float_to_half 量化成 2 字节再写，模拟 export --dtype f16）。
    void write_file(const std::string &path, const std::vector<T> &tensors,
                    bool corrupt_magic = false, uint32_t dtype = 0) {
        const uint64_t table_end = sizeof(TinyHeader) + tensors.size() * sizeof(TensorEntry);
        const uint64_t data_offset = align_up(table_end);
        const size_t elem = dtype == 0 ? 4 : 2;
        std::vector<uint64_t> offsets;
        uint64_t off = data_offset;
        for (const T &t: tensors) {
            offsets.push_back(off);
            off = align_up(off + t.data.size() * elem);
        }
        const uint64_t total = off;

        FILE *f = std::fopen(path.c_str(), "wb");
        EXPECT_TRUE(f != nullptr);

        TinyHeader h{};
        std::memcpy(h.magic, kMagic, 8);
        if (corrupt_magic) h.magic[0] = 'X';
        h.version = kFormatVersion;
        h.dtype = dtype;
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
            e.dtype = dtype;
            e.ndim = static_cast<uint32_t>(tensors[i].shape.size());
            for (int d = 0; d < 4; ++d) {
                e.shape[d] = d < static_cast<int>(e.ndim) ? tensors[i].shape[d] : 0;
            }
            e.offset = offsets[i];
            e.nbytes = tensors[i].data.size() * elem;
            std::fwrite(&e, 1, sizeof(e), f);
        }

        for (size_t i = 0; i < tensors.size(); ++i) {
            std::fseek(f, static_cast<long>(offsets[i]), SEEK_SET);
            if (dtype == 0) {
                std::fwrite(tensors[i].data.data(), 4, tensors[i].data.size(), f);
            } else {
                // f16：与 exporter 相同的量化路径（float_to_half）。
                std::vector<uint16_t> half(tensors[i].data.size());
                for (size_t k = 0; k < half.size(); ++k) {
                    half[k] = float_to_half(tensors[i].data[k]);
                }
                std::fwrite(half.data(), 2, half.size(), f);
            }
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

TEST (loader_accepts_f16) {
    // f16 文件：header/tensor dtype=1、nbytes=元素×2，加载应成功；
    // 视图按字节给出，量化值经 half_to_float 还原后与导出前一致（在
    // half 可表示范围内逐位相等——sample 值 0/1/2/.../7、-1、0.5、2.25 全部可精确表示）。
    const std::string path = "tinyqwen_test_f16.tqwen";
    write_file(path, sample_tensors(), false, 1);

    ModelFile file;
    std::string err;
    EXPECT_TRUE(file.load(path, &err));
    EXPECT_TRUE(err.empty());
    EXPECT_EQ(file.header().dtype, static_cast<uint32_t>(Dtype::kF16));

    const TensorView *a = file.get("a.weight");
    EXPECT_TRUE(a != nullptr);
    EXPECT_TRUE(a->dtype == Dtype::kF16);
    EXPECT_EQ(a->nbytes, (uint64_t) 6 * 2); // 6 个 half
    const uint8_t *raw = a->data;
    for (int i = 0; i < 6; ++i) {
        uint16_t h;
        std::memcpy(&h, raw + i * 2, sizeof(h));
        EXPECT_NEAR(half_to_float(h), (float) i, 0.0);
    }

    const TensorView *b = file.get("b");
    EXPECT_TRUE(b != nullptr);
    uint16_t h3;
    std::memcpy(&h3, b->data + 3 * 2, sizeof(h3));
    EXPECT_NEAR(half_to_float(h3), 7.0, 0.0);
    std::remove(path.c_str());
}

TEST (loader_rejects_mixed_dtype) {
    // v1 约定全文件单一 dtype：header 声明 f32 但 tensor 标 f16 必须拒绝。
    const std::string path = "tinyqwen_test_mixdtype.tqwen";
    write_file(path, sample_tensors()); // header dtype=0

    // 手工把第一个 tensor 条目的 dtype 改成 1。
    FILE *f = std::fopen(path.c_str(), "r+b");
    EXPECT_TRUE(f != nullptr);
    std::fseek(f, static_cast<long>(sizeof(TinyHeader) + offsetof(TensorEntry, dtype)),
               SEEK_SET);
    const uint32_t one = 1;
    std::fwrite(&one, sizeof(one), 1, f);
    std::fclose(f);

    ModelFile file;
    std::string err;
    EXPECT_TRUE(!file.load(path, &err));
    EXPECT_TRUE(err.find("dtype") != std::string::npos);
    std::remove(path.c_str());
}
