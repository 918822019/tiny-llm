// ============================================================================
// test_loader.cpp — 模型加载器（ModelFile）单元测试
// ============================================================================
// 本文件测试 ModelFile 类的模型加载功能，包括：
//   1. 完整读写往返测试（f32 / f16 格式）
//   2. 文件格式错误检测（魔术数错误、文件截断、重复张量名、dtype 不一致）
//   3. v2 格式（Qwen3.5 混合架构）扩展块验证
//   4. 层类型判断（linear vs full attention）和紧凑缓存索引
//
// 测试使用辅助函数 write_file() 按 exporter 相同的规则构造合法的 .tqwen
// 文件，然后测试 ModelFile::load() 的加载、验证和错误处理功能。
// ============================================================================

#include "test_framework.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "model_loader.h"
#include "ref_ops.h" // float_to_half（构造 f16 测试文件用，与 exporter 同款量化）

using namespace tinyqwen;

namespace {
    // =========================================================================
    // T — 辅助结构体：表示一个张量的元数据和数据
    // =========================================================================
    // 用于在测试中描述要写入 .tqwen 文件的张量信息。
    struct T {
        std::string name;               // 张量名称（如 "a.weight"）
        std::vector<uint64_t> shape;    // 形状（如 {2, 3}）
        std::vector<float> data;        // 浮点数据
    };

    // =========================================================================
    // align_up() — 64 字节对齐上取整
    // =========================================================================
    // 参数：
    //   x — 要对齐的值
    // 返回值：不小于 x 且是 kAlignment（64）的倍数的最小值
    // 说明：模型文件中的张量数据区域必须 64 字节对齐，以支持 mmap 和 SIMD 访问。
    uint64_t align_up(uint64_t x) { return (x + kAlignment - 1) / kAlignment * kAlignment; }

    // =========================================================================
    // write_file() — 构造合法的 .tqwen 测试文件
    // =========================================================================
    // 按 exporter 同样的规则写一个合法的 .tqwen 文件，供 loader 测试使用。
    //
    // 参数：
    //   path          — 输出文件路径
    //   tensors       — 要写入的张量列表
    //   corrupt_magic — 是否损坏魔术数（用于测试错误检测）
    //   dtype         — 0=f32（T::data 按 float 原样写）；1=f16（T::data 每个
    //                    float 先经 float_to_half 量化成 2 字节再写，模拟
    //                    export --dtype f16）
    // 说明：文件布局为 [TinyHeader][TensorEntry 表][padding][张量数据区域]，
    //       数据区域 64B 对齐，文件末尾填充到 total_bytes。
    void write_file(const std::string &path, const std::vector<T> &tensors,
                    bool corrupt_magic = false, uint32_t dtype = 0) {
        // 计算 TensorEntry 表的结束位置
        const uint64_t table_end = sizeof(TinyHeader) + tensors.size() * sizeof(TensorEntry);
        // 数据区域的起始偏移（64B 对齐）
        const uint64_t data_offset = align_up(table_end);
        // 每个元素的实际字节数（f32=4, f16=2）
        const size_t elem = dtype == 0 ? 4 : 2;
        // 计算每个张量的文件偏移和总文件大小
        std::vector<uint64_t> offsets;
        uint64_t off = data_offset;
        for (const T &t: tensors) {
            offsets.push_back(off);
            off = align_up(off + t.data.size() * elem);
        }
        const uint64_t total = off;

        // 打开文件以二进制写模式
        FILE *f = std::fopen(path.c_str(), "wb");
        EXPECT_TRUE(f != nullptr);

        // ---- 写入 TinyHeader ----
        TinyHeader h{};
        std::memcpy(h.magic, kMagic, 8);  // 写入魔数
        if (corrupt_magic) h.magic[0] = 'X'; // 故意损坏魔数以测试错误检测
        h.version = kFormatVersion;       // 版本号
        h.dtype = dtype;                  // 数据类型
        h.n_layers = 2;                   // 2 层
        h.hidden_size = 4;                // 隐藏层维度
        h.intermediate_size = 8;          // 中间层维度
        h.n_heads = 2;                    // 注意力头数
        h.n_kv_heads = 1;                 // KV 头数（GQA）
        h.head_dim = 2;                   // 每个头的维度
        h.vocab_size = 16;                // 词表大小
        h.max_seq_len = 32;               // 最大序列长度
        h.tied_embeddings = 1;            // 使用共享嵌入
        h.rms_norm_eps = 1e-6f;           // RMSNorm 的 epsilon
        h.rope_theta = 1e6f;              // RoPE 位置编码的 theta
        h.tensor_count = tensors.size();  // 张量数量
        h.tensor_table_offset = sizeof(TinyHeader); // 张量表偏移
        h.data_offset = data_offset;      // 数据区域偏移
        h.total_bytes = total;            // 文件总大小
        std::fwrite(&h, 1, sizeof(h), f);

        // ---- 写入 TensorEntry 表 ----
        for (size_t i = 0; i < tensors.size(); ++i) {
            TensorEntry e{};
            std::snprintf(e.name, sizeof(e.name), "%s", tensors[i].name.c_str());
            e.dtype = dtype;
            e.ndim = static_cast<uint32_t>(tensors[i].shape.size());
            // 填充 shape 数组（最多 4 维）
            for (int d = 0; d < 4; ++d) {
                e.shape[d] = d < static_cast<int>(e.ndim) ? tensors[i].shape[d] : 0;
            }
            e.offset = offsets[i];                          // 文件偏移
            e.nbytes = tensors[i].data.size() * elem;       // 字节数
            std::fwrite(&e, 1, sizeof(e), f);
        }

        // ---- 写入张量数据 ----
        for (size_t i = 0; i < tensors.size(); ++i) {
            std::fseek(f, static_cast<long>(offsets[i]), SEEK_SET);
            if (dtype == 0) {
                // f32：直接写入 float 数据
                std::fwrite(tensors[i].data.data(), 4, tensors[i].data.size(), f);
            } else {
                // f16：与 exporter 相同的量化路径（float_to_half），
                // 每个 float 转为 2 字节 half
                std::vector<uint16_t> half(tensors[i].data.size());
                for (size_t k = 0; k < half.size(); ++k) {
                    half[k] = float_to_half(tensors[i].data[k]);
                }
                std::fwrite(half.data(), 2, half.size(), f);
            }
        }
        // 保证文件长度恰为 total（中间的 padding 字节保持为 0）
        std::fseek(f, static_cast<long>(total) - 1, SEEK_SET);
        const char z = 0;
        std::fwrite(&z, 1, 1, f);
        std::fclose(f);
    }

    // =========================================================================
    // sample_tensors() — 返回一组示例张量
    // =========================================================================
    // 返回两个张量：
    //   "a.weight" 形状 [2, 3]，数据 [0, 1, 2, 3, 4, 5]
    //   "b"        形状 [4]，   数据 [-1, 0.5, 2.25, 7]
    std::vector<T> sample_tensors() {
        return {
            {"a.weight", {2, 3}, {0, 1, 2, 3, 4, 5}},
            {"b", {4}, {-1, 0.5f, 2.25f, 7}},
        };
    }
} // namespace

// =============================================================================
// loader_round_trip — 完整读写往返测试
// =============================================================================
// 测试写入 .tqwen 文件后再通过 ModelFile 加载回来的完整流程：
//   1. 验证模型配置参数（n_layers, hidden_size, n_heads 等）正确
//   2. 验证张量数量正确
//   3. 验证张量名称、形状、元素数量正确
//   4. 验证数据区域 64 字节对齐
//   5. 验证张量数据值正确
//   6. 验证不存在的张量返回 nullptr
TEST (loader_round_trip) {
    const std::string path = "tinyqwen_test_roundtrip.tqwen";
    write_file(path, sample_tensors());

    ModelFile file;
    std::string err;
    // 加载文件，应成功
    EXPECT_TRUE(file.load(path, &err));
    EXPECT_TRUE(err.empty());

    // 验证模型配置参数
    const ModelConfig &c = file.config();
    EXPECT_EQ(c.n_layers, 2u);
    EXPECT_EQ(c.hidden_size, 4u);
    EXPECT_EQ(c.n_heads, 2u);
    EXPECT_EQ(c.n_kv_heads, 1u);
    EXPECT_TRUE(c.tied_embeddings);
    EXPECT_NEAR(c.rms_norm_eps, 1e-6, 1e-9);
    EXPECT_NEAR(c.rope_theta, 1e6, 1e-3);

    // 验证张量数量
    EXPECT_EQ(file.tensor_count(), (size_t) 2);

    // 验证张量 "a.weight"
    const TensorView *a = file.get("a.weight");
    EXPECT_TRUE(a != nullptr);
    EXPECT_EQ(a->ndim, 2);
    EXPECT_EQ(a->shape[0], (uint64_t) 2);
    EXPECT_EQ(a->shape[1], (uint64_t) 3);
    EXPECT_EQ(a->numel(), (uint64_t) 6);
    // 文件内偏移保证 64B 对齐；指针是否对齐取决于分配基址，
    // 所以这里相对文件 base 检查偏移量是否对齐。
    EXPECT_TRUE(static_cast<uintptr_t>(a->data - file.base()) % kAlignment == 0);
    // 验证数据值：a.weight 的数据是 [0, 1, 2, 3, 4, 5]
    for (int i = 0; i < 6; ++i) EXPECT_NEAR(a->f32()[i], (float) i, 0.0);

    // 验证张量 "b"
    const TensorView *b = file.get("b");
    EXPECT_TRUE(b != nullptr);
    EXPECT_NEAR(b->f32()[3], 7.0, 0.0); // 第 4 个元素应为 7.0

    // 不存在的张量返回 nullptr
    EXPECT_TRUE(file.get("missing") == nullptr);
    std::remove(path.c_str());
}

// =============================================================================
// loader_rejects_bad_magic — 魔术数错误检测测试
// =============================================================================
// 验证当文件魔术数被损坏时，ModelFile::load() 正确返回错误，
// 且错误信息中包含 "magic" 关键字。
TEST (loader_rejects_bad_magic) {
    const std::string path = "tinyqwen_test_badmagic.tqwen";
    write_file(path, sample_tensors(), /*corrupt_magic=*/true);

    ModelFile file;
    std::string err;
    EXPECT_TRUE(!file.load(path, &err));  // 加载应失败
    EXPECT_TRUE(err.find("magic") != std::string::npos); // 错误信息应提到魔术数
    std::remove(path.c_str());
}

// =============================================================================
// loader_rejects_truncated_file — 截断文件错误检测测试
// =============================================================================
// 验证当文件被截断（最后 64 字节丢失）时，ModelFile::load() 正确报错。
// 测试方法：先写一个完整文件，然后复制时去掉最后 64 字节，模拟截断。
TEST (loader_rejects_truncated_file) {
    const std::string good = "tinyqwen_test_good.tqwen";
    const std::string bad = "tinyqwen_test_trunc.tqwen";
    write_file(good, sample_tensors());

    // 复制时砍掉最后 64 字节，模拟截断文件。
    FILE *in = std::fopen(good.c_str(), "rb");
    std::fseek(in, 0, SEEK_END);
    long size = std::ftell(in);
    std::fseek(in, 0, SEEK_SET);
    std::vector<char> buf(static_cast<size_t>(size) - 64); // 少读 64 字节
    size_t got = std::fread(buf.data(), 1, buf.size(), in);
    std::fclose(in);
    EXPECT_EQ(got, buf.size());
    FILE *out = std::fopen(bad.c_str(), "wb");
    std::fwrite(buf.data(), 1, buf.size(), out);
    std::fclose(out);

    ModelFile file;
    std::string err;
    EXPECT_TRUE(!file.load(bad, &err)); // 加载应失败
    EXPECT_TRUE(!err.empty());          // 错误信息非空
    std::remove(good.c_str());
    std::remove(bad.c_str());
}

// =============================================================================
// loader_rejects_duplicate_names — 重复张量名检测测试
// =============================================================================
// 验证当文件中存在重复张量名时，ModelFile::load() 正确报错。
// 错误信息中应包含 "duplicate" 关键字。
TEST (loader_rejects_duplicate_names) {
    const std::string path = "tinyqwen_test_dup.tqwen";
    std::vector<T> tensors = sample_tensors();
    tensors[1].name = "a.weight"; // 制造重名：两个张量都叫 "a.weight"
    write_file(path, tensors);

    ModelFile file;
    std::string err;
    EXPECT_TRUE(!file.load(path, &err));
    EXPECT_TRUE(err.find("duplicate") != std::string::npos);
    std::remove(path.c_str());
}

// =============================================================================
// loader_accepts_f16 — f16 格式文件加载测试
// =============================================================================
// 验证 f16 格式（dtype=1）的模型文件能被正确加载。
// 测试点：
//   1. header.dtype 正确为 kF16
//   2. 张量视图的 dtype 为 kF16
//   3. nbytes 为元素数 * 2（每个 half 2 字节）
//   4. 量化值经 half_to_float 还原后与原始 float 一致
// 注：sample 值 0/1/2/.../7、-1、0.5、2.25 全部可在 half 中精确表示。
TEST (loader_accepts_f16) {
    // f16 文件：header/tensor dtype=1、nbytes=元素×2，加载应成功；
    // 视图按字节给出，量化值经 half_to_float 还原后与导出前一致。
    const std::string path = "tinyqwen_test_f16.tqwen";
    write_file(path, sample_tensors(), false, 1);

    ModelFile file;
    std::string err;
    EXPECT_TRUE(file.load(path, &err));
    EXPECT_TRUE(err.empty());
    // 验证 header 中的 dtype 为 kF16
    EXPECT_EQ(file.header().dtype, static_cast<uint32_t>(Dtype::kF16));

    // 验证张量 "a.weight" 的 f16 数据
    const TensorView *a = file.get("a.weight");
    EXPECT_TRUE(a != nullptr);
    EXPECT_TRUE(a->dtype == Dtype::kF16);
    EXPECT_EQ(a->nbytes, (uint64_t) 6 * 2); // 6 个 half = 12 字节
    const uint8_t *raw = a->data;
    for (int i = 0; i < 6; ++i) {
        uint16_t h;
        std::memcpy(&h, raw + i * 2, sizeof(h)); // 提取第 i 个 half 位
        EXPECT_NEAR(half_to_float(h), (float) i, 0.0); // 还原后应与原始值一致
    }

    // 验证张量 "b" 的 f16 数据
    const TensorView *b = file.get("b");
    EXPECT_TRUE(b != nullptr);
    uint16_t h3;
    std::memcpy(&h3, b->data + 3 * 2, sizeof(h3)); // 第 4 个元素
    EXPECT_NEAR(half_to_float(h3), 7.0, 0.0);
    std::remove(path.c_str());
}

// =============================================================================
// loader_rejects_mixed_dtype — 混合 dtype 检测测试
// =============================================================================
// v1 约定全文件单一 dtype：header 声明 f32 但某个 tensor 条目标注 f16
// 必须被拒绝。测试通过手工修改 tensor 条目的 dtype 字段来模拟不一致。
TEST (loader_rejects_mixed_dtype) {
    // v1 约定全文件单一 dtype：header 声明 f32 但 tensor 标 f16 必须拒绝。
    const std::string path = "tinyqwen_test_mixdtype.tqwen";
    write_file(path, sample_tensors()); // header dtype=0 (f32)

    // 手工把第一个 tensor 条目的 dtype 改成 1 (f16)
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
    EXPECT_TRUE(err.find("dtype") != std::string::npos); // 错误信息应提到 dtype
    std::remove(path.c_str());
}

// ---- v2（Qwen3.5 混合架构）header 扩展块测试 ------------------------------

namespace {
    // =========================================================================
    // patch_v2_ext() — 在已写好的文件上覆写 v2 扩展块
    // =========================================================================
    // 在已写好的文件上覆写 header.reserved 为 v2 扩展块，模拟 qwen3_5 导出。
    // write_file 默认 n_layers=2，这里把 interval 设为 2 保证整除。
    //
    // 参数：
    //   path — 已存在的 .tqwen 文件路径
    // 说明：v2 扩展块包含 Qwen3.5 混合架构特有的参数，如线性注意力头数、
    //       卷积核维度、full attention 间隔等。
    void patch_v2_ext(const std::string &path) {
        TinyHeaderV2Ext ext{};
        ext.model_type = static_cast<uint32_t>(ModelType::kQwen35); // 模型类型：Qwen3.5
        ext.linear_num_qk_heads = 2;       // 线性注意力 QK 头数
        ext.linear_num_v_heads = 2;        // 线性注意力 V 头数
        ext.linear_qk_head_dim = 4;        // 线性注意力 QK 头维度
        ext.linear_v_head_dim = 4;         // 线性注意力 V 头维度
        ext.linear_conv_kernel_dim = 4;    // 线性注意力卷积核维度
        ext.full_attention_interval = 2;   // full attention 间隔：每 2 层插入 1 层
        ext.partial_rotary_factor = 0.25f; // 部分 RoPE 因子
        ext.eos_token_id = 248044;         // EOS token ID
        ext.pad = 0;                       // 填充

        FILE *f = std::fopen(path.c_str(), "r+b");
        EXPECT_TRUE(f != nullptr);
        // 覆写 reserved 字段为 v2 扩展块
        std::fseek(f, static_cast<long>(offsetof(TinyHeader, reserved)), SEEK_SET);
        std::fwrite(&ext, 1, sizeof(ext), f);
        std::fclose(f);
    }
} // namespace

// =============================================================================
// loader_v2_qwen35_ext — v2 Qwen3.5 扩展块测试
// =============================================================================
// 验证 v2 格式（Qwen3.5 混合架构）扩展块的加载和解析：
//   1. 模型类型正确识别为 kQwen35
//   2. 线性注意力参数正确解析
//   3. full attention 间隔正确解析
//   4. 层类型判断函数正确：is_linear_layer() 根据 (layer_idx+1) % interval 判断
//   5. 紧凑缓存索引正确：full/linear 层各自从 0 开始编号
//   6. n_full_layers() 正确计算 full attention 层数
//
// 配置：interval=2, n_layers=2
//   layer0: (0+1)%2=1 != 0 -> linear 层
//   layer1: (1+1)%2=0 == 0 -> full 层
TEST (loader_v2_qwen35_ext) {
    const std::string path = "tinyqwen_test_v2.tqwen";
    write_file(path, sample_tensors()); // version=kFormatVersion(2)，reserved 全 0
    patch_v2_ext(path);                 // 覆写 v2 扩展块

    ModelFile file;
    std::string err;
    EXPECT_TRUE(file.load(path, &err));
    EXPECT_TRUE(err.empty());

    const ModelConfig &c = file.config();
    // 验证模型类型
    EXPECT_TRUE(c.model_type == ModelType::kQwen35);
    // 验证线性注意力参数
    EXPECT_EQ(c.linear_num_qk_heads, 2u);
    EXPECT_EQ(c.linear_num_v_heads, 2u);
    EXPECT_EQ(c.linear_qk_head_dim, 4u);
    EXPECT_EQ(c.linear_v_head_dim, 4u);
    EXPECT_EQ(c.linear_conv_kernel_dim, 4u);
    EXPECT_EQ(c.full_attention_interval, 2u);
    EXPECT_NEAR(c.partial_rotary_factor, 0.25, 1e-9);
    EXPECT_EQ(c.eos_token_id, 248044u);

    // 层类型与紧凑 cache 下标。interval=2，n_layers=2：
    //   layer0 (0+1)%2=1 -> linear；layer1 (1+1)%2=0 -> full。
    EXPECT_TRUE(c.is_linear_layer(0));   // layer 0 是 linear 层
    EXPECT_TRUE(!c.is_linear_layer(1));  // layer 1 是 full 层
    EXPECT_EQ(c.n_full_layers(), 1);     // 1 个 full attention 层
    // 紧凑缓存索引：full 层从 0 开始编号
    EXPECT_EQ(c.full_layer_cache_index(1), 0);   // layer 1 是第 0 个 full 层
    EXPECT_EQ(c.linear_layer_cache_index(0), 0);  // layer 0 是第 0 个 linear 层
    std::remove(path.c_str());
}

// =============================================================================
// loader_v1_defaults_to_qwen2 — v1 默认模型类型测试
// =============================================================================
// 验证不写扩展块（reserved 全 0）的 v2 文件应解释为 Qwen2 模型。
// 这种情况下 is_linear_layer 对所有层都返回 false，n_full_layers 等于总层数。
TEST (loader_v1_defaults_to_qwen2) {
    // 不写扩展块（reserved 全 0）的 v2 文件应解释为 qwen2，且 is_linear_layer 恒 false。
    const std::string path = "tinyqwen_test_v2_qwen2.tqwen";
    write_file(path, sample_tensors()); // reserved 全 0 -> model_type=0

    ModelFile file;
    std::string err;
    EXPECT_TRUE(file.load(path, &err));
    EXPECT_TRUE(err.empty());
    const ModelConfig &c = file.config();
    // 默认模型类型为 kQwen2
    EXPECT_TRUE(c.model_type == ModelType::kQwen2);
    // Qwen2 没有线性注意力层
    EXPECT_TRUE(!c.is_linear_layer(0));
    EXPECT_TRUE(!c.is_linear_layer(1));
    // 所有层都是 full attention 层
    EXPECT_EQ(c.n_full_layers(), (int) c.n_layers);
    std::remove(path.c_str());
}

// =============================================================================
// loader_v2_rejects_bad_interval — 非法 interval 检测测试
// =============================================================================
// 验证 full_attention_interval 不能整除 n_layers 时加载被拒绝。
// interval=3 但 n_layers=2 不整除，应报错。
TEST (loader_v2_rejects_bad_interval) {
    // interval=3 但 n_layers=2 不整除 -> 必须拒绝。
    const std::string path = "tinyqwen_test_v2_badint.tqwen";
    write_file(path, sample_tensors());
    TinyHeaderV2Ext ext{};
    ext.model_type = static_cast<uint32_t>(ModelType::kQwen35);
    ext.linear_num_qk_heads = 2;
    ext.linear_num_v_heads = 2;
    ext.linear_qk_head_dim = 4;
    ext.linear_v_head_dim = 4;
    ext.linear_conv_kernel_dim = 4;
    ext.full_attention_interval = 3; // 2 % 3 != 0，不合法
    ext.partial_rotary_factor = 0.25f;
    FILE *f = std::fopen(path.c_str(), "r+b");
    std::fseek(f, static_cast<long>(offsetof(TinyHeader, reserved)), SEEK_SET);
    std::fwrite(&ext, 1, sizeof(ext), f);
    std::fclose(f);

    ModelFile file;
    std::string err;
    EXPECT_TRUE(!file.load(path, &err));
    EXPECT_TRUE(err.find("interval") != std::string::npos ||
                err.find("divisible") != std::string::npos);
    std::remove(path.c_str());
}
namespace {
    // 构造一个"像 MoE"的文件：路由门 + 共享专家（必须常驻）+ 若干路由专家
    // （可卸载，占大头，撑出明显的内存差）
    std::vector<T> moe_like_tensors() {
        std::vector<T> v;
        v.push_back({"model.layers.0.mlp.gate.weight", {4, 4}, std::vector<float>(16, 1.0f)});
        v.push_back({"model.layers.0.mlp.shared_experts.gate_proj.weight",
                     {8, 4}, std::vector<float>(32, 2.0f)});
        for (int e = 0; e < 4; ++e) {
            const std::string pe = "model.layers.0.mlp.experts." + std::to_string(e) + ".";
            v.push_back({pe + "gate_proj.weight", {64, 8}, std::vector<float>(512, 3.0f + e)});
            v.push_back({pe + "up_proj.weight", {64, 8}, std::vector<float>(512, 4.0f + e)});
        }
        return v;
    }

    // 与 write_file 同款算式，算出每个 tensor 的文件偏移，用于核对 file_offset
    std::vector<uint64_t> expected_offsets(const std::vector<T> &tensors) {
        const uint64_t data_off =
            align_up(sizeof(TinyHeader) + tensors.size() * sizeof(TensorEntry));
        std::vector<uint64_t> offs;
        uint64_t off = data_off;
        for (const T &t: tensors) {
            offs.push_back(off);
            off = align_up(off + t.data.size() * 4);
        }
        return offs;
    }
} // namespace

// =============================================================================
// loader_sparse_offloads_experts — 稀疏加载把路由专家留盘不读
// =============================================================================
// 核心命题：offload_experts=true 时名字含 ".mlp.experts." 的 tensor 一个字节
// 都不进 RAM（data==nullptr），只记 file_offset/nbytes；路由门与共享专家仍常驻。
TEST (loader_sparse_offloads_experts) {
    const std::string path = "/tmp/tq_sparse.tqwen";
    const std::vector<T> tensors = moe_like_tensors();
    write_file(path, tensors);

    ModelFile file;
    std::string err;
    EXPECT_TRUE(file.load(path, &err, true));
    if (!err.empty()) std::printf("err=%s\n", err.c_str());

    // 专家 tensor：留盘，无内存指针，但 offset/nbytes 必须正确
    const std::vector<uint64_t> offs = expected_offsets(tensors);
    size_t n_offloaded = 0;
    for (size_t i = 0; i < tensors.size(); ++i) {
        const TensorView *t = file.get(tensors[i].name);
        EXPECT_TRUE(t != nullptr);
        EXPECT_EQ(t->file_offset, offs[i]);
        EXPECT_EQ(t->nbytes, tensors[i].data.size() * 4);
        const bool is_expert = tensors[i].name.find(".mlp.experts.") != std::string::npos;
        if (is_expert) {
            EXPECT_TRUE(t->data == nullptr);
            ++n_offloaded;
        } else {
            EXPECT_TRUE(t->data != nullptr);
        }
    }
    // 8 个路由专家 tensor（4 专家 × gate/up）被卸载；路由门 + 共享专家常驻
    EXPECT_EQ(n_offloaded, (size_t) 8);
    EXPECT_EQ(file.offloaded_count(), (size_t) 8);
    EXPECT_TRUE(file.offloaded_bytes() > 0);
    // 内存占用必须显著小于文件（专家占大头）
    EXPECT_TRUE(file.resident_bytes() < file.file_bytes());
    EXPECT_TRUE(file.offloaded_bytes() > file.file_bytes() / 2);
    // 常驻 tensor 仍须 64B 对齐（紧凑重排不能丢对齐）
    const TensorView *r = file.get("model.layers.0.mlp.gate.weight");
    EXPECT_TRUE(static_cast<uintptr_t>(r->data - file.base()) % kAlignment == 0);

    std::remove(path.c_str());
}

// =============================================================================
// loader_sparse_resident_bytes_identical — 稀疏与全量读到的字节必须一致
// =============================================================================
// 紧凑重排只是换了落点，内容不能变。逐字节比对两种加载方式的同名 tensor。
TEST (loader_sparse_resident_bytes_identical) {
    const std::string path = "/tmp/tq_sparse_eq.tqwen";
    const std::vector<T> tensors = moe_like_tensors();
    write_file(path, tensors);

    ModelFile full, sparse;
    std::string err;
    EXPECT_TRUE(full.load(path, &err, false));
    EXPECT_TRUE(sparse.load(path, &err, true));

    for (const T &src: tensors) {
        const TensorView *f = full.get(src.name);
        const TensorView *s = sparse.get(src.name);
        EXPECT_TRUE(f != nullptr && s != nullptr);
        EXPECT_EQ(f->nbytes, s->nbytes);
        if (s->data == nullptr) continue;  // 卸载的不比（全量侧才有数据）
        EXPECT_EQ(std::memcmp(f->data, s->data, s->nbytes), 0);
    }
    std::remove(path.c_str());
}

// =============================================================================
// loader_full_keeps_entire_file — 全量路径回归：默认行为不变
// =============================================================================
// offload_experts=false 时必须整文件驻留，没有任何 tensor 被卸载。这是保证
// 非 MoE 模型与 MoE resident 正确性锚点不受影响的回归护栏。
TEST (loader_full_keeps_entire_file) {
    const std::string path = "/tmp/tq_full.tqwen";
    const std::vector<T> tensors = moe_like_tensors();
    write_file(path, tensors);

    ModelFile file;
    std::string err;
    EXPECT_TRUE(file.load(path, &err));  // 默认 offload_experts=false
    EXPECT_EQ(file.resident_bytes(), file.file_bytes());
    EXPECT_EQ(file.offloaded_count(), (size_t) 0);
    EXPECT_EQ(file.offloaded_bytes(), (size_t) 0);
    for (const T &src: tensors) {
        const TensorView *t = file.get(src.name);
        EXPECT_TRUE(t != nullptr && t->data != nullptr);
    }
    std::remove(path.c_str());
}

// =============================================================================
// loader_estimate_matches_actual — 预检估算必须逐字节等于真实 load() 结果
// =============================================================================
// 这是 estimate_resident_bytes() 唯一的硬不变量：内存预算预检靠它决定放行还是
// fail-fast。估算与实际不一致会让预检要么误放行（照样换页）要么误拒绝合法文件。
// 两个口径都要锁死。
TEST (loader_estimate_matches_actual) {
    const std::string path = "/tmp/tq_estimate.tqwen";
    const std::vector<T> tensors = moe_like_tensors();
    write_file(path, tensors);

    uint64_t est_full = 0, est_sparse = 0;
    std::string err;
    EXPECT_TRUE(ModelFile::estimate_resident_bytes(path, false, &est_full, &err));
    EXPECT_TRUE(ModelFile::estimate_resident_bytes(path, true, &est_sparse, &err));

    ModelFile full, sparse;
    EXPECT_TRUE(full.load(path, &err, false));
    EXPECT_TRUE(sparse.load(path, &err, true));

    EXPECT_EQ(est_full, full.resident_bytes());
    EXPECT_EQ(est_sparse, sparse.resident_bytes());

    std::remove(path.c_str());
}

// =============================================================================
// loader_estimate_two_regimes — 全量口径等于文件大小，稀疏口径严格更小
// =============================================================================
// 全量：data_ 就是整个文件，故估算 == file_bytes。
// 稀疏：路由专家留盘，故估算 < file_bytes 且省下的正是专家字节。
TEST (loader_estimate_two_regimes) {
    const std::string path = "/tmp/tq_estimate2.tqwen";
    const std::vector<T> tensors = moe_like_tensors();
    write_file(path, tensors);

    uint64_t est_full = 0, est_sparse = 0;
    std::string err;
    EXPECT_TRUE(ModelFile::estimate_resident_bytes(path, false, &est_full, &err));
    EXPECT_TRUE(ModelFile::estimate_resident_bytes(path, true, &est_sparse, &err));

    ModelFile file;
    EXPECT_TRUE(file.load(path, &err, true));
    EXPECT_EQ(est_full, file.file_bytes());
    EXPECT_TRUE(est_sparse < est_full);
    EXPECT_EQ(est_full - est_sparse, file.offloaded_bytes());

    std::remove(path.c_str());
}

// =============================================================================
// loader_estimate_rejects_bad_file — 预检的校验与 load() 同款
// =============================================================================
// 预检放行却在 load() 失败（或反之）会让 fail-fast 失去意义。非 .tqwen 文件
// 必须在预检阶段就被拒。
TEST (loader_estimate_rejects_bad_file) {
    const std::string path = "/tmp/tq_estimate_bad.bin";
    FILE *f = std::fopen(path.c_str(), "wb");
    EXPECT_TRUE(f != nullptr);
    const std::vector<uint8_t> junk(256, 0xAB);
    std::fwrite(junk.data(), 1, junk.size(), f);
    std::fclose(f);

    uint64_t est = 0;
    std::string err;
    EXPECT_TRUE(!ModelFile::estimate_resident_bytes(path, false, &est, &err));
    EXPECT_TRUE(!err.empty());

    std::remove(path.c_str());
}
