// ============================================================================
// model_loader.cpp — .tqwen 模型文件加载器
// ============================================================================
// 本文件实现 ModelFile 类，负责从 .tqwen 格式的二进制文件中加载模型权重。
//
// 先读 docs/infra_primer.md 第 5、9 节。本文件的核心思想是 **fail fast**：
// 在"读文件"这个边界上把一切能检查的都检查掉，任何不一致立刻报错退出，
// 绝不让坏数据悄悄流进后面的计算。权重文件约 2GB，读错一个偏移的代价
// 是"结果看起来正常但全错"，这种 bug 几乎无法排查，所以宁可在这里严格。
//
// 文件格式（.tqwen）：
//   [TinyHeader 192B][TensorEntry 表][padding...][tensor 数据区]
//   - TinyHeader: 魔数、版本、配置参数、tensor 表偏移等信息
//   - TensorEntry: 每个 tensor 的 name/shape/dtype/offset/nbytes
//   - 数据区: 所有 tensor 的实际权重数据，按 64B 对齐
//
// 加载流程：
//   1. 整个文件读进内存（一次性，简单可靠）
//   2. 校验文件头（魔数、版本、dtype、配置合理性）
//   3. 逐条校验 tensor 表（dtype、shape、边界、对齐）
//   4. 解析模型配置（Qwen2.x 或 Qwen3.5 混合架构）
//   5. INT4 量化环境校验（v2 文件）
//   6. 建立"名字 -> 视图"索引（零拷贝，直接指向文件内存）
// ============================================================================

#include "model_loader.h"

#include <cerrno>    // errno（strerror 获取错误描述）
#include <cstdio>    // 标准输入输出（fopen, fread, fseek, ftell, fclose）
#include <cstring>   // 内存操作（memcmp, memcpy）

namespace tinyqwen {
    namespace {
        // =====================================================================
        // fail() — 辅助函数：将错误信息写入输出参数
        // =====================================================================
        // 参数：
        //   err — 输出参数指针，为 nullptr 时不写入
        //   msg — 错误信息字符串
        void fail(std::string *err, const std::string &msg) {
            if (err) *err = msg;
        }

        // =====================================================================
        // read_entire_file() — 一次性将整个文件读入内存
        // =====================================================================
        // 参数：
        //   path — 文件路径
        //   out  — 输出参数，读取的内容存入此 vector<uint8_t>
        //   err  — 输出参数，出错时写入错误信息
        // 返回值：成功返回 true，失败返回 false
        // 说明：为什么整读而不是按需读？v1 追求简单：拿到全部字节后，后面所有
        //       校验和取数都只是"内存里算偏移"，不再碰文件系统。
        //       （mmap 是更省的方案，见 known_limitations，v1 不做。）
        bool read_entire_file(const std::string &path, std::vector<uint8_t> *out, std::string *err) {
            // 以二进制只读方式打开文件
            FILE *f = std::fopen(path.c_str(), "rb");
            if (!f) {
                fail(err, "cannot open file: " + path + " (" + std::strerror(errno) + ")");
                return false;
            }
            // 先跳到文件末尾量出总长度
            std::fseek(f, 0, SEEK_END);
            long size = std::ftell(f);
            // 再跳回开头准备读
            std::fseek(f, 0, SEEK_SET);
            if (size < 0) {
                std::fclose(f);
                fail(err, "ftell failed: " + path);
                return false;
            }
            // 一次性开好这么大的内存
            out->resize(static_cast<size_t>(size));
            // 读整个文件内容
            size_t got = out->empty() ? 0 : std::fread(out->data(), 1, out->size(), f);
            std::fclose(f);
            if (got != out->size()) {
                // 实际读到的 < 预期：文件被截断/损坏
                fail(err, "short read: " + path);
                return false;
            }
            return true;
        }

        // =====================================================================
        // entry_name() — 从 TensorEntry 中取出名字字符串
        // =====================================================================
        // 参数：
        //   e — TensorEntry 结构体
        // 返回值：tensor 名称（std::string）
        // 说明：注意坑：name 字段是"定长 64 字节、用 NUL 补齐"，但名字恰好
        //       占满 64 字符时就没有结尾的 '\0'，不能当普通 C 字符串用 strlen，
        //       必须带长度上限地扫描。
        std::string entry_name(const TensorEntry &e) {
            size_t len = 0;
            // 扫描直到遇到 NUL 字符或达到最大长度
            while (len < kMaxTensorName && e.name[len] != '\0') ++len;
            return std::string(e.name, len); // 用找到的长度构造 string
        }

        // =====================================================================
        // shape_str() — 将张量形状转成可读字符串（如 "[32000, 896]"）
        // =====================================================================
        std::string shape_str(const TensorView &t) {
            std::string s = "[";
            for (int i = 0; i < t.ndim; ++i) {
                if (i) s += ", ";
                s += std::to_string(t.shape[i]);
            }
            s += "]";
            return s;
        }
    } // namespace

    // =========================================================================
    // ModelFile::load() — 加载 .tqwen 模型文件
    // =========================================================================
    // 参数：
    //   path — 模型文件路径
    //   err  — 输出参数，出错时写入错误信息
    // 返回值：成功返回 true，失败返回 false
    // 说明：这是核心加载函数，分为 5 个阶段，每个阶段做严格的校验。
    //       任何校验失败立即返回 false，不给后面的计算留下脏数据。
    bool ModelFile::load(const std::string &path, std::string *err) {
        // 重置内部状态
        data_.clear();
        tensors_.clear();
        order_.clear();
        header_ = TinyHeader{};
        config_ = ModelConfig{};

        // 读取整个文件到内存
        if (!read_entire_file(path, &data_, err)) return false;

        // ---- 阶段 1：校验文件头（这是不是一个合法、完整、我们认识的文件）----

        // 文件必须至少包含一个 TinyHeader（192 字节）
        if (data_.size() < sizeof(TinyHeader)) {
            fail(err, "file too small for header: " + path);
            return false;
        }
        // 把前 192 字节按 TinyHeader 的布局解释成一个结构体
        std::memcpy(&header_, data_.data(), sizeof(TinyHeader));

        // 魔数校验：如果不是 "TQWN" 魔数，根本就不是我们的文件
        if (std::memcmp(header_.magic, kMagic, sizeof(kMagic)) != 0) {
            fail(err, "bad magic (not a .tqwen file): " + path);
            return false;
        }
        // 版本校验：必须在编译时支持的范围内
        if (header_.version < kFormatVersionMin || header_.version > kFormatVersion) {
            fail(err, "unsupported format version: " + std::to_string(header_.version) +
                      " (this build supports " + std::to_string(kFormatVersionMin) + ".." +
                      std::to_string(kFormatVersion) + ")");
            return false;
        }
        // dtype 校验：只支持 f32、f16、i4、vq2 四种
        if (header_.dtype != static_cast<uint32_t>(Dtype::kF32) &&
            header_.dtype != static_cast<uint32_t>(Dtype::kF16) &&
            header_.dtype != static_cast<uint32_t>(Dtype::kI4) &&
            header_.dtype != static_cast<uint32_t>(Dtype::kVQ2)) {
            fail(err, "loader supports dtype f32/f16/i4/vq2, got " + std::to_string(header_.dtype));
            return false;
        }
        // 头里记录的文件大小必须和磁盘上真实大小一致，否则文件被截断了
        if (header_.total_bytes != data_.size()) {
            fail(err, "total_bytes mismatch: header=" + std::to_string(header_.total_bytes) +
                      " actual=" + std::to_string(data_.size()));
            return false;
        }
        // tensor 表偏移必须是 TinyHeader 大小（v1 格式约定）
        if (header_.tensor_table_offset != sizeof(TinyHeader)) {
            fail(err, "tensor_table_offset must be 192 in v1");
            return false;
        }
        // tensor 数量必须在合理范围（0 或超过 10 万都不可信）
        if (header_.tensor_count == 0 || header_.tensor_count > 100000) {
            fail(err, "implausible tensor_count: " + std::to_string(header_.tensor_count));
            return false;
        }
        // tensor 表不能越出文件末尾
        const uint64_t table_bytes = header_.tensor_count * sizeof(TensorEntry);
        if (header_.tensor_table_offset + table_bytes > data_.size()) {
            fail(err, "tensor table runs past end of file");
            return false;
        }
        // 数据偏移必须 64B 对齐，且大于头大小
        if (header_.data_offset % kAlignment != 0 || header_.data_offset < sizeof(TinyHeader)) {
            fail(err, "bad data_offset: " + std::to_string(header_.data_offset));
            return false;
        }

        // 模型配置本身的合理性：这些值接下来要拿来建模，必须能用
        const TinyHeader &h = header_;
        if (h.n_layers == 0 || h.hidden_size == 0 || h.intermediate_size == 0 ||
            h.n_heads == 0 || h.n_kv_heads == 0 || h.head_dim == 0 || h.vocab_size == 0 ||
            h.max_seq_len == 0) {
            fail(err, "model config contains zero fields");
            return false;
        }
        // GQA 要求：Q 头数必须能被 KV 头数整除
        if (h.n_heads % h.n_kv_heads != 0) {
            fail(err, "n_heads % n_kv_heads != 0");
            return false;
        }
        // RMSNorm 和 RoPE 的超参数必须为正
        if (h.rms_norm_eps <= 0.0f || h.rope_theta <= 0.0f) {
            fail(err, "rms_norm_eps / rope_theta must be positive");
            return false;
        }

        // ---- 阶段 2：逐条校验 tensor 表，并建立"名字 -> 视图"索引 ----
        for (uint64_t i = 0; i < header_.tensor_count; ++i) {
            // 从数据区中读取第 i 个 tensor 的条目
            TensorEntry e;
            std::memcpy(&e, data_.data() + header_.tensor_table_offset + i * sizeof(TensorEntry),
                        sizeof(TensorEntry));

            // ndim 必须在 1-4 之间
            if (e.ndim < 1 || e.ndim > 4) {
                fail(err, "tensor #" + std::to_string(i) + ": bad ndim " + std::to_string(e.ndim));
                return false;
            }
            // dtype 校验：kI4 文件允许混合（大矩阵 kI4、小向量 kF32）；
            // f32/f16 文件仍要求全文件单一 dtype
            if (static_cast<Dtype>(header_.dtype) == Dtype::kI4) {
                if (e.dtype != static_cast<uint32_t>(Dtype::kI4) &&
                    e.dtype != static_cast<uint32_t>(Dtype::kF32)) {
                    fail(err, "tensor #" + std::to_string(i) + ": i4 file allows only "
                              "i4/f32 tensors, got dtype " + std::to_string(e.dtype));
                    return false;
                }
            } else if (static_cast<Dtype>(header_.dtype) == Dtype::kVQ2) {
                if (e.dtype != static_cast<uint32_t>(Dtype::kVQ2) &&
                    e.dtype != static_cast<uint32_t>(Dtype::kF32) &&
                    e.dtype != static_cast<uint32_t>(Dtype::kF16)) {
                    fail(err, "tensor #" + std::to_string(i) + ": vq2 file allows only "
                              "vq2/f32/f16 tensors, got dtype " + std::to_string(e.dtype));
                    return false;
                }
            } else {
                if (e.dtype != header_.dtype) {
                    fail(err, "tensor #" + std::to_string(i) + ": dtype " +
                                  std::to_string(e.dtype) + " != header dtype " +
                                  std::to_string(header_.dtype) + " (uniform-dtype)");
                    return false;
                }
            }
            // 元素个数 = 各维相乘；任何一维为 0 都是非法的
            uint64_t numel = 1;
            for (uint32_t d = 0; d < e.ndim; ++d) {
                if (e.shape[d] == 0) {
                    fail(err, "tensor #" + std::to_string(i) + ": zero in shape");
                    return false;
                }
                numel *= e.shape[d];
            }
            // 用不到的 shape 尾部必须清 0（防止垃圾值干扰后续判断）
            for (uint32_t d = e.ndim; d < 4; ++d) {
                if (e.shape[d] != 0) {
                    fail(err, "tensor #" + std::to_string(i) + ": trailing shape must be 0");
                    return false;
                }
            }
            // 声明的字节数校验：INT4/VQ2 用专门公式；f32/f16 用元素数 * 元素字节
            if (static_cast<Dtype>(e.dtype) == Dtype::kI4 ||
                static_cast<Dtype>(e.dtype) == Dtype::kVQ2) {
                if (e.ndim != 2) {
                    fail(err, "tensor #" + std::to_string(i) + ": i4/vq2 tensor must be 2D");
                    return false;
                }
                // nbytes 延迟到阶段 5 用对应公式再验（此处先记录）
            } else {
                if (e.nbytes != numel * dtype_size(static_cast<Dtype>(e.dtype))) {
                    fail(err, "tensor #" + std::to_string(i) + ": nbytes mismatch");
                    return false;
                }
            }
            // 数据偏移必须 64B 对齐、且落在数据区内
            if (e.offset % kAlignment != 0 || e.offset < header_.data_offset) {
                fail(err, "tensor #" + std::to_string(i) + ": unaligned / bad offset");
                return false;
            }
            // 数据不能越出文件末尾
            if (e.offset + e.nbytes > data_.size()) {
                fail(err, "tensor #" + std::to_string(i) + ": payload runs past end of file");
                return false;
            }

            // 提取 tensor 名字
            std::string name = entry_name(e);
            if (name.empty()) {
                fail(err, "tensor #" + std::to_string(i) + ": empty name");
                return false;
            }
            // 不允许重名 tensor
            if (tensors_.count(name)) {
                fail(err, "duplicate tensor name: " + name);
                return false;
            }

            // 这一条合法：构造一个指向文件内存的视图并存进索引
            // 注意 data 指针 = 文件内存起点 + 该 tensor 的偏移，零拷贝
            TensorView view;
            view.name = name;
            view.dtype = static_cast<Dtype>(e.dtype);
            view.ndim = static_cast<int>(e.ndim);
            for (int d = 0; d < 4; ++d) view.shape[d] = e.shape[d];
            view.data = data_.data() + e.offset; // 零拷贝：直接指向文件内存
            view.nbytes = e.nbytes;
            tensors_.emplace(name, std::move(view)); // 存入 map
            order_.push_back(std::move(name));       // 记录插入顺序（用于 print_summary）
        }

        // ---- 阶段 3：把配置导出成 ModelConfig 供建模使用 ----
        config_.n_layers = h.n_layers;
        config_.hidden_size = h.hidden_size;
        config_.intermediate_size = h.intermediate_size;
        config_.n_heads = h.n_heads;
        config_.n_kv_heads = h.n_kv_heads;
        config_.head_dim = h.head_dim;
        config_.vocab_size = h.vocab_size;
        config_.max_seq_len = h.max_seq_len;
        config_.rms_norm_eps = h.rms_norm_eps;
        config_.rope_theta = h.rope_theta;
        config_.tied_embeddings = h.tied_embeddings != 0; // 非零表示词嵌入与 lm_head 共享

        // ---- 阶段 4：v2 扩展头（混合架构字段）。v1 文件跳过，配置保持默认 ----
        if (header_.version >= 2) {
            // 从 TinyHeader 的 reserved 字段中读取 v2 扩展信息
            TinyHeaderV2Ext ext{};
            std::memcpy(&ext, header_.reserved, sizeof(ext));

            // model_type 必须在已知范围内
            if (ext.model_type > static_cast<uint32_t>(ModelType::kQwen35)) {
                fail(err, "unknown model_type: " + std::to_string(ext.model_type));
                return false;
            }
            // pad 字段必须为 0（预留对齐）
            if (ext.pad != 0) {
                fail(err, "v2 ext pad field must be 0");
                return false;
            }

            // 填充 Qwen3.5 混合架构参数
            config_.model_type = static_cast<ModelType>(ext.model_type);
            config_.linear_num_qk_heads = ext.linear_num_qk_heads;
            config_.linear_num_v_heads = ext.linear_num_v_heads;
            config_.linear_qk_head_dim = ext.linear_qk_head_dim;
            config_.linear_v_head_dim = ext.linear_v_head_dim;
            config_.linear_conv_kernel_dim = ext.linear_conv_kernel_dim;
            config_.full_attention_interval = ext.full_attention_interval;
            config_.partial_rotary_factor = ext.partial_rotary_factor;
            config_.eos_token_id = ext.eos_token_id;
            config_.quant_group_size = ext.quant_group_size;

            // 混合架构（qwen3_5）自洽性校验。fail fast：这些值接下来直接用来
            // 建模，错一个就是静默算错。
            if (config_.model_type == ModelType::kQwen35) {
                // linear attention 的形状字段必须全部设置
                if (config_.linear_num_qk_heads == 0 || config_.linear_num_v_heads == 0 ||
                    config_.linear_qk_head_dim == 0 || config_.linear_v_head_dim == 0 ||
                    config_.linear_conv_kernel_dim < 2) {
                    fail(err, "qwen3_5: linear attention shape fields must be set "
                              "(conv kernel >= 2)");
                    return false;
                }
                // full attention 间隔至少为 2
                if (config_.full_attention_interval < 2) {
                    fail(err, "qwen3_5: full_attention_interval must be >= 2");
                    return false;
                }
                // 层数必须能被 full_attention_interval 整除
                if (config_.n_layers % config_.full_attention_interval != 0) {
                    fail(err, "qwen3_5: n_layers must be divisible by "
                              "full_attention_interval");
                    return false;
                }
                // partial_rotary_factor 必须在 (0, 1] 范围内
                if (config_.partial_rotary_factor <= 0.0f ||
                    config_.partial_rotary_factor > 1.0f) {
                    fail(err, "qwen3_5: partial_rotary_factor must be in (0, 1]");
                    return false;
                }
                // GDN 的 query/key 头数要能对齐到 value 头数（repeat_interleave）
                if (config_.linear_num_v_heads % config_.linear_num_qk_heads != 0) {
                    fail(err, "qwen3_5: linear_num_v_heads % linear_num_qk_heads != 0");
                    return false;
                }
            }
        }

        // ---- 阶段 5：INT4 量化校验（quant_group_size + 每个 I4 tensor 的 nbytes）----
        if (static_cast<Dtype>(header_.dtype) == Dtype::kI4) {
            // quant_group_size 必须在 [1, 1024] 且为偶数
            if (config_.quant_group_size == 0 || config_.quant_group_size > 1024) {
                fail(err, "i4 file: quant_group_size must be in [1,1024], got " +
                          std::to_string(config_.quant_group_size));
                return false;
            }
            if (config_.quant_group_size % 2 != 0) {
                fail(err, "i4 file: quant_group_size must be even");
                return false;
            }
            const int gs = static_cast<int>(config_.quant_group_size);
            // 对每个 I4 tensor 校验其 nbytes
            for (const auto &kv : tensors_) {
                const TensorView &t = kv.second;
                if (t.dtype != Dtype::kI4) continue; // 只检查 I4 类型的 tensor
                // I4 行字节数 = 行数 * i4_row_bytes(列数, group_size)
                const uint64_t expected = t.shape[0] * i4_row_bytes(
                    static_cast<int>(t.shape[1]), gs);
                if (t.nbytes != expected) {
                    fail(err, "tensor '" + kv.first + "': i4 nbytes mismatch, expected " +
                              std::to_string(expected) + " got " + std::to_string(t.nbytes));
                    return false;
                }
            }
        }

        // ---- 阶段 5b：VQ2 量化校验（每个 VQ2 tensor 的 nbytes = 码本 + 索引）----
        if (static_cast<Dtype>(header_.dtype) == Dtype::kVQ2) {
            for (const auto &kv : tensors_) {
                const TensorView &t = kv.second;
                if (t.dtype != Dtype::kVQ2) continue; // 只检查 VQ2 类型的 tensor
                if (t.ndim != 2) {
                    fail(err, "tensor '" + kv.first + "': vq2 tensor must be 2D");
                    return false;
                }
                if (t.shape[1] % kVQ2BlockDim != 0) {
                    fail(err, "tensor '" + kv.first + "': vq2 in_dim must be divisible by " +
                              std::to_string(kVQ2BlockDim));
                    return false;
                }
                const uint64_t expected = vq2_tensor_bytes(
                    static_cast<int>(t.shape[0]), static_cast<int>(t.shape[1]));
                if (t.nbytes != expected) {
                    fail(err, "tensor '" + kv.first + "': vq2 nbytes mismatch, expected " +
                              std::to_string(expected) + " got " + std::to_string(t.nbytes));
                    return false;
                }
            }
        }

        return true; // 所有校验通过
    }

    // =========================================================================
    // ModelFile::get() — 按名字查找 tensor
    // =========================================================================
    // 参数：
    //   name — tensor 名称（如 "model.layers.0.self_attn.q_proj.weight"）
    // 返回值：找到返回 TensorView 指针，未找到返回 nullptr
    const TensorView *ModelFile::get(const std::string &name) const {
        auto it = tensors_.find(name);
        return it == tensors_.end() ? nullptr : &it->second;
    }

    // =========================================================================
    // ModelFile::print_summary() — 打印模型结构摘要
    // =========================================================================
    // 说明：打印模型配置信息、所有 tensor 的列表（名称、形状、dtype、偏移、
    //       字节数），以及文件总大小。用于 --verbose 模式下的信息输出。
    void ModelFile::print_summary() const {
        const ModelConfig &c = config_;
        // 打印模型基本配置
        std::printf("model: layers=%u hidden=%u inter=%u heads=%u kv_heads=%u head_dim=%u "
                    "vocab=%u max_seq_len=%u tied=%d\n",
                    c.n_layers, c.hidden_size, c.intermediate_size, c.n_heads, c.n_kv_heads,
                    c.head_dim, c.vocab_size, c.max_seq_len, (int) c.tied_embeddings);
        std::printf("rms_norm_eps=%g rope_theta=%g\n", c.rms_norm_eps, c.rope_theta);
        // Qwen3.5 混合架构额外信息
        if (c.model_type == ModelType::kQwen35) {
            std::printf("qwen3_5 hybrid: attn_interval=%u linear_qk=%ux%u linear_v=%ux%u "
                        "conv=%u rotary=%g eos=%u\n",
                        c.full_attention_interval, c.linear_num_qk_heads, c.linear_qk_head_dim,
                        c.linear_num_v_heads, c.linear_v_head_dim, c.linear_conv_kernel_dim,
                        (double) c.partial_rotary_factor, c.eos_token_id);
        }
        // 打印 tensor 列表表头
        std::printf("%-56s %-18s %-5s %12s %14s\n", "name", "shape", "dtype", "offset", "nbytes");
        for (const std::string &name: order_) {
            const TensorView &t = tensors_.at(name);
            // TensorView 没单独存 offset，这里用"指针 - 文件起点"反推出来
            uint64_t offset = static_cast<uint64_t>(t.data - data_.data());
            std::printf("%-56s %-18s %-5s %12llu %14llu\n", name.c_str(), shape_str(t).c_str(),
                        dtype_name(t.dtype), (unsigned long long) offset,
                        (unsigned long long) t.nbytes);
        }
        // 打印文件总大小
        std::printf("file size: %llu bytes (%.2f MB)\n", (unsigned long long) data_.size(),
                    (double) data_.size() / (1024.0 * 1024.0));
    }
} // namespace tinyqwen