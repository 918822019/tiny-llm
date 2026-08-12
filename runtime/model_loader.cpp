// .tqwen 文件加载器。
//
// 先读 docs/infra_primer.md 第 5、9 节。本文件的核心思想是 **fail fast**：
// 在"读文件"这个边界上把一切能检查的都检查掉，任何不一致立刻报错退出，
// 绝不让坏数据悄悄流进后面的计算。权重文件约 2GB，读错一个偏移的代价
// 是"结果看起来正常但全错"，这种 bug 几乎无法排查，所以宁可在这里严格。
//
// 流程：整个文件读进内存 -> 校验文件头 -> 校验 tensor 表 -> 建立索引。

#include "model_loader.h"

#include <cerrno>
#include <cstdio>
#include <cstring>

namespace tinyqwen {

namespace {

void fail(std::string* err, const std::string& msg) {
  if (err) *err = msg;
}

// 一次性把整个文件读进内存。
// 为什么整读而不是按需读？v1 追求简单：拿到全部字节后，后面所有校验和
// 取数都只是"内存里算偏移"，不再碰文件系统。（mmap 是更省的方案，见
// known_limitations，v1 不做。）
bool read_entire_file(const std::string& path, std::vector<uint8_t>* out, std::string* err) {
  FILE* f = std::fopen(path.c_str(), "rb");  // 二进制只读打开
  if (!f) {
    fail(err, "cannot open file: " + path + " (" + std::strerror(errno) + ")");
    return false;
  }
  // 先跳到文件末尾量出总长度，再跳回开头准备读。
  std::fseek(f, 0, SEEK_END);
  long size = std::ftell(f);
  std::fseek(f, 0, SEEK_SET);
  if (size < 0) {
    std::fclose(f);
    fail(err, "ftell failed: " + path);
    return false;
  }
  out->resize(static_cast<size_t>(size));  // 一次性开好这么大的内存
  size_t got = out->empty() ? 0 : std::fread(out->data(), 1, out->size(), f);
  std::fclose(f);
  if (got != out->size()) {  // 实际读到的 < 预期：文件被截断/损坏
    fail(err, "short read: " + path);
    return false;
  }
  return true;
}

// 从 TensorEntry 里取出名字字符串。
// 注意坑：name 字段是"定长 64 字节、用 NUL 补齐"，但名字恰好占满 64 字符时
// 就没有结尾的 '\0'，不能当普通 C 字符串用 strlen，必须带长度上限地扫描。
std::string entry_name(const TensorEntry& e) {
  size_t len = 0;
  while (len < kMaxTensorName && e.name[len] != '\0') ++len;
  return std::string(e.name, len);
}

std::string shape_str(const TensorView& t) {
  std::string s = "[";
  for (int i = 0; i < t.ndim; ++i) {
    if (i) s += ", ";
    s += std::to_string(t.shape[i]);
  }
  s += "]";
  return s;
}

}  // namespace

bool ModelFile::load(const std::string& path, std::string* err) {
  data_.clear();
  tensors_.clear();
  order_.clear();
  header_ = TinyHeader{};
  config_ = ModelConfig{};

  if (!read_entire_file(path, &data_, err)) return false;

  // ---- 阶段 1：校验文件头（这是不是一个合法、完整、我们认识的文件）----

  if (data_.size() < sizeof(TinyHeader)) {
    fail(err, "file too small for header: " + path);
    return false;
  }
  // 把前 192 字节按 TinyHeader 的布局"看"成一个结构体。
  std::memcpy(&header_, data_.data(), sizeof(TinyHeader));

  // 魔数不对 => 根本不是我们的文件（可能传错了文件）。
  if (std::memcmp(header_.magic, kMagic, sizeof(kMagic)) != 0) {
    fail(err, "bad magic (not a .tqwen file): " + path);
    return false;
  }
  if (header_.version != kFormatVersion) {
    fail(err, "unsupported format version: " + std::to_string(header_.version));
    return false;
  }
  if (header_.dtype != static_cast<uint32_t>(Dtype::kF32)) {
    fail(err, "v1 loader only supports dtype=f32, got " + std::to_string(header_.dtype));
    return false;
  }
  // 头里记录的文件大小必须和磁盘上真实大小一致，否则文件被截断了。
  if (header_.total_bytes != data_.size()) {
    fail(err, "total_bytes mismatch: header=" + std::to_string(header_.total_bytes) +
                  " actual=" + std::to_string(data_.size()));
    return false;
  }
  if (header_.tensor_table_offset != sizeof(TinyHeader)) {
    fail(err, "tensor_table_offset must be 192 in v1");
    return false;
  }
  if (header_.tensor_count == 0 || header_.tensor_count > 100000) {
    fail(err, "implausible tensor_count: " + std::to_string(header_.tensor_count));
    return false;
  }
  // tensor 表不能越出文件末尾。
  const uint64_t table_bytes = header_.tensor_count * sizeof(TensorEntry);
  if (header_.tensor_table_offset + table_bytes > data_.size()) {
    fail(err, "tensor table runs past end of file");
    return false;
  }
  if (header_.data_offset % kAlignment != 0 || header_.data_offset < sizeof(TinyHeader)) {
    fail(err, "bad data_offset: " + std::to_string(header_.data_offset));
    return false;
  }

  // 模型配置本身的合理性：这些值接下来要拿来建模，必须能用。
  const TinyHeader& h = header_;
  if (h.n_layers == 0 || h.hidden_size == 0 || h.intermediate_size == 0 ||
      h.n_heads == 0 || h.n_kv_heads == 0 || h.head_dim == 0 || h.vocab_size == 0 ||
      h.max_seq_len == 0) {
    fail(err, "model config contains zero fields");
    return false;
  }
  if (h.n_heads % h.n_kv_heads != 0) {  // GQA 要求 q 头数能被 kv 头数整除
    fail(err, "n_heads % n_kv_heads != 0");
    return false;
  }
  if (h.rms_norm_eps <= 0.0f || h.rope_theta <= 0.0f) {
    fail(err, "rms_norm_eps / rope_theta must be positive");
    return false;
  }

  // ---- 阶段 2：逐条校验 tensor 表，并建立"名字 -> 视图"索引 ----
  for (uint64_t i = 0; i < header_.tensor_count; ++i) {
    TensorEntry e;
    std::memcpy(&e, data_.data() + header_.tensor_table_offset + i * sizeof(TensorEntry),
                sizeof(TensorEntry));

    if (e.ndim < 1 || e.ndim > 4) {
      fail(err, "tensor #" + std::to_string(i) + ": bad ndim " + std::to_string(e.ndim));
      return false;
    }
    if (e.dtype != static_cast<uint32_t>(Dtype::kF32)) {
      fail(err, "tensor #" + std::to_string(i) + ": v1 supports f32 payloads only");
      return false;
    }
    // 元素个数 = 各维相乘；任何一维为 0 都是非法的。
    uint64_t numel = 1;
    for (uint32_t d = 0; d < e.ndim; ++d) {
      if (e.shape[d] == 0) {
        fail(err, "tensor #" + std::to_string(i) + ": zero in shape");
        return false;
      }
      numel *= e.shape[d];
    }
    // 用不到的 shape 尾部必须清 0（防止垃圾值）。
    for (uint32_t d = e.ndim; d < 4; ++d) {
      if (e.shape[d] != 0) {
        fail(err, "tensor #" + std::to_string(i) + ": trailing shape must be 0");
        return false;
      }
    }
    // 声明的字节数必须等于 元素个数 * 4（f32 每个 4 字节）。
    if (e.nbytes != numel * dtype_size(Dtype::kF32)) {
      fail(err, "tensor #" + std::to_string(i) + ": nbytes mismatch");
      return false;
    }
    // 数据偏移必须 64B 对齐、且落在数据区内。
    if (e.offset % kAlignment != 0 || e.offset < header_.data_offset) {
      fail(err, "tensor #" + std::to_string(i) + ": unaligned / bad offset");
      return false;
    }
    // 数据不能越出文件末尾。
    if (e.offset + e.nbytes > data_.size()) {
      fail(err, "tensor #" + std::to_string(i) + ": payload runs past end of file");
      return false;
    }

    std::string name = entry_name(e);
    if (name.empty()) {
      fail(err, "tensor #" + std::to_string(i) + ": empty name");
      return false;
    }
    if (tensors_.count(name)) {
      fail(err, "duplicate tensor name: " + name);
      return false;
    }

    // 这一条合法：构造一个指向文件内存的视图并存进索引。
    // 注意 data 指针 = 文件内存起点 + 该 tensor 的偏移，零拷贝。
    TensorView view;
    view.name = name;
    view.dtype = static_cast<Dtype>(e.dtype);
    view.ndim = static_cast<int>(e.ndim);
    for (int d = 0; d < 4; ++d) view.shape[d] = e.shape[d];
    view.data = data_.data() + e.offset;
    view.nbytes = e.nbytes;
    tensors_.emplace(name, std::move(view));
    order_.push_back(std::move(name));
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
  config_.tied_embeddings = h.tied_embeddings != 0;
  return true;
}

const TensorView* ModelFile::get(const std::string& name) const {
  auto it = tensors_.find(name);
  return it == tensors_.end() ? nullptr : &it->second;
}

void ModelFile::print_summary() const {
  const ModelConfig& c = config_;
  std::printf("model: layers=%u hidden=%u inter=%u heads=%u kv_heads=%u head_dim=%u "
              "vocab=%u max_seq_len=%u tied=%d\n",
              c.n_layers, c.hidden_size, c.intermediate_size, c.n_heads, c.n_kv_heads,
              c.head_dim, c.vocab_size, c.max_seq_len, (int)c.tied_embeddings);
  std::printf("rms_norm_eps=%g rope_theta=%g\n", c.rms_norm_eps, c.rope_theta);
  std::printf("%-56s %-18s %-5s %12s %14s\n", "name", "shape", "dtype", "offset", "nbytes");
  for (const std::string& name : order_) {
    const TensorView& t = tensors_.at(name);
    // TensorView 没单独存 offset，这里用"指针 - 文件起点"反推出来。
    uint64_t offset = static_cast<uint64_t>(t.data - data_.data());
    std::printf("%-56s %-18s %-5s %12llu %14llu\n", name.c_str(), shape_str(t).c_str(),
                dtype_name(t.dtype), (unsigned long long)offset,
                (unsigned long long)t.nbytes);
  }
  std::printf("file size: %llu bytes (%.2f MB)\n", (unsigned long long)data_.size(),
              (double)data_.size() / (1024.0 * 1024.0));
}

}  // namespace tinyqwen
