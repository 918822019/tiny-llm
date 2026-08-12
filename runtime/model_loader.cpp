#include "model_loader.h"

#include <cerrno>
#include <cstdio>
#include <cstring>

namespace tinyqwen {

namespace {

void fail(std::string* err, const std::string& msg) {
  if (err) *err = msg;
}

bool read_entire_file(const std::string& path, std::vector<uint8_t>* out, std::string* err) {
  FILE* f = std::fopen(path.c_str(), "rb");
  if (!f) {
    fail(err, "cannot open file: " + path + " (" + std::strerror(errno) + ")");
    return false;
  }
  std::fseek(f, 0, SEEK_END);
  long size = std::ftell(f);
  std::fseek(f, 0, SEEK_SET);
  if (size < 0) {
    std::fclose(f);
    fail(err, "ftell failed: " + path);
    return false;
  }
  out->resize(static_cast<size_t>(size));
  size_t got = out->empty() ? 0 : std::fread(out->data(), 1, out->size(), f);
  std::fclose(f);
  if (got != out->size()) {
    fail(err, "short read: " + path);
    return false;
  }
  return true;
}

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

  if (data_.size() < sizeof(TinyHeader)) {
    fail(err, "file too small for header: " + path);
    return false;
  }
  std::memcpy(&header_, data_.data(), sizeof(TinyHeader));

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
  const uint64_t table_bytes = header_.tensor_count * sizeof(TensorEntry);
  if (header_.tensor_table_offset + table_bytes > data_.size()) {
    fail(err, "tensor table runs past end of file");
    return false;
  }
  if (header_.data_offset % kAlignment != 0 || header_.data_offset < sizeof(TinyHeader)) {
    fail(err, "bad data_offset: " + std::to_string(header_.data_offset));
    return false;
  }

  // Header config sanity.
  const TinyHeader& h = header_;
  if (h.n_layers == 0 || h.hidden_size == 0 || h.intermediate_size == 0 ||
      h.n_heads == 0 || h.n_kv_heads == 0 || h.head_dim == 0 || h.vocab_size == 0 ||
      h.max_seq_len == 0) {
    fail(err, "model config contains zero fields");
    return false;
  }
  if (h.n_heads % h.n_kv_heads != 0) {
    fail(err, "n_heads % n_kv_heads != 0");
    return false;
  }
  if (h.rms_norm_eps <= 0.0f || h.rope_theta <= 0.0f) {
    fail(err, "rms_norm_eps / rope_theta must be positive");
    return false;
  }

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
    uint64_t numel = 1;
    for (uint32_t d = 0; d < e.ndim; ++d) {
      if (e.shape[d] == 0) {
        fail(err, "tensor #" + std::to_string(i) + ": zero in shape");
        return false;
      }
      numel *= e.shape[d];
    }
    for (uint32_t d = e.ndim; d < 4; ++d) {
      if (e.shape[d] != 0) {
        fail(err, "tensor #" + std::to_string(i) + ": trailing shape must be 0");
        return false;
      }
    }
    if (e.nbytes != numel * dtype_size(Dtype::kF32)) {
      fail(err, "tensor #" + std::to_string(i) + ": nbytes mismatch");
      return false;
    }
    if (e.offset % kAlignment != 0 || e.offset < header_.data_offset) {
      fail(err, "tensor #" + std::to_string(i) + ": unaligned / bad offset");
      return false;
    }
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
    // Offset is not stored in TensorView; recover it relative to file start.
    uint64_t offset = static_cast<uint64_t>(t.data - data_.data());
    std::printf("%-56s %-18s %-5s %12llu %14llu\n", name.c_str(), shape_str(t).c_str(),
                dtype_name(t.dtype), (unsigned long long)offset,
                (unsigned long long)t.nbytes);
  }
  std::printf("file size: %llu bytes (%.2f MB)\n", (unsigned long long)data_.size(),
              (double)data_.size() / (1024.0 * 1024.0));
}

}  // namespace tinyqwen
