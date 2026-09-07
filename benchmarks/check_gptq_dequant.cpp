// ============================================================================
// check_gptq_dequant.cpp — 用真 AutoGPTQ 权重校验 matvec_gptq 的数值正确性
// ============================================================================
// fake MoE 模型的 GPTQ 是手写随机打包，测不出与真 AutoGPTQ 产物的语义差异
// （qzeros 的 zp-1 约定、int4 打包顺序、g_idx）。本工具读 tools/
// validate_gptq_numeric.py 从真 checkpoint 转出的 in-band 块，跑 matvec_gptq，
// 把结果 dump 出来交 Python 与参考 dequant 逐位对比。
//
// 用法：
//   check_gptq_dequant --block <path> --out-dim N --in-dim K --group-size G
//                      --x <path> --out <path>
// block/x 是 fp32/裸字节文件；out 写 out_dim 个 fp32。
// ============================================================================

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "../kernels/dispatch.h"

namespace {

    bool read_raw(const std::string &path, std::vector<uint8_t> *buf) {
        FILE *f = std::fopen(path.c_str(), "rb");
        if (!f) return false;
        std::fseek(f, 0, SEEK_END);
        const long n = std::ftell(f);
        std::fseek(f, 0, SEEK_SET);
        if (n < 0) { std::fclose(f); return false; }
        buf->resize(static_cast<size_t>(n));
        const size_t got = buf->empty() ? 0 : std::fread(buf->data(), 1, buf->size(), f);
        std::fclose(f);
        return got == buf->size();
    }

    bool write_raw(const std::string &path, const void *data, size_t nbytes) {
        FILE *f = std::fopen(path.c_str(), "wb");
        if (!f) return false;
        const size_t put = std::fwrite(data, 1, nbytes, f);
        std::fclose(f);
        return put == nbytes;
    }

    const char *value(const char *flag, int argc, char **argv) {
        for (int i = 1; i < argc - 1; ++i) {
            if (std::strcmp(argv[i], flag) == 0) return argv[i + 1];
        }
        return nullptr;
    }

} // namespace

int main(int argc, char **argv) {
    const char *block_path = value("--block", argc, argv);
    const char *x_path = value("--x", argc, argv);
    const char *out_path = value("--out", argc, argv);
    const char *od = value("--out-dim", argc, argv);
    const char *id = value("--in-dim", argc, argv);
    const char *gs = value("--group-size", argc, argv);
    if (!block_path || !x_path || !out_path || !od || !id || !gs) {
        std::fprintf(stderr,
                     "usage: %s --block <path> --out-dim N --in-dim K --group-size G "
                     "--x <path> --out <path>\n", argv[0]);
        return 2;
    }
    const int out_dim = std::atoi(od);
    const int in_dim = std::atoi(id);
    const int group_size = std::atoi(gs);

    std::vector<uint8_t> block;
    if (!read_raw(block_path, &block)) {
        std::fprintf(stderr, "cannot read block: %s\n", block_path);
        return 1;
    }
    std::vector<uint8_t> xraw;
    if (!read_raw(x_path, &xraw)) {
        std::fprintf(stderr, "cannot read x: %s\n", x_path);
        return 1;
    }
    if (xraw.size() != static_cast<size_t>(in_dim) * sizeof(float)) {
        std::fprintf(stderr, "x size %zu != in_dim*4 (%d)\n", xraw.size(), in_dim * 4);
        return 1;
    }

    std::vector<float> y(static_cast<size_t>(out_dim), 0.0f);
    tinyqwen::matvec_gptq(block.data(),
                          reinterpret_cast<const float *>(xraw.data()),
                          y.data(), out_dim, in_dim, group_size);

    if (!write_raw(out_path, y.data(), y.size() * sizeof(float))) {
        std::fprintf(stderr, "cannot write out: %s\n", out_path);
        return 1;
    }
    std::fprintf(stderr, "[check_gptq_dequant] impl=%s out_dim=%d in_dim=%d gs=%d -> %s\n",
                 tinyqwen::matvec_gptq_impl_name(), out_dim, in_dim, group_size, out_path);
    return 0;
}
