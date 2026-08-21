# runtime/ 导读

文件看着多，其实就 **8 个角色**，每个干一件清楚的事。runtime 的本质是一条
流水线："把权重读进来 → 备好内存 → 循环计算 → 输出结果"。

## 角色地图

| 职责     | 文件                                            | 一句话                                      |
|--------|-----------------------------------------------|------------------------------------------|
| ① 格式契约 | `tiny_format.h`                               | 定义 .tqwen 长什么样（只是定义，不干活）                 |
| ② 加载   | `tensor.*` + `model_loader.*`                 | 把文件读进内存、校验、按名字找到每块权重                     |
| ③ 内存   | `kv_cache.*` + `gdn_state.*`                  | decode 的历史记忆：full attention 的 K/V 缓存 + Qwen3.5 GDN 的 O(1) 递归/conv 状态 |
| ④ 计算   | `qwen_model.*` + `qwen_forward_*.cpp`         | 真正的前向：一个 token 进、下一个 token 出             |
| ⑤ 后端   | `backend.h` + `backend_cpu.*` + `backend_cuda.*` | 算子抽象层：模型只调 IBackend，不关心 dtype/量化/硬件      |
| ⑥ 测量   | `profiler.*`                                  | 记录每步耗时（可关，关了零开销）                         |
| ⑦ 编排   | `main.cpp`                                    | CLI：选后端/engine、prefill + decode 循环       |
| 配置     | `config.*`                                    | key=value 配置解析（供 ⑦ 用）                    |

## 数据怎么流过它们

```text
   model.tqwen
        │
   ② model_loader ──读入+校验──► ModelFile（权重在内存里）
        │                              │
        ▼                              ▼
   ④ QwenModel::create() ◄──绑定权重视图──┘
        │  同时持有 ③ kv_cache/gdn_state、⑥ profiler、⑤ backend
        ▼
   forward_token()/forward_prefill()  ◄── main.cpp 循环调用
        │   按顺序调 ⑤ IBackend 算子接口（matvec/rmsnorm/rope/attention/GDN）
        │   ⑤ 按权重 dtype（f32/f16/i4）分发到 ../kernels/ 的通用入口
        │   顺便写 ③ cache、被 ⑥ profiler 计时
        ▼
   下一个 token id ──► main.cpp 打印
```

**关键认知**：
- 真正"算"的代码在 `../kernels/`，runtime 只是**按正确顺序调用 kernel 并把
  数据摆到位**。想懂数学看 kernels + `docs/qwen_forward.md`；想懂内存/加载看
  `model_loader` + `kv_cache` + `tiny_format.h`。
- ④ 不直接碰 kernel：所有算子经 ⑤ `IBackend` 接口调用（重构 commit `ebca4db`）。
  权重以 `WeightTensor`（data + QuantType + group_size）传递，模型层不感知量化；
  换后端（CPU/CUDA/...）不需要改模型代码。分层全景见 `../docs/architecture.md`。
- forward 拆成两个文件：`qwen_forward_token.cpp`（decode，matvec 路径）与
  `qwen_forward_prefill.cpp`（批量 prefill，GEMM 路径）；`qwen_model.cpp` 只剩
  `create()`（权重绑定/校验/workspace 分配）和公共逻辑。

## 两条 GPU 路径的区别（易混淆）

- `--backend cuda`（`backend_cuda.*`）：实现 IBackend，**逐算子**调 CUDA kernel，
  每次调用带 H2D/D2H 拷贝。供 A/B 与单算子调试，**不是性能路径**；INT4/partial
  RoPE/top_k/GDN 算子未实现（触发即 abort）。
- `--engine cuda`（`../kernels/cuda/gpu_engine.cu`）：**整段 forward** 常驻显存、
  单 stream 跑完，每步仅 4B argmax 过 PCIe。性能路径，但不走 IBackend。

## 建议阅读顺序（由浅入深）

1. `tiny_format.h` — 先懂"数据长什么样"（全是定义和注释）
2. `tensor.h` — 懂"视图"概念（最短）
3. `model_loader.cpp` — 文件怎么读进来、为什么 fail fast
4. `kv_cache.cpp` — K/V 那块内存怎么摆；`gdn_state.cpp` — GDN 的 O(1) 状态
5. `backend.h` — 算子抽象接口（对照 `../docs/architecture.md`）
6. `main.cpp` — 站最高处看流程怎么串（后端/engine 选择 + 循环编排）
7. `qwen_model.cpp` — create 与权重绑定；再看 `qwen_forward_token.cpp`
   核心前向（对照 `docs/qwen_forward.md`）
8. `profiler.cpp` — 纯工具，可选

每步对应的概念，见 `../docs/infra_primer.md`。
