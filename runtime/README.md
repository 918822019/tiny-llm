# runtime/ 导读

文件看着多，其实就 **6 个角色**，每个干一件清楚的事。runtime 的本质是一条
流水线："把权重读进来 → 备好内存 → 循环计算 → 输出结果"。

## 角色地图

| 职责     | 文件                            | 一句话                            |
|--------|-------------------------------|--------------------------------|
| ① 格式契约 | `tiny_format.h`               | 定义 .tqwen 长什么样（只是定义，不干活）       |
| ② 加载   | `tensor.*` + `model_loader.*` | 把文件读进内存、校验、按名字找到每块权重           |
| ③ 内存   | `kv_cache.*`                  | decode 时缓存历史 K/V 的那块内存         |
| ④ 计算   | `qwen_model.*`                | 真正的前向：一个 token 进、下一个 token 出   |
| ⑤ 测量   | `profiler.*`                  | 记录每步耗时（可关，关了零开销）               |
| ⑥ 编排   | `main.cpp`                    | CLI：把上面串起来，prefill + decode 循环 |
| 配置     | `config.*`                    | key=value 配置解析（供 ⑥ 用）          |

## 数据怎么流过它们

```text
   model.tqwen
        │
   ② model_loader ──读入+校验──► ModelFile（权重在内存里）
        │                              │
        ▼                              ▼
   ④ qwen_model.create() ◄──绑定权重视图──┘
        │  同时持有 ③ kv_cache、⑤ profiler
        ▼
   ④ forward_token()  ◄── main.cpp 循环调用
        │   内部按顺序调 kernels/ 里的算子（matvec/RMSNorm/RoPE/attention）
        │   顺便写 ③ kv_cache、被 ⑤ profiler 计时
        ▼
   下一个 token id ──► main.cpp 打印
```

**关键认知**：真正"算"的代码在 `../kernels/`，runtime 只是**按正确顺序调用
kernel 并把数据摆到位**。想懂数学看 kernels + `docs/qwen_forward.md`；
想懂内存/加载看 `model_loader` + `kv_cache` + `tiny_format.h`。

## 建议阅读顺序（由浅入深）

1. `tiny_format.h` — 先懂"数据长什么样"（全是定义和注释）
2. `tensor.h` — 懂"视图"概念（最短）
3. `model_loader.cpp` — 文件怎么读进来、为什么 fail fast
4. `kv_cache.cpp` — 那块内存怎么摆
5. `main.cpp` — 站最高处看流程怎么串
6. `qwen_model.cpp` — 最后看核心前向（对照 `docs/qwen_forward.md`）
7. `profiler.cpp` — 纯工具，可选

每步对应的概念，见 `../docs/infra_primer.md`。
