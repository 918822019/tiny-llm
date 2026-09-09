# MemCom-0.3B NPU Benchmark (SM8850 / Hexagon V81)

memcom-0.3b-core (437.6M params, step_00022000) 在 SM8850 NPU 上的端到端推理性能基准。

## 架构

- D=1152, 20 blocks, VOCAB=50000, FFN=3110, CACHE_LEN=256
- ATTN_LAYERS={4,9,14,19}: GQA 18Q/2KV, RoPE(theta=1e4, d=64)
- 其余 16 层: BilinearMemoryCell (BMC) — 状态递推 `S_new = g*S + i*(k⊗v)`，**时间维不可并行**
- 每步 input: 27 tensors (~1.70 MB) — ids[1] + cos/sin[64] + 16×s[9,64,64] + 8×kc/vc[2,64,256]
- 每步 output: 25 tensors (~1.80 MB) — logits[1,50000] + 16×s_out + 8×kc_out/vc_out
- 每步权重流式读取: ~388.7 MB

## 文件

| 文件 | 说明 |
|---|---|
| `memcom_model.py` | 模型定义（shape-faithful 重建） |
| `export.py` | ONNX 导出（decode 单步图，fp16） |
| `ref_compare.py` | 数值对比脚本（logits + state meanabs） |
| `memcom_e2e.cpp` | ARM64 e2e 解码驱动（dlopen HTP → 反量化 argmax → KV 回灌 → 逐 token 计时） |
| `memcom_tensors.inc` | 从 context binary 提取的 tensor spec 表（自动生成） |
| `w8a16_spec.json` | qnn-context-binary-utility 导出的完整 tensor 元数据 |
| `shim.cpp` | CPU backend 用：导出 `qnn_wrapper_api::strnDup`（手动链接 model .o 时缺失） |

## 构建流程

```bash
# 1. ONNX 导出（容器内）
python export.py   # → memcom/decode.onnx (fp16)

# 2. QNN converter（W8A16 量化）
qnn-onnx-converter --input_network memcom/decode.onnx \
  --output_path memcom/model_w8a16.cpp \
  --quantization_overrides htp.json \
  --restrict_quantization_steps "-0x8000 0x7F7F"

# 3. ARM64 交叉编译（NDK）
$NDK/toolchains/llvm/prebuilt/linux-x86_64/bin/clang++ \
  --target=aarch64-none-linux-android26 \
  -O2 -static-libstdc++ \
  -I$R/include/QNN \
  memcom_e2e.cpp -o memcom_e2e -ldl -lm

# 4. 手机端 context binary 生成
qnn-context-binary-generator --model libmodel_w8a16.so \
  --backend libQnnHtp.so --binary_file memcom_decode_w8a16.bin
```

## 运行

```bash
# 单独跑（NEON 量化，最优配置）
LD_LIBRARY_PATH=. ./memcom_e2e --prefill 128 --tokens 50 --warmup 3

# 参数
--prefill N    零初始化状态，串行处理 N 个 prompt token
--tokens M     自回归生成 M 个 token
--warmup K     预热步数（不计入统计）
--context PATH context binary 路径（默认 memcom_decode_w8a16.bin）
--inputs DIR   seed 输入目录（默认 inputs/）
--vote         HTP DDR 投票（当前设备不支持，会挂死）
```

## 性能结果（W8A16，SM8850，128 tok prefill + 50 tok decode）

| 配置 | Prefill tok/s | Decode tok/s | TTFT (128tok) |
|---|---|---|---|
| 基线（标量量化 + solo） | 21.91 | 22.02 | 5.84s |
| 标量量化 + warming 线程 | 42.71 | 44.20 | 3.00s |
| **NEON 量化 + solo（最优）** | **38.63** | **28.89** | **3.31s** |
| NEON 量化 + warming 线程 | 35.77 | 22.09 | 3.58s |

graph-only 单步（qnn-net-run b2b）：W8A16 min 18.4ms / avg 21.1ms, W8A8 min 17.4ms / avg 19.5ms

## 优化结论

1. **NEON 量化转换是最大单项优化**：24 对 KV/state 反馈从标量 float（~20ms/tok）改为 NEON SIMD 线性变换 `q_out = A*q_in + B`（~2ms/tok）。Prefill +76%，Decode +31%。

2. **Warming 线程与 NEON 互斥**：标量量化（CPU 重）时 warming 线程有正收益（DVFS 钉频 > DDR 竞争）；NEON 量化（CPU 轻）时 warming 线程有负收益（STREAM 64 GB/s 竞争 HTP 20 GB/s DDR 带宽）。

3. **算子融合无效**：98.2% 的 DDR 流量是权重读取（388.7 MB/步），激活中间张量 <0.1%。融合最多省 0.5%。

4. **Prefill = Decode**：BMC 架构（16/20 层）时间维不可并行，prefill 就是 N 次串行 decode step，每 token 耗时相同。

5. **HTP perf voting 不可用**：`QnnHtpPerfInfrastructure` 的 `setPowerConfig` 在 SM8850 挂死。SDK 无示例代码，仅头文件声明。

6. **理论下限**：388.7 MB / 20 GB/s DDR = 19.4 ms/tok（51.5 tok/s）。当前最优 28.89 tok/s（decode），距下限 44%。差距来源：DVFS 降频 + graph execute 调度开销。

## 待主模型训完后

- W8A8 量化：graph-only 预期 ~6% 提速（18.4→17.4ms）
- 重新跑完整 e2e 基准对比
- 评估 W4 可行性（HTP V81 不支持 SFXP4，当前不可行）
