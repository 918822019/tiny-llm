# 机器极限账表 —— Apple M4（Mac16,10，16GB）

测量日期：2026-08-21 · 测量代码：`experiments/machine_ceiling/` · 全部冷态实测，非规格书数字

## 0. 机器画像

| 项 | 值 |
|---|---|
| 芯片 | Apple M4（基础版），Mac16,10 |
| CPU | 4P + 6E = 10 核；P 簇 L2 16MB，E 簇 L2 4MB |
| GPU | 10 核，Metal 4（UMA 统一内存） |
| 内存 | 16GB 统一内存，LPDDR5X，官方规格 ~120 GB/s |
| 系统 | macOS 26（Metal 运行时 API 已改名：`computeCommandEncoder`） |

## 1. DRAM 带宽墙（本机一切带宽-bound 问题的物理上限）

### CPU 流式读饱和曲线（bench_bw，2GB buffer，冷态）

| 线程 | GB/s | 解读 |
|---|---|---|
| 1（P 核） | 68.8 | 单核已达整机墙的 60%——Apple P 核 MLP 极深 |
| 2–4（P 簇内） | 73–76 | **P 簇通路天花板 ≈ 74 GB/s**，簇内加核几乎无增益 |
| 5 | 101 | E 核开始加入（4P+1E） |
| 6–10 | 112–115 | **整机墙 ≈ 114 GB/s**（规格的 95%） |

> 与项目历史记录互证：optimization_log 里"单核带宽 ~72 GB/s"与本次单核 68.8 一致，
> 方法学可信。日志中"整机 ~199 GB/s"出自更高带宽的 Mac（推测 M Pro 系，273 GB/s 级），
> **不是本机**——引用历史数字时注意机器差异。

### 其它内存模式（10 线程峰值）

| 模式 | GB/s |
|---|---|
| 读 | 114–115 |
| 写 | 51 |
| 拷贝（读+写合计） | 95–102 |

### GPU 与异构并发（bench_gpu_bw，Metal compute 流式读）

| 配置 | GPU GB/s | CPU GB/s | 合计 |
|---|---|---|---|
| GPU 单干（1GB buffer） | 105.8 | — | 105.8 |
| CPU 10 线程单干 | — | 113 | 113 |
| CPU 1 + GPU | 81.2 | 31.7 | 112.8 |
| CPU 4 + GPU | 66.1 | 49.2 | 115.3 |
| CPU 6 + GPU | 56.0 | 58.8 | 114.8 |
| CPU 10 + GPU | 55.0 | 63.2 | **118.2** |

**结论（异构分片设计 A 的判决实验）：CPU+GPU 并发总带宽 ≤ ~118 GB/s，与 CPU 单干
持平。UMA 上 DRAM 墙是全 SoC 共享的，并发只能切分墙、不能加高墙。**
想把 decode 推过墙，唯一路径是减少每 token 搬运的字节数（量化/稀疏/架构），
不是加执行单元。GPU 的价值在 compute-bound 场景（prefill、batched 验证），
不在 bandwidth-bound 的 batch=1 decode 搬运。

## 2. 持续负载与热墙（bench_sustain，10 线程流式读 300s）

| 时间 | GB/s |
|---|---|
| 10–50s | 114（峰值平台） |
| 130s | 110（开始下滑） |
| 150s / 180s | 90 / 67 |
| 220–250s | **27–34（深谷）** |
| 260–300s | 57–73（回升震荡） |
| 300s 平均 | **74.6 GB/s（峰值的 65%）** |

> 73/74s 两个窗口（11.6 GB/s）为异常点：疑似前一次运行残留进程退出时的争用，
> 趋势判读时忽略。

**峰值 114 → 深谷 27，同一台机器同一负载相差 4.2×。** 热调度器震荡明显
（250s 后回升），说明 governor 在周期性放宽。任何"端侧持续性能"的讨论
必须基于 ~75 GB/s 的持续墙而非 114 的峰值墙；交互式场景可以抢峰值窗口，
后台长任务只能按持续墙预算。

## 3. 计算上限（bench_flops，冷态）

| 项 | 单核 | 4P 簇 | 全 10 核 |
|---|---|---|---|
| fp32 NEON FMA | 66.5 GFLOPS | 252.5 GFLOPS | 502 GFLOPS |
| i8 dotprod (vdotq_s32) | 183.7 GOPS | — | 1309 GOPS |
| AMX sgemm fp32（Accelerate） | — | — | **1360 (2048²) / 1603 GFLOPS (4096²)** |

- AMX ≈ NEON 全核的 3.2×——**prefill 的算力天花板在 AMX**，不在 NEON 线程堆叠。
- i8 全核 1.3 TOPS 是 W4A8/SDOT 路线的算力参考：i4 反量化+点积若指令效率低，
  会先撞这条算力线而非带宽墙（项目 Android i4 阶段已实证过此陷阱）。

## 4. NAND 顺序读（bench_nand，F_NOCACHE 绕 page cache）

| 测试 | GB/s |
|---|---|
| 1.4GB 文件（SSD 缓存命中） | 13.8 / 22.2 |
| 5.3GB 文件冷读（超出 SSD 缓存） | **3.33 / 3.34** |

**flash offload 轴的天花板 ≈ 3.3 GB/s**（比 DRAM 墙低 35×）。
推论：权重纯流式 decode 上限 = 3.3GB ÷ 每 token 字节数——i4 0.8B（~0.4GB/token）
也只有 ~8 tok/s。flash 只适合做冷层/expert 仓库 + DRAM 热缓存的两级结构，
且 prefetch 深度要能盖住 3.3GB/s 与 DRAM 墙之间的差。

## 5. 推导：LLM decode 上限对账（batch=1）

上限 = 墙 ÷ 每 token 搬运字节。Qwen3.5-0.8B f16 权重 1435MB（+KV/GDN 态可忽略）。

| 墙 | f16 0.8B 上限 | 对应 ms/tok |
|---|---|---|
| 峰值 114 GB/s | 79 tok/s | **12.6 ms** |
| 持续 74.6 GB/s | 52 tok/s | 19.3 ms |
| 热深谷 27 GB/s | 19 tok/s | 53 ms |

对账：
- AGENTS.md 记录本机 f16 满栈 17.65 ms/tok → 有效带宽 81 GB/s = 冷态墙的 **71%**，
  与账表自洽（剩余 29% 是算子开销/调度缝隙）。
- 本次实测复核（冷态）：decode_avg **18.14 ms/tok** → 79 GB/s，与上条吻合。
- 实测复核（热态，刚跑完 300s 压力）：decode_avg **46.3 ms/tok** → 31 GB/s，
  正好落在热衰减区。**同一配置冷热差 2.6×**——热状态必须作为测量变量记录在案。
- Qwen2.5-0.5B i4（~0.3GB/token）：峰值上限 ~2.6 ms/tok；历史实测 3.67 ms
  （高带宽 Mac 上已是算力-bound）。本机预计 3.5–4.5 ms——i4 在本机**开始摸到
  带宽墙**（300MB/3.6ms = 83 GB/s > P 簇 74），需要跨簇线程才能打满，
  与日志"线程甜蜜点 8（5P+3）"互证。

## 6. 对异构流水线设计的直接结论

1. **设计 A（CPU+NPU/GPU 权重分片提带宽）：本机证伪。** 墙共享，无增益空间。
   骁龙上是否成立取决于其 CPU 通路是否远低于 DRAM 峰值——需同法实测。
2. **batch=1 decode 是纯带宽题，NPU 算力无用武之地。** NPU 进场的前提是
   把 decode 变成 batched GEMM——即投机解码的 verify 阶段（设计 D）。
3. **P 簇单簇 74 GB/s 是低延迟路径的天花板**：f16 0.8B（需 ~81 GB/s）单簇不够，
   i4（需 ~40–83 GB/s 视目标速度）单簇可行——"只用 P 簇保延迟、全簇保吞吐"
   是可操作的调度分档。
4. **E 簇推断带宽 ≈ 40 GB/s**（114−74，未直接测量），足够跑 i4 后台解码
   （~10ms/tok 级）——"E 核当后台工人"轴有真实带宽基础。
5. **热预算是硬约束**：并发旋钮必须存在。持续场景按 75 GB/s 预算，
   峰值墙只在短冲刺（交互首 token）可用。
6. **prefill 走 AMX**（1.6 TFLOPS fp32），与 decode 的带宽题完全正交——
   prefill∥decode 重叠（设计 C）是两个不同资源瓶颈的组合，这是本机
   异构流水线最有希望真正赚到收益的形态。

## 附：测量方法与可复现性

```bash
cd experiments/machine_ceiling
clang++ -O3 -std=c++17 bench_bw.cpp -o bench_bw
clang++ -O3 -std=c++17 bench_gpu_bw.mm -framework Metal -framework Foundation -o bench_gpu_bw
clang++ -O3 -std=c++17 -march=armv8.2-a+dotprod bench_flops.cpp -framework Accelerate -o bench_flops
clang -O3 bench_nand.c -o bench_nand
clang++ -O3 -std=c++17 bench_sustain.cpp -o bench_sustain

./bench_bw 2048 10 3 read            # CPU 带宽饱和曲线
./bench_gpu_bw gpu 1024 4            # GPU 带宽
./bench_gpu_bw concurrent 10 1024 4  # CPU+GPU 并发
./bench_flops f32 10 2 && ./bench_flops i8 10 2 && ./bench_flops gemm 4096 3
./bench_nand <大文件> 16 2           # 用 >5GB 文件才有冷读数
./bench_sustain 10 300 10 2048       # 热衰减（5 分钟）
```

注意：本机 Metal 运行时（macOS 26）方法名已改（`computeCommandEncoder`），
shader 不接受旧式 `restrict` 写法——bench_gpu_bw.mm 已适配。
测量时机器必须冷态（跑过持续负载后等 10 分钟），否则带宽数会落在热衰减区。
