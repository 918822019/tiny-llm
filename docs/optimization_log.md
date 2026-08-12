# 优化日志

> 每一次性能优化都在这里留一条记录：**改了什么、提升了多少、为什么提升**。
> 方法论见 `docs/benchmarking.md`。所有数字来自 `scripts/bench.sh` 的标准负载
>（固定 prompt、decode 32 token、丢弃预热 4、取稳态中位数），跨版本可比。

## 汇总表

主指标 = **decode 延迟中位数（ms/token）**，越小越好。加速比相对 fp32-baseline。

| 版本 label | commit | decode 中位 ms/tok | p95 ms/tok | 加速比 | 原因（为什么快/慢） |
|---|---|---:|---:|---:|---|
| fp32-baseline | 5f679ea | 230.36 | 252.35 | 1.00× | 参考实现：fp32 标量、单线程、未量化，一切只求正确 |

<!-- 新的优化按时间顺序往上表追加行，并在下面补一个详细小节 -->

---

## 详细记录

### fp32-baseline（2026-08-12）

- **是什么**：v1 参考实现。FP32 标量 kernel、单线程、token-by-token、未做任何量化。
- **测量环境**：arm（Apple Silicon），commit `5f679ea`，Release `-O3`。
- **结果**：
  - decode 中位 **230.36 ms/token**（27 个稳态样本，mean 235.31，min 226.65，p95 252.35）
  - prefill：3 token 共 769 ms
- **时间花在哪**（top op）：`lm_head` ~2186ms（占比最大），其次是各层 `down_proj`。
  说明瓶颈在 matvec，且 decode 是**带宽瓶颈**（每 token 都要把全部权重读一遍）。
- **结论 / 下一步**：这是正确性基准，不追求快。优化主攻方向 = 量化 matvec
  （减少权重搬运）+ 后续多线程。复现命令：
  ```bash
  ./scripts/bench.sh fp32-baseline
  ```

---

<!-- 模板：复制下面这段，填好后追加 -->
<!--
### <优化名>（<日期>）

- **是什么**：<改了哪个 kernel / 数据结构 / 调度，一两句话>
- **假设**：<为什么预期会快，落在带宽/计算/并行/指令哪一类>
- **结果**：decode 中位 __ ms/token（样本 __），p95 __
- **加速比**：__× vs fp32-baseline
- **验证**：<和 reference 的对齐误差，确认没算错>
- **意外 / 教训**：<如果和预期不符，分析为什么——这往往是最值钱的部分>
- **复现**：`./scripts/bench.sh <优化名>`
-->
