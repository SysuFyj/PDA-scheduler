# 阶段成果：长度异构下的 Decode 放置

日期：2026-08-18

## 阶段结论

最近两次有效实验共同给出了 least-active 与 token-balance 的适用边界：

1. **存在完成反馈的持续负载**：周期性输入/输出长度会使固定 RR 产生条带热点；least-active 虽不读取长度，但长请求完成较慢，会通过 active request count 的变化获得**隐式完成速度反馈**，从而显著缓解 RR 的固定相位问题，并达到接近 token-balance 的负载均衡。
2. **分配早于完成的同步微突发**：大量长短请求在任何 Decode 请求完成前集中完成放置时，least-active 尚未获得长度反馈，只能按 unfinished request count 做表面均衡，可能把长请求集中到单卡；token-balance 直接使用 reserved target output tokens，因此能在放置时识别工作量差异。

这两点不能简化为“只有微突发时 token-balance 才有价值”。当前证据支持的是：**微突发是 token-balance 相对 least-active 优势最明确的负载区域；其他负载下二者仍可能存在吞吐、TTFT、TPOT 和预测误差方面的权衡。**

## 核心证据

### 持续长请求 Burst

- 拓扑：1P3D，Decode KV=0.55。
- 工作负载：900 普通请求 → 200 个长请求 token-load burst → 900 普通请求。
- Burst output-token CV：RR 19.4%，least-active 3.4%，token-balance 3.1%。
- TTFT P99：RR 665.5 秒，least-active 351.7 秒，token-balance 357.8 秒。
- 结论：least-active 的隐式反馈足以修复该持续负载下的 RR 条带热点；least-active 与 token-balance 没有全指标赢家。

### 2 秒异构微突发

- 拓扑：1P3D，Decode KV=0.55。
- 工作负载：约 70% 普通负载叠加 `[16000,512,512]×21`，63 个 burst 请求在 2 秒内到达。
- 严格门禁：三个策略的 63 次 burst 分配期间，Decode completed snapshot 始终为 `(3,3,2)`。
- 16K 分配：RR 14/7/0，least-active 19/2/0，token-balance 8/7/6。
- 请求级抢占：RR 85 次，least-active 112 次，token-balance 1 次。
- Background TTFT P99：RR 213.5 秒，least-active 247.6 秒，token-balance 0.241 秒。
- 结论：没有 completion 反馈时，least-active 退化为请求数均衡；token-balance 能在放置阶段避免 KV 热点。

## 目录索引

- 实验设计与代码：`experiments/1p3d_burst_20260817/`
- 紧凑正式结果：`outputs/1p3d_burst_20260817/`
- 微突发设计与代码：`experiments/1p3d_microburst_20260817/`
- 微突发紧凑正式结果：`outputs/1p3d_microburst_20260817/`
- 文件级清单：`EXPERIMENT_INDEX.md`
- 当前调度代码说明：`CODE_SNAPSHOT.md`

原始请求 CSV、服务日志、metrics JSONL 和生成 trace 继续保留在本地 `outputs/`，不纳入 Git；Git 只保存报告、审计、汇总 JSON、关键 CSV 和 PNG。
