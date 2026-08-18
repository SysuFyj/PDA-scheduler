# 1P3D 异构长度微突发调度实验

日期：2026-08-17

## 1. 结论摘要

本实验在 1P3D、Decode `gpu_memory_utilization=0.55` 下，保持约 70% 普通 offered load，并在 2 秒内叠加 `[16000,512,512]×21`，比较 RR、least-active 和 token-balance。

核心假设得到支持：

- 三策略在全部 63 个 burst Decode 分配期间，completed snapshot 始终为 `(3,3,2)`，严格满足“没有完成反馈”的门禁。
- RR 将 16K 请求分为 14/7/0；least-active 更集中，为 19/2/0；token-balance 为 8/7/6。
- least-active 虽把 burst 请求数保持为 21/21/21，却无法区分长度，热点卡静态目标 KV 需求达到容量的 243%。
- token-balance 的 burst output-token CV 为 10.1%，RR 为 74.3%，least-active 为 110.8%。
- token-balance 只有 1 次请求级抢占；RR 为 85 次，least-active 为 112 次。

因此，前一实验中 least-active 依靠隐式长度反馈接近 token-balance 的能力有明确边界：**当 burst 的所有 Decode 分配发生在任何请求完成之前时，隐式反馈尚未产生，least-active 会退化为请求数均衡。**

## 2. 环境与配置

- GPU：NVIDIA A100-SXM4-40GB；P0=GPU1，D0/D1/D2=GPU2/3/4。
- Decode KV：每实例 126,576 tokens，`gpu_memory_utilization=0.55`。
- 模型：Qwen2.5-7B-Instruct，TP=1，`max_model_len=32768`，`max_num_seqs=256`。
- Python 3.12.13；PyTorch 2.11.0+cu126；CUDA runtime 12.6。
- vLLM `0.1.dev15606+g994059927`；`--enforce-eager`、`temperature=0`、`ignore_eos=true`。
- Decode 在 Prefill 完成后基于最新 Router 状态选择。

## 3. 工作负载

- 普通请求：1200 条，平均 5.840661 req/s，保留 sample_1 的目标输出长度多重集合。
- Burst：63 条，`[16000,512,512]×21`，最后一条在 1.906 秒到达。
- 所有输入均为 131 tokens。
- 2 秒窗口内普通流量按四个三请求小组到达，平均 6 req/s；三请求小组不改变 RR 模 3 相位。
- 前 60 个普通请求仅重排为输出至少 1024 的样本，以防严格门禁窗口内出现普通请求 completion。
- 最后一个普通请求在 205.402 秒到达。

Smoke 使用 `×18`：token-balance 完成 354/354、0 失败，completed 门禁通过，16K 恰好为 6/6/6，无 waiting 或抢占，随后才启动正式矩阵。

## 4. Burst 放置与 KV 需求

| 策略 | 16K 分配 | Burst 请求数 | Burst output-token CV | 每卡估算目标 KV 占比 |
|---|---|---|---:|---|
| RR | 7 / 0 / 14 | 21 / 21 / 21 | 74.3% | 96% / 11% / 182% |
| least-active | 0 / 2 / 19 | 21 / 21 / 21 | 110.8% | 11% / 35% / 243% |
| token-balance | 7 / 6 / 8 | 15 / 26 / 22 | 10.1% | 93% / 87% / 109% |

估算目标 KV 为 `目标输出 token + 请求数×131 input tokens`。它代表所有请求完整生成时的静态需求，不是 vLLM 的即时预分配量。

RR 因实际 Prefill 完成顺序发生一次相位移动，没有形成预期的 21/0/0，但仍集中为 14/7/0。least-active 在 active count 完全相同时按 tie-break 选择，最终形成更严重的 19/2/0。token-balance 允许请求数不均，通过多放短请求来抵消长请求工作量。

## 5. 系统压力与请求级抢占

| 策略 | 峰值 waiting | waiting req-s | 抢占事件 | 唯一被抢占请求 | recompute tokens 近似 |
|---|---:|---:|---:|---:|---:|
| RR | 39 | 7,274 | 85 | 26 | 121,754 |
| least-active | 48 | 10,300 | 112 | 39 | 196,055 |
| token-balance | 1 | 69 | 1 | 1 | 15,749 |

RR 的 85 次抢占全部发生在热点 D2：78 次属于 19 个 background 请求，7 次属于 7 个长 burst 请求。least-active 的 112 次也全部发生在 D2：98 次属于 27 个 background 请求，14 次属于 12 个长 burst 请求。没有短 burst 请求被抢占。

token-balance 相对 RR 将 waiting request-seconds 降低 99.05%、抢占事件降低 98.82%、recompute-token 近似降低 87.06%；相对 least-active 分别降低 99.33%、99.11% 和 91.97%。

## 6. 性能结果

| 策略 | 吞吐 tok/s | Wall s | 到达结束后排空 s | TTFT mean/P99 s | TPOT mean/P99 ms | E2E mean/P99 s |
|---|---:|---:|---:|---:|---:|---:|
| RR | 2,128 | 548.5 | 343.1 | 3.082 / 213.210 | 29.9 / 118.6 | 29.8 / 408.6 |
| least-active | 1,957 | 596.7 | 391.3 | 4.203 / 246.435 | 30.2 / 162.9 | 32.0 / 398.8 |
| token-balance | 2,745 | 425.3 | 219.9 | 0.183 / 0.241 | 26.5 / 30.1 | 24.5 / 419.9 |

token-balance 相对 RR：吞吐提高 29.0%，wall 降低 22.5%，排空时间降低 35.9%。相对 least-active：吞吐提高 40.3%，wall 降低 28.7%，排空时间降低 43.8%。

总体 E2E P99 不适合作为本实验的唯一排序指标：21 个 16K 请求占总请求的 1.66%，恰好落在 P99 附近。应查看请求类别结果。

## 7. 请求类别延迟

### Background 请求

| 策略 | TTFT P99 s | TPOT P99 ms | E2E P99 s |
|---|---:|---:|---:|
| RR | 213.534 | 120.6 | 235.5 |
| least-active | 247.604 | 166.4 | 278.9 |
| token-balance | 0.241 | 30.1 | 66.1 |

热点不仅影响长 burst 自身，还使之后被放到热点卡的普通请求长时间 waiting/recompute。token-balance 将 background TTFT P99 相对 RR 和 least-active 均降低约 99.9%。

### 16K Burst 请求

| 策略 | TTFT P99 s | TPOT P99 ms | E2E P99 s |
|---|---:|---:|---:|
| RR | 0.204 | 34.0 | 544.6 |
| least-active | 0.198 | 37.1 | 593.5 |
| token-balance | 0.207 | 26.4 | 423.0 |

长请求本身都很早获得首 token，因此其主要损失体现在 TPOT 和总完成时间。token-balance 将长请求 E2E P99 相对 RR 降低 22.3%，相对 least-active 降低 28.7%。

### 512 Burst 请求

三策略的 512 请求 TTFT P99 均约 0.20–0.21 秒，E2E P99 约 8.4–8.6 秒。它们在 KV 压力建立前快速完成，因此不是 tail 的主要来源。

## 8. 机制解释

1. 63 个 burst 请求在 2 秒内完成 Decode 选择，期间 completed count 不变。
2. least-active 只能看到 unfinished request count；每个三元组增加三个 active 请求，使三卡在请求数上重新相等。
3. 由于 16K 与 512 在 active count 中权重相同，tie-break 会反复把长请求送到同一实例。
4. 请求完成后产生的隐式长度反馈来得太晚：错误放置已经完成，后续普通请求只能绕开热点，无法迁移已放置的长请求。
5. token-balance 在选择时立即计入 16K reserved work，因此无需等待 completion 就能打破周期相位。

## 9. 有效性限制

- 这是刻意保持模 3 相位的合成机制实验，不是自然 Poisson arrival。
- 普通流量在 2 秒窗口内按三请求小组到达；平均负载正确，但短时 arrival shape 被控制。
- token-balance 使用 oracle 目标输出长度；实际部署需要预测长度或安全上界。
- 静态目标 KV 百分比不代表 vLLM preallocation；实际峰值由动态 KV 增长决定。
- 单 trace、单次正式重复，没有 seed 间方差或置信区间。
- 请求级 recompute-token 是抢占前 computed tokens 的近似值。
- 根据用户要求未使用子 Agent，本次审计不是独立复核。

## 10. 结论

本实验明确划定了 least-active 的适用边界：

- 有完成反馈的长时间负载中，它可以通过隐式长度信息接近 token-balance。
- 在所有请求先完成放置、之后才出现 completion 的微突发中，它没有长度信号，甚至可能比 RR 更严重地集中长请求。
- token-balance 的优势主要不是改善 burst TTFT，而是避免错误 KV 放置、后续普通请求 waiting、抢占重算和长请求 TPOT/E2E tail。

下一步最有价值的实验是随机化 burst 相位、普通请求插入位置和 Prefill 完成顺序，运行 5 个 paired seeds，确认 least-active 的集中概率以及 token-balance 对预测误差的容忍范围。

## 11. 产物

- 计划：`experiments/1p3d_microburst_20260817/EXPERIMENT_PLAN.md`
- 正式汇总：`outputs/1p3d_microburst_20260817/formal-summary.json`
- 总体表：`outputs/1p3d_microburst_20260817/tables/overall.csv`
- 放置/KV 表：`outputs/1p3d_microburst_20260817/tables/burst-assignments.csv`
- 分类延迟：`outputs/1p3d_microburst_20260817/tables/latency-by-class.csv`
- 请求级抢占：`outputs/1p3d_microburst_20260817/tables/request-preemptions.csv`
- 图：`outputs/1p3d_microburst_20260817/figures/waiting-over-time.png`
- 图：`outputs/1p3d_microburst_20260817/figures/burst-kv-by-worker.png`
- 有效性审计：`outputs/1p3d_microburst_20260817/VALIDITY_AUDIT.md`
