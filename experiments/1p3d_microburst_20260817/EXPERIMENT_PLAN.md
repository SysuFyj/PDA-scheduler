# 1P3D 异构长度微突发实验计划

日期：2026-08-17

## 1. 研究问题

- 问题：在没有 burst 请求完成的 2 秒窗口内，least-active 是否因缺少长度反馈而退化为按请求数轮转，使 `[16000, 512, 512] × N` 的 16K 请求集中到单个 Decode。
- 可证伪假设：RR 与 least-active 会把大多数 16K 请求放到同一实例；token-balance 会把 16K 请求近似均分。
- 否定条件：严格 completed 门禁通过，但 least-active 的 16K 分配仍接近均分。
- 允许权衡：token-balance 可以用目标输出长度这一 oracle 信息换取更低的 KV 热点、waiting、preemption 和 tail latency。

## 2. 系统边界

- 模型：Qwen2.5-7B-Instruct。
- 引擎：本地独立 `.venv` 中的 vLLM，PyTorch CUDA 12.6 组合。
- 拓扑：1P3D，P0=GPU1，D0/D1/D2=GPU2/3/4。
- Decode `gpu_memory_utilization=0.55`，每实例 126,576 KV tokens。
- `max_model_len=32768`、`max_num_seqs=256`、TP=1、`--enforce-eager`。
- Scheduler 唯一策略变量：RR、least-active、token-balance；Decode 仍在 Prefill 完成后选择。

## 3. 工作负载

- 普通请求来源：`data/traces/sample_1_2000.jsonl`，保留原始输出长度多重集合。
- 普通请求平均到达率：5.840661 req/s，约等于前一实验最大观测吞吐的 70% token offered load。
- 前 60 个普通请求从原 trace 中选择输出长度至少 1024 的样本，只改变顺序，不改变普通请求长度多重集合，以避免 2 秒门禁窗口内普通请求完成。
- burst：固定输入 131 tokens，输出长度 `[16000, 512, 512] × N`，所有 burst 请求在 2 秒内发出。
- burst 以三元组发送；2 秒窗口内的 12 个普通请求按四个三元组发送，平均仍为 6 req/s。普通三元组只在 burst 三元组之间插入，因此不改变 RR 的模 3 相位。
- 2 秒后普通请求恢复固定间隔到达。
- Smoke：`N=18`，300 个普通请求，54 个 burst 请求，只运行 token-balance。
- Formal：`N=21`，1200 个普通请求，63 个 burst 请求，运行三种策略。

## 4. 容量计算

- Formal 理想均分时，每个 Decode 获得 7 个 16K、14 个 512。
- 每卡 burst KV 需求估算：`7×16000 + 14×512 + 21×131 = 121,919 tokens`。
- 相对 126,576 KV tokens，占用约 96.32%。普通背景请求会进一步制造 admission/preemption 压力。

## 5. 观测与门禁

- Router 每次选择记录 `decode_completed_before`、active count、score 和目标输出 token。
- 严格门禁：从第一个到最后一个 burst Decode 分配，三个 Decode 的 completed snapshot 必须完全不变。
- Smoke token-balance 门禁：54/54 burst 完成分配；每卡恰好 6 个 16K；请求完整；没有 timeout；观测补丁成功产生或确认零 request-level preemption。
- vLLM 只增加临时日志，不改变调度：每次 preemption 记录 request ID、preemption count 和被清零前的 computed tokens；实验后恢复原文件并校验 SHA256。
- 请求级 recompute 以抢占前 computed tokens 作为需要重新计算的近似工作量。

## 6. 指标

- 长/短 burst 请求在每卡的数量、目标输出 token、输入加输出 KV 估算及不均衡 CV。
- 每卡 KV、running、waiting 和累计 preemption 时序。
- 请求级 preemption 次数与 recompute-token 近似值。
- 总体、background、burst-long、burst-short 的 TTFT、TPOT、E2E mean/P95/P99。
- 最后一次普通请求到达后，waiting 和 active reservation 的稳定排空时间。

## 7. 有效性限制

- 合成长短周期和三请求普通流量小组是机制实验，不代表生产到达过程。
- token-balance 使用目标输出长度，属于 oracle 上界。
- completed 门禁失败的组不进入策略结论。
- request-level recompute 是 computed-token 近似值，不是 GPU kernel 时间。
- 单 trace、单正式重复只支持本工作负载的配对机制结论。

## 8. 资源预算与停止条件

- GPU：1–4，使用项目级持续 `flock`，锁覆盖 smoke、formal 和清理。
- Smoke：1 组，354 个测量请求，最长 12 分钟。
- Formal：3 组，每组 1263 个测量请求，单组最长 30 分钟。
- 总上限：4143 个测量请求，最长 90 分钟。
- 每组独立执行 clean → start → warmup → run → reconcile → stop → clean。
- Smoke 未通过 token-balance 分配或 completed 门禁时停止，不执行 formal。
- 正式运行需要用户对上述完整矩阵明确 `APPROVE`。
