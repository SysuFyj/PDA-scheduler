# 1P3D 70%-Long-Burst-70% 调度对照实验

日期：2026-08-17

## 1. 摘要

本实验在 1P3D、Decode `gpu_memory_utilization=0.55` 下，用同一份 2000 请求 trace 构造“普通负载 → 200 个长请求 token-load burst → 普通负载”，比较 RR、least-active 和 token-balance。

核心结果：

- RR 把全部 67 个 12K 请求放到 D2，使 burst 阶段每卡累计目标 token 的 CV 达到 19.4%，最大/最小为 1.49 倍。
- least-active 将 burst token CV 降至 3.4%，token-balance 为 3.1%；least-active 确实通过隐式长度反馈达到了与 oracle token-balance 接近的负载均衡。
- 相对 RR，least-active 吞吐提高 22.8%、TTFT P99 降低 47.2%、稳定排空时间降低 50.7%；token-balance 吞吐提高 28.1%、TTFT P99 降低 46.2%、稳定排空时间降低 47.2%。
- least-active 与 token-balance 没有全指标赢家：token-balance 吞吐高 4.3%、TPOT P99 低 26.7%；least-active TTFT mean 低 21.1%、TTFT P99 低 1.7%、waiting request-seconds 少 15.3%。

因此，本次实验支持“RR 无法处理周期条带长度，而 least-active 可接近 token-balance”的机制判断；不支持“token-balance 或 least-active 全面更优”。

## 2. 环境与拓扑

- GPU：NVIDIA A100-SXM4-40GB；驱动 560.35.05。
- GPU 映射：P0=GPU1，D0/D1/D2=GPU2/3/4；实验结束后均已清理。
- Python 3.12.13；PyTorch 2.11.0+cu126；CUDA runtime 12.6。
- vLLM `0.1.dev15606+g994059927`。
- 模型：Qwen2.5-7B-Instruct，TP=1，`max_model_len=32768`，`max_num_seqs=256`。
- Prefill memory utilization 0.90；Decode memory utilization 0.55；每个 Decode 126,576 KV tokens。
- 公共参数：`--enforce-eager`、`temperature=0`、`ignore_eos=true`。
- Git 基线：`669271e9510114d06719ab88ab4ebfa06fb146ce`；运行包含尚未提交的后置 Decode 选择与三策略 router 改动。

## 3. 工作负载

容量扫描使用完整的 sample_1 2000 请求，在 rate 8/10/12/14 下得到 4,114/4,716/5,177/5,592 output tok/s。正式实验将 5,592 tok/s 作为扫描范围内最大观测吞吐。

正式 trace：

| 阶段 | 请求数 | 到达率 | 平均输出长度 | token offered load |
|---|---:|---:|---:|---:|
| pre | 900 | 5.841 req/s | 689.0 | 4,024 tok/s（观测峰值的 72.0%） |
| burst | 200 | 0.591 req/s | 9,467.7 | 5,592 tok/s（观测峰值的 100%） |
| post | 900 | 5.841 req/s | 651.5 | 3,805 tok/s（观测峰值的 68.0%） |

burst 长度严格循环 `[12000, 8192, 8192]`。长请求的 request/s 小于 baseline，但其 token offered load 突升到观测峰值的 100%；这是 token-load burst，而非请求计数 burst。

## 4. 总体结果

| 策略 | 吞吐 tok/s | Wall s | TTFT mean/P99 s | TPOT mean/P99 ms | E2E mean/P99 s | 稳定排空 s |
|---|---:|---:|---:|---:|---:|---:|
| RR + FCFS | 2,148 | 1,443.0 | 110.8 / 665.5 | 57.5 / 707.1 | 178.7 / 768.1 | 668.4 |
| least-active | 2,638 | 1,175.2 | 103.3 / 351.7 | 101.6 / 1,173.1 | 180.5 / 579.2 | 329.5 |
| token-balance | 2,752 | 1,126.6 | 131.0 / 357.8 | 77.4 / 860.3 | 205.1 / 566.3 | 352.8 |

RR 的 TPOT 中心值看似较好，但它把最严重的拥塞集中在单个 Decode，代价是极高的 TTFT/E2E tail 和两倍左右的恢复时间。该结果不能解释为 RR 的整体流式体验更好。

## 5. 阶段结果

pre 阶段三策略 TTFT P99 均约 0.21 秒、TPOT P99 约 27 ms、无 preemption，说明基础服务性能相当。

burst 请求自身：

| 策略 | TTFT P99 s | TPOT P99 ms | E2E P99 s |
|---|---:|---:|---:|
| RR + FCFS | 179.7 | 74.0 | 933.4 |
| least-active | 202.3 | 70.3 | 714.0 |
| token-balance | 133.6 | 66.4 | 641.9 |

post 请求最能反映 burst 的遗留排队：RR post TTFT P99 为 668.1 秒，least-active 为 355.3 秒，token-balance 为 362.2 秒。least-active 和 token-balance 都避免了 RR 的单卡长尾继续阻塞后续条带请求。

## 6. 放置与机制

burst 阶段累计目标 token：

| 策略 | D0 | D1 | D2 | CV | max/min |
|---|---:|---:|---:|---:|---:|
| RR + FCFS | 548,864 | 540,672 | 804,000 | 19.4% | 1.49 |
| least-active | 628,032 | 606,560 | 658,944 | 3.4% | 1.09 |
| token-balance | 645,216 | 645,216 | 603,104 | 3.1% | 1.07 |

RR 的根因是周期对齐：三卡 RR 周期与 `[12K, 8K, 8K]` 长度周期相同，warmup 后 12K 槽固定落到 D2。least-active 虽然不知道目标长度，但长请求完成慢、active count 下降慢，后续请求会被偏向其他实例，从而形成隐式负反馈。token-balance 直接使用 reserved target output tokens，burst 期间均衡最稳定。

三策略均达到 100% Decode KV 峰值，并出现大量 preemption：RR 3,405 次，least-active 3,110 次，token-balance 3,418 次。token-balance 的 waiting request-seconds 更高并不与其吞吐优势矛盾：它把压力更均匀地铺到三卡，系统范围内同时 waiting 的请求更多，但避免了 RR 的单卡长期尾部热点。

## 7. 有效性与限制

- 三组均完成 2000/2000 请求，无失败、无缺失或重复 ID；实际生成 token 与目标完全一致。
- rate 14 时吞吐仍在上升且 KV 峰值仅 52%，所以 5,592 tok/s 不是严格饱和峰值，只是有限扫描的最大观测值。
- 单 trace、单顺序、单次重复，没有 seed 方差和置信区间。
- burst 是合成、周期性、oracle 长度压力测试；token-balance 使用不可在线准确获得的目标输出长度。
- TPOT 为客户端 streaming 指标，不是引擎内部精确 token scheduling gap。
- 32K 和 16K 探索版本因 RR 形成过于极端、长时间不能排空的热点而排除，不进入正式比较。
- 根据用户要求未使用子 Agent，本次有效性审计不是独立 reviewer 审计。

## 8. 结论

在这份专门针对 RR 周期条带缺陷构造的 1P3D 压力 trace 上：

1. RR 的确会将同一长度槽持续映射到同一 Decode，造成 1.49 倍 token 负载差、长时间 KV 满载和后续 TTFT 积压。
2. least-active 不需要显式输出长度，也能把 burst token 不均衡压到与 token-balance 相近，并在 TTFT mean、TTFT P99 和总 waiting 上略优于 token-balance。
3. token-balance 在吞吐、TPOT、burst 请求 tail 和总体 E2E P99 上略优，说明显式 token 信息仍有价值。
4. 下一步若要判断一般性，应随机化长请求位置与周期，至少运行 3–5 个 arrival/stripe seeds，并把容量扫描延伸到吞吐平台或排队拐点。

## 9. 产物与复现

- 计划：`experiments/1p3d_burst_20260817/EXPERIMENT_PLAN.md`
- 正式汇总：`outputs/1p3d_burst_20260817/formal-summary.json`
- 主表：`outputs/1p3d_burst_20260817/tables/overall.csv`
- 阶段表：`outputs/1p3d_burst_20260817/tables/phase-latency-and-pressure.csv`
- 放置表：`outputs/1p3d_burst_20260817/tables/assignments.csv`
- 时序图：`outputs/1p3d_burst_20260817/figures/waiting-over-time.png`、`kv-over-time.png`
- 审计：`outputs/1p3d_burst_20260817/VALIDITY_AUDIT.md`
- 分析脚本：`experiments/1p3d_burst_20260817/analyze_results.py`

复现分析：

```bash
cd /data/fyj/project/PDA-scheduler/vllm_pd
.venv/bin/python experiments/1p3d_burst_20260817/analyze_results.py
```
