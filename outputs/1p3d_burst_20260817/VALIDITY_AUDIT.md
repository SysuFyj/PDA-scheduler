# 1P3D 长请求 Burst 实验有效性审计

日期：2026-08-17

## 审计结论

分类：**VALID_POSITIVE（仅限本次合成、单 trace、单次配对运行的机制结论）**。

本实验有效支持以下限定结论：固定 RR 在周期为 3 的长请求条带下产生明显 Decode token 负载热点；least-active 与 token-balance 都显著降低 RR 的排空时间和 TTFT/E2E 尾延迟。实验不支持“least-active 或 token-balance 在一般工作负载下全局最优”的结论。

## Gate A：运行完整性 — PASS

- 三个策略均为 `submitted=2000, completed=2000, failed=0`。
- 每组有 2000 个唯一 request ID，无重复终态。
- 每组实际 completion token 总数与目标 token 总数均为 3,099,988。
- 所有请求均以 `finish_reason=length` 完成，无 timeout、取消或 EOS 提前结束。
- 正式运行后 GPU 1–4 无残留进程，端口 8101/8203/8204/8205/8300 均已释放。

## Gate B：处理完整性 — PASS

- Router 日志包含每个请求的策略名、Decode 选择、选择前 active 数和 reserved output token score。
- RR 的 200 个 burst 请求按 67/66/67 分配，全部 67 个 12K 请求落到 D2。
- least-active 和 token-balance 的决策分别使用 active request count 和 reserved target output tokens。

## Gate C：瓶颈有效性 — PASS WITH LIMITATION

- 三个策略的 Decode KV 峰值均达到 100%，并出现 3,110–3,418 次累计 preemption，说明实验确实触发原生 KV/continuous-batching 压力。
- pre 阶段三策略 TTFT P99 均约 0.21 秒且无 preemption，压力来自 burst 后的 Decode，而非冷启动或 Prefill 主导。
- 容量扫描只到 rate 14；吞吐仍单调上升且 KV 峰值仅 52%。因此 5,592 output tok/s 是“扫描范围内最大观测吞吐”，不是严格证明的系统峰值。burst 的 100% 负载是相对该观测值定义。

## Gate D：指标有效性 — PASS

- 报告分别使用 TTFT、TPOT、E2E、吞吐、waiting、KV 和 preemption。
- TPOT 来自客户端流式 chunk 间隔，只用于端到端流式表现，不解释为精确引擎 token scheduling gap。
- 原聚合的 recovery 字段是首次 waiting 瞬时归零；最终报告改用最后一次正 waiting 后的稳定恢复时间。

## Gate E：统计有效性 — PASS FOR PAIRED TRACE, UNKNOWN FOR GENERALIZATION

- 三策略使用完全相同的 2000 prompts、目标长度和绝对到达时间，适合本 trace 的配对比较。
- 只有一个 trace 顺序和一次正式重复，无 seed 间方差、置信区间或 win count；禁止推广到其他请求顺序或生产分布。

## Gate F：外部有效性 — LIMITED

- burst 输出长度为人为指定的 12K/8192/8192 周期，且 `ignore_eos=true`，属于机制压力测试。
- token-balance 使用目标输出长度，属于 oracle 信息策略；least-active 不直接读取长度。
- 使用本项目 patched router 和 P→D 后置 Decode 选择，不等同于未经修改的上游 vLLM 部署。
- 根据用户要求未使用子 Agent；本审计由执行者同一上下文完成，不构成独立复核。

## 支持与禁止结论

支持：RR 对周期性长度条带敏感；least-active 可通过完成速度的隐式反馈把 burst token CV 从 19.4% 降至 3.4%，接近 token-balance 的 3.1%；两者均将稳定排空时间约减半。

禁止：不能声称 5,592 tok/s 是真实峰值；不能声称 token-balance 或 least-active 全面优于另一方；不能从单 trace 推广到随机长请求 burst 或真实 EOS 输出。
