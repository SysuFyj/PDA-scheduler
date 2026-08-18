# 调度代码快照

## 当前行为

- 请求到达时只选择 Prefill。
- Prefill 完成后，基于最新共享状态选择 Decode。
- `default_fcfs`：跨 Decode Round-Robin，实例内部使用 vLLM FCFS。
- `least_active`：选择 unfinished request count 最小的 Decode。
- `token_balance`：选择 reserved target output tokens 总量最小的 Decode。

## 代码结构

- `src/vllm_pd_router/decode_scheduling.py`：metric provider、feasibility filter、selector 和 reservation manager。
- `src/vllm_pd_router/scheduler.py`：Prefill/Decode 两阶段绑定与共享状态。
- `src/vllm_pd_router/scheduler_service.py`：多 Router worker 共享 scheduler 控制面。
- `src/vllm_pd_router/proxy.py`：Prefill 执行、Decode 后置选择、决策日志和请求转发。
- `src/vllm_pd_router/config.py`：策略与 reservation timeout 配置。

## 观测能力

- 每次 Decode 选择记录 active count、completed snapshot、策略 score 和 reserved output tokens。
- dispatch、completion、failure 和 reservation timeout 使用同一 reservation 状态账本。
- 微突发实验可临时记录 vLLM 请求级 preemption 与被清零前 computed tokens，运行后恢复环境文件。

## 测试

- 配置与策略校验。
- Prefill 后置 Decode 选择。
- 最新共享状态与多 worker scheduler。
- RR、least-active、token-balance 选择语义。
- reservation 幂等释放与超时回收。
- completed snapshot 观测。
- 微突发到达窗口、普通流量三元组和 vLLM instrumentation 恢复。

阶段提交前全量测试：27 项通过。
