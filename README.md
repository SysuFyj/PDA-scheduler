# Minimal vLLM PD Router

这是一个尽量贴近 vLLM 默认服务行为的 Prefill/Decode 分离 Router。
Router 只负责两件事：请求到达时选择 Prefill，Prefill 完成后选择 Decode。

## 支持策略

配置中的 `router.strategy` 只接受以下三个名称：

- `default_fcfs`：Round-Robin 跨 Decode 实例放置，实例内部保持 vLLM 默认 FCFS。
- `least_active`：选择 Router 记录的未完成 Decode 请求数最少的实例。
- `token_balance`：选择 Router 记录的未完成请求原始输出 token 预留总量最少的实例。

`token_balance` 是简单的请求级预留策略，不声称等价于 vLLM 的真实 KV 使用量或剩余 token 工作量。
所有策略都在 Prefill 完成后才进行 Decode 选择。

Decode 调度内部按 `metric → feasibility filter → selector → atomic reservation`
执行。请求完成或失败时按 reservation ID 幂等释放；可通过
`router.reservation_timeout_s` 启用惰性超时回收。该配置默认关闭，避免对超长
Decode 请求引入行为变化；启用后，过期 reservation 会在下一次调度、状态查询或
显式回收时释放并计为失败。

## 项目结构

```text
src/vllm_pd_router/  Router、共享调度器、NIXL 协议和本地启动器
configs/             2P4D 示例配置
scripts/             启停、状态和 smoke 请求脚本
tests/               标准库 unittest 测试
data/traces/         三份可复现实验 trace
docs/                已完成实验的计划和实现清单
```

## 安装

先准备与机器 CUDA/驱动匹配的 vLLM 和 NIXL 环境，再安装 Router：

```bash
cd vllm_pd
python -m pip install -e .
```

不要把模型、虚拟环境、运行日志或结果提交到 Git。模型路径通过环境变量提供：

```bash
export VLLM_PD_MODEL=/path/to/Qwen2.5-7B-Instruct
```

## 运行

默认配置使用 GPU 1–6 的 2P4D 拓扑：

```bash
./scripts/start.sh configs/2p4d.yaml outputs/local
./scripts/status.sh outputs/local
./scripts/smoke_request.sh
./scripts/stop.sh outputs/local
```

也可以使用单 Router worker 配置 `configs/2p4d_single_router.yaml`。配置中的 GPU、端口和模型路径必须按本机环境修改。

## 测试

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

## 设计边界

- 不实现 vLLM Engine 内部 scheduler；Decode 实例内部仍由 vLLM 负责 FCFS、continuous batching 和 KV 管理。
- Router 状态只在请求分配、Prefill 完成和 Decode 流结束时更新。
- SSE 客户端收到 `[DONE]` 后仍必须继续消费到 EOF，确保服务端 Decode 完成回调执行。
- 任何正式实验都必须记录请求完整性、策略决策、到达序列和清理状态。

## 数据

`data/traces/` 中的 trace 来自本地 PDA Scheduler 实验，仅用于复现实验和开发测试。trace 中的模型输出长度是工作负载元数据，不代表 Router 能在线获得真实剩余输出长度。
