# 开发规范

## 修改原则

1. 保持 Router 边界简单：Prefill 到达时选 Prefill，Prefill 成功后选 Decode。
2. 不把 vLLM Engine 内部状态伪装成 Router 已知状态；代理指标必须在文档中说明语义和局限。
3. 不在 Router 中复制 vLLM scheduler、KV allocator 或 continuous batching 逻辑。
4. 请求失败、取消、超时、重试和流结束路径都必须释放对应状态。
5. 不提交模型、虚拟环境、运行输出、日志、缓存和机器特定绝对路径。

## 策略规范

- 公共策略名只能是 `default_fcfs`、`least_active`、`token_balance`。
- 新策略必须添加选择行为测试、失败释放测试和配置解析测试。
- 策略选择不得改变 Prefill/Decode 协议格式。
- 策略日志必须包含请求 ID、Prefill、Decode、选择前分数和预留量。

## 测试门禁

提交前运行：

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
python -m compileall -q src tests
```

如果修改 streaming、状态回收或共享 Scheduler，必须覆盖：

- non-streaming 成功；
- streaming 成功并读到 EOF；
- Prefill 失败时不保留 Decode；
- Decode 失败、取消和完成都释放状态；
- 多 Router worker 共享状态。

## 提交规范

提交标题使用动词开头，例如：

```text
Add token-balance routing test
Fix decode state release on stream EOF
Document PD trace provenance
```

每个提交只解决一个主题，不提交实验日志或临时调试文件。
