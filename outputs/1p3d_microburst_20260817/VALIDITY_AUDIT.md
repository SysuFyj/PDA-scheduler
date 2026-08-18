# 1P3D 异构长度微突发实验有效性审计

日期：2026-08-17

## 审计结论

分类：**VALID_POSITIVE（仅限本次合成、单 trace、单次配对机制实验）**。

实验有效支持核心假设：在最后一个 burst 请求完成 Decode 分配前没有任何 Decode completion 增长时，least-active 无法获得 burst 长度反馈，按 active 请求数维持表面均衡，却将 19/21 个 16K 请求集中到同一 Decode；token-balance 将长请求控制在 8/7/6。

## Gate A：运行完整性 — PASS

- Smoke 完成 354/354，0 失败；三组 Formal 均完成 1263/1263，0 失败。
- 所有请求 ID 唯一，全部以目标长度完成，无 timeout 或缺失终态。
- 每组独立重启服务；实验后 GPU 1–4、端口和登记进程均已清理。
- vLLM 请求级抢占日志补丁已恢复，恢复后 SHA256 与原文件一致。
- 首次路径匹配 attempt 在启动服务前失败，已隔离且不进入聚合。

## Gate B：处理完整性 — PASS

- 三策略 Router 日志均包含 63 个 burst Decode 决策。
- RR：长请求 14/7/0；least-active：19/2/0；token-balance：8/7/6。
- 三策略对 burst 的请求数和目标 token 评分符合各自定义；token-balance 没有按请求数均衡，而是按 outstanding target output token 选择。

## Gate C：机制门禁 — PASS

- 三策略 63 个 burst 决策的 `decode_completed_before` 均恒定为 `(3,3,2)`，该计数来自 8 个 warmup 请求。
- 因此 RR 和 least-active 的 burst 放置过程没有任何 completion 反馈；least-active 的长请求集中不是完成速度反馈造成的。
- RR 和 least-active 的热点卡实际达到 100% KV；token-balance 峰值为 99.84%。
- RR/least-active 峰值 waiting 为 39/48，token-balance 为 1。

## Gate D：指标有效性 — PASS WITH LIMITATION

- 请求级抢占事件直接记录 request ID、抢占次数和清零前 computed tokens。
- recompute token 是“抢占前已计算、随后归零”的工作量近似，不是 GPU kernel 时间。
- 目标 KV 百分比是 `target output + input` 的静态需求估算；vLLM 动态分配 KV，不会一次性预留完整目标长度。
- TPOT 是客户端流式指标，不解释为精确引擎 token scheduling gap。

## Gate E：统计有效性 — PASS FOR THIS PAIRED TRACE

- 三策略使用完全相同的 prompt、目标长度和绝对到达时间。
- 单 trace、单次正式重复，没有 seed 方差或置信区间；禁止推广到随机 arrival order。

## Gate F：外部有效性 — LIMITED

- burst 为 `[16000,512,512]×21` 合成周期。
- 为保持普通负载且不改变 RR 模 3 相位，2 秒窗口内普通请求按四个三请求小组到达；平均为 6 req/s，但不是严格逐请求 uniform。
- 前 60 个普通请求从相同 trace 中重排为输出至少 1024 的请求，以满足严格 completion 门禁；普通长度多重集合不变。
- token-balance 使用目标输出长度，属于 oracle 信息策略。
- 根据用户此前要求，本实验未使用子 Agent；审计不是独立 reviewer 复核。

## 支持与禁止结论

支持：没有 burst completion 反馈时，least-active 可退化为表面请求数均衡并产生比 RR 更严重的长度集中；token-balance 显著降低热点、waiting、抢占、重算和背景请求 TTFT tail。

禁止：不能声称 least-active 在所有 burst 中都比 RR 差；不能把静态目标 KV 百分比当作 vLLM 实时预分配量；不能从单一相位和单次运行推广到随机微突发。
