# 1P3D 70%-Long-Burst-70% Experiment

## Question

在 1P3D、Decode `gpu_memory_utilization=0.55` 下，连续约 200 个长请求形成的 token-load burst 是否会让 RR 产生局部 KV 热点；least-active 能否继续利用隐式长度反馈接近 token-balance。

## Topology

- P0: GPU 1
- D0/D1/D2: GPU 2/3/4
- Qwen2.5-7B-Instruct, TP=1
- `max_model_len=32768`, `max_num_seqs=256`
- Prefill memory utilization 0.90
- Decode memory utilization 0.55
- `temperature=0`, `ignore_eos=true`, `--enforce-eager`

## Phase 1: Peak Scan

- Use all 2000 requests from `sample_1_2000.jsonl`. An initial 800-request scan was excluded because tail drain made it underestimate sustained throughput.
- Preserve the original reference output length.
- Effective uniform rates: 8/10/12/14 req/s. The full-trace scan did not repeat rate 6 because rate 8 was already well below saturation.
- Use token-balance for the scan to avoid placement artifacts.
- Peak throughput is the maximum completed output-token throughput among valid runs.

## Phase 2: Burst Trace

- Exactly 2000 source prompts, each used once.
- Pre phase: requests 0-899, original output distribution, offered token load = 70% of measured peak.
- Burst phase: requests 900-1099, output pattern `[12000, 8192, 8192]`, burst stream offered token load = 100% of measured peak.
- Post phase: requests 1100-1999, original output distribution, offered token load = 70% of measured peak.
- Each phase is uniform; phases are contiguous with no idle gap.
- All strategies use the same generated trace and absolute arrival offsets.

## Strategies

- `default_fcfs`: RR Decode placement + vLLM FCFS.
- `least_active`: minimum unfinished Decode request count.
- `token_balance`: minimum reserved target output tokens.

## Metrics

- Overall and per phase TTFT/TPOT/E2E mean/P95/P99.
- Throughput, success rate, wall time.
- Per Decode assignments and assigned target tokens by phase.
- Peak KV, running, waiting, preemptions.
- Waiting request-seconds during pre/burst/post and recovery time after burst.

## Falsification

- RR hypothesis fails if it does not create larger token/KV imbalance than the other policies.
- Implicit-length hypothesis fails if least-active is materially worse than token-balance in waiting and TTFT tail.
- Token-balance superiority is not claimed unless it improves both mechanism metrics and latency tails.
