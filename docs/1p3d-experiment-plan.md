# vLLM 1P3D Deterministic Decode Placement Scan

Date: 2026-08-11

## Objective

Compare post-Prefill least-active, reserved-output-token balance, and
round-robin Decode placement under a reduced vLLM Decode KV budget.

## System

- Model: Qwen2.5-7B-Instruct, BF16 KV cache, TP=1.
- Topology: P0 on GPU 0; D0-D2 on GPU 1-3.
- Decode GPU memory utilization: 0.45.
- Four router workers and one shared scheduler for every strategy.
- Decode is selected only after Prefill completes.

## Deterministic-like Configuration

- `--enforce-eager`.
- `--no-async-scheduling`.
- `--no-enable-prefix-caching`.
- `--attention-backend TRITON_ATTN`.
- Engine seed 0, request temperature 0, native FCFS, and EOS ignored.

This configuration reduces optional execution variability but is not claimed
to be identical to SGLang batch-invariant deterministic inference.

## Workload

- Trace: `sample_1_2000.jsonl`.
- Poisson rates: 6, 8, 10, 12, and 14 requests/s.
- Paired arrival seed: 20260810.
- Per run: one 16-token warmup request and 2000 measured requests.
- Output work: reference output length capped at 32768.

## Matrix

- 5 rates x 3 strategies = 15 formal run units.
- Every run unit performs clean launch, health check, warmup, measurement,
  reconciliation, and cleanup.
- Formal hard stop: 18000 seconds.

## Metrics and Gates

- TTFT, TPOT, E2E, throughput, success, failure, and timeout.
- Prefill elapsed time and post-Prefill Decode placement evidence.
- Per-Decode assignment and reserved output-token state.
- Actual KV capacity from all three Decode startup logs.
- Exactly 2000 submitted and completed requests, zero failures, complete
  scheduler decisions, clean counters, no missing IDs, no OOM/NIXL failure,
  and complete service cleanup.

## Approval

The user approved GPU 0-3, topology 1P3D, rates 6/8/10/12/14, three
strategies, 15 groups, and a maximum duration of five hours on
2026-08-11 at 12:37 Asia/Shanghai.

