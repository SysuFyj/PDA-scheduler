# Implementation Manifest

- Experiment: vLLM deterministic 1P3D strategy scan
- Date: 2026-08-11
- GPUs: 0-3 (P on GPU 0; D on GPUs 1-3)
- Trace: `sample_1_2000.jsonl`
- Rates: 6, 8, 10, 12, 14 requests/s
- Strategies: least active, output-token balance, round-robin + vLLM FCFS
- Requests per group: 1 warmup + 2,000 measured
- Groups: 15
- Deterministic controls: seed 0, temperature 0, eager execution, async scheduling disabled, prefix caching disabled, Triton attention
- Decode GPU memory utilization: 0.45
- Decode KV capacity: 52,976 tokens per instance
- Result root: `/data/fyj/project/PDA-scheduler/outputs/vllm_pd_deterministic_1p3d_scan_20260811/formal`
- Reconciliation: 30,000 submitted, 30,000 completed, 0 failed
- Scheduler decisions: 30,015, including warmups
- Retries/timeouts: none
- Arrival schedules: identical across strategies at each rate
- Residual processes/ports: none after completion
- GPU locks: released after completion
- Validity decision: approved

## Runtime Fixes Applied Before Formal Run

- The benchmark now consumes each streaming response through EOF after receiving `[DONE]`, allowing the router response generator to execute its decode-finish callback.
- The case runner waits for shared scheduler counters to drain before recording final validity state.
