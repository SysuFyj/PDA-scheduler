from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SHARED_EXPERIMENT = ROOT / "experiments/2p4d_kv_striped_20260815"
sys.path.insert(0, str(SHARED_EXPERIMENT))

from run_experiment import (  # noqa: E402
    BASE_TRACE,
    atomic_json,
    make_config,
    run_benchmark,
    run_monitored,
    start_cluster,
    stop_cluster,
    summarize_metrics,
    write_jsonl,
)


STRATEGIES = ("default_fcfs", "least_active", "token_balance")
GPU_MAP = ("1", "2", "3", "4")
PREFILL_COUNT = 1


def load_source() -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in BASE_TRACE.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != 2000:
        raise ValueError(f"Expected 2000 requests, found {len(rows)}")
    return rows


def original_output_tokens(row: dict[str, Any]) -> int:
    return int(row.get("oracle_output_tokens") or row.get("qwen_generated_tokens") or 1)


def build_warmup(rows: list[dict[str, Any]], path: Path) -> None:
    warmup = []
    for index, source in enumerate(rows[:8]):
        row = dict(source)
        row["request_id"] = f"warmup-{index:02d}"
        row["target_output_tokens"] = 16
        row["phase"] = "warmup"
        warmup.append(row)
    write_jsonl(path, warmup)


def build_scan_trace(rows: list[dict[str, Any]], path: Path, count: int) -> None:
    scan = []
    for index, source in enumerate(rows[:count]):
        row = dict(source)
        row["request_id"] = f"scan-{index:04d}"
        row["target_output_tokens"] = original_output_tokens(row)
        row["phase"] = "scan"
        scan.append(row)
    write_jsonl(path, scan)


def build_burst_trace(rows: list[dict[str, Any]], path: Path, peak: float) -> dict[str, Any]:
    pre_sources = rows[:900]
    burst_sources = rows[900:1100]
    post_sources = rows[1100:]
    baseline_lengths = [original_output_tokens(row) for row in pre_sources + post_sources]
    baseline_mean = statistics.mean(baseline_lengths)
    burst_pattern = (12000, 8192, 8192)
    burst_lengths = [burst_pattern[index % 3] for index in range(200)]
    burst_mean = statistics.mean(burst_lengths)
    baseline_rate = 0.70 * peak / baseline_mean
    burst_rate = peak / burst_mean
    trace: list[dict[str, Any]] = []
    offset = 0.0
    phases: dict[str, dict[str, float | int]] = {}

    def append_phase(name: str, sources: list[dict[str, Any]], lengths: list[int], rate: float) -> None:
        nonlocal offset
        start = offset
        for index, (source, length) in enumerate(zip(sources, lengths)):
            row = dict(source)
            row["request_id"] = f"{name}-{index:04d}"
            row["target_output_tokens"] = length
            row["phase"] = name
            row["stream_type"] = "long_burst" if name == "burst" else "baseline"
            row["trace_class"] = f"{name}_output_{length}"
            row["arrival_offset_s"] = offset
            trace.append(row)
            offset += 1.0 / rate
        phases[name] = {
            "start_s": start,
            "end_s": offset,
            "requests": len(sources),
            "request_rate": rate,
            "mean_output_tokens": statistics.mean(lengths),
            "offered_token_rate": rate * statistics.mean(lengths),
        }

    append_phase("pre", pre_sources, [original_output_tokens(row) for row in pre_sources], baseline_rate)
    append_phase("burst", burst_sources, burst_lengths, burst_rate)
    append_phase("post", post_sources, [original_output_tokens(row) for row in post_sources], baseline_rate)
    write_jsonl(path, trace)
    metadata = {
        "peak_token_throughput": peak,
        "baseline_mean_output_tokens": baseline_mean,
        "burst_pattern": list(burst_pattern),
        "burst_mean_output_tokens": burst_mean,
        "phases": phases,
        "total_requests": len(trace),
        "total_target_tokens": sum(int(row["target_output_tokens"]) for row in trace),
        "last_arrival_offset_s": trace[-1]["arrival_offset_s"],
    }
    atomic_json(path.with_name("burst-trace-metadata.json"), metadata)
    return metadata


def percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    position = (len(values) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    fraction = position - lower
    return values[lower] * (1 - fraction) + values[upper] * fraction


def phase_latency_summary(request_csv: Path) -> dict[str, Any]:
    with request_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    result: dict[str, Any] = {}
    for phase in ("pre", "burst", "post"):
        selected = [row for row in rows if row["phase"] == phase and row["success"] == "True"]
        phase_result: dict[str, Any] = {"completed": len(selected)}
        for key in ("ttft_s", "tpot_s", "e2e_s"):
            values = sorted(float(row[key]) for row in selected if row[key])
            phase_result[key] = {
                "mean": statistics.mean(values) if values else None,
                "p95": percentile(values, 0.95),
                "p99": percentile(values, 0.99),
            }
        result[phase] = phase_result
    return result


def summarize_metric_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    waiting_request_seconds = 0.0
    peak_waiting = 0.0
    peak_kv = 0.0
    peak_running = 0.0
    first_waiting_s = None
    first_counters: dict[str, float] = {}
    last_counters: dict[str, float] = {}
    for sample in samples:
        waiting = 0.0
        for worker in sample.get("workers", []):
            if "error" in worker:
                continue
            worker_id = str(worker["worker"])
            waiting += float(worker.get("waiting") or 0)
            peak_kv = max(peak_kv, float(worker.get("kv") or 0))
            peak_running = max(peak_running, float(worker.get("running") or 0))
            counter = float(worker.get("preemptions") or 0)
            first_counters.setdefault(worker_id, counter)
            last_counters[worker_id] = counter
        waiting_request_seconds += waiting
        peak_waiting = max(peak_waiting, waiting)
        if waiting and first_waiting_s is None:
            first_waiting_s = float(sample["elapsed_s"])
    return {
        "samples": len(samples),
        "waiting_request_seconds_approx": waiting_request_seconds,
        "peak_waiting": peak_waiting,
        "peak_kv": peak_kv,
        "peak_running": peak_running,
        "first_waiting_s": first_waiting_s,
        "preemptions_delta": sum(last_counters.get(worker, value) - value for worker, value in first_counters.items()),
    }


def phase_metric_summary(metrics_path: Path, metadata: dict[str, Any]) -> dict[str, Any]:
    samples = [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    result: dict[str, Any] = {}
    for phase, boundary in metadata["phases"].items():
        start_s = float(boundary["start_s"])
        end_s = float(boundary["end_s"])
        selected = [sample for sample in samples if start_s <= float(sample["elapsed_s"]) < end_s]
        result[phase] = summarize_metric_samples(selected)
    post_end = float(metadata["phases"]["post"]["end_s"])
    drain = [sample for sample in samples if float(sample["elapsed_s"]) >= post_end]
    result["drain"] = summarize_metric_samples(drain)
    result["recovery_to_zero_waiting_s"] = next(
        (
            float(sample["elapsed_s"]) - post_end
            for sample in drain
            if sum(float(worker.get("waiting") or 0) for worker in sample.get("workers", []) if "error" not in worker) == 0
        ),
        None,
    )
    return result


def parse_assignments(router_log: Path) -> dict[str, Any]:
    by_phase: dict[str, dict[str, dict[str, int]]] = {}
    for line in router_log.read_text(encoding="utf-8", errors="replace").splitlines():
        if "route request_id=" not in line:
            continue
        fields = {}
        for item in line.split():
            if "=" in item:
                key, value = item.split("=", 1)
                fields[key] = value.rstrip(",")
        request_id = fields.get("request_id", "")
        phase = request_id.split("-", 1)[0]
        if phase not in {"pre", "burst", "post"}:
            continue
        worker = fields["decode"]
        tokens = int(fields["reserved_output_tokens"])
        state = by_phase.setdefault(phase, {}).setdefault(worker, {"requests": 0, "tokens": 0, "long_12k": 0})
        state["requests"] += 1
        state["tokens"] += tokens
        if tokens == 12000:
            state["long_12k"] += 1
    return by_phase


def run_peak_scan(output_root: Path, scan_trace: Path, warmup_trace: Path, rates: list[float]) -> float:
    summaries = []
    for rate in rates:
        run_root = output_root / "peak_scan" / f"rate_{rate:g}"
        config = make_config(run_root, "token_balance", GPU_MAP, PREFILL_COUNT)
        state_dir = run_root / "service"
        start_cluster(config, state_dir, run_root / "startup.log")
        try:
            run_benchmark(warmup_trace, 2.0, run_root / "warmup", timeout_s=300)
            summary = run_monitored(scan_trace, rate, run_root / "run", config)
            summary["request_rate"] = rate
            summary["metrics"] = summarize_metrics(run_root / "run/metrics.jsonl")
            summaries.append(summary)
            atomic_json(output_root / "peak-scan-summary.json", summaries)
        finally:
            stop_cluster(state_dir)
    return max(float(summary["token_throughput"]) for summary in summaries)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--scan-rates", type=float, nargs="+", default=[6, 8, 10, 12, 14])
    parser.add_argument("--scan-requests", type=int, default=800)
    parser.add_argument("--skip-scan", action="store_true")
    parser.add_argument("--peak-token-throughput", type=float)
    args = parser.parse_args()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    trace_dir = output_root / "traces"
    trace_dir.mkdir(parents=True, exist_ok=True)
    source = load_source()
    warmup_trace = trace_dir / "warmup_8.jsonl"
    scan_trace = trace_dir / f"peak_scan_{args.scan_requests}.jsonl"
    burst_trace = trace_dir / "sample_1_2000_70_burst_70.jsonl"
    build_warmup(source, warmup_trace)
    build_scan_trace(source, scan_trace, args.scan_requests)
    manifest: dict[str, Any] = {
        "started_at": time.time(),
        "status": "running",
        "topology": "1P3D",
        "gpu_map": GPU_MAP,
        "decode_gpu_memory_utilization": 0.55,
        "scan_rates": args.scan_rates,
        "scan_requests": args.scan_requests,
    }
    atomic_json(output_root / "manifest.json", manifest)
    try:
        peak = args.peak_token_throughput
        if not args.skip_scan:
            peak = run_peak_scan(output_root, scan_trace, warmup_trace, args.scan_rates)
        if peak is None:
            raise ValueError("--peak-token-throughput is required with --skip-scan")
        metadata = build_burst_trace(source, burst_trace, peak)
        manifest["peak_token_throughput"] = peak
        manifest["trace_metadata"] = metadata
        atomic_json(output_root / "manifest.json", manifest)
        formal = []
        for strategy in STRATEGIES:
            run_root = output_root / "formal" / strategy
            config = make_config(run_root, strategy, GPU_MAP, PREFILL_COUNT)
            state_dir = run_root / "service"
            start_cluster(config, state_dir, run_root / "startup.log")
            try:
                run_benchmark(warmup_trace, 2.0, run_root / "warmup", timeout_s=300)
                summary = run_monitored(burst_trace, 1.0, run_root / "run", config)
                summary["strategy"] = strategy
                summary["overall_metrics"] = summarize_metrics(run_root / "run/metrics.jsonl")
                summary["phase_metrics"] = phase_metric_summary(run_root / "run/metrics.jsonl", metadata)
                summary["phase_latency"] = phase_latency_summary(run_root / "run/benchmark/requests.csv")
                summary["assignments"] = parse_assignments(state_dir / "router.log")
                formal.append(summary)
                atomic_json(output_root / "formal-summary.json", formal)
            finally:
                stop_cluster(state_dir)
        manifest["status"] = "completed"
        manifest["finished_at"] = time.time()
        atomic_json(output_root / "manifest.json", manifest)
    except BaseException as error:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(error).__name__}: {error}"
        manifest["finished_at"] = time.time()
        atomic_json(output_root / "manifest.json", manifest)
        raise


if __name__ == "__main__":
    main()
