from __future__ import annotations

import argparse
import contextlib
import csv
import fcntl
import json
import os
import re
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Iterator


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = Path(__file__).resolve().parent
SHARED_EXPERIMENT = ROOT / "experiments/2p4d_kv_striped_20260815"
sys.path.insert(0, str(EXPERIMENT_DIR))
sys.path.insert(0, str(SHARED_EXPERIMENT))

from run_experiment import (  # noqa: E402
    BASE_TRACE,
    BENCHMARK,
    ENV,
    MODEL,
    atomic_json,
    environment,
    make_config,
    monitor,
    run_benchmark,
    start_cluster,
    stop_cluster,
    summarize_metrics,
    write_jsonl,
)
from vllm_preemption_instrumentation import instrument  # noqa: E402


GPU_MAP = ("1", "2", "3", "4")
PREFILL_COUNT = 1
BACKGROUND_RATE = 5.840661277496174
BURST_WINDOW_S = 2.0
FORMAL_STRATEGIES = ("default_fcfs", "least_active", "token_balance")
ROUTE_PATTERN = re.compile(
    r"route request_id=(\S+) strategy=(\S+) prefill=(\S+) decode=(\S+).*"
    r"decode_completed_before=\(([^)]*)\).*reserved_output_tokens=(\d+)"
)
PREEMPT_PATTERN = re.compile(
    r"PD_REQUEST_PREEMPT request_id=(\S+) computed_tokens=(\d+) preemption_count=(\d+)"
)


def original_output_tokens(row: dict[str, Any]) -> int:
    return int(row.get("oracle_output_tokens") or row.get("qwen_generated_tokens") or 1)


def load_source() -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in BASE_TRACE.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != 2000:
        raise ValueError(f"Expected 2000 source rows, found {len(rows)}")
    return rows


def ordered_background(source: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    long_first = [row for row in source if original_output_tokens(row) >= 1024][:60]
    selected_ids = {id(row) for row in long_first}
    remaining = [row for row in source if id(row) not in selected_ids]
    result = (long_first + remaining)[:count]
    if len(result) != count:
        raise ValueError(f"Could not build {count} background requests")
    return result


def background_offsets(count: int) -> list[float]:
    triplet_centers = (0.24, 0.72, 1.20, 1.68)
    offsets = [center + member * 0.0005 for center in triplet_centers for member in range(3)]
    offset = BURST_WINDOW_S + 1.0 / BACKGROUND_RATE
    while len(offsets) < count:
        offsets.append(offset)
        offset += 1.0 / BACKGROUND_RATE
    return offsets[:count]


def burst_offsets(cycles: int) -> list[float]:
    offsets = []
    for cycle in range(cycles):
        start = cycle * BURST_WINDOW_S / cycles
        offsets.extend(start + member * 0.0005 for member in range(3))
    if offsets[-1] >= BURST_WINDOW_S:
        raise AssertionError("Burst escaped the two-second window")
    return offsets


def build_trace(
    source: list[dict[str, Any]], path: Path, background_count: int, cycles: int
) -> dict[str, Any]:
    background_source = ordered_background(source, background_count)
    used_ids = {id(row) for row in background_source}
    burst_source = [row for row in source if id(row) not in used_ids][: cycles * 3]
    combined: list[tuple[float, int, dict[str, Any]]] = []
    for index, (row, offset) in enumerate(zip(background_source, background_offsets(background_count))):
        item = dict(row)
        item["request_id"] = f"background-{index:04d}"
        item["target_output_tokens"] = original_output_tokens(row)
        item["phase"] = "background"
        item["stream_type"] = "background"
        item["trace_class"] = f"background_output_{item['target_output_tokens']}"
        item["arrival_offset_s"] = offset
        combined.append((offset, 1, item))
    pattern = (16000, 512, 512)
    for index, (row, offset) in enumerate(zip(burst_source, burst_offsets(cycles))):
        item = dict(row)
        output_tokens = pattern[index % 3]
        item["request_id"] = f"microburst-{index:03d}"
        item["target_output_tokens"] = output_tokens
        item["phase"] = "microburst"
        item["stream_type"] = "burst_long" if output_tokens == 16000 else "burst_short"
        item["trace_class"] = f"microburst_output_{output_tokens}"
        item["arrival_offset_s"] = offset
        combined.append((offset, 0, item))
    combined.sort(key=lambda entry: (entry[0], entry[1]))
    rows = [entry[2] for entry in combined]
    write_jsonl(path, rows)
    ideal_requests = cycles
    ideal_kv_tokens = (cycles // 3) * 16000 + (2 * cycles // 3) * 512 + ideal_requests * 131
    metadata = {
        "source": str(BASE_TRACE),
        "background_requests": background_count,
        "background_rate": BACKGROUND_RATE,
        "burst_cycles": cycles,
        "burst_requests": cycles * 3,
        "burst_window_s": BURST_WINDOW_S,
        "burst_pattern": list(pattern),
        "last_burst_offset_s": max(row["arrival_offset_s"] for row in rows if row["phase"] == "microburst"),
        "last_arrival_offset_s": rows[-1]["arrival_offset_s"],
        "ideal_per_decode_burst_requests": ideal_requests,
        "ideal_per_decode_kv_tokens": ideal_kv_tokens,
        "ideal_per_decode_kv_fraction": ideal_kv_tokens / 126576,
        "total_requests": len(rows),
        "total_target_output_tokens": sum(int(row["target_output_tokens"]) for row in rows),
    }
    atomic_json(path.with_suffix(".metadata.json"), metadata)
    return metadata


def build_warmup(source: list[dict[str, Any]], path: Path) -> None:
    rows = []
    for index, source_row in enumerate(source[:8]):
        row = dict(source_row)
        row["request_id"] = f"warmup-{index:02d}"
        row["target_output_tokens"] = 16
        row["phase"] = "warmup"
        rows.append(row)
    write_jsonl(path, rows)


def parse_completed_tuple(text: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in text.split(",") if item.strip())


def parse_routes(router_log: Path) -> dict[str, Any]:
    routes = []
    assignments: dict[str, dict[str, int]] = {}
    for line in router_log.read_text(encoding="utf-8", errors="replace").splitlines():
        match = ROUTE_PATTERN.search(line)
        if not match:
            continue
        request_id, strategy, _, worker, completed_text, token_text = match.groups()
        if not request_id.startswith("microburst-"):
            continue
        token_count = int(token_text)
        routes.append(
            {
                "request_id": request_id,
                "strategy": strategy,
                "worker": worker,
                "completed_before": parse_completed_tuple(completed_text),
                "target_output_tokens": token_count,
            }
        )
        worker_state = assignments.setdefault(worker, {"requests": 0, "long": 0, "short": 0, "output_tokens": 0})
        worker_state["requests"] += 1
        worker_state["long" if token_count == 16000 else "short"] += 1
        worker_state["output_tokens"] += token_count
    snapshots = [route["completed_before"] for route in routes]
    return {
        "routes": routes,
        "assignments": assignments,
        "strict_completed_gate": bool(snapshots) and all(snapshot == snapshots[0] for snapshot in snapshots),
        "first_completed_snapshot": snapshots[0] if snapshots else None,
        "last_completed_snapshot": snapshots[-1] if snapshots else None,
    }


def external_request_id(engine_request_id: str) -> str | None:
    match = re.search(r"(microburst-\d{3}|background-\d{4})", engine_request_id)
    return match.group(1) if match else None


def parse_preemptions(service_dir: Path) -> dict[str, Any]:
    per_request: dict[str, dict[str, int]] = {}
    records = []
    for log_path in sorted(service_dir.glob("D*.log")):
        worker = log_path.stem
        for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
            match = PREEMPT_PATTERN.search(line)
            if not match:
                continue
            engine_id, computed_text, count_text = match.groups()
            request_id = external_request_id(engine_id)
            if request_id is None:
                continue
            computed_tokens = int(computed_text)
            record = {
                "worker": worker,
                "request_id": request_id,
                "computed_tokens": computed_tokens,
                "preemption_count": int(count_text),
            }
            records.append(record)
            state = per_request.setdefault(request_id, {"preemptions": 0, "recompute_tokens_approx": 0})
            state["preemptions"] += 1
            state["recompute_tokens_approx"] += computed_tokens
    return {
        "records": records,
        "per_request": per_request,
        "total_preemptions": len(records),
        "total_recompute_tokens_approx": sum(record["computed_tokens"] for record in records),
    }


def percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    position = (len(values) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    fraction = position - lower
    return values[lower] * (1 - fraction) + values[upper] * fraction


def latency_classes(request_csv: Path) -> dict[str, Any]:
    rows = list(csv.DictReader(request_csv.open(newline="", encoding="utf-8")))
    result: dict[str, Any] = {}
    selectors = {
        "all": lambda row: True,
        "background": lambda row: row["phase"] == "background",
        "burst_long": lambda row: row["stream_type"] == "burst_long",
        "burst_short": lambda row: row["stream_type"] == "burst_short",
    }
    for name, selector in selectors.items():
        selected = [row for row in rows if row["success"] == "True" and selector(row)]
        item: dict[str, Any] = {"completed": len(selected)}
        for metric in ("ttft_s", "tpot_s", "e2e_s"):
            values = [float(row[metric]) for row in selected if row[metric]]
            item[metric] = {
                "mean": statistics.mean(values) if values else None,
                "p95": percentile(values, 0.95),
                "p99": percentile(values, 0.99),
            }
        result[name] = item
    return result


def vllm_scheduler_path() -> Path:
    python_dir = f"python{sys.version_info.major}.{sys.version_info.minor}"
    path = ENV / "lib" / python_dir / "site-packages/vllm/v1/core/sched/scheduler.py"
    if not path.is_file():
        raise RuntimeError(f"vLLM scheduler path does not exist: {path}")
    return path


def gpu_processes() -> dict[int, list[dict[str, Any]]]:
    gpu_rows = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"],
        text=True,
        capture_output=True,
        check=True,
    ).stdout.splitlines()
    uuid_to_index = {uuid.strip(): int(index.strip()) for index, uuid in (row.split(",", 1) for row in gpu_rows)}
    process_rows = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory", "--format=csv,noheader,nounits"],
        text=True,
        capture_output=True,
        check=True,
    ).stdout.splitlines()
    result: dict[int, list[dict[str, Any]]] = {int(gpu): [] for gpu in GPU_MAP}
    for row in process_rows:
        gpu_uuid, pid_text, memory_text = (item.strip() for item in row.split(",", 2))
        index = uuid_to_index.get(gpu_uuid)
        if index not in result:
            continue
        pid = int(pid_text)
        owner = subprocess.run(["ps", "-o", "user=", "-p", str(pid)], text=True, capture_output=True).stdout.strip()
        result[index].append({"pid": pid, "owner": owner, "memory_mib": int(memory_text)})
    return result


@contextlib.contextmanager
def gpu_locks() -> Iterator[None]:
    before = gpu_processes()
    if any(before.values()):
        raise RuntimeError(f"Target GPUs already have compute processes: {before}")
    lock_dir = Path("/tmp/pda-scheduler-gpu-locks")
    lock_dir.mkdir(parents=True, exist_ok=True)
    handles = []
    try:
        for gpu in GPU_MAP:
            handle = (lock_dir / f"gpu-{gpu}.lock").open("w")
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            handles.append(handle)
        after = gpu_processes()
        if any(after.values()):
            raise RuntimeError(f"Target GPUs became occupied after locking: {after}")
        yield
    finally:
        for handle in reversed(handles):
            fcntl.flock(handle, fcntl.LOCK_UN)
            handle.close()


def run_group(
    strategy: str, trace: Path, warmup: Path, output_root: Path, expected_burst: int, timeout_s: float
) -> dict[str, Any]:
    config = make_config(output_root, strategy, GPU_MAP, PREFILL_COUNT)
    service_dir = output_root / "service"
    started = False
    try:
        start_cluster(config, service_dir, output_root / "startup.log")
        started = True
        run_benchmark(warmup, 2.0, output_root / "warmup", timeout_s=300)
        run_dir = output_root / "run"
        run_dir.mkdir(parents=True, exist_ok=True)
        stop_event = threading.Event()
        monitor_thread = threading.Thread(
            target=monitor,
            args=(config, run_dir / "metrics.jsonl", stop_event),
            daemon=True,
        )
        monitor_thread.start()
        try:
            benchmark_dir = run_dir / "benchmark"
            command = [
                str(ENV / "bin/python"),
                str(BENCHMARK),
                "--model",
                str(MODEL),
                "--trace",
                str(trace),
                "--request-rate",
                "1.0",
                "--output-dir",
                str(benchmark_dir),
                "--max-concurrency",
                "2048",
                "--timeout-s",
                str(timeout_s),
                "--ignore-eos",
            ]
            completed = subprocess.run(
                command,
                cwd=ROOT,
                env=environment(),
                check=False,
                timeout=timeout_s,
            )
            summary_path = benchmark_dir / "summary.json"
            if not summary_path.exists():
                raise RuntimeError(
                    f"Benchmark failed before summary creation: rc={completed.returncode}"
                )
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary["returncode"] = completed.returncode
        finally:
            stop_event.set()
            monitor_thread.join(timeout=10)
        routes = parse_routes(service_dir / "router.log")
        preemptions = parse_preemptions(service_dir)
        summary.update(
            {
                "strategy": strategy,
                "routes": routes,
                "preemptions": preemptions,
                "metrics": summarize_metrics(output_root / "run/metrics.jsonl"),
                "latency_classes": latency_classes(output_root / "run/benchmark/requests.csv"),
                "burst_route_count_gate": len(routes["routes"]) == expected_burst,
            }
        )
        atomic_json(output_root / "group-summary.json", summary)
        return summary
    finally:
        if started or service_dir.exists():
            stop_cluster(service_dir)


def smoke_passed(summary: dict[str, Any], cycles: int) -> bool:
    assignments = summary["routes"]["assignments"]
    return (
        summary["submitted"] == summary["completed"]
        and summary["failed"] == 0
        and summary["burst_route_count_gate"]
        and summary["routes"]["strict_completed_gate"]
        and sorted(worker.get("long", 0) for worker in assignments.values()) == [cycles // 3] * 3
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs/1p3d_microburst_20260817")
    parser.add_argument("--smoke-only", action="store_true")
    args = parser.parse_args()
    output_root = args.output_root.resolve()
    trace_dir = output_root / "traces"
    trace_dir.mkdir(parents=True, exist_ok=True)
    source = load_source()
    warmup = trace_dir / "warmup_8.jsonl"
    smoke_trace = trace_dir / "smoke_300_background_18cycles.jsonl"
    formal_trace = trace_dir / "formal_1200_background_21cycles.jsonl"
    build_warmup(source, warmup)
    smoke_metadata = build_trace(source, smoke_trace, background_count=300, cycles=18)
    formal_metadata = build_trace(source, formal_trace, background_count=1200, cycles=21)
    manifest = {
        "started_at": time.time(),
        "status": "running",
        "gpu_map": GPU_MAP,
        "topology": "1P3D",
        "decode_gpu_memory_utilization": 0.55,
        "smoke_metadata": smoke_metadata,
        "formal_metadata": formal_metadata,
    }
    atomic_json(output_root / "manifest.json", manifest)
    instrumented = None
    try:
        with gpu_locks():
            instrumented = instrument(vllm_scheduler_path())
            manifest["vllm_instrumentation"] = {
                "path": str(instrumented.path),
                "original_sha256": instrumented.original_sha256,
                "instrumented_sha256": instrumented.instrumented_sha256,
            }
            atomic_json(output_root / "manifest.json", manifest)
            smoke = run_group(
                "token_balance",
                smoke_trace,
                warmup,
                output_root / "smoke/token_balance",
                expected_burst=54,
                timeout_s=720,
            )
            manifest["smoke_passed"] = smoke_passed(smoke, cycles=18)
            atomic_json(output_root / "manifest.json", manifest)
            if not manifest["smoke_passed"]:
                raise RuntimeError("Smoke validity gate failed; formal matrix was not started")
            if not args.smoke_only:
                formal = []
                for strategy in FORMAL_STRATEGIES:
                    summary = run_group(
                        strategy,
                        formal_trace,
                        warmup,
                        output_root / f"formal/{strategy}",
                        expected_burst=63,
                        timeout_s=1800,
                    )
                    formal.append(summary)
                    atomic_json(output_root / "formal-summary.json", formal)
            manifest["status"] = "completed"
    except BaseException as error:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        if instrumented is not None:
            instrumented.restore()
            manifest["vllm_instrumentation_restored"] = True
        manifest["finished_at"] = time.time()
        atomic_json(output_root / "manifest.json", manifest)


if __name__ == "__main__":
    main()
