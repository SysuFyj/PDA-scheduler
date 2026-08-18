from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import signal
import statistics
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

import httpx
import yaml


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = Path(__file__).resolve().parent
ENV = ROOT / ".venv"
MODEL = ROOT.parent / "_staging/20260813/real_pd_experiment/models/Qwen__Qwen2.5-7B-Instruct"
BASE_TRACE = ROOT / "data/traces/sample_1_2000.jsonl"
BASE_CONFIG = ROOT / "configs/2p4d-default-fcfs.yaml"
BENCHMARK = EXPERIMENT_DIR / "benchmark.py"
METRICS = (
    "vllm:kv_cache_usage_perc",
    "vllm:num_requests_running",
    "vllm:num_requests_waiting",
    "vllm:num_preemptions",
)
STRATEGIES = ("default_fcfs", "least_active", "token_balance")


def atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def environment() -> dict[str, str]:
    result = dict(os.environ)
    result["PATH"] = f"{ENV / 'bin'}:{result.get('PATH', '')}"
    result["PYTHONPATH"] = str(ROOT / "src")
    result["PYTHONNOUSERSITE"] = "1"
    result["VLLM_PD_MODEL"] = str(MODEL)
    result.pop("PYTHONHOME", None)
    result.pop("CONDA_PREFIX", None)
    return result


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_trace(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != 2000:
        raise ValueError(f"Expected 2000 requests in {path}, found {len(rows)}")
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")


def build_traces(output_root: Path, decode_count: int) -> dict[str, Path]:
    source = load_trace(BASE_TRACE)
    trace_dir = output_root / "traces"
    capacity = []
    for index, row in enumerate(source[:300]):
        item = dict(row)
        item["request_id"] = f"capacity-{index:04d}"
        item["target_output_tokens"] = 1024
        item["trace_class"] = "capacity_1024"
        capacity.append(item)
    warmup = []
    for index, row in enumerate(source[:8]):
        item = dict(row)
        item["request_id"] = f"warmup-{index:02d}"
        item["target_output_tokens"] = 16
        item["trace_class"] = "warmup"
        warmup.append(item)
    if decode_count == 4:
        pattern = (32000, 128, 128, 128, 8192, 128, 128, 128, 8192, 128, 128, 128, 8192, 128, 128, 128)
    elif decode_count == 2:
        pattern = tuple(
            32000 if index == 0 else (2304 if index % 2 == 0 else 1024)
            for index in range(64)
        )
    else:
        raise ValueError(f"Unsupported decode count: {decode_count}")
    striped = []
    for index, row in enumerate(source):
        item = dict(row)
        output_tokens = pattern[index % len(pattern)]
        item["request_id"] = f"striped-{index:04d}"
        item["target_output_tokens"] = output_tokens
        item["trace_class"] = f"output_{output_tokens}"
        striped.append(item)
    paths = {
        "warmup": trace_dir / "warmup_8.jsonl",
        "capacity": trace_dir / "capacity_300x1024.jsonl",
        "striped": trace_dir / "sample_1_2000_striped.jsonl",
    }
    write_jsonl(paths["warmup"], warmup)
    write_jsonl(paths["capacity"], capacity)
    write_jsonl(paths["striped"], striped)
    atomic_json(
        trace_dir / "trace-summary.json",
        {
            "source": str(BASE_TRACE),
            "source_sha256": sha256(BASE_TRACE),
            "pattern": list(pattern),
            "requests": len(striped),
            "mean_output_tokens": statistics.mean(row["target_output_tokens"] for row in striped),
            "total_output_tokens": sum(row["target_output_tokens"] for row in striped),
            "class_counts": {
                f"output_{token_count}": sum(row["target_output_tokens"] == token_count for row in striped)
                for token_count in sorted(set(pattern))
            },
        },
    )
    return paths


def make_config(
    output_dir: Path,
    strategy: str,
    gpu_map: tuple[str, ...],
    prefill_count: int,
) -> Path:
    config = yaml.safe_load(BASE_CONFIG.read_text(encoding="utf-8"))
    decode_count = len(gpu_map) - prefill_count
    config["prefill"] = config["prefill"][:prefill_count]
    config["decode"] = config["decode"][:decode_count]
    config["model"] = str(MODEL)
    config["router"]["strategy"] = strategy
    config["router"]["workers"] = 1
    config["engine"]["decode_gpu_memory_utilization"] = 0.55
    for worker, gpu in zip(config["prefill"] + config["decode"], gpu_map):
        worker["gpu"] = gpu
    path = output_dir / "config.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def parse_metric(text: str, name: str) -> float | None:
    values: list[float] = []
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        if not (line.startswith(name + "{") or line.startswith(name + " ") or line.startswith(name + "_total")):
            continue
        try:
            values.append(float(line.rsplit(None, 1)[1]))
        except (IndexError, ValueError):
            pass
    return max(values) if values else None


def worker_metrics(client: httpx.Client, worker_id: str, url: str) -> dict[str, Any]:
    response = client.get(f"{url}/metrics")
    response.raise_for_status()
    values = {name: parse_metric(response.text, name) for name in METRICS}
    return {
        "worker": worker_id,
        "kv": values["vllm:kv_cache_usage_perc"],
        "running": values["vllm:num_requests_running"],
        "waiting": values["vllm:num_requests_waiting"],
        "preemptions": values["vllm:num_preemptions"],
    }


def monitor(config_path: Path, output_path: Path, stop_event: threading.Event) -> None:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    started = time.monotonic()
    with httpx.Client(timeout=5) as client, output_path.open("w", encoding="utf-8") as handle:
        while not stop_event.is_set():
            sample: dict[str, Any] = {"elapsed_s": time.monotonic() - started, "workers": []}
            for worker in config["decode"]:
                try:
                    sample["workers"].append(worker_metrics(client, worker["id"], worker["url"]))
                except Exception as error:
                    sample["workers"].append({"worker": worker["id"], "error": f"{type(error).__name__}: {error}"})
            try:
                response = client.get(f"http://127.0.0.1:{config['router']['port']}/router/state")
                response.raise_for_status()
                sample["router"] = response.json()
            except Exception as error:
                sample["router_error"] = f"{type(error).__name__}: {error}"
            handle.write(json.dumps(sample, separators=(",", ":")) + "\n")
            handle.flush()
            stop_event.wait(1.0)


def cluster_command(action: str, config: Path, state_dir: Path) -> list[str]:
    command = [str(ENV / "bin/python"), "-m", "vllm_pd_router.cluster", action]
    if action == "start":
        command.extend(["--config", str(config), "--state-dir", str(state_dir)])
    else:
        command.extend(["--state-dir", str(state_dir)])
    return command


def run_benchmark(trace: Path, rate: float, output_dir: Path, timeout_s: float = 3600) -> dict[str, Any]:
    command = [
        str(ENV / "bin/python"), str(BENCHMARK), "--model", str(MODEL),
        "--trace", str(trace), "--request-rate", str(rate), "--output-dir", str(output_dir),
        "--max-concurrency", "2048", "--timeout-s", str(timeout_s), "--ignore-eos",
    ]
    completed = subprocess.run(command, cwd=ROOT, env=environment(), check=False)
    summary_path = output_dir / "summary.json"
    if not summary_path.exists():
        raise RuntimeError(f"Benchmark failed before summary creation: rc={completed.returncode}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["returncode"] = completed.returncode
    summary["request_rate"] = rate
    return summary


def run_monitored(trace: Path, rate: float, run_dir: Path, config: Path) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    stop_event = threading.Event()
    thread = threading.Thread(target=monitor, args=(config, run_dir / "metrics.jsonl", stop_event), daemon=True)
    thread.start()
    try:
        return run_benchmark(trace, rate, run_dir / "benchmark")
    finally:
        stop_event.set()
        thread.join(timeout=10)


def start_cluster(config: Path, state_dir: Path, log_path: Path) -> None:
    with log_path.open("w", encoding="utf-8") as handle:
        completed = subprocess.run(
            cluster_command("start", config, state_dir), cwd=ROOT, env=environment(),
            stdout=handle, stderr=subprocess.STDOUT, check=False, timeout=1200,
        )
    if completed.returncode:
        raise RuntimeError(f"Cluster startup failed: {log_path}")


def stop_cluster(state_dir: Path) -> None:
    subprocess.run(
        cluster_command("stop", Path(), state_dir), cwd=ROOT, env=environment(),
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False, timeout=180,
    )


def select_capacity(rows: list[dict[str, Any]]) -> float:
    passing = [
        row for row in rows
        if row["success_rate"] == 1.0
        and row["token_throughput"] >= 0.95 * row["offered_token_rate"]
        and (row["ttft_s"]["p99"] or float("inf")) <= 2.0
    ]
    if passing:
        return max(row["offered_token_rate"] for row in passing)
    return max(row["token_throughput"] for row in rows)


def summarize_metrics(path: Path) -> dict[str, Any]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    per_worker: dict[str, dict[str, float | None]] = {}
    waiting_request_seconds = 0.0
    first_waiting_s: float | None = None
    for row in rows:
        elapsed = float(row["elapsed_s"])
        sample_waiting = 0.0
        for worker in row.get("workers", []):
            if "error" in worker:
                continue
            state = per_worker.setdefault(worker["worker"], {"peak_kv": 0.0, "peak_running": 0.0, "peak_waiting": 0.0, "preemptions": 0.0})
            for source, target in (("kv", "peak_kv"), ("running", "peak_running"), ("waiting", "peak_waiting"), ("preemptions", "preemptions")):
                value = worker.get(source)
                if value is not None:
                    state[target] = max(float(state[target] or 0), float(value))
            sample_waiting += float(worker.get("waiting") or 0)
        waiting_request_seconds += sample_waiting
        if sample_waiting > 0 and first_waiting_s is None:
            first_waiting_s = elapsed
    return {
        "samples": len(rows),
        "first_waiting_s": first_waiting_s,
        "waiting_request_seconds_approx": waiting_request_seconds,
        "workers": per_worker,
    }


def parse_assignments(router_log: Path) -> dict[str, Any]:
    pattern = re.compile(r"route request_id=(\S+) strategy=(\S+) prefill=(\S+) decode=(\S+).*reserved_output_tokens=(\d+)")
    counts: dict[str, int] = {}
    tokens: dict[str, int] = {}
    classes: dict[str, dict[str, int]] = {}
    for line in router_log.read_text(encoding="utf-8", errors="replace").splitlines():
        match = pattern.search(line)
        if not match or not match.group(1).startswith("striped-"):
            continue
        request_id, _, _, worker, token_text = match.groups()
        token_count = int(token_text)
        counts[worker] = counts.get(worker, 0) + 1
        tokens[worker] = tokens.get(worker, 0) + token_count
        class_name = f"output_{token_count}"
        classes.setdefault(worker, {})[class_name] = classes.setdefault(worker, {}).get(class_name, 0) + 1
    return {"request_counts": counts, "assigned_tokens": tokens, "class_counts": classes}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs/2p4d_kv_striped_20260815")
    parser.add_argument("--capacity-rates", type=float, nargs="+", default=[3, 4, 5, 6, 8])
    parser.add_argument("--overload-factor", type=float, default=1.05)
    parser.add_argument("--skip-capacity", action="store_true")
    parser.add_argument("--capacity-token-rate", type=float)
    parser.add_argument("--gpu-map", nargs="+", default=["0", "1", "2", "4", "6", "7"])
    parser.add_argument("--prefill-count", type=int, default=2)
    args = parser.parse_args()
    gpu_map = tuple(args.gpu_map)
    decode_count = len(gpu_map) - args.prefill_count
    if (args.prefill_count, decode_count) not in {(2, 4), (1, 2)}:
        raise ValueError("Only 2P4D and 1P2D topologies are supported")
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    traces = build_traces(output_root, decode_count)
    manifest: dict[str, Any] = {
        "started_at": time.time(), "gpu_map": gpu_map, "model": str(MODEL),
        "topology": f"{args.prefill_count}P{decode_count}D",
        "strategies": STRATEGIES, "capacity_rates": args.capacity_rates,
        "overload_factor": args.overload_factor, "status": "running",
    }
    atomic_json(output_root / "manifest.json", manifest)
    capacity_rows: list[dict[str, Any]] = []
    capacity_token_rate = args.capacity_token_rate
    try:
        if not args.skip_capacity:
            capacity_root = output_root / "capacity"
            config = make_config(capacity_root, "default_fcfs", gpu_map, args.prefill_count)
            state_dir = capacity_root / "service"
            start_cluster(config, state_dir, capacity_root / "startup.log")
            try:
                run_benchmark(traces["warmup"], 2.0, capacity_root / "warmup", timeout_s=300)
                for rate in args.capacity_rates:
                    run_dir = capacity_root / f"rate_{rate:g}"
                    summary = run_monitored(traces["capacity"], rate, run_dir, config)
                    summary["offered_token_rate"] = rate * 1024
                    summary["metrics"] = summarize_metrics(run_dir / "metrics.jsonl")
                    capacity_rows.append(summary)
                    atomic_json(output_root / "capacity-summary.json", capacity_rows)
            finally:
                stop_cluster(state_dir)
            capacity_token_rate = select_capacity(capacity_rows)
        if capacity_token_rate is None:
            raise ValueError("--capacity-token-rate is required with --skip-capacity")
        trace_summary = json.loads((output_root / "traces/trace-summary.json").read_text(encoding="utf-8"))
        mean_output_tokens = float(trace_summary["mean_output_tokens"])
        formal_rate = args.overload_factor * capacity_token_rate / mean_output_tokens
        manifest["capacity_token_rate"] = capacity_token_rate
        manifest["formal_request_rate"] = formal_rate
        atomic_json(output_root / "manifest.json", manifest)
        formal_rows = []
        for strategy in STRATEGIES:
            run_root = output_root / "formal" / strategy
            config = make_config(run_root, strategy, gpu_map, args.prefill_count)
            state_dir = run_root / "service"
            start_cluster(config, state_dir, run_root / "startup.log")
            try:
                run_benchmark(traces["warmup"], 2.0, run_root / "warmup", timeout_s=300)
                summary = run_monitored(traces["striped"], formal_rate, run_root / "run", config)
                summary["strategy"] = strategy
                summary["offered_token_rate"] = formal_rate * mean_output_tokens
                summary["metrics"] = summarize_metrics(run_root / "run/metrics.jsonl")
                summary["assignments"] = parse_assignments(state_dir / "router.log")
                formal_rows.append(summary)
                atomic_json(output_root / "formal-summary.json", formal_rows)
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
