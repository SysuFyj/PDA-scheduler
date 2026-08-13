from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .config import ClusterConfig, WorkerConfig, load_config


def _model_path(config: ClusterConfig) -> str:
    model = Path(config.model).expanduser()
    if model.is_absolute():
        return str(model)
    if config.source_path is not None:
        relative_to_config = (config.source_path.parent / model).resolve()
        if relative_to_config.exists():
            return str(relative_to_config)
    return config.model


def _engine_command(config: ClusterConfig, worker: WorkerConfig, role: str) -> list[str]:
    parsed = urlparse(worker.url)
    memory = (
        config.engine.prefill_gpu_memory_utilization
        if role == "prefill"
        else config.engine.decode_gpu_memory_utilization
    )
    kv_role = "kv_producer" if role == "prefill" else "kv_consumer"
    command = [
        "vllm",
        "serve",
        _model_path(config),
        "--host",
        parsed.hostname or config.engine.host,
        "--port",
        str(parsed.port),
        "--tensor-parallel-size",
        str(config.engine.tensor_parallel_size),
        "--max-model-len",
        str(config.engine.max_model_len),
        "--max-num-seqs",
        str(config.engine.max_num_seqs),
        "--gpu-memory-utilization",
        str(memory),
        "--kv-transfer-config",
        json.dumps({"kv_connector": "NixlConnector", "kv_role": kv_role}),
    ]
    command.extend(config.engine.common_args)
    command.extend(config.engine.prefill_args if role == "prefill" else config.engine.decode_args)
    command.extend(worker.extra_args)
    return command


def _worker_environment(worker: WorkerConfig) -> dict[str, str]:
    environment = dict(os.environ)
    if worker.gpu:
        environment["CUDA_VISIBLE_DEVICES"] = worker.gpu
    if worker.side_channel_port is not None:
        environment["VLLM_NIXL_SIDE_CHANNEL_PORT"] = str(worker.side_channel_port)
    if worker.transfer_host:
        environment["VLLM_NIXL_SIDE_CHANNEL_HOST"] = worker.transfer_host
    environment.setdefault("PYTHONUNBUFFERED", "1")
    return environment


def _healthy(url: str) -> bool:
    try:
        with urllib.request.urlopen(f"{url}/health", timeout=2) as response:
            return 200 <= response.status < 300
    except (OSError, urllib.error.URLError):
        return False


def _wait_for_workers(config: ClusterConfig, processes: list[subprocess.Popen[Any]]) -> None:
    workers = (*config.prefill, *config.decode)
    deadline = time.monotonic() + config.engine.startup_timeout_s
    pending = {worker.worker_id: worker for worker in workers}
    while pending and time.monotonic() < deadline:
        for process in processes:
            if process.poll() is not None:
                raise RuntimeError(f"vLLM worker exited during startup with code {process.returncode}")
        for worker_id, worker in list(pending.items()):
            if _healthy(worker.url):
                print(f"ready {worker_id} {worker.url}", flush=True)
                pending.pop(worker_id)
        if pending:
            time.sleep(2)
    if pending:
        names = ", ".join(pending)
        raise TimeoutError(f"Timed out waiting for vLLM workers: {names}")


def _read_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _terminate(pid: int, timeout_s: float = 20.0) -> None:
    if not _pid_alive(pid):
        return
    try:
        os.killpg(pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if not _pid_alive(pid):
            return
        time.sleep(0.25)
    try:
        os.killpg(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def start(config_path: str, state_dir: str, no_router: bool) -> None:
    config = load_config(config_path)
    output_dir = Path(state_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    state_path = output_dir / "pids.json"
    existing = _read_state(state_path)
    live = [item for item in existing.get("processes", []) if _pid_alive(int(item["pid"]))]
    if live:
        raise RuntimeError(f"Cluster already has live processes in {state_path}")

    process_records: list[dict[str, Any]] = []
    processes: list[subprocess.Popen[Any]] = []
    try:
        for role, workers in (("prefill", config.prefill), ("decode", config.decode)):
            for worker in workers:
                log_path = output_dir / f"{worker.worker_id}.log"
                log_handle = log_path.open("ab", buffering=0)
                command = _engine_command(config, worker, role)
                print(f"starting {worker.worker_id}: {' '.join(command)}", flush=True)
                process = subprocess.Popen(
                    command,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    env=_worker_environment(worker),
                    start_new_session=True,
                )
                processes.append(process)
                process_records.append(
                    {
                        "id": worker.worker_id,
                        "role": role,
                        "pid": process.pid,
                        "url": worker.url,
                        "log": str(log_path),
                        "command": command,
                    }
                )
        _wait_for_workers(config, processes)

        if not no_router:
            if config.router.workers > 1:
                scheduler_log = output_dir / "scheduler.log"
                scheduler_handle = scheduler_log.open("ab", buffering=0)
                scheduler_command = [
                    sys.executable,
                    "-m",
                    "vllm_pd_router.scheduler_service",
                    "--config",
                    str(Path(config_path).expanduser().resolve()),
                ]
                scheduler_process = subprocess.Popen(
                    scheduler_command,
                    stdout=scheduler_handle,
                    stderr=subprocess.STDOUT,
                    env=dict(os.environ, PYTHONUNBUFFERED="1"),
                    start_new_session=True,
                )
                processes.append(scheduler_process)
                process_records.append(
                    {
                        "id": "scheduler",
                        "role": "scheduler",
                        "pid": scheduler_process.pid,
                        "url": f"http://127.0.0.1:{config.router.scheduler_port}",
                        "log": str(scheduler_log),
                        "command": scheduler_command,
                    }
                )
                deadline = time.monotonic() + 60
                while time.monotonic() < deadline and not _healthy(
                    f"http://127.0.0.1:{config.router.scheduler_port}"
                ):
                    if scheduler_process.poll() is not None:
                        raise RuntimeError(
                            f"Scheduler exited with code {scheduler_process.returncode}"
                        )
                    time.sleep(0.2)
                if not _healthy(
                    f"http://127.0.0.1:{config.router.scheduler_port}"
                ):
                    raise TimeoutError("Timed out waiting for shared scheduler")

            router_log = output_dir / "router.log"
            router_handle = router_log.open("ab", buffering=0)
            command = [
                sys.executable,
                "-m",
                "vllm_pd_router.proxy",
                "--config",
                str(Path(config_path).expanduser().resolve()),
            ]
            process = subprocess.Popen(
                command,
                stdout=router_handle,
                stderr=subprocess.STDOUT,
                env=dict(os.environ, PYTHONUNBUFFERED="1"),
                start_new_session=True,
            )
            processes.append(process)
            process_records.append(
                {
                    "id": "router",
                    "role": "router",
                    "pid": process.pid,
                    "url": f"http://127.0.0.1:{config.router.port}",
                    "log": str(router_log),
                    "command": command,
                }
            )
            deadline = time.monotonic() + 60
            while time.monotonic() < deadline and not _healthy(
                f"http://127.0.0.1:{config.router.port}"
            ):
                if process.poll() is not None:
                    raise RuntimeError(f"Router exited with code {process.returncode}")
                time.sleep(1)
            if not _healthy(f"http://127.0.0.1:{config.router.port}"):
                raise TimeoutError("Timed out waiting for router")

        with state_path.open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "config": str(Path(config_path).expanduser().resolve()),
                    "created_at": time.time(),
                    "processes": process_records,
                },
                handle,
                indent=2,
            )
        print(f"cluster ready; state={state_path}", flush=True)
    except Exception:
        for process in reversed(processes):
            _terminate(process.pid)
        raise


def stop(state_dir: str) -> None:
    output_dir = Path(state_dir).expanduser().resolve()
    state_path = output_dir / "pids.json"
    state = _read_state(state_path)
    for item in reversed(state.get("processes", [])):
        pid = int(item["pid"])
        print(f"stopping {item['id']} pid={pid}", flush=True)
        _terminate(pid)
    if state_path.exists():
        state_path.unlink()


def status(state_dir: str) -> int:
    state_path = Path(state_dir).expanduser().resolve() / "pids.json"
    state = _read_state(state_path)
    if not state:
        print("cluster is not running")
        return 1
    all_alive = True
    for item in state.get("processes", []):
        alive = _pid_alive(int(item["pid"]))
        all_alive = all_alive and alive
        print(f"{item['id']:>8} pid={item['pid']} alive={alive} url={item['url']}")
    return 0 if all_alive else 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Manage a local vLLM PD cluster")
    subparsers = parser.add_subparsers(dest="command", required=True)
    start_parser = subparsers.add_parser("start")
    start_parser.add_argument("--config", required=True)
    start_parser.add_argument("--state-dir", default="outputs/vllm_pd/latest")
    start_parser.add_argument("--no-router", action="store_true")
    stop_parser = subparsers.add_parser("stop")
    stop_parser.add_argument("--state-dir", default="outputs/vllm_pd/latest")
    status_parser = subparsers.add_parser("status")
    status_parser.add_argument("--state-dir", default="outputs/vllm_pd/latest")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "start":
        start(args.config, args.state_dir, args.no_router)
    elif args.command == "stop":
        stop(args.state_dir)
    else:
        raise SystemExit(status(args.state_dir))


if __name__ == "__main__":
    main()
