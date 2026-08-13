from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import yaml


@dataclass(frozen=True)
class WorkerConfig:
    worker_id: str
    url: str
    gpu: str = ""
    side_channel_port: int | None = None
    transfer_host: str | None = None
    extra_args: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, raw: dict[str, Any], role: str, index: int) -> "WorkerConfig":
        worker_id = str(raw.get("id") or f"{role[0].upper()}{index}")
        url = str(raw.get("url") or "").rstrip("/")
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname or not parsed.port:
            raise ValueError(f"Invalid {role} worker URL for {worker_id}: {url!r}")
        side_channel_port = raw.get("side_channel_port")
        return cls(
            worker_id=worker_id,
            url=url,
            gpu=str(raw.get("gpu") or ""),
            side_channel_port=int(side_channel_port) if side_channel_port is not None else None,
            transfer_host=str(raw["transfer_host"]) if raw.get("transfer_host") else None,
            extra_args=tuple(str(item) for item in raw.get("extra_args", [])),
        )


@dataclass(frozen=True)
class RouterConfig:
    host: str = "0.0.0.0"
    port: int = 8300
    strategy: str = "least_active"
    request_timeout_s: float = 1800.0
    connect_timeout_s: float = 10.0
    max_connections: int = 4096
    api_key: str | None = None
    workers: int = 1
    scheduler_port: int = 8299
    access_log: bool = False


@dataclass(frozen=True)
class EngineConfig:
    host: str = "127.0.0.1"
    tensor_parallel_size: int = 1
    max_model_len: int = 32768
    max_num_seqs: int = 256
    prefill_gpu_memory_utilization: float = 0.90
    decode_gpu_memory_utilization: float = 0.55
    startup_timeout_s: float = 900.0
    common_args: tuple[str, ...] = ()
    prefill_args: tuple[str, ...] = ()
    decode_args: tuple[str, ...] = ()


@dataclass(frozen=True)
class ClusterConfig:
    model: str
    prefill: tuple[WorkerConfig, ...]
    decode: tuple[WorkerConfig, ...]
    router: RouterConfig = field(default_factory=RouterConfig)
    engine: EngineConfig = field(default_factory=EngineConfig)
    source_path: Path | None = None


def _tuple(raw: dict[str, Any], key: str) -> tuple[str, ...]:
    return tuple(str(item) for item in raw.get(key, []))


def load_config(path: str | Path) -> ClusterConfig:
    source_path = Path(path).expanduser().resolve()
    with source_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    model = os.path.expandvars(str(raw.get("model") or "")).strip()
    if not model:
        raise ValueError("Config must define a model path or Hugging Face model name")

    prefill_raw = raw.get("prefill") or []
    decode_raw = raw.get("decode") or []
    if not prefill_raw or not decode_raw:
        raise ValueError("Config must define at least one prefill and one decode worker")
    prefill = tuple(
        WorkerConfig.from_dict(item, "prefill", index)
        for index, item in enumerate(prefill_raw)
    )
    decode = tuple(
        WorkerConfig.from_dict(item, "decode", index)
        for index, item in enumerate(decode_raw)
    )
    ids = [worker.worker_id for worker in (*prefill, *decode)]
    if len(ids) != len(set(ids)):
        raise ValueError("Worker ids must be unique")

    router_raw = raw.get("router") or {}
    router = RouterConfig(
        host=str(router_raw.get("host", "0.0.0.0")),
        port=int(router_raw.get("port", 8300)),
        strategy=str(router_raw.get("strategy", "least_active")),
        request_timeout_s=float(router_raw.get("request_timeout_s", 1800.0)),
        connect_timeout_s=float(router_raw.get("connect_timeout_s", 10.0)),
        max_connections=int(router_raw.get("max_connections", 4096)),
        api_key=str(router_raw["api_key"]) if router_raw.get("api_key") else None,
        workers=int(router_raw.get("workers", 1)),
        scheduler_port=int(router_raw.get("scheduler_port", 8299)),
        access_log=bool(router_raw.get("access_log", False)),
    )
    supported = {
        "default_fcfs",
        "least_active",
        "token_balance",
    }
    if router.strategy not in supported:
        raise ValueError(
            f"Unsupported strategy: {router.strategy!r}; expected "
            f"one of {sorted(supported)}"
        )
    router = RouterConfig(
        host=router.host,
        port=router.port,
        strategy=router.strategy,
        request_timeout_s=router.request_timeout_s,
        connect_timeout_s=router.connect_timeout_s,
        max_connections=router.max_connections,
        api_key=router.api_key,
        workers=router.workers,
        scheduler_port=router.scheduler_port,
        access_log=router.access_log,
    )
    if router.workers < 1:
        raise ValueError("router.workers must be at least 1")
    if router.workers > 1 and router.scheduler_port == router.port:
        raise ValueError("router.scheduler_port must differ from router.port")

    engine_raw = raw.get("engine") or {}
    engine = EngineConfig(
        host=str(engine_raw.get("host", "127.0.0.1")),
        tensor_parallel_size=int(engine_raw.get("tensor_parallel_size", 1)),
        max_model_len=int(engine_raw.get("max_model_len", 32768)),
        max_num_seqs=int(engine_raw.get("max_num_seqs", 256)),
        prefill_gpu_memory_utilization=float(
            engine_raw.get("prefill_gpu_memory_utilization", 0.90)
        ),
        decode_gpu_memory_utilization=float(
            engine_raw.get("decode_gpu_memory_utilization", 0.55)
        ),
        startup_timeout_s=float(engine_raw.get("startup_timeout_s", 900.0)),
        common_args=_tuple(engine_raw, "common_args"),
        prefill_args=_tuple(engine_raw, "prefill_args"),
        decode_args=_tuple(engine_raw, "decode_args"),
    )
    return ClusterConfig(
        model=model,
        prefill=prefill,
        decode=decode,
        router=router,
        engine=engine,
        source_path=source_path,
    )
