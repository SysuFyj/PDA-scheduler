from __future__ import annotations

import argparse
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

import uvicorn
from fastapi import FastAPI

from .config import ClusterConfig, load_config
from .scheduler import LeastActiveScheduler, PairSelection, PrefillSelection


def _prefill_payload(selection: PrefillSelection) -> dict[str, int]:
    return {"prefill_index": selection.prefill_index}


def _pair_payload(selection: PairSelection) -> dict[str, Any]:
    return {
        "prefill_index": selection.prefill_index,
        "decode_index": selection.decode_index,
        "decode_active_before": selection.decode_active_before,
        "decode_scores_before": selection.decode_scores_before,
        "decode_score_name": selection.decode_score_name,
        "reserved_output_tokens": selection.reserved_output_tokens,
    }


def create_app(config: ClusterConfig) -> FastAPI:
    scheduler = LeastActiveScheduler(
        config.prefill, config.decode, strategy=config.router.strategy
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        yield

    app = FastAPI(title="PDA vLLM Scheduler Control Plane", lifespan=lifespan)
    app.state.scheduler = scheduler

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/state")
    async def state() -> dict[str, Any]:
        return await scheduler.snapshot()

    @app.post("/acquire/prefill")
    async def acquire_prefill() -> dict[str, int]:
        return _prefill_payload(await scheduler.acquire_prefill())

    @app.post("/acquire/decode")
    async def acquire_decode(payload: dict[str, Any]) -> dict[str, Any]:
        prefill_index = int(payload["prefill_index"])
        prefill = PrefillSelection(prefill_index, config.prefill[prefill_index])
        return _pair_payload(
            await scheduler.acquire_decode(prefill, int(payload.get("output_tokens", 0)))
        )

    @app.post("/acquire/pair")
    async def acquire_pair(payload: dict[str, Any]) -> dict[str, Any]:
        return _pair_payload(await scheduler.acquire(int(payload.get("output_tokens", 0))))

    @app.post("/finish/prefill")
    async def finish_prefill(payload: dict[str, Any]) -> dict[str, str]:
        index = int(payload["prefill_index"])
        await scheduler.finish_prefill(
            PrefillSelection(index, config.prefill[index]), bool(payload["success"])
        )
        return {"status": "ok"}

    @app.post("/finish/decode")
    async def finish_decode(payload: dict[str, Any]) -> dict[str, str]:
        index = int(payload["decode_index"])
        selection = PairSelection(
            prefill_index=0,
            prefill=config.prefill[0],
            decode_index=index,
            decode=config.decode[index],
            decode_active_before=(),
            decode_scores_before=(),
            decode_score_name="",
            reserved_output_tokens=int(payload.get("reserved_output_tokens", 0)),
        )
        await scheduler.finish_decode(selection, bool(payload["success"]))
        return {"status": "ok"}

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Shared vLLM PD scheduler")
    parser.add_argument("--config", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int)
    parser.add_argument("--log-level", default="warning")
    args = parser.parse_args()
    config = load_config(args.config)
    uvicorn.run(
        create_app(config),
        host=args.host,
        port=args.port or config.router.scheduler_port,
        log_level=args.log_level,
        access_log=False,
    )


if __name__ == "__main__":
    main()
