from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from .config import ClusterConfig, load_config
from .protocol import (
    build_decode_payload,
    build_prefill_payload,
    extract_kv_transfer_params,
    forwarded_headers,
)
from .scheduler import LeastActiveScheduler, PairSelection, PrefillSelection, RemoteScheduler

LOGGER = logging.getLogger("vllm_pd_router")
SUPPORTED_ENDPOINTS = {"v1/completions", "v1/chat/completions"}


class ProxyRuntime:
    def __init__(self, config: ClusterConfig) -> None:
        self.config = config
        timeout = httpx.Timeout(
            connect=config.router.connect_timeout_s,
            read=None,
            write=config.router.request_timeout_s,
            pool=config.router.connect_timeout_s,
        )
        limits = httpx.Limits(
            max_connections=config.router.max_connections,
            max_keepalive_connections=max(32, config.router.max_connections // 4),
        )
        self.client = httpx.AsyncClient(timeout=timeout, limits=limits)
        if config.router.workers > 1:
            self.scheduler = RemoteScheduler(
                f"http://127.0.0.1:{config.router.scheduler_port}",
                config.prefill,
                config.decode,
                timeout_s=config.router.connect_timeout_s,
            )
        else:
            self.scheduler = LeastActiveScheduler(
                config.prefill, config.decode, strategy=config.router.strategy
            )
        self.active_streams = 0

    async def close(self) -> None:
        await self.client.aclose()
        close_scheduler = getattr(self.scheduler, "close", None)
        if close_scheduler is not None:
            await close_scheduler()

    def _headers(self, request: Request, request_id: str) -> dict[str, str]:
        headers = forwarded_headers(dict(request.headers), request_id)
        if self.config.router.api_key and "authorization" not in headers:
            headers["authorization"] = f"Bearer {self.config.router.api_key}"
        return headers

    async def _run_prefill(
        self,
        endpoint: str,
        request_data: dict[str, Any],
        selection: PrefillSelection,
        headers: dict[str, str],
    ) -> dict[str, Any]:
        response: httpx.Response | None = None
        success = False
        try:
            response = await self.client.post(
                f"{selection.prefill.url}/{endpoint}",
                json=build_prefill_payload(request_data),
                headers={**headers, "x-request-id": f"{headers['x-request-id']}-prefill"},
            )
            if response.status_code >= 400:
                raise UpstreamFailure.from_response("prefill", response)
            try:
                payload = response.json()
            except json.JSONDecodeError as error:
                raise UpstreamFailure(
                    stage="prefill",
                    status_code=502,
                    content=f"Invalid JSON from prefill worker: {error}".encode(),
                    media_type="text/plain",
                ) from error
            params = extract_kv_transfer_params(
                payload,
                selection.prefill.url,
                selection.prefill.transfer_host,
            )
            success = True
            return params
        finally:
            await self.scheduler.finish_prefill(selection, success=success)

    async def forward(self, request: Request, endpoint: str) -> Response:
        router_enter_monotonic = time.monotonic()
        router_enter_wall = time.time()
        if endpoint not in SUPPORTED_ENDPOINTS:
            return JSONResponse({"error": "Unsupported endpoint"}, status_code=404)
        try:
            request_data = await request.json()
        except Exception as error:
            return JSONResponse({"error": f"Invalid JSON request: {error}"}, status_code=400)
        if not isinstance(request_data, dict):
            return JSONResponse({"error": "Request body must be a JSON object"}, status_code=400)

        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex
        timing: dict[str, float | int] | None = None
        if request.headers.get("x-pd-stage-timing") == "1":
            timing = {
                "router_enter_monotonic": router_enter_monotonic,
                "router_enter_wall": router_enter_wall,
                "request_parsed_monotonic": time.monotonic(),
            }
            try:
                timing["client_send_wall"] = float(
                    request.headers["x-pd-client-send-time"]
                )
            except (KeyError, ValueError):
                pass
        requested_output_tokens = int(
            request_data.get("max_tokens")
            or request_data.get("max_completion_tokens")
            or 0
        )
        prefill_started = time.monotonic()
        if timing is not None:
            timing["prefill_started_monotonic"] = prefill_started
        prefill_selection = await self.scheduler.acquire_prefill()
        selection: PairSelection | None = None
        headers = self._headers(request, request_id)
        try:
            kv_transfer_params = await self._run_prefill(
                endpoint, request_data, prefill_selection, headers
            )
            if timing is not None:
                timing["prefill_done_monotonic"] = time.monotonic()
        except asyncio.CancelledError:
            if selection is not None:
                await self.scheduler.finish_decode(selection, success=False)
            raise
        except UpstreamFailure as error:
            if selection is not None:
                await self.scheduler.finish_decode(selection, success=False)
            return error.as_response()
        except Exception as error:
            if selection is not None:
                await self.scheduler.finish_decode(selection, success=False)
            LOGGER.exception("Prefill stage failed for request %s", request_id)
            return JSONResponse({"error": f"Prefill stage failed: {error}"}, status_code=502)

        if selection is None:
            selection = await self.scheduler.acquire_decode(
                prefill_selection, requested_output_tokens
            )
        if timing is not None:
            timing["decode_selected_monotonic"] = time.monotonic()
        LOGGER.info(
            "route request_id=%s strategy=%s prefill=%s decode=%s "
            "prefill_elapsed_s=%.6f decode_active_before=%s "
            "decode_score_name=%s decode_scores_before=%s reserved_output_tokens=%s",
            request_id,
            self.config.router.strategy,
            selection.prefill.worker_id,
            selection.decode.worker_id,
            time.monotonic() - prefill_started,
            selection.decode_active_before,
            selection.decode_score_name,
            selection.decode_scores_before,
            selection.reserved_output_tokens,
        )

        decode_payload = build_decode_payload(request_data, kv_transfer_params)
        if bool(request_data.get("stream")):
            return await self._forward_stream(
                endpoint, decode_payload, selection, headers, request_id, timing
            )
        return await self._forward_json(
            endpoint, decode_payload, selection, headers, request_id
        )

    async def _forward_json(
        self,
        endpoint: str,
        payload: dict[str, Any],
        selection: PairSelection,
        headers: dict[str, str],
        request_id: str,
    ) -> Response:
        success = False
        try:
            response = await self.client.post(
                f"{selection.decode.url}/{endpoint}", json=payload, headers=headers
            )
            success = response.status_code < 400
            return Response(
                content=response.content,
                status_code=response.status_code,
                media_type=response.headers.get("content-type"),
                headers={"x-request-id": request_id},
            )
        except Exception as error:
            LOGGER.exception("Decode stage failed for request %s", request_id)
            return JSONResponse({"error": f"Decode stage failed: {error}"}, status_code=502)
        finally:
            await self.scheduler.finish_decode(selection, success=success)

    async def _forward_stream(
        self,
        endpoint: str,
        payload: dict[str, Any],
        selection: PairSelection,
        headers: dict[str, str],
        request_id: str,
        timing: dict[str, float | int] | None = None,
    ) -> Response:
        try:
            if timing is not None:
                timing["decode_send_started_monotonic"] = time.monotonic()
            upstream_request = self.client.build_request(
                "POST", f"{selection.decode.url}/{endpoint}", json=payload, headers=headers
            )
            upstream = await self.client.send(upstream_request, stream=True)
            if timing is not None:
                timing["decode_headers_monotonic"] = time.monotonic()
        except asyncio.CancelledError:
            await self.scheduler.finish_decode(selection, success=False)
            raise
        except Exception as error:
            await self.scheduler.finish_decode(selection, success=False)
            LOGGER.exception("Decode stream setup failed for request %s", request_id)
            return JSONResponse({"error": f"Decode stage failed: {error}"}, status_code=502)

        if upstream.status_code >= 400:
            content = await upstream.aread()
            await upstream.aclose()
            await self.scheduler.finish_decode(selection, success=False)
            return Response(
                content=content,
                status_code=upstream.status_code,
                media_type=upstream.headers.get("content-type"),
                headers={"x-request-id": request_id},
            )

        async def body() -> AsyncIterator[bytes]:
            success = False
            first_chunk = True
            try:
                async for chunk in upstream.aiter_raw():
                    if first_chunk:
                        first_chunk = False
                        if timing is not None:
                            timing["first_chunk_monotonic"] = time.monotonic()
                            timing["active_streams"] = self.active_streams
                            self._log_stage_timing(request_id, timing)
                    yield chunk
                success = True
            finally:
                await upstream.aclose()
                await self.scheduler.finish_decode(selection, success=success)
                self.active_streams = max(0, self.active_streams - 1)

        self.active_streams += 1

        return StreamingResponse(
            body(),
            status_code=upstream.status_code,
            media_type=upstream.headers.get("content-type", "text/event-stream"),
            headers={"x-request-id": request_id},
        )

    def _log_stage_timing(
        self, request_id: str, timing: dict[str, float | int]
    ) -> None:
        router_enter = float(timing["router_enter_monotonic"])
        prefill_started = float(timing["prefill_started_monotonic"])
        prefill_done = float(timing["prefill_done_monotonic"])
        decode_send_started = float(timing["decode_send_started_monotonic"])
        decode_headers = float(timing["decode_headers_monotonic"])
        first_chunk = float(timing["first_chunk_monotonic"])
        payload = {
            "request_id": request_id,
            "router_worker_pid": os.getpid(),
            "router_enter_wall": timing["router_enter_wall"],
            "client_to_router_s": (
                float(timing["router_enter_wall"])
                - float(timing["client_send_wall"])
                if "client_send_wall" in timing
                else None
            ),
            "request_parse_s": float(timing["request_parsed_monotonic"])
            - router_enter,
            "router_to_prefill_start_s": prefill_started - router_enter,
            "prefill_s": prefill_done - prefill_started,
            "prefill_to_decode_send_s": decode_send_started - prefill_done,
            "decode_headers_s": decode_headers - decode_send_started,
            "decode_headers_to_first_chunk_s": first_chunk - decode_headers,
            "router_to_first_chunk_s": first_chunk - router_enter,
            "active_streams": timing["active_streams"],
        }
        LOGGER.info(
            "stage_timing %s",
            json.dumps(payload, separators=(",", ":"), sort_keys=True),
        )


class UpstreamFailure(Exception):
    def __init__(self, stage: str, status_code: int, content: bytes, media_type: str) -> None:
        super().__init__(f"{stage} upstream returned HTTP {status_code}")
        self.stage = stage
        self.status_code = status_code
        self.content = content
        self.media_type = media_type

    @classmethod
    def from_response(cls, stage: str, response: httpx.Response) -> "UpstreamFailure":
        return cls(
            stage=stage,
            status_code=response.status_code,
            content=response.content,
            media_type=response.headers.get("content-type", "application/json"),
        )

    def as_response(self) -> Response:
        return Response(
            content=self.content,
            status_code=self.status_code,
            media_type=self.media_type,
            headers={"x-vllm-pd-stage": self.stage},
        )


def create_app(config: ClusterConfig) -> FastAPI:
    runtime = ProxyRuntime(config)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        yield
        await runtime.close()

    app = FastAPI(title="PDA vLLM PD Router", lifespan=lifespan)
    app.state.runtime = runtime

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/router/state")
    async def state() -> dict[str, Any]:
        return await runtime.scheduler.snapshot()

    @app.get("/metrics")
    async def metrics() -> Response:
        snapshot = await runtime.scheduler.snapshot()
        lines = ["# TYPE vllm_pd_worker_active gauge"]
        for role in ("prefill", "decode"):
            for worker in snapshot[role]:
                labels = f'role="{role}",worker="{worker["id"]}"'
                lines.append(f"vllm_pd_worker_active{{{labels}}} {worker['active']}")
                lines.append(f"vllm_pd_worker_assigned_total{{{labels}}} {worker['assigned']}")
                lines.append(f"vllm_pd_worker_completed_total{{{labels}}} {worker['completed']}")
                lines.append(f"vllm_pd_worker_failed_total{{{labels}}} {worker['failed']}")
        return Response("\n".join(lines) + "\n", media_type="text/plain")

    @app.get("/v1/models")
    async def models(request: Request) -> Response:
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex
        try:
            response = await runtime.client.get(
                f"{config.decode[0].url}/v1/models",
                headers=runtime._headers(request, request_id),
            )
            return Response(
                response.content,
                status_code=response.status_code,
                media_type=response.headers.get("content-type"),
            )
        except Exception as error:
            return JSONResponse({"error": str(error)}, status_code=502)

    @app.post("/v1/completions")
    async def completions(request: Request) -> Response:
        return await runtime.forward(request, "v1/completions")

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> Response:
        return await runtime.forward(request, "v1/chat/completions")

    return app


def create_app_from_env() -> FastAPI:
    config_path = os.environ.get("VLLM_PD_CONFIG")
    if not config_path:
        raise RuntimeError("VLLM_PD_CONFIG is required for multi-worker router startup")
    log_level = os.environ.get("VLLM_PD_LOG_LEVEL", "INFO").upper()
    logging.basicConfig(level=log_level)
    LOGGER.setLevel(log_level)
    return create_app(load_config(config_path))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="vLLM disaggregated PD router")
    parser.add_argument("--config", required=True)
    parser.add_argument("--host")
    parser.add_argument("--port", type=int)
    parser.add_argument("--log-level", default="info")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    logging.basicConfig(level=args.log_level.upper())
    if config.router.workers > 1:
        os.environ["VLLM_PD_CONFIG"] = str(config.source_path)
        os.environ["VLLM_PD_LOG_LEVEL"] = args.log_level
        app = "vllm_pd_router.proxy:create_app_from_env"
    else:
        app = create_app(config)
    uvicorn.run(
        app,
        factory=config.router.workers > 1,
        workers=config.router.workers,
        host=args.host or config.router.host,
        port=args.port or config.router.port,
        log_level=args.log_level,
        access_log=config.router.access_log,
    )


if __name__ == "__main__":
    main()
