from __future__ import annotations

import json
import unittest

try:
    import httpx
except ImportError:
    httpx = None

if httpx is not None:
    from vllm_pd_router.config import ClusterConfig, RouterConfig, WorkerConfig
    from vllm_pd_router.proxy import create_app


@unittest.skipIf(httpx is None, "httpx/FastAPI dependencies are not installed")
class ProxyIntegrationTest(unittest.IsolatedAsyncioTestCase):
    async def test_stream_request_emits_stage_timing(self) -> None:
        class MockAsyncStream(httpx.AsyncByteStream):
            async def __aiter__(self):
                yield (
                    b'data: {"choices":[{"text":"x","finish_reason":null}]}\n\n'
                    b"data: [DONE]\n\n"
                )

        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.port == 8101:
                return httpx.Response(
                    200,
                    json={
                        "kv_transfer_params": {
                            "remote_engine_id": "producer-0",
                            "remote_port": 5601,
                            "remote_block_ids": [1, 2],
                        }
                    },
                )
            return httpx.Response(
                200,
                stream=MockAsyncStream(),
                headers={"content-type": "text/event-stream"},
            )

        config = ClusterConfig(
            model="test-model",
            prefill=(WorkerConfig("P0", "http://prefill.local:8101"),),
            decode=(WorkerConfig("D0", "http://decode.local:8201"),),
            router=RouterConfig(strategy="least_active"),
        )
        app = create_app(config)
        await app.state.runtime.client.aclose()
        app.state.runtime.client = httpx.AsyncClient(
            transport=httpx.MockTransport(handler), timeout=None
        )
        client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://router.local"
        )
        try:
            with self.assertLogs("vllm_pd_router", level="INFO") as captured:
                async with client.stream(
                    "POST",
                    "/v1/completions",
                    json={
                        "model": "test-model",
                        "prompt": "x",
                        "max_tokens": 2,
                        "stream": True,
                    },
                    headers={
                        "x-request-id": "timed-request",
                        "x-pd-stage-timing": "1",
                        "x-pd-client-send-time": "1.0",
                    },
                ) as response:
                    self.assertEqual(response.status_code, 200)
                    await response.aread()
            timing_lines = [
                line for line in captured.output if "stage_timing" in line
            ]
            self.assertEqual(len(timing_lines), 1)
            self.assertIn('"request_id":"timed-request"', timing_lines[0])
            self.assertEqual(app.state.runtime.active_streams, 0)
        finally:
            await client.aclose()
            await app.state.runtime.close()

    async def test_nonstream_request_runs_prefill_then_least_active_decode(self) -> None:
        calls: list[tuple[str, dict[str, object]]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            calls.append((str(request.url), body))
            if request.url.port == 8101:
                return httpx.Response(
                    200,
                    json={
                        "kv_transfer_params": {
                            "remote_engine_id": "producer-0",
                            "remote_port": 5601,
                            "remote_block_ids": [1, 2],
                        }
                    },
                )
            return httpx.Response(
                200,
                json={"id": "completion", "choices": [{"text": " Paris"}]},
            )

        config = ClusterConfig(
            model="test-model",
            prefill=(WorkerConfig("P0", "http://prefill.local:8101"),),
            decode=(
                WorkerConfig("D0", "http://decode-0.local:8201"),
                WorkerConfig("D1", "http://decode-1.local:8202"),
            ),
            router=RouterConfig(),
        )
        app = create_app(config)
        await app.state.runtime.client.aclose()
        app.state.runtime.client = httpx.AsyncClient(
            transport=httpx.MockTransport(handler), timeout=None
        )
        client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://router.local"
        )
        try:
            response = await client.post(
                "/v1/completions",
                json={
                    "model": "test-model",
                    "prompt": "The capital of France is",
                    "max_tokens": 8,
                    "stream": False,
                },
            )
            self.assertEqual(response.status_code, 200)
            self.assertEqual(len(calls), 2)
            self.assertEqual(calls[0][1]["max_tokens"], 1)
            self.assertFalse(calls[0][1]["stream"])
            self.assertEqual(
                calls[1][1]["kv_transfer_params"]["remote_engine_id"], "producer-0"
            )
            self.assertEqual(calls[1][1]["kv_transfer_params"]["remote_host"], "prefill.local")
            state = (await client.get("/router/state")).json()
            self.assertEqual([worker["active"] for worker in state["decode"]], [0, 0])
            self.assertEqual(state["decode"][0]["completed"], 1)
        finally:
            await client.aclose()
            await app.state.runtime.close()

    async def test_post_prefill_strategy_selects_decode_after_prefill(self) -> None:
        events: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.port == 8101:
                events.append("prefill")
                return httpx.Response(
                    200,
                    json={
                        "kv_transfer_params": {
                            "remote_engine_id": "producer-0",
                            "remote_port": 5601,
                            "remote_block_ids": [1, 2],
                        }
                    },
                )
            events.append("decode")
            return httpx.Response(200, json={"id": "completion", "choices": []})

        config = ClusterConfig(
            model="test-model",
            prefill=(WorkerConfig("P0", "http://prefill.local:8101"),),
            decode=(WorkerConfig("D0", "http://decode.local:8201"),),
            router=RouterConfig(strategy="least_active"),
        )
        app = create_app(config)
        await app.state.runtime.client.aclose()
        app.state.runtime.client = httpx.AsyncClient(
            transport=httpx.MockTransport(handler), timeout=None
        )
        original_acquire_decode = app.state.runtime.scheduler.acquire_decode

        async def acquire_decode(prefill, output_tokens=0):
            events.append("select_decode")
            return await original_acquire_decode(prefill, output_tokens)

        app.state.runtime.scheduler.acquire_decode = acquire_decode
        client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://router.local"
        )
        try:
            response = await client.post(
                "/v1/completions",
                json={"model": "test-model", "prompt": "x", "max_tokens": 2},
            )
            self.assertEqual(response.status_code, 200)
            self.assertEqual(events, ["prefill", "select_decode", "decode"])
            state = (await client.get("/router/state")).json()
            self.assertEqual(state["strategy"], "least_active")
            self.assertEqual(state["decode"][0]["completed"], 1)
        finally:
            await client.aclose()
            await app.state.runtime.close()

    async def test_post_prefill_failure_does_not_assign_decode(self) -> None:
        def handler(_: httpx.Request) -> httpx.Response:
            return httpx.Response(500, json={"error": "prefill failed"})

        config = ClusterConfig(
            model="test-model",
            prefill=(WorkerConfig("P0", "http://prefill.local:8101"),),
            decode=(WorkerConfig("D0", "http://decode.local:8201"),),
            router=RouterConfig(strategy="least_active"),
        )
        app = create_app(config)
        await app.state.runtime.client.aclose()
        app.state.runtime.client = httpx.AsyncClient(
            transport=httpx.MockTransport(handler), timeout=None
        )
        client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://router.local"
        )
        try:
            response = await client.post(
                "/v1/completions",
                json={"model": "test-model", "prompt": "x", "max_tokens": 2},
            )
            self.assertEqual(response.status_code, 500)
            state = (await client.get("/router/state")).json()
            self.assertEqual(state["prefill"][0]["failed"], 1)
            self.assertEqual(state["decode"][0]["assigned"], 0)
        finally:
            await client.aclose()
            await app.state.runtime.close()


if __name__ == "__main__":
    unittest.main()
