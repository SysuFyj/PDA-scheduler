from __future__ import annotations

import unittest

import httpx

from vllm_pd_router.config import ClusterConfig, RouterConfig, WorkerConfig
from vllm_pd_router.scheduler import RemoteScheduler
from vllm_pd_router.scheduler_service import create_app


class RemoteSchedulerTest(unittest.IsolatedAsyncioTestCase):
    async def test_workers_share_post_prefill_decode_state(self) -> None:
        config = ClusterConfig(
            model="test-model",
            prefill=(WorkerConfig("P0", "http://prefill.local:8101"),),
            decode=(
                WorkerConfig("D0", "http://decode-0.local:8201"),
                WorkerConfig("D1", "http://decode-1.local:8202"),
            ),
            router=RouterConfig(strategy="least_active", workers=2),
        )
        app = create_app(config)

        async def make_client() -> RemoteScheduler:
            scheduler = RemoteScheduler(
                "http://scheduler.local", config.prefill, config.decode
            )
            await scheduler.client.aclose()
            scheduler.client = httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app),
                base_url="http://scheduler.local",
            )
            return scheduler

        worker_a = await make_client()
        worker_b = await make_client()
        try:
            prefill_a = await worker_a.acquire_prefill()
            await worker_a.finish_prefill(prefill_a, success=True)
            decode_a = await worker_a.acquire_decode(prefill_a, output_tokens=100)

            prefill_b = await worker_b.acquire_prefill()
            await worker_b.finish_prefill(prefill_b, success=True)
            decode_b = await worker_b.acquire_decode(prefill_b, output_tokens=200)

            self.assertEqual(decode_a.decode.worker_id, "D0")
            self.assertEqual(decode_b.decode.worker_id, "D1")
            state = await worker_b.snapshot()
            self.assertEqual([item["active"] for item in state["decode"]], [1, 1])

            await worker_a.finish_decode(decode_a, success=True)
            await worker_b.finish_decode(decode_b, success=True)
            state = await worker_a.snapshot()
            self.assertEqual([item["active"] for item in state["decode"]], [0, 0])
        finally:
            await worker_a.close()
            await worker_b.close()


if __name__ == "__main__":
    unittest.main()
