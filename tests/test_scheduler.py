from __future__ import annotations

import unittest

from vllm_pd_router.config import WorkerConfig
from vllm_pd_router.scheduler import LeastActiveScheduler


def worker(worker_id: str, port: int) -> WorkerConfig:
    return WorkerConfig(worker_id=worker_id, url=f"http://127.0.0.1:{port}")


class LeastActiveSchedulerTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.scheduler = LeastActiveScheduler(
            (worker("P0", 8101), worker("P1", 8102)),
            tuple(worker(f"D{index}", 8200 + index) for index in range(4)),
        )

    async def test_round_robin_prefill_and_balanced_decode_ties(self) -> None:
        selections = [await self.scheduler.acquire() for _ in range(4)]
        self.assertEqual([item.prefill.worker_id for item in selections], ["P0", "P1", "P0", "P1"])
        self.assertEqual([item.decode.worker_id for item in selections], ["D0", "D1", "D2", "D3"])

    async def test_freed_decoder_is_selected_first(self) -> None:
        selections = [await self.scheduler.acquire() for _ in range(4)]
        await self.scheduler.finish_prefill(selections[1], success=True)
        await self.scheduler.finish_decode(selections[1], success=True)
        selected = await self.scheduler.acquire()
        self.assertEqual(selected.decode.worker_id, "D1")

    async def test_snapshot_tracks_active_and_failures(self) -> None:
        selected = await self.scheduler.acquire(output_tokens=128)
        await self.scheduler.finish_prefill(selected, success=False)
        await self.scheduler.finish_decode(selected, success=False)
        snapshot = await self.scheduler.snapshot()
        self.assertEqual(snapshot["prefill"][0]["active"], 0)
        self.assertEqual(snapshot["decode"][0]["active"], 0)
        self.assertEqual(snapshot["prefill"][0]["failed"], 1)
        self.assertEqual(snapshot["decode"][0]["failed"], 1)
        self.assertEqual(snapshot["decode"][0]["reserved_output_tokens"], 0)

    async def test_decode_binding_does_not_reserve_decode_early(self) -> None:
        prefill = await self.scheduler.acquire_prefill()
        snapshot = await self.scheduler.snapshot()
        self.assertEqual(sum(worker["active"] for worker in snapshot["prefill"]), 1)
        self.assertEqual(sum(worker["active"] for worker in snapshot["decode"]), 0)
        await self.scheduler.finish_prefill(prefill, success=True)
        selection = await self.scheduler.acquire_decode(prefill)
        self.assertEqual(selection.decode.worker_id, "D0")
        await self.scheduler.finish_decode(selection, success=True)

    async def test_decode_binding_uses_latest_decode_state(self) -> None:
        occupied = await self.scheduler.acquire()
        prefill = await self.scheduler.acquire_prefill()
        await self.scheduler.finish_prefill(prefill, success=True)
        selected = await self.scheduler.acquire_decode(prefill)
        self.assertNotEqual(selected.decode.worker_id, occupied.decode.worker_id)
        await self.scheduler.finish_prefill(occupied, success=True)
        await self.scheduler.finish_decode(occupied, success=True)
        await self.scheduler.finish_decode(selected, success=True)

    async def test_output_token_balance_uses_reserved_output_sum(self) -> None:
        scheduler = LeastActiveScheduler(
            (worker("P0", 8101),),
            (worker("D0", 8200), worker("D1", 8201)),
            strategy="token_balance",
        )
        first_prefill = await scheduler.acquire_prefill()
        first = await scheduler.acquire_decode(first_prefill, output_tokens=100)
        second_prefill = await scheduler.acquire_prefill()
        second = await scheduler.acquire_decode(second_prefill, output_tokens=10)
        third_prefill = await scheduler.acquire_prefill()
        third = await scheduler.acquire_decode(third_prefill, output_tokens=20)
        self.assertEqual(first.decode.worker_id, "D0")
        self.assertEqual(second.decode.worker_id, "D1")
        self.assertEqual(third.decode.worker_id, "D1")
        snapshot = await scheduler.snapshot()
        self.assertEqual(
            [item["reserved_output_tokens"] for item in snapshot["decode"]],
            [100, 30],
        )
        for selection in (first, second, third):
            await scheduler.finish_decode(selection, success=True)
        snapshot = await scheduler.snapshot()
        self.assertEqual(
            [item["reserved_output_tokens"] for item in snapshot["decode"]],
            [0, 0],
        )

    async def test_round_robin_decode_ignores_current_load(self) -> None:
        scheduler = LeastActiveScheduler(
            (worker("P0", 8101),),
            (worker("D0", 8200), worker("D1", 8201)),
            strategy="default_fcfs",
        )
        selections = []
        for _ in range(4):
            prefill = await scheduler.acquire_prefill()
            selections.append(await scheduler.acquire_decode(prefill, output_tokens=10))
        self.assertEqual(
            [selection.decode.worker_id for selection in selections],
            ["D0", "D1", "D0", "D1"],
        )


if __name__ == "__main__":
    unittest.main()
