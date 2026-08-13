from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

import httpx

from .config import WorkerConfig


@dataclass
class WorkerState:
    config: WorkerConfig
    active: int = 0
    assigned: int = 0
    completed: int = 0
    failed: int = 0
    reserved_output_tokens: int = 0

    def snapshot(self) -> dict[str, Any]:
        return {
            "id": self.config.worker_id,
            "url": self.config.url,
            "active": self.active,
            "assigned": self.assigned,
            "completed": self.completed,
            "failed": self.failed,
            "reserved_output_tokens": self.reserved_output_tokens,
        }


@dataclass(frozen=True)
class PrefillSelection:
    prefill_index: int
    prefill: WorkerConfig


@dataclass(frozen=True)
class PairSelection(PrefillSelection):
    decode_index: int
    decode: WorkerConfig
    decode_active_before: tuple[int, ...]
    decode_scores_before: tuple[int, ...]
    decode_score_name: str
    reserved_output_tokens: int


class LeastActiveScheduler:
    """Small shared scheduler for post-Prefill Decode placement."""

    def __init__(
        self,
        prefill_workers: tuple[WorkerConfig, ...],
        decode_workers: tuple[WorkerConfig, ...],
        strategy: str = "least_active",
    ) -> None:
        if not prefill_workers or not decode_workers:
            raise ValueError("At least one prefill and decode worker are required")
        self.prefill = [WorkerState(worker) for worker in prefill_workers]
        self.decode = [WorkerState(worker) for worker in decode_workers]
        self.strategy = strategy
        self._prefill_rr = 0
        self._decode_rr = 0
        self._lock = asyncio.Lock()

    def _decode_distance(self, index: int) -> int:
        return (index - self._decode_rr) % len(self.decode)

    def _acquire_prefill_locked(self) -> PrefillSelection:
        prefill_index = self._prefill_rr % len(self.prefill)
        self._prefill_rr = (prefill_index + 1) % len(self.prefill)
        prefill_state = self.prefill[prefill_index]
        prefill_state.active += 1
        prefill_state.assigned += 1
        return PrefillSelection(prefill_index=prefill_index, prefill=prefill_state.config)

    def _acquire_decode_locked(
        self, prefill: PrefillSelection, output_tokens: int
    ) -> PairSelection:
        active_before = tuple(state.active for state in self.decode)
        reserved_before = tuple(state.reserved_output_tokens for state in self.decode)
        if self.strategy == "default_fcfs":
            score_name = "round_robin"
            scores_before = tuple(self._decode_distance(index) for index in range(len(self.decode)))
            decode_index = self._decode_rr
        elif self.strategy == "token_balance":
            score_name = "reserved_output_tokens"
            scores_before = reserved_before
            decode_index = min(
                range(len(self.decode)),
                key=lambda index: (reserved_before[index], self._decode_distance(index)),
            )
        else:
            score_name = "active_requests"
            scores_before = active_before
            decode_index = min(
                range(len(self.decode)),
                key=lambda index: (active_before[index], self._decode_distance(index)),
            )
        self._decode_rr = (decode_index + 1) % len(self.decode)
        decode_state = self.decode[decode_index]
        decode_state.active += 1
        decode_state.assigned += 1
        reserved_output_tokens = max(0, int(output_tokens))
        decode_state.reserved_output_tokens += reserved_output_tokens
        return PairSelection(
            prefill_index=prefill.prefill_index,
            prefill=prefill.prefill,
            decode_index=decode_index,
            decode=decode_state.config,
            decode_active_before=active_before,
            decode_scores_before=scores_before,
            decode_score_name=score_name,
            reserved_output_tokens=reserved_output_tokens,
        )

    async def acquire_prefill(self) -> PrefillSelection:
        async with self._lock:
            return self._acquire_prefill_locked()

    async def acquire_decode(
        self, prefill: PrefillSelection, output_tokens: int = 0
    ) -> PairSelection:
        async with self._lock:
            return self._acquire_decode_locked(prefill, output_tokens)

    async def acquire(self, output_tokens: int = 0) -> PairSelection:
        async with self._lock:
            prefill = self._acquire_prefill_locked()
            return self._acquire_decode_locked(prefill, output_tokens)

    async def finish_prefill(
        self, selection: PrefillSelection, success: bool
    ) -> None:
        async with self._lock:
            state = self.prefill[selection.prefill_index]
            state.active = max(0, state.active - 1)
            if success:
                state.completed += 1
            else:
                state.failed += 1

    async def finish_decode(self, selection: PairSelection, success: bool) -> None:
        async with self._lock:
            state = self.decode[selection.decode_index]
            state.active = max(0, state.active - 1)
            state.reserved_output_tokens = max(
                0, state.reserved_output_tokens - selection.reserved_output_tokens
            )
            if success:
                state.completed += 1
            else:
                state.failed += 1

    async def snapshot(self) -> dict[str, Any]:
        async with self._lock:
            return {
                "strategy": self.strategy,
                "prefill": [state.snapshot() for state in self.prefill],
                "decode": [state.snapshot() for state in self.decode],
            }


class RemoteScheduler:
    """Client for the single scheduler shared by proxy worker processes."""

    def __init__(
        self,
        base_url: str,
        prefill_workers: tuple[WorkerConfig, ...],
        decode_workers: tuple[WorkerConfig, ...],
        timeout_s: float = 10.0,
    ) -> None:
        self.prefill = prefill_workers
        self.decode = decode_workers
        self.client = httpx.AsyncClient(base_url=base_url, timeout=timeout_s)

    async def close(self) -> None:
        await self.client.aclose()

    async def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        response = await self.client.post(path, json=payload)
        response.raise_for_status()
        return response.json()

    def _prefill_selection(self, payload: dict[str, Any]) -> PrefillSelection:
        index = int(payload["prefill_index"])
        return PrefillSelection(index, self.prefill[index])

    def _pair_selection(self, payload: dict[str, Any]) -> PairSelection:
        prefill_index = int(payload["prefill_index"])
        decode_index = int(payload["decode_index"])
        return PairSelection(
            prefill_index=prefill_index,
            prefill=self.prefill[prefill_index],
            decode_index=decode_index,
            decode=self.decode[decode_index],
            decode_active_before=tuple(payload["decode_active_before"]),
            decode_scores_before=tuple(payload["decode_scores_before"]),
            decode_score_name=str(payload["decode_score_name"]),
            reserved_output_tokens=int(payload["reserved_output_tokens"]),
        )

    async def acquire_prefill(self) -> PrefillSelection:
        return self._prefill_selection(await self._post("/acquire/prefill", {}))

    async def acquire_decode(
        self, prefill: PrefillSelection, output_tokens: int = 0
    ) -> PairSelection:
        payload = await self._post(
            "/acquire/decode",
            {"prefill_index": prefill.prefill_index, "output_tokens": output_tokens},
        )
        return self._pair_selection(payload)

    async def acquire(self, output_tokens: int = 0) -> PairSelection:
        return self._pair_selection(
            await self._post("/acquire/pair", {"output_tokens": output_tokens})
        )

    async def finish_prefill(
        self, selection: PrefillSelection, success: bool
    ) -> None:
        await self._post(
            "/finish/prefill",
            {"prefill_index": selection.prefill_index, "success": success},
        )

    async def finish_decode(self, selection: PairSelection, success: bool) -> None:
        await self._post(
            "/finish/decode",
            {
                "decode_index": selection.decode_index,
                "reserved_output_tokens": selection.reserved_output_tokens,
                "success": success,
            },
        )

    async def snapshot(self) -> dict[str, Any]:
        response = await self.client.get("/state")
        response.raise_for_status()
        return response.json()
