from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

import httpx

from .config import WorkerConfig
from .decode_scheduling import (
    DecodeFeasibilityFilter,
    DecodeMetricProvider,
    DecodeRequest,
    DecodeReservationManager,
    DecodeSelector,
    WorkerState,
    create_decode_selector,
)


@dataclass(frozen=True)
class PrefillSelection:
    prefill_index: int
    prefill: WorkerConfig


@dataclass(frozen=True)
class PairSelection(PrefillSelection):
    decode_index: int
    decode: WorkerConfig
    decode_active_before: tuple[int, ...]
    decode_completed_before: tuple[int, ...]
    decode_scores_before: tuple[int, ...]
    decode_score_name: str
    reserved_output_tokens: int
    reservation_id: str


class LeastActiveScheduler:
    """Small shared scheduler for post-Prefill Decode placement."""

    def __init__(
        self,
        prefill_workers: tuple[WorkerConfig, ...],
        decode_workers: tuple[WorkerConfig, ...],
        strategy: str = "least_active",
        metric_provider: DecodeMetricProvider | None = None,
        feasibility_filter: DecodeFeasibilityFilter | None = None,
        selector: DecodeSelector | None = None,
        reservation_timeout_s: float | None = None,
    ) -> None:
        if not prefill_workers or not decode_workers:
            raise ValueError("At least one prefill and decode worker are required")
        self.prefill = [WorkerState(worker) for worker in prefill_workers]
        self.decode = [WorkerState(worker) for worker in decode_workers]
        self.strategy = strategy
        self._prefill_rr = 0
        self._decode_rr = 0
        self.metric_provider = metric_provider or DecodeMetricProvider()
        self.feasibility_filter = feasibility_filter or DecodeFeasibilityFilter()
        self.selector = selector or create_decode_selector(strategy)
        self.reservations = DecodeReservationManager(
            self.decode, timeout_s=reservation_timeout_s
        )
        self._lock = asyncio.Lock()

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
        self.reservations.reclaim_expired()
        request = DecodeRequest(output_tokens=output_tokens)
        all_candidates = self.metric_provider.collect(self.decode, self._decode_rr)
        active_before = tuple(
            candidate.metrics.active_requests for candidate in all_candidates
        )
        completed_before = tuple(
            candidate.state.completed for candidate in all_candidates
        )
        feasible_candidates = self.feasibility_filter.filter(all_candidates, request)
        decision = self.selector.select(feasible_candidates, request)
        decode_index = decision.candidate.index
        self._decode_rr = (decode_index + 1) % len(self.decode)
        reservation = self.reservations.reserve(decision, request)
        return PairSelection(
            prefill_index=prefill.prefill_index,
            prefill=prefill.prefill,
            decode_index=decode_index,
            decode=decision.candidate.state.config,
            decode_active_before=active_before,
            decode_completed_before=completed_before,
            decode_scores_before=tuple(
                self.selector.score(candidate) for candidate in all_candidates
            ),
            decode_score_name=self.selector.score_name,
            reserved_output_tokens=reservation.reserved_output_tokens,
            reservation_id=reservation.reservation_id,
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
            self.reservations.release(selection.reservation_id, success)

    async def reclaim_expired_decode_reservations(self) -> int:
        async with self._lock:
            return self.reservations.reclaim_expired()

    async def snapshot(self) -> dict[str, Any]:
        async with self._lock:
            self.reservations.reclaim_expired()
            return {
                "strategy": self.strategy,
                "prefill": [state.snapshot() for state in self.prefill],
                "decode": [state.snapshot() for state in self.decode],
                "active_decode_reservations": self.reservations.active_count,
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
            decode_completed_before=tuple(payload["decode_completed_before"]),
            decode_scores_before=tuple(payload["decode_scores_before"]),
            decode_score_name=str(payload["decode_score_name"]),
            reserved_output_tokens=int(payload["reserved_output_tokens"]),
            reservation_id=str(payload["reservation_id"]),
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
                "reservation_id": selection.reservation_id,
                "success": success,
            },
        )

    async def snapshot(self) -> dict[str, Any]:
        response = await self.client.get("/state")
        response.raise_for_status()
        return response.json()
