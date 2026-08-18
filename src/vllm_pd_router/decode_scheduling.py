from __future__ import annotations

import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Sequence

from .config import WorkerConfig


@dataclass
class WorkerState:
    config: WorkerConfig
    active: int = 0
    assigned: int = 0
    completed: int = 0
    failed: int = 0
    reserved_output_tokens: int = 0

    def snapshot(self) -> dict[str, int | str]:
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
class DecodeRequest:
    output_tokens: int = 0

    @property
    def reserved_output_tokens(self) -> int:
        return max(0, int(self.output_tokens))


@dataclass(frozen=True)
class DecodeMetrics:
    active_requests: int
    reserved_output_tokens: int


@dataclass(frozen=True)
class DecodeCandidate:
    index: int
    state: WorkerState
    metrics: DecodeMetrics
    round_robin_distance: int


@dataclass(frozen=True)
class DecodeSelectionDecision:
    candidate: DecodeCandidate


@dataclass(frozen=True)
class DecodeReservation:
    reservation_id: str
    decode_index: int
    reserved_output_tokens: int
    created_at: float
    expires_at: float | None


class DecodeMetricProvider:
    def collect(
        self,
        workers: Sequence[WorkerState],
        round_robin_start: int,
    ) -> tuple[DecodeCandidate, ...]:
        worker_count = len(workers)
        return tuple(
            DecodeCandidate(
                index=index,
                state=state,
                metrics=DecodeMetrics(
                    active_requests=state.active,
                    reserved_output_tokens=state.reserved_output_tokens,
                ),
                round_robin_distance=(index - round_robin_start) % worker_count,
            )
            for index, state in enumerate(workers)
        )


class DecodeFeasibilityFilter:
    def filter(
        self,
        candidates: Sequence[DecodeCandidate],
        request: DecodeRequest,
    ) -> tuple[DecodeCandidate, ...]:
        del request
        return tuple(candidates)


class DecodeSelector(ABC):
    score_name: str

    @abstractmethod
    def score(self, candidate: DecodeCandidate) -> int:
        raise NotImplementedError()

    @abstractmethod
    def select(
        self,
        candidates: Sequence[DecodeCandidate],
        request: DecodeRequest,
    ) -> DecodeSelectionDecision:
        raise NotImplementedError()

    @staticmethod
    def _require_candidates(
        candidates: Sequence[DecodeCandidate],
    ) -> None:
        if not candidates:
            raise RuntimeError("No feasible decode workers are available")


class RoundRobinDecodeSelector(DecodeSelector):
    score_name = "round_robin"

    def score(self, candidate: DecodeCandidate) -> int:
        return candidate.round_robin_distance

    def select(
        self,
        candidates: Sequence[DecodeCandidate],
        request: DecodeRequest,
    ) -> DecodeSelectionDecision:
        del request
        self._require_candidates(candidates)
        selected = min(candidates, key=self.score)
        return DecodeSelectionDecision(selected)


class LeastActiveDecodeSelector(DecodeSelector):
    score_name = "active_requests"

    def score(self, candidate: DecodeCandidate) -> int:
        return candidate.metrics.active_requests

    def select(
        self,
        candidates: Sequence[DecodeCandidate],
        request: DecodeRequest,
    ) -> DecodeSelectionDecision:
        del request
        self._require_candidates(candidates)
        selected = min(
            candidates,
            key=lambda candidate: (
                self.score(candidate),
                candidate.round_robin_distance,
            ),
        )
        return DecodeSelectionDecision(selected)


class TokenBalanceDecodeSelector(DecodeSelector):
    score_name = "reserved_output_tokens"

    def score(self, candidate: DecodeCandidate) -> int:
        return candidate.metrics.reserved_output_tokens

    def select(
        self,
        candidates: Sequence[DecodeCandidate],
        request: DecodeRequest,
    ) -> DecodeSelectionDecision:
        del request
        self._require_candidates(candidates)
        selected = min(
            candidates,
            key=lambda candidate: (
                self.score(candidate),
                candidate.round_robin_distance,
            ),
        )
        return DecodeSelectionDecision(selected)


def create_decode_selector(strategy: str) -> DecodeSelector:
    if strategy == "default_fcfs":
        return RoundRobinDecodeSelector()
    if strategy == "token_balance":
        return TokenBalanceDecodeSelector()
    return LeastActiveDecodeSelector()


class DecodeReservationManager:
    def __init__(
        self,
        workers: Sequence[WorkerState],
        timeout_s: float | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if timeout_s is not None and timeout_s <= 0:
            raise ValueError("reservation timeout must be positive or None")
        self.workers = workers
        self.timeout_s = timeout_s
        self.clock = clock
        self._reservations: dict[str, DecodeReservation] = {}

    def reserve(
        self,
        decision: DecodeSelectionDecision,
        request: DecodeRequest,
    ) -> DecodeReservation:
        now = self.clock()
        reservation = DecodeReservation(
            reservation_id=uuid.uuid4().hex,
            decode_index=decision.candidate.index,
            reserved_output_tokens=request.reserved_output_tokens,
            created_at=now,
            expires_at=now + self.timeout_s if self.timeout_s is not None else None,
        )
        state = self.workers[reservation.decode_index]
        state.active += 1
        state.assigned += 1
        state.reserved_output_tokens += reservation.reserved_output_tokens
        self._reservations[reservation.reservation_id] = reservation
        return reservation

    def release(self, reservation_id: str, success: bool) -> bool:
        reservation = self._reservations.pop(reservation_id, None)
        if reservation is None:
            return False
        self._release_state(reservation, success)
        return True

    def reclaim_expired(self) -> int:
        now = self.clock()
        expired_ids = [
            reservation_id
            for reservation_id, reservation in self._reservations.items()
            if reservation.expires_at is not None and reservation.expires_at <= now
        ]
        for reservation_id in expired_ids:
            reservation = self._reservations.pop(reservation_id)
            self._release_state(reservation, success=False)
        return len(expired_ids)

    def _release_state(
        self,
        reservation: DecodeReservation,
        success: bool,
    ) -> None:
        state = self.workers[reservation.decode_index]
        state.active = max(0, state.active - 1)
        state.reserved_output_tokens = max(
            0,
            state.reserved_output_tokens - reservation.reserved_output_tokens,
        )
        if success:
            state.completed += 1
        else:
            state.failed += 1

    @property
    def active_count(self) -> int:
        return len(self._reservations)
