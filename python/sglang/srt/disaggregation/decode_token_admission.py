"""CPU-only state for Decode-side token admission.

This module intentionally knows nothing about KV layouts. The scheduler passes
the current allocator-derived token budget in and receives one opaque scalar.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Callable, Deque, Dict, Tuple


@dataclass(frozen=True)
class ReservationResult:
    accepted: bool
    admittable_tokens: int
    suggested_output_tokens: int


class DecodeTokenAdmissionState:
    def __init__(
        self,
        *,
        fallback_output_tokens: int,
        sample_capacity: int = 100,
        min_samples: int = 20,
        reservation_ttl_s: float = 30.0,
        max_output_reserve_tokens: int = 4096,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._fallback_output_tokens = max(0, fallback_output_tokens)
        self._min_samples = min_samples
        self._reservation_ttl_s = reservation_ttl_s
        self._max_output_reserve_tokens = max_output_reserve_tokens
        self._clock = clock
        self._completed_output_tokens: Deque[int] = deque(maxlen=sample_capacity)
        self._reservations: Dict[str, Tuple[int, float]] = {}

    def record_completed_output(self, output_tokens: int) -> None:
        self._completed_output_tokens.append(max(0, int(output_tokens)))

    def suggested_output_tokens(self) -> int:
        if len(self._completed_output_tokens) < self._min_samples:
            return self._fallback_output_tokens
        values = sorted(self._completed_output_tokens)
        # Nearest-rank P90: ceil(0.9 * n) - 1.
        p90 = values[(9 * len(values) - 1) // 10]
        # Never regress below the existing operational safety reserve. The
        # dynamic estimator is intended to fix under-reservation, not to claim
        # capacity from a small low-output sample.
        return min(
            self._max_output_reserve_tokens,
            max(self._fallback_output_tokens, p90),
        )

    def _prune_expired(self) -> None:
        now = self._clock()
        self._reservations = {
            key: value for key, value in self._reservations.items() if value[1] > now
        }

    def admittable_tokens(self, allocator_budget: int) -> int:
        self._prune_expired()
        reserved = sum(tokens for tokens, _deadline in self._reservations.values())
        return max(0, int(allocator_budget) - reserved)

    def reserve(
        self, reservation_id: str, tokens: int, allocator_budget: int
    ) -> ReservationResult:
        self._prune_expired()
        tokens = max(0, int(tokens))
        previous = self._reservations.get(reservation_id)
        if previous is not None:
            accepted = previous[0] == tokens
        elif tokens <= self.admittable_tokens(allocator_budget):
            self._reservations[reservation_id] = (
                tokens,
                self._clock() + self._reservation_ttl_s,
            )
            accepted = True
        else:
            accepted = False
        return ReservationResult(
            accepted=accepted,
            admittable_tokens=self.admittable_tokens(allocator_budget),
            suggested_output_tokens=self.suggested_output_tokens(),
        )

    def release(self, reservation_id: str, allocator_budget: int) -> ReservationResult:
        self._reservations.pop(reservation_id, None)
        return ReservationResult(
            accepted=True,
            admittable_tokens=self.admittable_tokens(allocator_budget),
            suggested_output_tokens=self.suggested_output_tokens(),
        )
