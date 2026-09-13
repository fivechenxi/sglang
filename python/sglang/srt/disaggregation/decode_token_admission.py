"""CPU-only state for Decode-side token admission.

This module intentionally knows nothing about KV layouts. The scheduler passes
the current allocator-derived token budget in and receives one opaque scalar.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Dict, Tuple


@dataclass(frozen=True)
class ReservationResult:
    accepted: bool
    admittable_tokens: int


class DecodeTokenAdmissionState:
    def __init__(
        self,
        *,
        reservation_ttl_s: float = 30.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._reservation_ttl_s = reservation_ttl_s
        self._clock = clock
        self._reservations: Dict[str, Tuple[int, float]] = {}

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
        )

    def release(self, reservation_id: str, allocator_budget: int) -> ReservationResult:
        self._reservations.pop(reservation_id, None)
        return ReservationResult(
            accepted=True,
            admittable_tokens=self.admittable_tokens(allocator_budget),
        )
