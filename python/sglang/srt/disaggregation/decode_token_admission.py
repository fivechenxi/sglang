"""CPU-only state for Decode-side token admission.

This module intentionally knows nothing about KV layouts. The scheduler passes
the current allocator-derived token budget in and receives one opaque scalar.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Dict, Optional


@dataclass(frozen=True)
class ReservationResult:
    accepted: bool
    admittable_tokens: int


@dataclass
class _Reservation:
    tokens: int
    deadline: Optional[float]
    committed: bool = False


class DecodeTokenAdmissionState:
    def __init__(
        self,
        *,
        reservation_ttl_s: float = 30.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._reservation_ttl_s = reservation_ttl_s
        self._clock = clock
        self._reservations: Dict[str, _Reservation] = {}

    def _prune_expired(self) -> None:
        now = self._clock()
        self._reservations = {
            key: value
            for key, value in self._reservations.items()
            if value.deadline is None or value.deadline > now
        }

    def admittable_tokens(self, allocator_budget: int) -> int:
        self._prune_expired()
        return max(0, int(allocator_budget) - self.reserved_tokens())

    def reserved_tokens(self, exclude_reservation_id: Optional[str] = None) -> int:
        """Return outstanding lease tokens, optionally excluding one lease.

        Decode's own allocation paths use this value too. Otherwise a locally
        queued request could consume capacity that the reservation endpoint had
        already promised to an HTTP request which has not reached Decode yet.
        """
        self._prune_expired()
        return sum(
            reservation.tokens
            for reservation_id, reservation in self._reservations.items()
            if reservation_id != exclude_reservation_id
        )

    def reservation_tokens(self, reservation_id: Optional[str]) -> int:
        if reservation_id is None:
            return 0
        self._prune_expired()
        reservation = self._reservations.get(reservation_id)
        return reservation.tokens if reservation is not None else 0

    def reserve(
        self, reservation_id: str, tokens: int, allocator_budget: int
    ) -> ReservationResult:
        self._prune_expired()
        tokens = max(0, int(tokens))
        previous = self._reservations.get(reservation_id)
        if previous is not None:
            accepted = previous.tokens == tokens
        elif tokens <= self.admittable_tokens(allocator_budget):
            self._reservations[reservation_id] = _Reservation(
                tokens=tokens,
                deadline=self._clock() + self._reservation_ttl_s,
            )
            accepted = True
        else:
            accepted = False
        return ReservationResult(
            accepted=accepted,
            admittable_tokens=self.admittable_tokens(allocator_budget),
        )

    def commit(self, reservation_id: str) -> bool:
        """Transfer a pending lease to the Decode queue exactly once.

        A committed reservation no longer expires: it remains part of the
        admission budget until physical preallocation succeeds or the request
        is aborted. This closes the enqueue-to-allocation over-admission gap.
        """
        self._prune_expired()
        reservation = self._reservations.get(reservation_id)
        if reservation is None or reservation.committed:
            return False
        reservation.committed = True
        reservation.deadline = None
        return True

    def is_committed(self, reservation_id: str) -> bool:
        self._prune_expired()
        reservation = self._reservations.get(reservation_id)
        return reservation is not None and reservation.committed

    def planned_extra_after_materialization(
        self,
        reservation_id: str,
        *,
        materialized_tokens: int,
        baseline_output_tokens: int,
    ) -> Optional[int]:
        """Return the dynamic headroom that remains after preallocation."""
        self._prune_expired()
        reservation = self._reservations.get(reservation_id)
        if reservation is None or not reservation.committed:
            return None
        return max(
            0,
            reservation.tokens
            - max(0, int(materialized_tokens))
            - max(0, int(baseline_output_tokens)),
        )

    def extra_after_materialization(
        self,
        reservation_id: str,
        *,
        materialized_tokens: int,
        baseline_output_tokens: int,
    ) -> Optional[int]:
        """Consume a committed lease and return dynamic output headroom.

        The allocator reflects ``materialized_tokens`` after preallocation and
        Decode already reserves ``baseline_output_tokens`` per active request.
        Only the remaining amount above those two facts must stay reserved.
        """
        extra = self.planned_extra_after_materialization(
            reservation_id,
            materialized_tokens=materialized_tokens,
            baseline_output_tokens=baseline_output_tokens,
        )
        if extra is not None:
            del self._reservations[reservation_id]
        return extra

    def release(self, reservation_id: str, allocator_budget: int) -> ReservationResult:
        self._reservations.pop(reservation_id, None)
        return ReservationResult(
            accepted=True,
            admittable_tokens=self.admittable_tokens(allocator_budget),
        )

    def clear(self) -> None:
        self._reservations.clear()
