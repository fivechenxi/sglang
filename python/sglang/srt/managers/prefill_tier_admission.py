# SPDX-License-Identifier: Apache-2.0

"""Actual-cache-tier admission accounting for disaggregated prefill."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PrefillTierCost:
    cold_tokens: int = 0
    load_back_tokens: int = 0
    storage_tokens: int = 0

    @classmethod
    def from_cache_lookup(
        cls,
        *,
        total_tokens: int,
        device_tokens: int,
        host_tokens: int,
        storage_tokens: int,
        load_back_threshold: int = 0,
    ) -> "PrefillTierCost":
        """Classify one P-side lookup without double-counting L3 as L2.

        After an L3 prefetch completes SGLang reports those pages as part of
        ``host_tokens`` as well as ``storage_tokens``.  Storage therefore has
        to be subtracted from the host total before charging L2 load-back.
        """

        total = max(0, total_tokens)
        device = min(max(0, device_tokens), total)
        host_total = min(max(0, host_tokens), total - device)
        storage = min(max(0, storage_tokens), host_total)
        if host_total < max(0, load_back_threshold):
            # L3 lookup/fetch work may already have happened, but this prefix is
            # too short for H2D load-back and will still be recomputed.
            return cls(
                cold_tokens=max(0, total - device),
                storage_tokens=storage,
            )
        return cls(
            cold_tokens=max(0, total - device - host_total),
            load_back_tokens=host_total - storage,
            storage_tokens=storage,
        )


class PrefillTierAdmission:
    """Per-P-DP reservations split by the work that will really be performed.

    The scheduler owns one instance, so no cross-rank lock is necessary.  A
    reservation lives from the P-side cache lookup until KV transfer reaches a
    terminal state.  Reconciliation replaces an earlier estimate atomically;
    it is used because an L1/L2 entry can be evicted between request receipt and
    scheduling.
    """

    def __init__(
        self,
        *,
        max_cold_tokens: int,
        max_load_back_tokens: int,
        max_storage_tokens: int,
    ) -> None:
        if min(
            max_cold_tokens,
            max_load_back_tokens,
            max_storage_tokens,
        ) < 0:
            raise ValueError("P-side cache-tier admission limits must be non-negative")
        self.limits = PrefillTierCost(
            max_cold_tokens,
            max_load_back_tokens,
            max_storage_tokens,
        )
        self.used = PrefillTierCost()
        self._reservations: dict[str, PrefillTierCost] = {}

    @property
    def enabled(self) -> bool:
        return any(
            (
                self.limits.cold_tokens,
                self.limits.load_back_tokens,
                self.limits.storage_tokens,
            )
        )

    @staticmethod
    def _add(a: PrefillTierCost, b: PrefillTierCost) -> PrefillTierCost:
        return PrefillTierCost(
            a.cold_tokens + b.cold_tokens,
            a.load_back_tokens + b.load_back_tokens,
            a.storage_tokens + b.storage_tokens,
        )

    @staticmethod
    def _sub(a: PrefillTierCost, b: PrefillTierCost) -> PrefillTierCost:
        return PrefillTierCost(
            max(0, a.cold_tokens - b.cold_tokens),
            max(0, a.load_back_tokens - b.load_back_tokens),
            max(0, a.storage_tokens - b.storage_tokens),
        )

    def _within_limits(self, cost: PrefillTierCost) -> bool:
        return all(
            limit == 0 or value <= limit
            for value, limit in (
                (cost.cold_tokens, self.limits.cold_tokens),
                (cost.load_back_tokens, self.limits.load_back_tokens),
                (cost.storage_tokens, self.limits.storage_tokens),
            )
        )

    def exceeded_tiers(self, rid: str, cost: PrefillTierCost) -> tuple[str, ...]:
        previous = self._reservations.get(rid, PrefillTierCost())
        candidate = self._add(self._sub(self.used, previous), cost)
        return tuple(
            tier
            for tier, value, limit in (
                ("cold", candidate.cold_tokens, self.limits.cold_tokens),
                (
                    "load_back",
                    candidate.load_back_tokens,
                    self.limits.load_back_tokens,
                ),
                ("storage", candidate.storage_tokens, self.limits.storage_tokens),
            )
            if limit > 0 and value > limit
        )

    def reserve_or_reconcile(self, rid: str, cost: PrefillTierCost) -> bool:
        if self.exceeded_tiers(rid, cost):
            return False
        previous = self._reservations.get(rid, PrefillTierCost())
        candidate = self._add(self._sub(self.used, previous), cost)
        self.used = candidate
        self._reservations[rid] = cost
        return True

    def release(self, rid: str) -> None:
        previous = self._reservations.pop(rid, None)
        if previous is not None:
            self.used = self._sub(self.used, previous)

    def get(self, rid: str) -> PrefillTierCost | None:
        return self._reservations.get(rid)

    def reset(self) -> None:
        self.used = PrefillTierCost()
        self._reservations.clear()

    def release_matching(self, rid: str, *, all_requests: bool = False) -> None:
        for reserved_rid in list(self._reservations):
            if all_requests or reserved_rid.startswith(rid):
                self.release(reserved_rid)
