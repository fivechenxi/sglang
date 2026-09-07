import pytest

from sglang.srt.managers.prefill_tier_admission import (
    PrefillTierAdmission,
    PrefillTierCost,
)


def test_negative_limit_is_rejected() -> None:
    with pytest.raises(ValueError, match="must be non-negative"):
        PrefillTierAdmission(
            max_cold_tokens=-1,
            max_load_back_tokens=0,
            max_storage_tokens=0,
        )


def test_tier_limits_are_independent_and_release() -> None:
    admission = PrefillTierAdmission(
        max_cold_tokens=100,
        max_load_back_tokens=200,
        max_storage_tokens=300,
    )
    assert admission.reserve_or_reconcile(
        "cold", PrefillTierCost(cold_tokens=100)
    )
    assert admission.reserve_or_reconcile(
        "l2", PrefillTierCost(load_back_tokens=200)
    )
    assert admission.reserve_or_reconcile(
        "l3", PrefillTierCost(storage_tokens=300)
    )
    assert not admission.reserve_or_reconcile(
        "extra-cold", PrefillTierCost(cold_tokens=1)
    )
    assert admission.exceeded_tiers(
        "extra-cold", PrefillTierCost(cold_tokens=1)
    ) == ("cold",)
    admission.release("cold")
    assert admission.reserve_or_reconcile(
        "extra-cold", PrefillTierCost(cold_tokens=1)
    )


def test_reconcile_is_atomic_when_cache_tier_changes() -> None:
    admission = PrefillTierAdmission(
        max_cold_tokens=100,
        max_load_back_tokens=100,
        max_storage_tokens=0,
    )
    assert admission.reserve_or_reconcile(
        "request", PrefillTierCost(load_back_tokens=80)
    )
    assert admission.reserve_or_reconcile(
        "other", PrefillTierCost(cold_tokens=90)
    )

    # The L2 entry disappeared before scheduling. Reconciliation must fail
    # without losing the original reservation.
    assert not admission.reserve_or_reconcile(
        "request", PrefillTierCost(cold_tokens=80)
    )
    assert admission.get("request") == PrefillTierCost(load_back_tokens=80)
    assert admission.used == PrefillTierCost(
        cold_tokens=90, load_back_tokens=80
    )


def test_zero_limit_disables_only_that_tier() -> None:
    admission = PrefillTierAdmission(
        max_cold_tokens=0,
        max_load_back_tokens=10,
        max_storage_tokens=0,
    )
    assert admission.enabled
    assert admission.reserve_or_reconcile(
        "unbounded", PrefillTierCost(cold_tokens=1_000_000, storage_tokens=1_000_000)
    )
    assert not admission.reserve_or_reconcile(
        "bounded", PrefillTierCost(load_back_tokens=11)
    )


def test_cache_lookup_classification_does_not_double_count_storage() -> None:
    assert PrefillTierCost.from_cache_lookup(
        total_tokens=1_000,
        device_tokens=200,
        host_tokens=500,
        storage_tokens=300,
    ) == PrefillTierCost(
        cold_tokens=300,
        load_back_tokens=200,
        storage_tokens=300,
    )


def test_cache_lookup_below_load_back_threshold_is_recomputed() -> None:
    assert PrefillTierCost.from_cache_lookup(
        total_tokens=1_000,
        device_tokens=800,
        host_tokens=100,
        storage_tokens=20,
        load_back_threshold=128,
    ) == PrefillTierCost(
        cold_tokens=200,
        storage_tokens=20,
    )


def test_storage_can_make_combined_host_restore_cross_threshold() -> None:
    assert PrefillTierCost.from_cache_lookup(
        total_tokens=1_000,
        device_tokens=700,
        host_tokens=200,
        storage_tokens=150,
        load_back_threshold=128,
    ) == PrefillTierCost(
        cold_tokens=100,
        load_back_tokens=50,
        storage_tokens=150,
    )
