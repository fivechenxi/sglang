from sglang.srt.disaggregation.decode_token_admission import (
    DecodeTokenAdmissionState,
)
from sglang.srt.managers.io_struct import DecodeTokenReservationReqOutput
from sglang.srt.utils.msgspec_utils import msgspec_to_builtins


def test_reservation_output_is_http_json_serializable():
    output = DecodeTokenReservationReqOutput(
        dp_rank=2,
        handled=True,
        accepted=True,
        reserved_tokens=4096,
        admittable_tokens=8192,
    )

    assert msgspec_to_builtins(output) == {
        "dp_rank": 2,
        "handled": True,
        "accepted": True,
        "reserved_tokens": 4096,
        "admittable_tokens": 8192,
        "error": "",
    }


def test_reservations_are_atomic_idempotent_and_expire():
    now = [10.0]
    state = DecodeTokenAdmissionState(
        reservation_ttl_s=30,
        clock=lambda: now[0],
    )

    assert state.reserve("a", 700, 1000).accepted
    assert state.reserve("a", 700, 1000).accepted
    assert not state.reserve("a", 701, 1000).accepted
    assert not state.reserve("b", 301, 1000).accepted
    assert state.admittable_tokens(1000) == 300

    now[0] = 41.0
    assert state.admittable_tokens(1000) == 1000


def test_committed_reservation_survives_ttl_until_released():
    now = [10.0]
    state = DecodeTokenAdmissionState(
        reservation_ttl_s=30,
        clock=lambda: now[0],
    )

    assert state.reserve("queued", 700, 1000).accepted
    assert state.commit("queued")
    assert not state.commit("queued")

    now[0] = 41.0
    assert state.admittable_tokens(1000) == 300

    state.release("queued", 1000)
    assert state.admittable_tokens(1000) == 1000


def test_materialization_keeps_only_dynamic_output_headroom():
    state = DecodeTokenAdmissionState()
    assert state.reserve("r", 3700, 4000).accepted
    assert state.commit("r")
    assert (
        state.planned_extra_after_materialization(
            "r", materialized_tokens=1000, baseline_output_tokens=640
        )
        == 2060
    )
    assert state.admittable_tokens(4000) == 300
    assert (
        state.extra_after_materialization(
            "r", materialized_tokens=1000, baseline_output_tokens=640
        )
        == 2060
    )
    assert state.admittable_tokens(4000) == 4000
    assert (
        state.extra_after_materialization(
            "r", materialized_tokens=1000, baseline_output_tokens=640
        )
        is None
    )


def test_reserved_tokens_can_exclude_request_being_materialized():
    state = DecodeTokenAdmissionState()
    assert state.reserve("current", 700, 2000).accepted
    assert state.reserve("in-flight", 800, 2000).accepted

    assert state.reserved_tokens() == 1500
    assert state.reserved_tokens("current") == 800
    assert state.reservation_tokens("current") == 700
    assert state.reservation_tokens("missing") == 0


def test_materialization_preserves_capacity_promised_to_other_request():
    state = DecodeTokenAdmissionState()
    assert state.reserve("current", 700, 2000).accepted
    assert state.commit("current")
    assert state.reserve("in-flight", 800, 2000).accepted

    # D's ordinary allocator paths see only the unpromised 500 tokens. While
    # materializing `current`, its own 700-token lease is added back, but the
    # other 800-token promise remains protected.
    protected_budget = state.admittable_tokens(2000)
    assert protected_budget == 500
    assert protected_budget + state.reservation_tokens("current") == 1200

    assert (
        state.extra_after_materialization(
            "current", materialized_tokens=500, baseline_output_tokens=200
        )
        == 0
    )
    # Allocating the current request consumes 700 physical tokens. The raw
    # allocator budget is now 1300, and the other lease still leaves 500.
    assert state.admittable_tokens(1300) == 500
