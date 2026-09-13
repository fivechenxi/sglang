from sglang.srt.disaggregation.decode_token_admission import (
    DecodeTokenAdmissionState,
)


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
