from sglang.srt.disaggregation.decode_token_admission import (
    DecodeTokenAdmissionState,
)


def test_reservations_are_atomic_idempotent_and_expire():
    now = [10.0]
    state = DecodeTokenAdmissionState(
        fallback_output_tokens=640,
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


def test_output_p90_uses_last_100_normal_completions():
    state = DecodeTokenAdmissionState(fallback_output_tokens=640)
    for value in range(1, 20):
        state.record_completed_output(value)
    assert state.suggested_output_tokens() == 640

    state.record_completed_output(20)
    assert state.suggested_output_tokens() == 640

    for value in range(21, 121):
        state.record_completed_output(value)
    # Window now contains 21..120; nearest-rank P90 is its 90th item.
    assert state.suggested_output_tokens() == 640

    for value in range(1000, 1100):
        state.record_completed_output(value)
    assert state.suggested_output_tokens() == 1089

    for value in range(10_000, 10_100):
        state.record_completed_output(value)
    assert state.suggested_output_tokens() == 4096
