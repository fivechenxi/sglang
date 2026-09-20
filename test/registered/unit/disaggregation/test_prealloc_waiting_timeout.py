"""Decode preallocation and KV transfer use distinct timeout phases."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from sglang.srt.disaggregation.base import KVPoll
from sglang.srt.disaggregation.decode import DecodePreallocQueue, DecodeRequest
from sglang.srt.disaggregation.mooncake.conn import (
    MooncakeKVManager,
    MooncakeKVReceiver,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def test_prealloc_queue_keeps_polling_after_handshake_ready():
    """A blocked D-side preallocation must keep driving receiver timeouts."""
    queue = object.__new__(DecodePreallocQueue)
    queue.gloo_group = MagicMock()
    queue.pp_size = 1
    req = SimpleNamespace(time_stats=MagicMock(), rid="rid-1", bootstrap_room=1)
    receiver = MagicMock(conclude_state=None)
    queue.queue = [DecodeRequest(req=req, kv_receiver=receiver, waiting_for_input=True)]

    with patch(
        "sglang.srt.disaggregation.decode.poll_and_all_reduce",
        return_value=[KVPoll.WaitingForInput],
    ) as poll_all_ranks:
        queue._update_handshake_waiters()

    poll_all_ranks.assert_called_once_with([receiver], queue.gloo_group)


def test_waiting_timeout_starts_before_metadata_is_sent():
    statuses = [KVPoll.WaitingForInput]
    receiver = object.__new__(MooncakeKVReceiver)
    receiver.kv_mgr = SimpleNamespace(
        waiting_timeout=30,
        prealloc_timeout=15,
        transfer_progress_time={},
        check_status=lambda room: statuses[0],
        record_failure=lambda room, message: None,
        update_status=lambda room, value: statuses.__setitem__(0, value),
        request_status={},
        required_prefill_response_num_table={},
        prefill_response_tracker={},
        failure_lock=__import__("threading").Lock(),
        failure_records={},
        connection_lock=__import__("threading").Lock(),
        connection_pool={},
        connection_pool_generations={},
    )
    receiver.bootstrap_room = 1
    receiver.init_time = None
    receiver.metadata_sent = False
    receiver.prealloc_blocked_snapshot = {"reason": "projected_memory"}
    receiver.conclude_state = None
    receiver.abort_notified = True
    receiver._connection_pool_entries = {}

    with (
        patch("sglang.srt.disaggregation.common.conn.time.monotonic") as clock,
        patch("sglang.srt.disaggregation.common.conn.logger.warning") as warning,
    ):
        clock.return_value = 100.0
        assert receiver.poll() == KVPoll.WaitingForInput
        assert receiver.init_time == 100.0

        clock.return_value = 114.9
        assert receiver.poll() == KVPoll.WaitingForInput

        clock.return_value = 115.0
        assert receiver.poll() == KVPoll.Failed

    structured_warning = next(
        call
        for call in warning.call_args_list
        if call.args and call.args[0].startswith("PD_PREALLOC_TIMEOUT")
    )
    assert structured_warning.args[-1] == {"reason": "projected_memory"}


def test_prefill_compute_does_not_consume_transfer_timeout():
    statuses = [KVPoll.WaitingForInput]
    receiver = object.__new__(MooncakeKVReceiver)
    receiver.kv_mgr = SimpleNamespace(
        waiting_timeout=30,
        prealloc_timeout=15,
        transfer_progress_time={},
        check_status=lambda room: statuses[0],
        record_failure=lambda room, message: None,
        update_status=lambda room, value: statuses.__setitem__(0, value),
        request_status={},
        required_prefill_response_num_table={},
        prefill_response_tracker={},
        failure_lock=__import__("threading").Lock(),
        failure_records={},
        connection_lock=__import__("threading").Lock(),
        connection_pool={},
        connection_pool_generations={},
    )
    receiver.bootstrap_room = 1
    receiver.init_time = None
    receiver.metadata_sent = True
    receiver.conclude_state = None
    receiver.abort_notified = True
    receiver._connection_pool_entries = {}

    with patch("sglang.srt.disaggregation.common.conn.time.monotonic") as clock:
        # P can queue/compute for longer than waiting_timeout without being
        # mistaken for a stalled KV transfer.
        clock.return_value = 1000.0
        assert receiver.poll() == KVPoll.WaitingForInput

        statuses[0] = KVPoll.Transferring
        receiver.kv_mgr.transfer_progress_time[1] = 1000.0
        clock.return_value = 1029.9
        assert receiver.poll() == KVPoll.Transferring

        # A later chunk refreshes the inactivity deadline.
        receiver.kv_mgr.transfer_progress_time[1] = 1029.9
        clock.return_value = 1059.8
        assert receiver.poll() == KVPoll.Transferring

        clock.return_value = 1060.0
        assert receiver.poll() == KVPoll.Failed


def test_transfer_progress_message_refreshes_decode_deadline():
    manager = object.__new__(MooncakeKVManager)
    manager.request_status = {1: KVPoll.WaitingForInput}
    manager.transfer_progress_time = {}

    with patch("sglang.srt.disaggregation.common.conn.time.monotonic") as clock:
        clock.return_value = 123.0
        manager.record_transfer_progress(1)

    assert manager.request_status[1] == KVPoll.Transferring
    assert manager.transfer_progress_time[1] == 123.0

    # A late progress packet must not resurrect a request already cleaned up.
    manager.request_status.clear()
    manager.transfer_progress_time.clear()
    manager.record_transfer_progress(1)
    assert manager.transfer_progress_time == {}


def test_blocked_prealloc_snapshot_is_kept_without_debug_logging():
    queue = object.__new__(DecodePreallocQueue)
    receiver = SimpleNamespace()
    req = SimpleNamespace(
        rid="rid-snapshot",
        bootstrap_room=7,
        origin_input_ids=[1, 2, 3],
        sampling_params=SimpleNamespace(max_new_tokens=640),
    )
    decode_req = DecodeRequest(
        req=req,
        kv_receiver=receiver,
        waiting_for_input=True,
        trace_created_at=10.0,
    )
    queue.queue = [decode_req]
    queue.pending_reqs = []
    queue.retracted_queue = []
    queue.transfer_queue = SimpleNamespace(queue=[])
    queue.scheduler = SimpleNamespace(
        ps=SimpleNamespace(dp_rank=2),
        running_batch=SimpleNamespace(reqs=[object()]),
        waiting_queue=[],
    )
    queue.tp_rank = 8
    queue.token_to_kv_pool_allocator = SimpleNamespace(available_size=lambda: 1234)
    queue.token_admission = SimpleNamespace(reserved_tokens=lambda: 321)
    queue.num_reserved_decode_tokens = 512
    queue._bootstrap_trace_last_log = {}

    with (
        patch(
            "sglang.srt.disaggregation.decode.envs.SGLANG_DISAGGREGATION_BOOTSTRAP_TRACE.get",
            return_value=False,
        ),
        patch("sglang.srt.disaggregation.decode.time.monotonic", return_value=12.0),
    ):
        queue._trace_prealloc_blocked(
            "projected_memory",
            decode_req,
            required_tokens_for_request=2000,
            full_allocatable_tokens=1234,
        )

    assert receiver.prealloc_blocked_snapshot == {
        "reason": "projected_memory",
        "rid": "rid-snapshot",
        "room": 7,
        "dp_rank": 2,
        "tp_rank": 8,
        "age_ms": 2000.0,
        "input_tokens": 3,
        "max_new_tokens": 640,
        "prealloc_queue": 1,
        "queue_position": 0,
        "pending_queue": 0,
        "transfer_queue": 0,
        "retracted_queue": 0,
        "running_reqs": 1,
        "waiting_reqs": 0,
        "allocator_available_tokens": 1234,
        "reservation_tokens": 321,
        "baseline_decode_reserve": 512,
        "required_tokens_for_request": 2000,
        "full_allocatable_tokens": 1234,
    }


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
