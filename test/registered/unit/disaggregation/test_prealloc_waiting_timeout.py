"""Decode preallocation and KV transfer use distinct timeout phases."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

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
    req = SimpleNamespace(time_stats=MagicMock(), rid="rid-1", bootstrap_room=1)
    receiver = MagicMock(conclude_state=None)
    queue.queue = [
        DecodeRequest(req=req, kv_receiver=receiver, waiting_for_input=True)
    ]

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
    receiver.conclude_state = None
    receiver.abort_notified = True
    receiver._connection_pool_entries = {}

    with patch("sglang.srt.disaggregation.common.conn.time.monotonic") as clock:
        clock.return_value = 100.0
        assert receiver.poll() == KVPoll.WaitingForInput
        assert receiver.init_time == 100.0

        clock.return_value = 129.9
        assert receiver.poll() == KVPoll.WaitingForInput

        clock.return_value = 130.0
        assert receiver.poll() == KVPoll.Failed


def test_prefill_compute_does_not_consume_transfer_timeout():
    statuses = [KVPoll.WaitingForInput]
    receiver = object.__new__(MooncakeKVReceiver)
    receiver.kv_mgr = SimpleNamespace(
        waiting_timeout=30,
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
