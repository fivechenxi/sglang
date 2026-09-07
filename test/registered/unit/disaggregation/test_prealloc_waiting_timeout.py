"""Decode KV preallocation must be covered by the waiting timeout."""

from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.disaggregation.base import KVPoll
from sglang.srt.disaggregation.mooncake.conn import MooncakeKVReceiver


def test_waiting_timeout_starts_before_metadata_is_sent():
    statuses = [KVPoll.WaitingForInput]
    receiver = object.__new__(MooncakeKVReceiver)
    receiver.kv_mgr = SimpleNamespace(
        waiting_timeout=30,
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
    receiver.conclude_state = None
    receiver.abort_notified = True
    receiver._connection_pool_entries = {}

    with patch("sglang.srt.disaggregation.common.conn.time.time") as clock:
        clock.return_value = 100.0
        assert receiver.poll() == KVPoll.WaitingForInput
        assert receiver.init_time == 100.0

        clock.return_value = 129.9
        assert receiver.poll() == KVPoll.WaitingForInput

        clock.return_value = 130.0
        assert receiver.poll() == KVPoll.Failed
