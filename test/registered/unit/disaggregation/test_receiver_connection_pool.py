"""Unit tests for srt/disaggregation/common/conn — receiver connection_pool invalidation."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


import threading
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

import zmq

from sglang.srt.disaggregation.base.conn import KVPoll
from sglang.srt.disaggregation.common.conn import CommonKVManager, CommonKVReceiver
from sglang.test.test_utils import CustomTestCase


class _ConcreteReceiver(CommonKVReceiver):
    def poll(self) -> KVPoll:
        raise NotImplementedError

    def failure_exception(self):
        raise NotImplementedError


def _receiver(connection_pool, entries):
    receiver = object.__new__(_ConcreteReceiver)
    receiver.kv_mgr = SimpleNamespace(
        connection_pool=connection_pool,
        connection_pool_generations={key: "current:4" for key in connection_pool},
        connection_lock=threading.Lock(),
    )
    receiver._connection_pool_entries = entries
    return receiver


class _FetchingReceiver(_ConcreteReceiver):
    def _get_bootstrap_info_from_server(
        self, prefill_dp_rank, prefill_cp_rank, target_tp_rank, target_pp_rank
    ):
        self.fetch_count += 1
        return {"rank_ip": "10.0.0.1", "rank_port": 2001, "pp_rank": target_pp_rank}

    def _register_kv_args(self):
        return True


class TestReceiverSocketConfiguration(CustomTestCase):
    def test_bootstrap_metadata_socket_keeps_automatic_reconnect_enabled(self):
        context = Mock()
        socket = context.socket.return_value

        with (
            patch.object(CommonKVReceiver, "_ctx", context),
            patch.object(CommonKVReceiver, "_socket_cache", {}),
            patch.object(CommonKVReceiver, "_socket_locks", {}),
            patch.object(CommonKVReceiver, "_global_lock", threading.Lock()),
        ):
            CommonKVReceiver._connect("tcp://127.0.0.1:12345")

        self.assertNotIn(call(zmq.RECONNECT_IVL, -1), socket.setsockopt.call_args_list)
        self.assertIn(call(zmq.LINGER, 0), socket.setsockopt.call_args_list)


def _fetching_receiver(connection_pool):
    receiver = object.__new__(_FetchingReceiver)
    receiver.kv_mgr = SimpleNamespace(
        connection_pool=connection_pool,
        connection_pool_generations={},
        connection_lock=threading.Lock(),
        is_mla_backend=False,
        bootstrap_generations={},
        record_failure=Mock(),
        update_status=Mock(),
    )
    receiver.bootstrap_addr = "prefill:8998"
    receiver.bootstrap_room = 1
    receiver.prefill_dp_rank = 0
    receiver.target_cp_ranks = [0]
    receiver.target_tp_rank = 0
    receiver.target_tp_ranks = [0]
    receiver.target_pp_ranks = [0]
    receiver._connection_pool_entries = {}
    receiver._fetched_bootstrap_generation = None
    receiver.fetch_count = 0
    return receiver


class TestReceiverConnectionPool(CustomTestCase):
    @patch.object(CommonKVReceiver, "disconnect_endpoint")
    def test_generation_change_invalidates_all_routes_for_prefill(
        self, mock_disconnect
    ):
        stale_a = [{"rank_ip": "10.0.0.1", "rank_port": 1001}]
        stale_b = [{"rank_ip": "10.0.0.1", "rank_port": 1002}]
        retained = [{"rank_ip": "10.0.0.2", "rank_port": 2001}]
        manager = object.__new__(CommonKVManager)
        manager.connection_pool = {
            "prefill:8998_0_0_0": stale_a,
            "prefill:8998_0_0_1": stale_b,
            "other:8998_0_0_0": retained,
        }
        manager.connection_lock = threading.Lock()
        manager.bootstrap_generations = {"prefill:8998": "old:4"}
        manager.connection_pool_generations = {
            "prefill:8998_0_0_0": "old:4",
            "prefill:8998_0_0_1": "old:4",
            "other:8998_0_0_0": "other:4",
        }

        manager._observe_bootstrap_generation("prefill:8998", "new:4")

        self.assertEqual(manager.connection_pool, {"other:8998_0_0_0": retained})
        self.assertEqual(manager.bootstrap_generations["prefill:8998"], "new:4")
        self.assertCountEqual(
            [call.args[0] for call in mock_disconnect.call_args_list],
            ["tcp://10.0.0.1:1001", "tcp://10.0.0.1:1002"],
        )

    @patch.object(CommonKVReceiver, "disconnect_endpoint")
    def test_initial_generation_invalidates_untagged_cached_routes(
        self, mock_disconnect
    ):
        cached = [{"rank_ip": "10.0.0.1", "rank_port": 1001}]
        manager = object.__new__(CommonKVManager)
        manager.connection_pool = {"prefill:8998_0_0_0": cached}
        manager.connection_pool_generations = {}
        manager.connection_lock = threading.Lock()
        manager.bootstrap_generations = {}

        manager._observe_bootstrap_generation("prefill:8998", "current:4")

        self.assertEqual(manager.connection_pool, {})
        self.assertEqual(manager.connection_pool_generations, {})
        mock_disconnect.assert_called_once_with("tcp://10.0.0.1:1001")

    @patch.object(CommonKVReceiver, "disconnect_endpoint")
    def test_same_generation_keeps_tagged_cached_routes(self, mock_disconnect):
        cached = [{"rank_ip": "10.0.0.1", "rank_port": 1001}]
        manager = object.__new__(CommonKVManager)
        manager.connection_pool = {"prefill:8998_0_0_0": cached}
        manager.connection_pool_generations = {"prefill:8998_0_0_0": "current:4"}
        manager.connection_lock = threading.Lock()
        manager.bootstrap_generations = {"prefill:8998": "current:4"}

        manager._observe_bootstrap_generation("prefill:8998", "current:4")

        self.assertEqual(manager.connection_pool, {"prefill:8998_0_0_0": cached})
        mock_disconnect.assert_not_called()

    def test_invalidate_removes_matching_generation(self):
        stale = [
            {"rank_ip": "10.0.0.1", "rank_port": 1001},
            {"rank_ip": "10.0.0.1", "rank_port": 1002},
        ]
        retained = [{"rank_ip": "10.0.0.2", "rank_port": 2001}]
        receiver = _receiver(
            {"stale": stale, "retained": retained},
            {"stale": stale},
        )

        receiver.invalidate_cached_bootstrap_infos()

        self.assertEqual(receiver.kv_mgr.connection_pool, {"retained": retained})
        self.assertEqual(
            receiver.kv_mgr.connection_pool_generations, {"retained": "current:4"}
        )
        self.assertEqual(receiver._connection_pool_entries, {})

    def test_invalidate_preserves_concurrent_replacement_generation(self):
        stale = [{"rank_ip": "10.0.0.1", "rank_port": 1001}]
        replacement = [{"rank_ip": "10.0.0.1", "rank_port": 2001}]
        receiver = _receiver(
            {"key": replacement},
            {"key": stale},
        )

        receiver.invalidate_cached_bootstrap_infos()

        self.assertEqual(receiver.kv_mgr.connection_pool, {"key": replacement})
        self.assertEqual(
            receiver.kv_mgr.connection_pool_generations, {"key": "current:4"}
        )

    def test_invalidate_removes_all_matching_cp_entries(self):
        stale_cp0 = [{"rank_ip": "10.0.0.1", "rank_port": 1001}]
        stale_cp1 = [{"rank_ip": "10.0.0.1", "rank_port": 1002}]
        receiver = _receiver(
            {"cp0": stale_cp0, "cp1": stale_cp1},
            {"cp0": stale_cp0, "cp1": stale_cp1},
        )

        receiver.invalidate_cached_bootstrap_infos()

        self.assertEqual(receiver.kv_mgr.connection_pool, {})

    def test_next_receiver_refetches_after_invalidation(self):
        stale = [{"rank_ip": "10.0.0.1", "rank_port": 1001}]
        connection_pool = {"prefill:8998_0_0_0": stale}
        stale_receiver = _receiver(
            connection_pool,
            {"prefill:8998_0_0_0": stale},
        )
        stale_receiver.invalidate_cached_bootstrap_infos()

        receiver = _fetching_receiver(connection_pool)
        receiver._setup_bootstrap_infos()

        self.assertEqual(receiver.fetch_count, 1)
        self.assertEqual(receiver.bootstrap_infos[0]["rank_port"], 2001)
        self.assertIs(
            connection_pool["prefill:8998_0_0_0"],
            receiver._connection_pool_entries["prefill:8998_0_0_0"],
        )

    def test_cached_route_with_wrong_generation_is_refetched(self):
        stale = [{"rank_ip": "10.0.0.1", "rank_port": 1000}]
        key = "prefill:8998_0_0_0"
        receiver = _fetching_receiver({key: stale})
        receiver.kv_mgr.bootstrap_generations = {"prefill:8998": "new:4"}
        receiver.kv_mgr.connection_pool_generations = {key: "old:4"}

        receiver._setup_bootstrap_infos()

        self.assertEqual(receiver.fetch_count, 1)
        self.assertEqual(receiver.bootstrap_infos[0]["rank_port"], 2001)
        self.assertEqual(receiver.kv_mgr.connection_pool_generations[key], "new:4")

    @patch("sglang.srt.disaggregation.common.conn.time.time", return_value=3.0)
    def test_waiting_timeout_invalidates_cached_generation(self, _mock_time):
        stale = [{"rank_ip": "10.0.0.1", "rank_port": 1001}]
        receiver = _receiver({"key": stale}, {"key": stale})
        receiver.bootstrap_room = 1
        receiver.bootstrap_infos = stale
        receiver.init_time = 1.0
        receiver.abort_notified = True
        receiver.kv_mgr.waiting_timeout = 1.0
        receiver.kv_mgr.record_failure = Mock()
        receiver.kv_mgr.update_status = Mock()

        self.assertEqual(receiver._check_waiting_timeout(), KVPoll.Failed)
        self.assertEqual(receiver.kv_mgr.connection_pool, {})

    def test_abort_invalidates_route_for_next_request(self):
        stale = [{"rank_ip": "10.0.0.1", "rank_port": 1001}]
        receiver = _receiver({"key": stale}, {"key": stale})
        receiver.bootstrap_room = 1
        receiver.bootstrap_infos = stale
        receiver.abort_notified = True
        receiver.conclude_state = KVPoll.Bootstrapping
        receiver.kv_mgr.record_failure = Mock()
        receiver.kv_mgr.update_status = Mock()

        receiver.abort()

        self.assertEqual(receiver.kv_mgr.connection_pool, {})
        self.assertEqual(receiver.kv_mgr.connection_pool_generations, {})


if __name__ == "__main__":
    unittest.main()
