"""Regression coverage for HiCache failure consensus and transfer ownership."""

import unittest
from types import SimpleNamespace as NS
from unittest.mock import MagicMock, patch

from sglang.srt.disaggregation.base import KVPoll
from sglang.srt.disaggregation.decode import DecodePreallocQueue, DecodeTransferQueue
from sglang.srt.disaggregation.decode_hicache_mixin import (
    HiCacheRestoreGatedKVReceiver,
    HiCacheRestoreResult,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


def request(poll, status):
    return NS(
        req=NS(rid="restore", bootstrap_room=7, return_logprob=False),
        metadata_buffer_index=3,
        hicache_restore_status=status,
        kv_receiver=NS(
            poll=lambda: poll,
            require_staging=False,
            failure_exception=lambda: None,
            clear=lambda: None,
        ),
    )


class TestRestoreConsensus(unittest.TestCase):
    def test_gate_preserves_inflight_and_propagates_terminal_failure(self):
        for poll in (
            KVPoll.Failed,
            KVPoll.Bootstrapping,
            KVPoll.WaitingForInput,
            KVPoll.Transferring,
            KVPoll.Success,
        ):
            for status in HiCacheRestoreResult:
                expected = poll
                if poll == KVPoll.Success:
                    if status == HiCacheRestoreResult.PENDING:
                        expected = KVPoll.Transferring
                    elif status == HiCacheRestoreResult.FAILED:
                        expected = KVPoll.Failed
                with self.subTest(poll=poll, status=status):
                    self.assertEqual(
                        HiCacheRestoreGatedKVReceiver(request(poll, status)).poll(),
                        expected,
                    )

    def test_failed_restore_keeps_admission_reservation(self):
        queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
        queue.scheduler = NS(enable_decode_hicache=True)
        queue.transfer_queue = NS(
            queue=[
                NS(
                    prefix_match=NS(restore_token_count=256),
                    hicache_restore_status=status,
                    hicache_restored_node=node,
                )
                for status, node in [
                    (HiCacheRestoreResult.PENDING, None),
                    (HiCacheRestoreResult.FAILED, None),
                    (HiCacheRestoreResult.READY, object()),
                ]
            ]
        )
        self.assertEqual(queue._hicache_pending_restore_tokens(), 512)

    def make_queue(self, dr):
        q = DecodeTransferQueue.__new__(DecodeTransferQueue)
        q.queue = [dr]
        q.enable_staging = False
        q.enable_deferred_kv_release = False
        q.gloo_group = MagicMock()
        q.req_to_metadata_buffer_idx_allocator = MagicMock()
        q.tp_rank = 0
        q.tree_cache = MagicMock()
        q.metadata_buffers = NS(bootstrap_room=[0, 0, 0, 7])
        q._clean_hicache_prefetch_resources = MagicMock()
        q._commit_transfer_to_req = MagicMock()
        q.scheduler = MagicMock()
        q.scheduler.enable_decode_hicache = True
        q.scheduler.enable_hisparse = False
        q.scheduler.metrics_reporter.enable_metrics = False
        return q

    def test_staging_path_obeys_restore_gate(self):
        q = self.make_queue(request(KVPoll.Success, HiCacheRestoreResult.PENDING))
        q.staging_handler = MagicMock()
        q.metadata_buffers = None
        with patch(
            "sglang.srt.disaggregation.utils._all_reduce_polls",
            side_effect=lambda p, g: p,
        ):
            self.assertEqual(q._poll_with_staging(), [KVPoll.Transferring])

    def test_failed_restore_does_not_free_inflight_destination(self):
        dr = request(KVPoll.Transferring, HiCacheRestoreResult.FAILED)
        q = self.make_queue(dr)
        with patch(
            "sglang.srt.disaggregation.decode.poll_and_all_reduce",
            return_value=[KVPoll.Transferring],
        ), patch("sglang.srt.disaggregation.decode.release_kv_cache") as release, patch(
            "sglang.srt.disaggregation.decode.prepare_abort"
        ) as abort:
            self.assertEqual(q.pop_transferred(), [])
            self.assertEqual(q.queue, [dr])
            release.assert_not_called()
            abort.assert_not_called()


if __name__ == "__main__":
    unittest.main()
