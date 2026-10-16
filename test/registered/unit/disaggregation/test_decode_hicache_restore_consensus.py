"""Regression coverage for HiCache failure consensus and transfer ownership."""

import unittest
from types import SimpleNamespace as NS
from unittest.mock import MagicMock, patch

from sglang.srt.disaggregation.base import KVPoll
from sglang.srt.disaggregation.decode import DecodePreallocQueue, DecodeTransferQueue
from sglang.srt.disaggregation.decode_hicache_mixin import (
    HiCacheRestoreResult,
    reduce_hicache_restore_polls,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


def request(poll, status):
    return NS(
        req=NS(
            rid="restore",
            bootstrap_room=7,
            bootstrap_host="prefill.test",
            return_logprob=False,
        ),
        metadata_buffer_index=3,
        hicache_restore_status=status,
        kv_receiver=NS(
            poll=lambda: poll,
            require_staging=False,
            failure_exception=lambda: None,
            clear=lambda: None,
            abort_notified=False,
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
                    with patch(
                        "sglang.srt.disaggregation.utils._all_reduce_polls",
                        side_effect=lambda p, g: p,
                    ):
                        self.assertEqual(
                            reduce_hicache_restore_polls(
                                [poll], [request(poll, status)], None
                            ),
                            [expected],
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

    def test_peer_restore_failure_waits_for_all_raw_transfers(self):
        # Rank 0 has finished RDMA but failed its restore; rank 1 is still
        # receiving into its destination pages. Exercise the real gate/reducer
        # order, rather than mocking poll_and_all_reduce's final answer.
        for local_raw, peer_raw, local_status, peer_status in [
            (
                KVPoll.Success,
                KVPoll.Transferring,
                HiCacheRestoreResult.FAILED,
                HiCacheRestoreResult.READY,
            ),
            (
                KVPoll.Transferring,
                KVPoll.Success,
                HiCacheRestoreResult.READY,
                HiCacheRestoreResult.FAILED,
            ),
        ]:
            with self.subTest(local_raw=local_raw):
                q = self.make_queue(request(local_raw, local_status))
                q.enable_deferred_kv_release = True
                q.metadata_buffers = NS(bootstrap_room=MagicMock())
                q.metadata_buffers.bootstrap_room.__getitem__.return_value.item.return_value = (
                    7
                )
                q._process_hicache_local_restores = MagicMock()
                peer_restore = (
                    KVPoll.Failed
                    if peer_status == HiCacheRestoreResult.FAILED
                    else KVPoll.Success
                )
                calls = []

                def reduce(polls, group):
                    peer = peer_raw if not calls else peer_restore
                    calls.append(list(polls))
                    return [min(polls[0], peer)]

                with patch(
                    "sglang.srt.disaggregation.utils._all_reduce_polls",
                    side_effect=reduce,
                ), patch(
                    "sglang.srt.disaggregation.decode.release_kv_cache"
                ) as release:
                    q.pop_transferred()
                    release.assert_not_called()
                    self.assertEqual(len(q.queue), 1)
                    self.assertEqual(calls[0], [local_raw])

    def test_only_globally_finished_request_observes_restore_failure(self):
        reqs = [request(KVPoll.Success, HiCacheRestoreResult.FAILED) for _ in range(2)]
        with patch(
            "sglang.srt.disaggregation.utils._all_reduce_polls",
            return_value=[KVPoll.Failed] * 2,
        ):
            self.assertEqual(
                reduce_hicache_restore_polls(
                    [KVPoll.Success, KVPoll.Transferring], reqs, None
                ),
                [KVPoll.Failed, KVPoll.Transferring],
            )

    def test_real_two_rank_gloo_consensus(self):
        import tempfile
        import torch.multiprocessing as mp

        with tempfile.TemporaryDirectory(prefix="restore-consensus-") as tmp:
            mp.spawn(
                _gloo_restore_worker,
                args=("file://" + tmp + "/init",),
                nprocs=2,
                join=True,
            )


def _gloo_restore_worker(rank, init_url):
    from datetime import timedelta
    import torch.distributed as dist
    from sglang.srt.disaggregation.utils import poll_and_all_reduce

    dist.init_process_group(
        "gloo",
        rank=rank,
        world_size=2,
        init_method=init_url,
        timeout=timedelta(seconds=30),
    )
    try:
        for in_flight_rank in (0, 1):
            for raw_finished in (False, True):
                raw = (
                    KVPoll.Transferring
                    if rank == in_flight_rank and not raw_finished
                    else KVPoll.Success
                )
                status = (
                    HiCacheRestoreResult.READY
                    if rank == in_flight_rank
                    else HiCacheRestoreResult.FAILED
                )
                # Include a second, independently READY request in the same
                # vector so both peers must enter the restore collective even
                # when the first request still has an in-flight writer.
                reqs = [
                    request(raw, status),
                    request(KVPoll.Success, HiCacheRestoreResult.READY),
                ]
                polls = poll_and_all_reduce(
                    [dr.kv_receiver for dr in reqs], dist.group.WORLD
                )
                result = reduce_hicache_restore_polls(polls, reqs, dist.group.WORLD)
                expected = [
                    KVPoll.Failed if raw_finished else KVPoll.Transferring,
                    KVPoll.Success,
                ]
                assert result == expected, (rank, result, expected)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    unittest.main()
