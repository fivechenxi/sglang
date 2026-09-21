import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.disaggregation.prefill import SchedulerDisaggregationPrefillMixin
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.schedule_batch import FINISH_ABORT, ReqKvInfo
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=8, suite="base-a-test-cpu")


class _FakeReq:
    """Minimal Req standing in for the fields the retire path touches."""

    def __init__(
        self,
        *,
        inflight_middle_chunks=0,
        allocated=True,
        to_finish=None,
        finished_reason=None,
        grammar=None,
    ):
        self.rid = "aborted-prefill-test"
        self.inflight_middle_chunks = inflight_middle_chunks
        self.req_pool_idx = 1 if allocated else None
        self.kv = (
            ReqKvInfo(kv_allocated_len=1, swa_evicted_seqlen=0) if allocated else None
        )
        self.mamba_pool_idx = torch.tensor([0]) if allocated else None
        self.metadata_buffer_index = 7 if allocated else -1
        self.pending_bootstrap = False
        self.disagg_kv_sender = Mock()
        self.to_finish = to_finish
        self.finished_reason = finished_reason
        self.return_logprob = False
        self.return_sampling_mask = False
        self.grammar = grammar
        self.output_ids = []
        self.origin_input_ids = list(range(64))
        self.extend_range = None
        self.time_stats = SimpleNamespace(
            set_prefill_finished_time=Mock(),
            set_completion_time=Mock(),
            set_last_chunked_prefill_finish_time=Mock(),
        )

    def finished(self):
        return self.finished_reason is not None

    def update_finish_state(self):
        if self.finished():
            return
        if self.to_finish:
            self.finished_reason = self.to_finish
            self.to_finish = None


class _Scheduler(SchedulerDisaggregationPrefillMixin):
    def __init__(self):
        self.batch_result_processor = SimpleNamespace(move_logprobs_to_cpu=Mock())
        self.spec_algorithm = SimpleNamespace(is_eagle=lambda: False)
        self.tree_cache = Mock()
        self.disagg_prefill_inflight_queue = []
        self.send_kv_chunk = Mock()
        self.output_streamer = Mock()
        self.metrics_reporter = SimpleNamespace(report_prefill_stats=Mock())
        self.req_to_metadata_buffer_idx_allocator = Mock()
        self.enable_hicache_storage = False
        self.chunked_req = None
        self._release_prefill_tier_admission = Mock()


def _batch(req):
    return SimpleNamespace(
        reqs=[req],
        spec_info=object(),
        prefill_stats=None,
        dp_cooperation_info=None,
    )


def _result(batch):
    result = GenerationBatchResult(next_token_ids=torch.tensor([11]))
    result.next_draft_input = batch.spec_info
    return result


def _free_req(req, tree_cache, is_insert=True):
    # The retire path must never insert an aborted request's KV into the radix
    # cache: it holds a rewritten 1-token prompt, not the request's real input.
    assert is_insert is False
    req.req_pool_idx = None
    req.kv = None
    req.mamba_pool_idx = None


class TestPrefillAbortResultCleanup(unittest.TestCase):
    @patch("sglang.srt.disaggregation.prefill.release_kv_cache", side_effect=_free_req)
    def test_aborted_final_result_retires_and_skips_kv(self, release_kv_cache):
        scheduler = _Scheduler()
        req = _FakeReq(to_finish=FINISH_ABORT(message="input too long"))
        batch = _batch(req)

        scheduler.process_batch_result_disagg_prefill(batch, _result(batch))

        scheduler.send_kv_chunk.assert_not_called()
        release_kv_cache.assert_called_once_with(
            req, scheduler.tree_cache, is_insert=False
        )
        scheduler.output_streamer.stream_output.assert_called_once_with([req], False)
        scheduler._release_prefill_tier_admission.assert_called_once_with(req.rid)
        scheduler.req_to_metadata_buffer_idx_allocator.free.assert_called_once_with(7)
        self.assertEqual(scheduler.disagg_prefill_inflight_queue, [])
        # The abort staged in to_finish was promoted so the streamed output
        # carries the terminal reason (required by the admission ACK protocol).
        self.assertIsInstance(req.finished_reason, FINISH_ABORT)
        self.assertIsNone(req.to_finish)
        self.assertEqual(req.metadata_buffer_index, -1)

    @patch("sglang.srt.disaggregation.prefill.release_kv_cache", side_effect=_free_req)
    def test_delayed_result_ignores_already_retired_request(self, release_kv_cache):
        scheduler = _Scheduler()
        req = _FakeReq(to_finish=FINISH_ABORT(message="input too long"))
        batch = _batch(req)

        scheduler.process_batch_result_disagg_prefill(batch, _result(batch))
        self.assertEqual(release_kv_cache.call_count, 1)

        # A delayed batch result for the same request must be a no-op.
        scheduler.output_streamer.stream_output.reset_mock()
        scheduler.process_batch_result_disagg_prefill(batch, _result(batch))
        self.assertEqual(release_kv_cache.call_count, 1)
        scheduler.output_streamer.stream_output.assert_not_called()
        scheduler.send_kv_chunk.assert_not_called()

    @patch("sglang.srt.disaggregation.prefill.release_kv_cache", side_effect=_free_req)
    def test_already_finished_abort_without_resources_is_ignored(
        self, release_kv_cache
    ):
        scheduler = _Scheduler()
        req = _FakeReq(allocated=False, finished_reason=FINISH_ABORT(message="done"))
        batch = _batch(req)

        scheduler.process_batch_result_disagg_prefill(batch, _result(batch))

        release_kv_cache.assert_not_called()
        scheduler.output_streamer.stream_output.assert_not_called()
        scheduler.send_kv_chunk.assert_not_called()
        self.assertEqual(scheduler.disagg_prefill_inflight_queue, [])

    @patch("sglang.srt.disaggregation.prefill.release_kv_cache", side_effect=_free_req)
    def test_grammar_rejection_retires_before_transfer(self, release_kv_cache):
        scheduler = _Scheduler()
        req = _FakeReq(grammar=Mock())
        req.grammar.accept_token.side_effect = ValueError("invalid token")
        batch = _batch(req)

        scheduler.process_batch_result_disagg_prefill(batch, _result(batch))

        req.grammar.accept_token.assert_called_once_with(11)
        scheduler.send_kv_chunk.assert_not_called()
        release_kv_cache.assert_called_once_with(
            req, scheduler.tree_cache, is_insert=False
        )
        scheduler.output_streamer.stream_output.assert_called_once_with([req], False)
        self.assertTrue(req.finished())
        self.assertEqual(scheduler.disagg_prefill_inflight_queue, [])

    @patch("sglang.srt.disaggregation.prefill.release_kv_cache", side_effect=_free_req)
    def test_sender_abort_failure_does_not_skip_local_cleanup(self, release_kv_cache):
        scheduler = _Scheduler()
        req = _FakeReq(to_finish=FINISH_ABORT(message="input too long"))
        req.disagg_kv_sender.abort.side_effect = RuntimeError("transport is down")
        batch = _batch(req)

        scheduler.process_batch_result_disagg_prefill(batch, _result(batch))

        release_kv_cache.assert_called_once_with(
            req, scheduler.tree_cache, is_insert=False
        )
        scheduler.req_to_metadata_buffer_idx_allocator.free.assert_called_once_with(7)
        scheduler.output_streamer.stream_output.assert_called_once_with([req], False)
        self.assertTrue(req.finished())

    @patch("sglang.srt.disaggregation.prefill.release_kv_cache", side_effect=_free_req)
    def test_aborted_middle_result_is_never_optimistically_requeued(
        self, release_kv_cache
    ):
        scheduler = _Scheduler()
        scheduler.optimistic_release_and_requeue = Mock()
        req = _FakeReq(
            inflight_middle_chunks=1,
            to_finish=FINISH_ABORT(message="bootstrap failed"),
        )
        req.pending_bootstrap = True
        req.extend_range = SimpleNamespace(end=32)
        batch = _batch(req)

        scheduler.process_batch_result_disagg_prefill(batch, _result(batch))

        scheduler.optimistic_release_and_requeue.assert_not_called()
        release_kv_cache.assert_called_once_with(
            req, scheduler.tree_cache, is_insert=False
        )
        scheduler.output_streamer.stream_output.assert_called_once_with([req], False)

    @patch("sglang.srt.disaggregation.prefill.release_kv_cache", side_effect=_free_req)
    def test_aborted_middle_result_waits_for_last_inflight_chunk(
        self, release_kv_cache
    ):
        scheduler = _Scheduler()
        scheduler.optimistic_release_and_requeue = Mock()
        req = _FakeReq(
            inflight_middle_chunks=1,
            to_finish=FINISH_ABORT(message="bootstrap failed"),
        )
        req.pending_bootstrap = True
        req.extend_range = SimpleNamespace(end=len(req.origin_input_ids))
        batch = _batch(req)

        scheduler.process_batch_result_disagg_prefill(batch, _result(batch))

        release_kv_cache.assert_not_called()
        scheduler.output_streamer.stream_output.assert_not_called()
        scheduler.optimistic_release_and_requeue.assert_not_called()

        scheduler.process_batch_result_disagg_prefill(batch, _result(batch))

        release_kv_cache.assert_called_once_with(
            req, scheduler.tree_cache, is_insert=False
        )
        scheduler.output_streamer.stream_output.assert_called_once_with([req], False)
        scheduler.optimistic_release_and_requeue.assert_not_called()

    def test_mamba_only_allocation_is_released(self):
        scheduler = _Scheduler()
        scheduler.tree_cache.supports_mamba.return_value = True
        req = _FakeReq(
            allocated=False,
            to_finish=FINISH_ABORT(message="input too long"),
        )
        req.mamba_pool_idx = torch.tensor([3])
        batch = _batch(req)

        scheduler.process_batch_result_disagg_prefill(batch, _result(batch))

        scheduler.tree_cache.req_to_token_pool.mamba_allocator.free.assert_called_once()
        self.assertIsNone(req.mamba_pool_idx)
        scheduler.output_streamer.stream_output.assert_called_once_with([req], False)

    @patch("sglang.srt.disaggregation.prefill.release_kv_cache")
    def test_prequeue_abort_skips_admission_ack_prefetch_and_bootstrap(
        self, release_kv_cache
    ):
        scheduler = _Scheduler()
        scheduler._set_or_validate_priority = Mock(return_value=True)
        scheduler.disaggregation_mode = DisaggregationMode.PREFILL
        scheduler._reserve_prefill_tier_admission = Mock(return_value=True)
        scheduler._prefetch_kvcache = Mock()
        scheduler.disagg_prefill_bootstrap_queue = Mock()
        scheduler.ipc_channels = Mock()
        scheduler.model_config = SimpleNamespace(num_key_value_heads=8)
        req = _FakeReq(
            allocated=False,
            to_finish=FINISH_ABORT(message="input too long"),
        )

        Scheduler._add_request_to_queue(scheduler, req)

        scheduler._reserve_prefill_tier_admission.assert_not_called()
        scheduler.ipc_channels.send_to_tokenizer.send_output.assert_not_called()
        scheduler._prefetch_kvcache.assert_not_called()
        scheduler.disagg_prefill_bootstrap_queue.add.assert_not_called()
        scheduler.output_streamer.stream_output.assert_called_once_with([req], False)
        release_kv_cache.assert_not_called()

        # A delayed duplicate reaches the guard after local state has already
        # been retired. It must not emit a second terminal response.
        Scheduler._add_request_to_queue(scheduler, req)
        scheduler.output_streamer.stream_output.assert_called_once_with([req], False)


if __name__ == "__main__":
    unittest.main()
