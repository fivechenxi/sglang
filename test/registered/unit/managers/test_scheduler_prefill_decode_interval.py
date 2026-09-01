"""Tests for scheduler prefill/decode interleaving."""

import unittest
from types import SimpleNamespace

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.scheduler_components.dp_attn import MLPSyncBatchInfo

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _make_scheduler(*, interval: int, require_mlp_sync: bool) -> Scheduler:
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.prefill_decode_interval = interval
    scheduler._prefill_decode_interval_remaining = 0
    scheduler._prefill_decode_global_decode_active = False
    scheduler.require_mlp_sync = require_mlp_sync
    return scheduler


def _make_batch(*, prefill: bool, decode_work: bool):
    return SimpleNamespace(
        is_prefill_in_batch=prefill,
        has_decode_work=decode_work,
    )


class TestPrefillDecodeInterval(unittest.TestCase):
    def test_dp_sync_payload_carries_prefill_and_decode_state(self):
        info = MLPSyncBatchInfo(
            dp_size=2,
            tp_size=1,
            cp_size=1,
            num_tokens=8,
            num_tokens_for_logprob=1,
            can_cuda_graph=False,
            is_extend_in_batch=True,
            local_can_run_tbo=False,
            local_forward_mode=1,
            can_run_breakable_cuda_graph=True,
            is_prefill_in_batch=True,
            has_decode_work=True,
        )
        self.assertEqual(info._get_local_tensor("cpu").tolist()[-2:], [1, 1])
        self.assertEqual(info._get_fallback_tensor("cpu").tolist()[-2:], [0, 0])

    def test_disabled_interval_does_not_arm(self):
        scheduler = _make_scheduler(interval=0, require_mlp_sync=False)
        scheduler._arm_prefill_decode_interval(
            _make_batch(prefill=True, decode_work=True)
        )
        self.assertFalse(scheduler._should_defer_prefill())

    def test_interval_arms_for_prefill_with_decode_work(self):
        scheduler = _make_scheduler(interval=2, require_mlp_sync=False)
        scheduler._arm_prefill_decode_interval(
            _make_batch(prefill=True, decode_work=True)
        )
        self.assertTrue(scheduler._should_defer_prefill())
        self.assertTrue(scheduler._should_defer_prefill())
        self.assertFalse(scheduler._should_defer_prefill())

    def test_dp_interval_uses_globally_synchronized_prefill_flag(self):
        scheduler = _make_scheduler(interval=2, require_mlp_sync=True)
        scheduler._arm_prefill_decode_interval(
            _make_batch(prefill=True, decode_work=True)
        )
        self.assertEqual(scheduler._prefill_decode_interval_remaining, 2)
        self.assertTrue(scheduler._should_defer_prefill())

    def test_decode_batch_does_not_rearm_interval(self):
        scheduler = _make_scheduler(interval=2, require_mlp_sync=True)
        scheduler._prefill_decode_interval_remaining = 1
        scheduler._prefill_decode_global_decode_active = True
        scheduler._arm_prefill_decode_interval(
            _make_batch(prefill=False, decode_work=True)
        )
        self.assertEqual(scheduler._prefill_decode_interval_remaining, 1)

    def test_pure_prefill_does_not_arm_or_idle(self):
        scheduler = _make_scheduler(interval=2, require_mlp_sync=True)
        scheduler._arm_prefill_decode_interval(
            _make_batch(prefill=True, decode_work=False)
        )
        self.assertEqual(scheduler._prefill_decode_interval_remaining, 0)
        self.assertFalse(scheduler._should_defer_prefill())

    def test_speculative_verify_does_not_rearm_prefill_interval(self):
        scheduler = _make_scheduler(interval=2, require_mlp_sync=True)
        scheduler._prefill_decode_interval_remaining = 1
        scheduler._prefill_decode_global_decode_active = True
        scheduler._arm_prefill_decode_interval(
            _make_batch(prefill=False, decode_work=True)
        )
        self.assertEqual(scheduler._prefill_decode_interval_remaining, 1)

    def test_prefill_protection_ends_when_global_decode_drains(self):
        scheduler = _make_scheduler(interval=2, require_mlp_sync=True)
        scheduler._prefill_decode_interval_remaining = 2
        scheduler._prefill_decode_global_decode_active = True
        scheduler._arm_prefill_decode_interval(
            _make_batch(prefill=False, decode_work=False)
        )
        self.assertFalse(scheduler._should_defer_prefill())
        self.assertEqual(scheduler._prefill_decode_interval_remaining, 0)

    def test_idle_batch_clears_stale_protection_window(self):
        scheduler = _make_scheduler(interval=2, require_mlp_sync=True)
        scheduler._prefill_decode_interval_remaining = 2
        scheduler._prefill_decode_global_decode_active = True
        scheduler._arm_prefill_decode_interval(None)
        self.assertFalse(scheduler._prefill_decode_global_decode_active)
        self.assertEqual(scheduler._prefill_decode_interval_remaining, 0)


if __name__ == "__main__":
    unittest.main()
