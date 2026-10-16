"""Shared load-back consumers wait for data, not just published addresses."""

import unittest
from types import SimpleNamespace as NS

from sglang.srt.disaggregation.decode_hicache_mixin import (
    DecodeHiCacheTransferMixin,
    HiCacheRestoreResult,
)
from sglang.srt.mem_cache.base_prefix_cache import load_back_paths_ready
from sglang.srt.mem_cache.hiradix_cache import HiRadixCache
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestSharedRestore(unittest.TestCase):
    def setUp(self):
        self.root = NS(id=0, parent=None)
        self.shared = NS(id=1, parent=self.root)
        self.tail = NS(id=2, parent=self.shared)
        self.unrelated = NS(id=3, parent=self.root)
        self.pending = [self.tail]

    def test_shared_ancestor_waits_but_unrelated_prefix_does_not(self):
        self.assertFalse(load_back_paths_ready(self.shared, self.pending))
        self.assertTrue(load_back_paths_ready(self.unrelated, self.pending))
        self.assertTrue(load_back_paths_ready(self.root, self.pending))
        self.pending.clear()
        self.assertTrue(load_back_paths_ready(self.shared, self.pending))

    def test_split_fragment_inherits_live_path_dependency(self):
        split = NS(id=4, parent=self.shared)
        self.tail.parent = split
        self.assertFalse(load_back_paths_ready(split, self.pending))
        self.pending.clear()
        self.assertTrue(load_back_paths_ready(split, self.pending))

    def test_local_dma_completion_does_not_wait_for_collective_ack_reaping(self):
        for cls in (HiRadixCache, UnifiedRadixCache):
            cache = NS(
                ongoing_load_back={2: self.tail},
                cache_controller=NS(ack_load_queue=[]),
                resolve_node_handle=lambda n: self.tail if n == 2 else n,
            )
            self.assertFalse(cls.is_load_back_ready(cache, self.shared))
            done = False
            cache.cache_controller.ack_load_queue.append(
                NS(node_ids=[2], finish_event=NS(query=lambda: done))
            )
            self.assertFalse(cls.is_load_back_ready(cache, self.shared))
            done = True
            self.assertTrue(cls.is_load_back_ready(cache, self.shared))
            self.assertTrue(cache.ongoing_load_back)  # ack has not been reaped

    def test_late_l1_consumer_and_original_restore_wait_for_ack(self):
        mixin = DecodeHiCacheTransferMixin()
        mixin.tree_cache = NS(
            is_load_back_event_done=lambda i: True,  # recyclable slot is not proof
            is_load_back_ready=lambda n: load_back_paths_ready(n, self.pending),
            cache_controller=NS(
                layer_done_counter=NS(producer_index=0, num_counters=3)
            ),
        )
        late = NS(
            hicache_restore_status=HiCacheRestoreResult.PENDING,
            prefix_match=NS(needs_local_restore=False, last_device_node=self.shared),
        )
        original = NS(
            hicache_restore_status=HiCacheRestoreResult.PENDING,
            prefix_match=NS(needs_local_restore=True),
            hicache_restored_node=self.tail,
            hicache_load_consumer_index=-1,
        )
        for ready in (False, True):
            if ready:
                self.pending.clear()
            mixin._process_hicache_local_restores([late, original])
            expected = (
                HiCacheRestoreResult.READY if ready else HiCacheRestoreResult.PENDING
            )
            self.assertEqual(late.hicache_restore_status, expected)
            self.assertEqual(original.hicache_restore_status, expected)


if __name__ == "__main__":
    unittest.main()
