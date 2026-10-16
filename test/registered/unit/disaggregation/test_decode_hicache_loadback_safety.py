"""A rank-local CUDA event query must never issue a collective."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from sglang.srt.mem_cache.hiradix_cache import HiRadixCache
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestLoadBackEventSafety(unittest.TestCase):
    def test_rank_local_readiness_has_no_collective_side_effect(self):
        for cache_class in (HiRadixCache, UnifiedRadixCache):
            for ready in (False, True):
                with self.subTest(cache=cache_class.__name__, ready=ready):
                    event = Mock()
                    event.query.return_value = ready
                    cache = SimpleNamespace(
                        cache_controller=SimpleNamespace(
                            layer_done_counter=SimpleNamespace(
                                events=[SimpleNamespace(finish_event=event)]
                            )
                        ),
                        loading_check=Mock(),
                    )
                    self.assertEqual(
                        cache_class.is_load_back_event_done(cache, 0), ready
                    )
                    cache.loading_check.assert_not_called()
                    self.assertTrue(cache_class.is_load_back_event_done(cache, -1))


if __name__ == "__main__":
    unittest.main()
