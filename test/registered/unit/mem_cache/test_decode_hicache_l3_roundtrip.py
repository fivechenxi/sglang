"""Decode L3 promises must survive a real CUDA -> file -> CUDA round trip."""

import json
import tempfile
import unittest
from array import array
from dataclasses import replace
from types import SimpleNamespace

import torch
from test_unified_radix_cache_unittest import (
    CacheConfig,
    UnifiedRadixCacheSuite,
    build_fixture,
)

from sglang.srt.disaggregation.decode_hicache_mixin import (
    DecodeHiCachePreallocMixin,
    DecodeHiCacheTransferMixin,
    HiCacheRestoreResult,
)
from sglang.srt.mem_cache.base_prefix_cache import InsertParams, MatchPrefixParams
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=20, stage="base-b", runner_config="1-gpu-small")


@unittest.skipUnless(torch.cuda.is_available(), "requires a real CUDA KV pool")
class TestDecodeL3RoundTrip(CustomTestCase):
    cfg = CacheConfig(page_size=64, kv_size=4096, max_context_len=4096, num_layers=2)
    _rid = 0
    _init_hicache = UnifiedRadixCacheSuite._init_hicache
    _make_req = UnifiedRadixCacheSuite._make_req
    _path_chain = UnifiedRadixCacheSuite._path_chain
    _backup_node = UnifiedRadixCacheSuite._backup_node
    _write_path_to_l3 = UnifiedRadixCacheSuite._write_path_to_l3
    _ongoing_l3_backups = UnifiedRadixCacheSuite._ongoing_l3_backups
    _flush_l3_backups = UnifiedRadixCacheSuite._flush_l3_backups
    _run_prefetch_to_completion = UnifiedRadixCacheSuite._run_prefetch_to_completion

    def _round_trip(
        self, *, bigram, hit, expected_promise=None, local_prefix=0, missing_tail=False
    ):
        self.cfg = replace(self.cfg, is_eagle=bigram)
        storage = tempfile.TemporaryDirectory(prefix="decode-l3-regression-")
        self.addCleanup(storage.cleanup)
        raw = array("q", range(1, hit + 1 + int(bigram)))
        # A longer query can find an L3 hit shorter than prefetch_threshold.
        prompt = raw + array("q", range(20000, 21024))
        prod, allocator, _ = build_fixture(self.cfg)
        self._init_hicache(
            prod,
            storage_backend="file",
            storage_dir=storage.name,
            prefetch_threshold=256,
        )
        indices = allocator.alloc(hit)
        prod.insert(InsertParams(key=RadixKey(raw), value=indices))
        match = prod.match_prefix(MatchPrefixParams(key=RadixKey(raw)))
        self.assertEqual(len(match.device_indices), hit)
        pool = allocator.get_kvcache()
        expected = []
        for layer in range(self.cfg.num_layers):
            k = pool.get_key_buffer(layer)
            v = pool.get_value_buffer(layer)
            # Advanced indexing assignment, not fill_ on an indexed copy.
            k[indices] = layer + 3
            v[indices] = layer + 17
            expected.append((k[indices].clone(), v[indices].clone()))
        self._backup_node(prod, match.last_device_node)
        self._write_path_to_l3(prod, match.last_device_node)
        self._flush_l3_backups(prod)

        cons, cons_allocator, req_pool = build_fixture(self.cfg)
        self._init_hicache(
            cons,
            storage_backend="file",
            storage_dir=storage.name,
            prefetch_threshold=256,
        )
        if local_prefix:
            local_indices = cons_allocator.alloc(local_prefix)
            cons.insert(
                InsertParams(
                    key=RadixKey(raw[: local_prefix + int(bigram)]),
                    value=local_indices,
                )
            )
            local_pool = cons_allocator.get_kvcache()
            for layer, (k, v) in enumerate(expected):
                local_pool.get_key_buffer(layer)[local_indices] = k[:local_prefix]
                local_pool.get_value_buffer(layer)[local_indices] = v[:local_prefix]
        req = self._make_req(req_pool)
        req.origin_input_ids = prompt
        result = cons.match_prefix(MatchPrefixParams(key=RadixKey(prompt)))
        self.assertEqual(len(result.device_indices), local_prefix)
        self.assertEqual(result.host_hit_length, 0)
        if local_prefix:
            # Decode only probes past a backed-up anchor (hash chain is known).
            self._backup_node(cons, result.last_device_node)
        harness = SimpleNamespace(
            tree_cache=cons, scheduler=SimpleNamespace(enable_decode_hicache=True)
        )
        pm = DecodeHiCachePreallocMixin._build_decode_prefix_match(harness, req, result)
        self.assertEqual(pm.l3_storage_hit_length, hit - local_prefix)
        DecodeHiCachePreallocMixin._start_hicache_prefetch(harness, req, pm)
        if pm.prefetch_registered:
            self._run_prefetch_to_completion(cons, req.rid)
        host = cons.match_prefix(MatchPrefixParams(key=RadixKey(prompt)))
        print(
            json.dumps(
                dict(
                    bigram=bigram,
                    queried=hit,
                    promised=pm.decode_prefix_len,
                    registered=pm.prefetch_registered,
                    restored_host=host.host_hit_length,
                )
            ),
            flush=True,
        )
        if expected_promise is None:
            expected_promise = hit
        self.assertEqual(pm.decode_prefix_len, expected_promise)
        if expected_promise == 0:
            self.assertFalse(pm.prefetch_registered)
            return
        dr = SimpleNamespace(
            req=req,
            prefix_match=pm,
            hicache_restore_status=HiCacheRestoreResult.PENDING,
        )
        if missing_tail:
            # Simulate a page present at lookup but missing from the final read.
            # The actual loaded span is shorter than the already-advertised end.
            pm.l3_storage_hit_length += self.cfg.page_size
            queued = DecodeHiCacheTransferMixin._try_hicache_queue_load_back(
                harness, dr
            )
            self.assertFalse(queued)
            self.assertEqual(dr.hicache_restore_status, HiCacheRestoreResult.FAILED)
            self.assertFalse(cons.ongoing_load_back)
            cons.sanity_check()
            return
        queued = DecodeHiCacheTransferMixin._try_hicache_queue_load_back(harness, dr)
        self.assertNotEqual(dr.hicache_restore_status, HiCacheRestoreResult.FAILED)
        self.assertTrue(queued)
        cons.ready_to_load_host_cache()
        for ack in list(cons.cache_controller.ack_load_queue):
            ack.finish_event.synchronize()
        cons.loading_check()
        restored = dr.hicache_restored_kv_indices
        self.assertEqual(len(restored), hit - local_prefix)
        restored_pool = cons_allocator.get_kvcache()
        for layer, (k, v) in enumerate(expected):
            torch.testing.assert_close(
                restored_pool.get_key_buffer(layer)[restored],
                k[local_prefix:],
                rtol=0,
                atol=0,
            )
            torch.testing.assert_close(
                restored_pool.get_value_buffer(layer)[restored],
                v[local_prefix:],
                rtol=0,
                atol=0,
            )
        cons.dec_lock_ref(dr.hicache_restored_node)
        cons.sanity_check()

    def test_unigram_control(self):
        self._round_trip(bigram=False, hit=1024)

    def test_bigram_full_hit(self):
        self._round_trip(bigram=True, hit=1024)

    def test_bigram_threshold_boundary(self):
        self._round_trip(bigram=True, hit=256)

    def test_bigram_l1_plus_l3_hit(self):
        self._round_trip(bigram=True, hit=1024, local_prefix=256)

    def test_short_restore_does_not_orphan_dma(self):
        self._round_trip(bigram=True, hit=960, missing_tail=True)

    def test_short_unigram_hit_is_not_promised(self):
        self._round_trip(bigram=False, hit=128, expected_promise=0)


if __name__ == "__main__":
    unittest.main()
