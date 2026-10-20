"""Real deployed methods with small tensors; CPU negative controls + CUDA roundtrip."""
import ast
from collections import deque
import gc
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

import torch
from sglang.srt.layers.dcp.layout import maybe_dcp_kernel_indices
from sglang.srt.mem_cache.memory_pool import DSATokenToKVPool, MLATokenToKVPool
from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.runtime_context import get_parallel
from sglang.srt.speculative.base_spec_worker import BaseSpecWorker
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

ROOT = Path(__file__).resolve().parent


def original_method(filename, clsname, name, namespace):
    tree = ast.parse((ROOT/'fixtures'/filename).read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == clsname)
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)
    method.decorator_list = []
    exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])), filename, 'exec'), namespace)
    return namespace[name]


def pool(seed, device, dcp, dsa=True):
    p = object.__new__(DSATokenToKVPool if dsa else MLATokenToKVPool)
    p.layer_num = 2
    p.page_size = 4
    p.cpu_offloading_chunk_size = 13
    p.kv_buffer = [(torch.arange(128*16, device=device).reshape(128, 1, 16)+seed+i).to(torch.uint8)
                   for i in range(p.layer_num)]
    if dsa:
        # Packed key AND scale bytes; globally indexed, unlike MLA physical rows.
        p.index_k_with_scale_buffer = [
            (torch.arange(128*dcp*132, device=device).reshape(128*dcp//4, 4*132)+seed+i).to(torch.uint8)
            for i in range(p.layer_num)]
    return p


def request(indices):
    r = object.__new__(Req)
    r.origin_input_ids = [1]*(len(indices)+1)
    r.output_ids = []
    r.req_pool_idx = 0
    r.mamba_pool_idx = None
    return r, NS(req_to_token=indices[None].clone())


class RetractionTests(unittest.TestCase):
    def test_mapping_all_ranks_and_ragged_tails(self):
        for size in (1, 2, 4, 8):
            page = 4*size
            for n in (0, 1, size-1, size, page-1, page, page+3):
                ids = torch.cat([torch.arange(3*page, 4*page), torch.arange(page, 2*page)])[:n]
                for rank in range(size):
                    expected = ids[ids % size == rank] // size
                    self.assertTrue(torch.equal(maybe_dcp_kernel_indices(ids, size, rank), expected))

    def test_original_negative_oob_and_silent_wrong_rows(self):
        old = original_method('memory_pool.original.py', 'MLATokenToKVPool', 'get_cpu_copy',
                              {'current_platform': NS(synchronize=lambda: None)})
        p = pool(5, 'cpu', 2, dsa=False)
        with self.assertRaises(IndexError):
            old(p, torch.arange(192, 208))
        ids = torch.arange(32, 48)
        actual = torch.cat(old(p, ids)[0])
        correct = p.kv_buffer[0][ids[0::2]//2]
        self.assertNotEqual(actual.shape, correct.shape)

    def test_target_and_all_drafts_poison_relocate(self):
        for device in ('cpu', 'cuda'):
            sync_ctx = patch('torch.cuda.synchronize') if device == 'cpu' else __import__('contextlib').nullcontext()
            with sync_ctx:
                for size in (1, 2, 4, 8):
                    for rank in range(size):
                        with self.subTest(device=device, size=size, rank=rank), get_parallel().override(attn_dcp_size=size, attn_dcp_rank=rank):
                            page = 4*size
                            old = torch.cat([torch.arange(3*page, 4*page), torch.arange(page, 2*page)])[:page+3].to(device)
                            new = old + 16*page
                            pools = [pool(seed, device, size) for seed in (1, 17, 39)]
                            allocator = TokenToKVPoolAllocator(128*size, torch.uint8, device, pools[0], False)
                            allocator.register_cpu_retraction_draft_pools(tuple(pools))
                            self.assertEqual(allocator.cpu_retraction_draft_pools, tuple(pools[1:]))
                            local_old = old[old % size == rank]//size
                            local_new = new[new % size == rank]//size
                            expected = [[x[local_old].clone() for x in p.kv_buffer] for p in pools]
                            idx_expected = [[x[old[::4]//4].clone() for x in p.index_k_with_scale_buffer] for p in pools]
                            r, reqpool = request(old)
                            r.offload_kv_cache(reqpool, allocator)
                            for p in pools:
                                for x in p.kv_buffer+p.index_k_with_scale_buffer:
                                    x.fill_(251)
                            reqpool.req_to_token = new[None]
                            r.load_kv_cache(reqpool, allocator)
                            self.assertFalse(hasattr(r, 'kv_cache_cpu'))
                            for j, p in enumerate(pools):
                                for i, x in enumerate(p.kv_buffer):
                                    self.assertTrue(torch.equal(x[local_new], expected[j][i]))
                                    guard = torch.ones(len(x), device=device, dtype=torch.bool)
                                    guard[local_new] = False
                                    self.assertTrue(torch.all(x[guard] == 251).item())
                                for i, x in enumerate(p.index_k_with_scale_buffer):
                                    self.assertTrue(torch.equal(x[new[::4]//4], idx_expected[j][i]))

    def test_original_req_negative_draft_is_lost(self):
        old_offload = original_method('schedule_batch.original.py', 'Req', 'offload_kv_cache', {})
        old_load = original_method('schedule_batch.original.py', 'Req', 'load_kv_cache', {})
        with patch('torch.cuda.synchronize'), get_parallel().override(attn_dcp_size=1, attn_dcp_rank=0):
            target, draft = pool(1, 'cpu', 1), pool(17, 'cpu', 1)
            a = TokenToKVPoolAllocator(128, torch.uint8, 'cpu', target, False)
            a.register_cpu_retraction_draft_pools((draft,))
            ids = torch.arange(8, 16)
            r, rp = request(ids)
            expected = draft.kv_buffer[0][ids].clone()
            old_offload(r, rp, a)
            draft.kv_buffer[0].fill_(251)
            rp.req_to_token = (ids+64)[None]
            old_load(r, rp, a)
            self.assertFalse(torch.equal(draft.kv_buffer[0][ids+64], expected))

    def test_registration_called_by_actual_base_alloc(self):
        for algorithm, unified, should in ((SpeculativeAlgorithm.EAGLE, False, True),
                                           (SpeculativeAlgorithm.EAGLE3, False, True),
                                           (SpeculativeAlgorithm.EAGLE, True, False)):
            obj = NS(_target_worker=NS(model_runner=NS(spec_algorithm=algorithm)),
                     _draft_worker=NS(alloc_memory_pool=Mock(), draft_runners=[NS(token_to_kv_pool=object())]))
            obj.target_worker, obj.draft_worker = obj._target_worker, obj._draft_worker
            allocator = Mock()
            with patch('sglang.srt.speculative.base_spec_worker.get_memory', return_value=NS(enable_unified_memory=unified)):
                BaseSpecWorker.alloc_memory_pool(obj, req_to_token_pool=object(), token_to_kv_pool_allocator=allocator)
            self.assertEqual(allocator.register_cpu_retraction_draft_pools.called, should)

    def test_overlap_checkpoint_before_retract(self):
        for mode, overlap, mem_ok, finished in ((DisaggregationMode.DECODE, True, False, False),
                                               (DisaggregationMode.DECODE, True, True, False),
                                               (DisaggregationMode.DECODE, False, False, False),
                                               (DisaggregationMode.NULL, True, False, False),
                                               (DisaggregationMode.DECODE, True, False, True)):
            with self.subTest(mode=mode, overlap=overlap, mem_ok=mem_ok, finished=finished):
                events = []
                b = NS(batch_size=lambda: 1, filter_batch=lambda: None,
                       is_empty=lambda: finished and 'settle' in events,
                       check_decode_mem=lambda: mem_ok,
                       retract_decode=lambda args: (events.append('retract') or ([], .3, [])),
                       prepare_for_decode=lambda: events.append('prepare'))
                q = deque([(object(), object())])
                s = NS(disaggregation_mode=mode, enable_overlap=overlap, result_queue=q,
                       process_batch_result=lambda *a: events.append('settle'), last_batch=object(),
                       forward_ct=1, token_to_kv_pool_allocator=NS(available_size=lambda: 100),
                       new_token_ratio_tracker=NS(current=.4, decay_step=lambda: None),
                       tree_cache=NS(req_to_token_pool=NS()), server_args=NS(),
                       metrics_reporter=NS(enable_metrics=False), _add_request_to_queue=lambda *a, **kw: None)
                with patch('sglang.srt.managers.scheduler.TEST_RETRACT', False):
                    Scheduler.update_running_batch(s, b)
                expected = mode == DisaggregationMode.DECODE and overlap and not mem_ok
                self.assertEqual('settle' in events, expected)
                if expected:
                    self.assertFalse(q)
                    self.assertIsNone(s.last_batch)
                    if not finished:
                        self.assertLess(events.index('settle'), events.index('retract'))
                    else:
                        self.assertNotIn('retract', events)


if __name__ == '__main__':
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(RetractionTests))
    report = {'passed': result.wasSuccessful(), 'tests': result.testsRun,
              'errors': len(result.errors), 'failures': len(result.failures),
              'cuda': torch.cuda.get_device_name(0),
              'note': 'Run against the patched isolated D runtime; not against this checkout.'}
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if result.wasSuccessful() else 1)
