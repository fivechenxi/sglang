import ast
import copy
import os
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from operations.draft_seed import _init_dsa_index_share_state as initialize


def worker():
    return NS(
        draft_runner=NS(model_config=NS(hf_config=NS(
            index_share_for_mtp_iteration=True, index_topk=2048))),
        server_args=NS(disaggregation_mode='decode', tp_size=8, dp_size=4, dcp_size=2),
        topk=1, speculative_num_steps=5, speculative_num_draft_tokens=6, gpu_id=0)


class DraftSeedTests(unittest.TestCase):
    def run_worker(self, value, enabled=True):
        with patch.dict(os.environ, {'GLM53_RESEARCH_D_RECOMPUTE_SEED': '1' if enabled else '0'}):
            initialize(value)

    def test_default_unchanged(self):
        value = worker()
        self.run_worker(value, False)
        self.assertTrue(value.index_share_for_mtp_iteration)
        self.assertTrue(value.seed_dsa_topk_from_draft_extend)

    def test_only_cross_phase_seed_changes(self):
        value = worker()
        self.run_worker(value)
        self.assertTrue(value.index_share_for_mtp_iteration)
        self.assertFalse(value.seed_dsa_topk_from_draft_extend)
        self.assertEqual(value.dsa_index_topk, 2048)

    def test_topology_guard(self):
        for field, item in [('disaggregation_mode', 'prefill'), ('tp_size', 4), ('dp_size', 2), ('dcp_size', 1)]:
            with self.subTest(field=field):
                value = worker()
                setattr(value.server_args, field, item)
                with self.assertRaises(RuntimeError):
                    self.run_worker(value)

    def test_speculation_guard(self):
        for field, item in [('topk', 2), ('speculative_num_steps', 4), ('speculative_num_draft_tokens', 5)]:
            with self.subTest(field=field):
                value = worker()
                setattr(value, field, item)
                with self.assertRaises(RuntimeError):
                    self.run_worker(value)

    def test_model_guard(self):
        for field, item in [('index_share_for_mtp_iteration', False), ('index_topk', 1024)]:
            value = worker()
            setattr(value.draft_runner.model_config.hf_config, field, item)
            with self.assertRaises(RuntimeError):
                self.run_worker(value)

    def test_original_initializer_preserved(self):
        root = Path(__file__).resolve().parents[1]
        candidate = ast.parse((root/'operations/draft_seed.py').read_text())
        fn = copy.deepcopy(next(n for n in candidate.body if isinstance(n, ast.FunctionDef)))
        self.assertIsInstance(fn.body[-1], ast.If)
        fn.body.pop()
        original = ast.parse((root/'tests/seed_original.py').read_text()).body[0]
        self.assertEqual(ast.dump(fn), ast.dump(original))


if __name__ == '__main__':
    unittest.main()
