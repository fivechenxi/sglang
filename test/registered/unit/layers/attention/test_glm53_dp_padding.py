import unittest

import torch

from sglang.srt.layers.attention.dsa.dsa_indexer import (
    _make_eager_idle_topk_result,
)
from sglang.srt.layers.attention.dsa_backend import (
    _restore_decode_dp_padding,
    _trim_trtllm_decode_dp_padding,
)
from sglang.srt.speculative.eagle_worker_v2 import (
    _slice_draft_output_to_local_tokens,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestGLM53DPAttentionPadding(unittest.TestCase):
    def test_idle_indexer_returns_only_dummy_topk(self):
        x = torch.empty((3, 16))
        result = _make_eager_idle_topk_result(x, index_topk=4, return_indices=True)
        self.assertEqual(result.shape, (3, 4))
        self.assertEqual(result.dtype, torch.int32)
        self.assertTrue(torch.all(result == -1))
        self.assertIsNone(
            _make_eager_idle_topk_result(x, index_topk=4, return_indices=False)
        )

    def test_sparse_decode_trims_to_metadata_rows_and_restores_padding(self):
        q = torch.arange(12).reshape(3, 1, 4)
        topk = torch.arange(15).reshape(3, 5)
        local_q, local_topk, padding = _trim_trtllm_decode_dp_padding(q, topk, 2)
        self.assertEqual(local_q.shape[0], 2)
        self.assertEqual(local_topk.shape[0], 2)
        self.assertEqual(padding, 1)
        torch.testing.assert_close(local_q, q[:2])

        output = _restore_decode_dp_padding(local_q, padding)
        self.assertEqual(output.shape, q.shape)
        torch.testing.assert_close(output[:2], q[:2])
        self.assertTrue(torch.all(output[2] == 0))

    def test_draft_postprocessing_ignores_dp_padding(self):
        logits = torch.arange(24).reshape(3, 8)
        hidden = torch.arange(12).reshape(3, 4)
        positions = torch.tensor([10, 20, 0])
        local_logits, local_hidden, local_positions = (
            _slice_draft_output_to_local_tokens(logits, hidden, positions, 2)
        )
        self.assertEqual(local_logits.shape, (2, 8))
        self.assertEqual(local_hidden.shape, (2, 4))
        self.assertEqual(local_positions.tolist(), [10, 20])
        with self.assertRaises(RuntimeError):
            _slice_draft_output_to_local_tokens(logits[:1], hidden, positions, 2)


if __name__ == "__main__":
    unittest.main()
