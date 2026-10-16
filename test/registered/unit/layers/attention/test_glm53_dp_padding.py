import unittest
from types import SimpleNamespace as NS
from unittest.mock import MagicMock

import torch

from sglang.srt.layers.attention.dsa.dsa_indexer import (
    _make_eager_idle_topk_result,
)
from sglang.srt.layers.attention.dsa_backend import (
    DeepseekSparseAttnBackend,
    _restore_decode_dp_padding,
    _trim_trtllm_decode_dp_padding,
)
from sglang.srt.speculative.eagle_worker_v2 import (
    _slice_draft_output_to_local_tokens,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestGLM53DPAttentionPadding(unittest.TestCase):
    def test_flashinfer_decode_without_wide_page_table(self):
        for real_rows, physical_rows, wide_table in [
            (2, 4, False),
            (4, 4, False),
            (0, 4, False),
            (6, 8, True),
        ]:
            with self.subTest(real_rows=real_rows, wide_table=wide_table):
                backend = DeepseekSparseAttnBackend.__new__(DeepseekSparseAttnBackend)
                backend.forward_metadata = NS(
                    page_table_1=torch.zeros(real_rows, 8) if wide_table else None,
                    real_page_table=torch.zeros(real_rows, 1, dtype=torch.int32),
                    cache_seqlens_int32=torch.ones(2, dtype=torch.int32),
                    dsa_cache_seqlens_int32=torch.cat(
                        [
                            torch.ones(real_rows, dtype=torch.int32),
                            torch.zeros(physical_rows - real_rows, dtype=torch.int32),
                        ]
                    ),
                )
                backend.dsa_decode_impl = "flashinfer_sparse_mla"
                backend.token_to_kv_pool = NS(get_key_buffer=lambda _: None)
                backend.hisparse_coordinator = None
                backend.use_fused_topk = True
                backend._get_fused_topk_page_table = lambda _: None
                backend._forward_flashinfer_sparse_mla = MagicMock(
                    side_effect=lambda **kw: kw["q_all"][:, :, :4].contiguous()
                )
                layer = NS(
                    is_cross_attention=False,
                    layer_id=0,
                    tp_q_head_num=1,
                    head_dim=6,
                    v_head_dim=4,
                    scaling=1.0,
                )
                q = torch.arange(physical_rows * 6).reshape(physical_rows, 1, 6)
                out = backend.forward_decode(q, None, None, layer, None)
                self.assertEqual(out.shape, (physical_rows, 1, 4))
                torch.testing.assert_close(out[:real_rows], q[:real_rows, :, :4])
                self.assertTrue(torch.all(out[real_rows:] == 0))
                if real_rows:
                    self.assertEqual(
                        backend._forward_flashinfer_sparse_mla.call_args.kwargs[
                            "seq_lens"
                        ].shape[0],
                        real_rows,
                    )
                else:
                    backend._forward_flashinfer_sparse_mla.assert_not_called()

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
