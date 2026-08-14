import unittest
from unittest.mock import patch

import torch

from sglang.srt.hardware_backend.npu.attention.sfa_c8 import (
    get_sfa_c8_packed_head_dim,
    pack_sfa_c8_kv,
    run_sfa_c8_attention,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestNPUSFAC8(unittest.TestCase):
    def test_glm_packed_head_dim_is_656_bytes(self):
        self.assertEqual(656, get_sfa_c8_packed_head_dim(512, 64))
        with self.assertRaisesRegex(ValueError, "divisible"):
            get_sfa_c8_packed_head_dim(510, 64)

    @patch("torch_npu.npu_dynamic_block_quant")
    def test_pack_layout_is_int8_rope_bytes_then_fp32_scale_bytes(self, quant):
        k_nope = torch.zeros(2, 1, 512, dtype=torch.bfloat16)
        k_rope = torch.arange(128, dtype=torch.uint8).view(torch.bfloat16)
        k_rope = k_rope.view(1, 1, 64).expand(2, -1, -1).clone()
        quantized = torch.arange(512, dtype=torch.int16).to(torch.int8)
        quantized = quantized.view(1, 1, 512).expand(2, -1, -1).clone()
        scales = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32)
        scales = scales.view(1, 1, 4).expand(2, -1, -1).clone()
        quant.return_value = quantized, scales

        packed = pack_sfa_c8_kv(
            k_nope,
            k_rope,
            kv_lora_rank=512,
            qk_rope_head_dim=64,
        )

        self.assertEqual(torch.int8, packed.dtype)
        self.assertEqual((2, 1, 656), packed.shape)
        torch.testing.assert_close(packed[..., :512], quantized)
        torch.testing.assert_close(
            packed[..., 512:640], k_rope.contiguous().view(torch.int8)
        )
        torch.testing.assert_close(
            packed[..., 640:], scales.contiguous().view(torch.int8)
        )
        self.assertEqual(1, quant.call_args.kwargs["dst_type"])
        self.assertEqual(128, quant.call_args.kwargs["col_block_size"])

    @patch("torch_npu.npu_kv_quant_sparse_flash_attention")
    def test_native_qsfa_uses_packed_cache_and_required_modes(self, qsfa):
        expected = torch.randn(2, 4, 512, dtype=torch.bfloat16)
        qsfa.return_value = expected
        q_nope = torch.randn(2, 4, 512, dtype=torch.bfloat16)
        q_rope = torch.randn(2, 4, 64, dtype=torch.bfloat16)
        packed = torch.empty(4, 128, 1, 656, dtype=torch.int8)
        topk = torch.zeros(2, 1, 8, dtype=torch.int32)
        block_table = torch.zeros(2, 4, dtype=torch.int32)
        q_lens = torch.tensor([1, 2], dtype=torch.int32)
        kv_lens = torch.tensor([32, 64], dtype=torch.int32)

        result = run_sfa_c8_attention(
            q_nope,
            q_rope,
            packed,
            topk,
            scale_value=0.125,
            block_table=block_table,
            actual_seq_lengths_query=q_lens,
            actual_seq_lengths_kv=kv_lens,
            rope_head_dim=64,
        )

        self.assertIs(result, expected)
        kwargs = qsfa.call_args.kwargs
        self.assertIs(kwargs["key"], packed)
        self.assertIs(kwargs["value"], packed)
        self.assertEqual((2, 4, 576), kwargs["query"].shape)
        self.assertEqual(2, kwargs["key_quant_mode"])
        self.assertEqual(2, kwargs["value_quant_mode"])
        self.assertEqual(1, kwargs["quant_scale_repo_mode"])
        self.assertEqual(128, kwargs["tile_size"])
        self.assertEqual(64, kwargs["rope_head_dim"])


if __name__ == "__main__":
    unittest.main()
