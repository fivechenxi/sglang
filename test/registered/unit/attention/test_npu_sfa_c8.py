import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.environ import envs
from sglang.srt.hardware_backend.npu.attention.sfa_c8 import (
    get_sfa_c8_packed_head_dim,
    get_sfa_c8_phase1_incompatibilities,
    pack_sfa_c8_kv,
    run_sfa_c8_attention,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestNPUSFAC8(unittest.TestCase):
    def test_phase1_requires_eager_standalone_configuration(self):
        common = dict(
            page_size=128,
            mlapo_enabled=False,
            dcp_size=1,
            prefill_cp_enabled=False,
            disaggregation_mode="null",
            hierarchical_cache_enabled=False,
            cpu_offload_gb=0,
            speculative_decoding_enabled=False,
            graph_enabled=False,
        )
        self.assertEqual([], get_sfa_c8_phase1_incompatibilities(**common))

        common["graph_enabled"] = True
        self.assertEqual(
            ["graph capture/replay (pass --disable-cuda-graph)"],
            get_sfa_c8_phase1_incompatibilities(**common),
        )

        self.assertEqual(
            [
                "page_size other than 128",
                "SGLANG_NPU_USE_MLAPO",
                "decode context parallelism/DCP",
                "DSA prefill context parallelism",
                "PD disaggregation",
                "hierarchical cache",
                "CPU offload",
                "speculative decoding/MTP",
                "graph capture/replay (pass --disable-cuda-graph)",
            ],
            get_sfa_c8_phase1_incompatibilities(
                page_size=64,
                mlapo_enabled=True,
                dcp_size=2,
                prefill_cp_enabled=True,
                disaggregation_mode="decode",
                hierarchical_cache_enabled=True,
                cpu_offload_gb=1,
                speculative_decoding_enabled=True,
                graph_enabled=True,
            ),
        )

    def test_sfa_and_indexer_c8_switches_are_independent(self):
        with patch.dict(
            "os.environ",
            {
                "SGLANG_NPU_ENABLE_SFA_C8": "1",
                "SGLANG_NPU_ENABLE_LI_C8": "0",
            },
        ):
            self.assertTrue(envs.SGLANG_NPU_ENABLE_SFA_C8.get())
            self.assertFalse(envs.SGLANG_NPU_ENABLE_LI_C8.get())

    def test_glm_packed_head_dim_is_656_bytes(self):
        self.assertEqual(656, get_sfa_c8_packed_head_dim(512, 64))
        with self.assertRaisesRegex(ValueError, "divisible"):
            get_sfa_c8_packed_head_dim(510, 64)

    def test_pack_layout_is_int8_rope_bytes_then_fp32_scale_bytes(self):
        k_nope = torch.zeros(2, 1, 512, dtype=torch.bfloat16)
        k_rope = torch.arange(128, dtype=torch.uint8).view(torch.bfloat16)
        k_rope = k_rope.view(1, 1, 64).expand(2, -1, -1).clone()
        quantized = torch.arange(512, dtype=torch.int16).to(torch.int8)
        quantized = quantized.view(1, 1, 512).expand(2, -1, -1).clone()
        scales = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32)
        scales = scales.view(1, 1, 4).expand(2, -1, -1).clone()
        quant = Mock(return_value=(quantized, scales))
        fake_torch_npu = SimpleNamespace(npu_dynamic_block_quant=quant)

        with patch.dict(sys.modules, {"torch_npu": fake_torch_npu}):
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

    def test_native_qsfa_uses_packed_cache_and_required_modes(self):
        expected = torch.randn(2, 4, 512, dtype=torch.bfloat16)
        qsfa = Mock(return_value=expected)
        fake_torch_npu = SimpleNamespace(npu_kv_quant_sparse_flash_attention=qsfa)
        q_nope = torch.randn(2, 4, 512, dtype=torch.bfloat16)
        q_rope = torch.randn(2, 4, 64, dtype=torch.bfloat16)
        packed = torch.empty(4, 128, 1, 656, dtype=torch.int8)
        topk = torch.zeros(2, 1, 8, dtype=torch.int32)
        block_table = torch.zeros(2, 4, dtype=torch.int32)
        q_lens = torch.tensor([1, 2], dtype=torch.int32)
        kv_lens = torch.tensor([32, 64], dtype=torch.int32)

        with patch.dict(sys.modules, {"torch_npu": fake_torch_npu}):
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
