"""Preserve legacy CP defaults and reject unsupported decode resharding."""

import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch

from sglang.srt.arg_groups.model_overrides.deepseek_v2 import _deepseek_family_overrides
from sglang.srt.arg_groups.parallel_hook import handle_context_parallelism
from sglang.srt.connector import ConnectorType
from sglang.srt.runtime_context import override_platform
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def args(cp=1, decode_tp=False):
    return NS(
        model_path="unused",
        attention_backend=None,
        prefill_attention_backend=None,
        decode_attention_backend=None,
        enable_prefill_cp=True,
        cp_strategy="interleave",
        tp_size=8,
        dp_size=1,
        attn_cp_size=cp,
        ep_size=8,
        kv_cache_dtype="auto",
        moe_a2a_backend="none",
        enable_cp_decode_attn_tp=decode_tp,
        enable_prefill_context_parallel=False,
        enable_dsa_prefill_context_parallel=False,
        enable_aiter_allreduce_fusion=False,
        moe_dp_size=1,
        pp_size=1,
    )


class TestGLM53CPConfig(unittest.TestCase):
    def resolve(self, cfg):
        with patch(
            "sglang.srt.configs.model_config.is_deepseek_dsa", return_value=True
        ), override_platform(is_npu=False, is_xpu=False, is_hip=False):
            declared = _deepseek_family_overrides(cfg, None)
        cfg._resolved_overrides = [("model", declared)]
        return declared

    def test_default_infers_cp8_explicit_cp4_keeps_tp2(self):
        for cp, expected in [(1, 8), (4, 4), (8, 8)]:
            with self.subTest(cp=cp):
                cfg = args(cp)
                self.assertEqual(self.resolve(cfg)["attn_cp_size"], expected)
                with patch(
                    "sglang.srt.arg_groups.parallel_hook.parse_connector_type",
                    return_value=ConnectorType.INSTANCE,
                ), patch("sglang.srt.layers.cp.base.init_cp_strategy") as init:
                    handle_context_parallelism(cfg)
                    self.assertEqual(init.call_args.kwargs["cp_size"], expected)

    def test_decode_resharding_rejects_only_composite_cp_tp(self):
        for cp in (1, 4, 8):
            cfg = args(cp, decode_tp=True)
            self.resolve(cfg)
            with patch(
                "sglang.srt.arg_groups.parallel_hook.parse_connector_type",
                return_value=ConnectorType.INSTANCE,
            ), patch("sglang.srt.layers.cp.base.init_cp_strategy"):
                if cp == 4:
                    with self.assertRaisesRegex(ValueError, "attention TP=1"):
                        handle_context_parallelism(cfg)
                else:
                    handle_context_parallelism(cfg)


if __name__ == "__main__":
    unittest.main()
