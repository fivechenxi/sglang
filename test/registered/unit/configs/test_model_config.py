"""Unit tests for hybrid attention model configuration."""

import unittest
from types import SimpleNamespace

from sglang.srt.configs.model_config import (
    get_dsa_indexer_layer_ids,
    get_hybrid_layer_ids,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestHybridLayerIds(CustomTestCase):
    def test_layer_type_architectures(self):
        config = SimpleNamespace(
            num_hidden_layers=4,
            layer_types=[
                "sliding_attention",
                "full_attention",
                "sliding_attention",
                "full_attention",
            ],
        )

        for architecture in (
            "Gemma4ForCausalLM",
            "Gemma4ForConditionalGeneration",
            "LagunaForCausalLM",
            "MellumForCausalLM",
        ):
            with self.subTest(architecture=architecture):
                self.assertEqual(
                    get_hybrid_layer_ids([architecture], config),
                    ([0, 2], [1, 3]),
                )


class TestDSAIndexerLayerIds(CustomTestCase):
    def test_glm52_topk_sharing_uses_21_physical_indexer_layers(self):
        config = SimpleNamespace(
            architectures=["GlmMoeDsaForCausalLM"],
            num_hidden_layers=78,
            index_topk=2048,
            index_topk_freq=4,
            index_skip_topk_offset=3,
        )

        layer_ids = get_dsa_indexer_layer_ids(config)

        self.assertEqual(len(layer_ids), 21)
        self.assertEqual(layer_ids[:5], [0, 1, 2, 6, 10])
        self.assertEqual(layer_ids[-1], 74)

    def test_pp_range_keeps_global_layer_ids(self):
        config = SimpleNamespace(
            architectures=["GlmMoeDsaForCausalLM"],
            num_hidden_layers=78,
            index_topk=2048,
            index_topk_freq=4,
            index_skip_topk_offset=3,
        )

        self.assertEqual(
            get_dsa_indexer_layer_ids(config, start_layer=20, end_layer=32),
            [22, 26, 30],
        )


if __name__ == "__main__":
    unittest.main()
