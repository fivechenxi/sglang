import unittest

import torch

from sglang.srt.hardware_backend.npu.memory_pool_npu import NPUMLATokenToKVPool
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestNPUMLACompactIndexerPool(unittest.TestCase):
    def test_indexer_layer_ids_map_to_compact_slots(self):
        pool = NPUMLATokenToKVPool(
            size=4,
            page_size=1,
            dtype=torch.bfloat16,
            kv_lora_rank=2,
            qk_rope_head_dim=1,
            index_head_dim=3,
            layer_num=4,
            device="cpu",
            enable_memory_saver=False,
            start_layer=0,
            end_layer=4,
            indexer_layer_ids=[0, 2],
        )

        self.assertEqual(pool.index_k_buffer.shape[0], 2)
        self.assertEqual(pool.indexer_layer_id_to_slot, {0: 0, 2: 1})
        self.assertEqual(
            pool.get_index_k_buffer(2).data_ptr(), pool.index_k_buffer[1].data_ptr()
        )
        with self.assertRaisesRegex(ValueError, "does not own"):
            pool.get_index_k_buffer(1)

    def test_sfa_c8_allocates_packed_main_cache_and_bf16_indexer(self):
        pool = NPUMLATokenToKVPool(
            size=4,
            page_size=1,
            dtype=torch.bfloat16,
            kv_lora_rank=128,
            qk_rope_head_dim=16,
            index_head_dim=32,
            layer_num=2,
            device="cpu",
            enable_memory_saver=False,
            start_layer=0,
            end_layer=2,
            indexer_layer_ids=[0],
            sfa_c8_enabled=True,
        )

        self.assertIsNone(pool.k_buffer)
        self.assertIsNone(pool.v_buffer)
        self.assertEqual(torch.int8, pool.packed_kv_buffer.dtype)
        self.assertEqual((2, 5, 1, 1, 164), pool.packed_kv_buffer.shape)
        self.assertEqual(torch.bfloat16, pool.index_k_buffer.dtype)
        self.assertEqual((1, 5, 1, 1, 32), pool.index_k_buffer.shape)
        with self.assertRaisesRegex(RuntimeError, "packed"):
            pool.get_kv_buffer(0)


if __name__ == "__main__":
    unittest.main()
