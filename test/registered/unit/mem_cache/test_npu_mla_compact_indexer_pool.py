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


if __name__ == "__main__":
    unittest.main()
