import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.managers.cache_controller import HiCacheController
from sglang.srt.mem_cache.pool_host import sfa_c8 as sfa_c8_host
from sglang.srt.mem_cache.pool_host.sfa_c8 import NPUSFAC8TokenToKVPoolHost
from sglang.srt.mem_cache.storage.mooncake_store.mooncake_store import MooncakeStore
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _device_pool(*, indexer_layer_ids=None):
    page_size = 4
    size = 8
    if indexer_layer_ids is None:
        indexer_layer_ids = [0]
    pool = SimpleNamespace(
        sfa_c8_enabled=True,
        sfa_c8_packed_head_dim=6,
        index_head_dim=2,
        indexer_layer_num=len(indexer_layer_ids),
        indexer_layer_ids=indexer_layer_ids,
        indexer_layer_id_to_slot={
            layer_id: slot for slot, layer_id in enumerate(indexer_layer_ids)
        },
        layer_num=2,
        page_size=page_size,
        layer_shard_enabled=False,
        start_layer=0,
        end_layer=2,
        size=size,
        device="cpu",
        store_dtype=torch.bfloat16,
        packed_kv_buffer=torch.zeros(
            2, size // page_size + 1, page_size, 1, 6, dtype=torch.int8
        ),
        index_k_buffer=torch.zeros(
            len(indexer_layer_ids),
            size // page_size + 1,
            page_size,
            1,
            2,
            dtype=torch.bfloat16,
        ),
    )
    pool.get_sfa_c8_page_payload_descriptor = lambda: (
        {
            "name": "sfa",
            "buffer": pool.packed_kv_buffer,
            "dtype": pool.packed_kv_buffer.dtype,
            "layers": pool.layer_num,
            "page_bytes": pool.packed_kv_buffer[0, 0].nbytes,
        },
        {
            "name": "lightning_indexer",
            "buffer": pool.index_k_buffer,
            "dtype": pool.index_k_buffer.dtype,
            "layers": pool.indexer_layer_num,
            "page_bytes": pool.index_k_buffer[0, 0].nbytes,
        },
    )
    return pool


class TestNPUSFAC8HostPool(unittest.TestCase):
    def test_npu_path_uses_dim_exchange_without_bf16_expansion(self):
        device = _device_pool()
        host = NPUSFAC8TokenToKVPoolHost(
            device, 1, 0, 4, "page_first_kv_split", pin_memory=False
        )
        transfer = Mock()
        directions = SimpleNamespace(D2H="d2h", H2D="h2d")
        with ExitStack() as stack:
            stack.enter_context(patch.object(sfa_c8_host, "_is_npu", True))
            stack.enter_context(
                patch.object(
                    sfa_c8_host,
                    "transfer_kv_dim_exchange",
                    transfer,
                    create=True,
                )
            )
            stack.enter_context(
                patch.object(sfa_c8_host, "TransferDirection", directions, create=True)
            )
            host.backup_from_device_all_layer(
                device, torch.arange(4), torch.arange(4, 8), "kernel_ascend"
            )

        kwargs = transfer.call_args.kwargs
        self.assertIs(kwargs["device_k"], device.packed_kv_buffer)
        self.assertIs(kwargs["host_k"], host.packed_kv_buffer)
        self.assertEqual(0, kwargs["device_v"].numel())
        self.assertIs(kwargs["device_index_k"], device.index_k_buffer)
        self.assertEqual("d2h", kwargs["direction"])

    def test_physical_page_remap_round_trip_preserves_both_regions(self):
        device = _device_pool()
        host = NPUSFAC8TokenToKVPoolHost(
            device,
            host_to_device_ratio=1,
            host_size=0,
            page_size=4,
            layout="page_first_kv_split",
            pin_memory=False,
        )
        device.packed_kv_buffer[:, 1].fill_(7)
        device.index_k_buffer[:, 1].fill_(3)
        host_indices = torch.arange(8, 12)
        device_indices = torch.arange(4, 8)

        host.backup_from_device_all_layer(
            device, host_indices, device_indices, "kernel_ascend"
        )
        device.packed_kv_buffer[:, 1].zero_()
        device.index_k_buffer[:, 1].zero_()
        host.load_to_device_per_layer(
            device, host_indices, device_indices, 0, "kernel_ascend"
        )

        self.assertTrue(torch.all(device.packed_kv_buffer[:, 1] == 7))
        self.assertTrue(torch.all(device.index_k_buffer[:, 1] == 3))

    def test_logical_page_metadata_has_one_iovec_per_region(self):
        host = NPUSFAC8TokenToKVPoolHost(
            _device_pool(),
            host_to_device_ratio=1,
            host_size=0,
            page_size=4,
            layout="page_first_kv_split",
            pin_memory=False,
        )
        pages = host.get_logical_page_buffer_meta(torch.arange(4, 8))
        self.assertEqual(1, len(pages))
        self.assertEqual(2, len(pages[0]))
        self.assertEqual(host.packed_kv_buffer[0].nbytes, pages[0][0][1])
        self.assertEqual(host.index_k_buffer[0].nbytes, pages[0][1][1])

        flat_ptrs, flat_sizes = host.get_page_buffer_meta(torch.arange(4, 8))
        self.assertEqual([item[0] for item in pages[0]], flat_ptrs)
        self.assertEqual([item[1] for item in pages[0]], flat_sizes)

    def test_draft_regions_join_the_same_logical_page(self):
        target = NPUSFAC8TokenToKVPoolHost(
            _device_pool(), 1, 0, 4, "page_first_kv_split", pin_memory=False
        )
        draft = NPUSFAC8TokenToKVPoolHost(
            _device_pool(), 1, 0, 4, "page_first_kv_split", pin_memory=False
        )
        target.attach_draft_host_pool(draft)
        self.assertIs(draft.logical_page_anchor, target)
        pages = target.get_logical_page_buffer_meta(torch.arange(4))
        self.assertEqual(4, len(pages[0]))
        self.assertEqual(
            target.size_per_token + draft.size_per_token,
            target.get_ksize_per_token(),
        )

        wrapper = SimpleNamespace(
            anchor_entry=SimpleNamespace(host_pool=target),
            page_size=target.page_size,
            get_page_buffer_meta=target.get_page_buffer_meta,
        )
        store = object.__new__(MooncakeStore)
        store.mem_pool_host = wrapper
        store.mla_suffix = ""
        keys, ptrs, sizes = store._get_mla_buffer_meta(["prefix"], torch.arange(4))
        self.assertEqual([f"prefix__k_{target.storage_key_suffix}"], keys)
        self.assertEqual(1, len(ptrs))
        self.assertEqual(4, len(ptrs[0]))
        self.assertEqual(4, len(sizes[0]))

    def test_controller_folds_draft_into_anchor_instead_of_sidecar(self):
        anchor = Mock()
        group = SimpleNamespace(anchor_entry=SimpleNamespace(host_pool=anchor))
        storage = Mock()
        storage.config.standalone_storage = False
        controller = object.__new__(HiCacheController)
        controller.has_draft = True
        controller.mem_pool_host = group
        controller.mem_pool_host_draft = Mock()
        controller.enable_storage = True
        controller.storage_backend_type = "mooncake"
        controller.storage_backend = storage

        controller._maybe_register_draft_with_storage()

        anchor.attach_draft_host_pool.assert_called_once_with(
            controller.mem_pool_host_draft
        )
        storage.register_logical_page_pool_extension.assert_called_once_with(
            controller.mem_pool_host_draft
        )
        self.assertIsNone(controller.draft_page_get_func)
        self.assertIsNone(controller.draft_page_set_func)

    def test_layout_fingerprint_covers_indexer_mapping_and_draft(self):
        target = NPUSFAC8TokenToKVPoolHost(
            _device_pool(indexer_layer_ids=[0]),
            1,
            0,
            4,
            "page_first_kv_split",
            pin_memory=False,
        )
        other_mapping = NPUSFAC8TokenToKVPoolHost(
            _device_pool(indexer_layer_ids=[1]),
            1,
            0,
            4,
            "page_first_kv_split",
            pin_memory=False,
        )
        before_draft = target.storage_key_suffix

        self.assertNotEqual(before_draft, other_mapping.storage_key_suffix)
        self.assertTrue(before_draft.startswith("sfa_c8_logical_v2_p4_"))

        target.attach_draft_host_pool(other_mapping)
        self.assertNotEqual(before_draft, target.storage_key_suffix)

    def test_device_payload_descriptor_is_validated(self):
        device = _device_pool()
        payloads = list(device.get_sfa_c8_page_payload_descriptor())
        payloads[0] = {**payloads[0], "page_bytes": payloads[0]["page_bytes"] + 1}
        device.get_sfa_c8_page_payload_descriptor = lambda: tuple(payloads)

        with self.assertRaisesRegex(ValueError, "page bytes mismatch"):
            NPUSFAC8TokenToKVPoolHost(
                device, 1, 0, 4, "page_first_kv_split", pin_memory=False
            )

        with self.assertRaisesRegex(ValueError, "page sizes differ"):
            NPUSFAC8TokenToKVPoolHost(
                _device_pool(), 1, 0, 8, "page_first_kv_split", pin_memory=False
            )

    def test_page_indices_must_be_aligned_contiguous_and_in_range(self):
        device = _device_pool()
        host = NPUSFAC8TokenToKVPoolHost(
            device, 1, 0, 4, "page_first_kv_split", pin_memory=False
        )

        with self.assertRaisesRegex(ValueError, "contiguous"):
            host.get_logical_page_buffer_meta(torch.tensor([0, 1, 3, 4]))
        with self.assertRaisesRegex(ValueError, "page boundary"):
            host.get_logical_page_buffer_meta(torch.tensor([1, 2, 3, 4]))
        with self.assertRaisesRegex(ValueError, "out-of-range"):
            host.get_logical_page_buffer_meta(torch.arange(12, 16))
        with self.assertRaisesRegex(ValueError, "duplicate pages"):
            host.get_logical_page_buffer_meta(torch.tensor([0, 1, 2, 3, 0, 1, 2, 3]))
        with self.assertRaisesRegex(TypeError, "integer dtype"):
            host.get_logical_page_buffer_meta(torch.arange(4, dtype=torch.float32))
        with self.assertRaisesRegex(ValueError, "out-of-range"):
            host.backup_from_device_all_layer(
                device, torch.arange(4), torch.arange(4), "kernel_ascend"
            )

    def test_draft_attachment_rejects_ambiguous_ownership(self):
        target = NPUSFAC8TokenToKVPoolHost(
            _device_pool(), 1, 0, 4, "page_first_kv_split", pin_memory=False
        )
        draft = NPUSFAC8TokenToKVPoolHost(
            _device_pool(), 1, 0, 4, "page_first_kv_split", pin_memory=False
        )
        other = NPUSFAC8TokenToKVPoolHost(
            _device_pool(), 1, 0, 4, "page_first_kv_split", pin_memory=False
        )

        with self.assertRaisesRegex(ValueError, "cannot attach itself"):
            target.attach_draft_host_pool(target)
        target.attach_draft_host_pool(draft)
        target.attach_draft_host_pool(draft)
        with self.assertRaisesRegex(ValueError, "different draft"):
            target.attach_draft_host_pool(other)
        with self.assertRaisesRegex(ValueError, "cannot own another"):
            draft.attach_draft_host_pool(other)

        smaller_device = _device_pool()
        smaller_device.size = 4
        smaller_device.packed_kv_buffer = torch.zeros(2, 2, 4, 1, 6, dtype=torch.int8)
        smaller_device.index_k_buffer = torch.zeros(1, 2, 4, 1, 2, dtype=torch.bfloat16)
        smaller = NPUSFAC8TokenToKVPoolHost(
            smaller_device, 2, 0, 4, "page_first_kv_split", pin_memory=False
        )
        fresh_target = NPUSFAC8TokenToKVPoolHost(
            _device_pool(), 1, 0, 4, "page_first_kv_split", pin_memory=False
        )
        with self.assertRaisesRegex(ValueError, "device capacities differ"):
            fresh_target.attach_draft_host_pool(smaller)

    def test_mooncake_l3_requires_explicit_cache_identity(self):
        host = NPUSFAC8TokenToKVPoolHost(
            _device_pool(), 1, 0, 4, "page_first_kv_split", pin_memory=False
        )
        backend = SimpleNamespace(
            batch_put_from_multi_buffers=lambda *args: [0],
            batch_get_into_multi_buffers=lambda *args: [1],
            register_buffer=lambda *args: 0,
        )
        store = object.__new__(MooncakeStore)
        store.store = backend
        store.extra_backend_tag = None
        store._replicate_config_cls = SimpleNamespace

        with self.assertRaisesRegex(ValueError, "extra_backend_tag"):
            store.register_mem_pool_host(host)

        store.extra_backend_tag = "model-tokenizer-release"
        store.register_mem_pool_host(host)
        self.assertGreater(store.gb_per_page, 0)

    def test_mooncake_multi_buffer_result_count_is_validated(self):
        store = object.__new__(MooncakeStore)
        store._use_group_semantics = False
        store._replicate_config_cls = SimpleNamespace
        store.store = SimpleNamespace(
            batch_put_from_multi_buffers=lambda *args: [],
            batch_get_into_multi_buffers=lambda *args: [],
        )
        keys = ["page"]
        ptrs = [[1, 2]]
        sizes = [[8, 16]]

        with self.assertRaisesRegex(RuntimeError, "put returned"):
            store._put_batch_zero_copy_impl(keys, ptrs, sizes)
        with self.assertRaisesRegex(RuntimeError, "get returned"):
            store._get_batch_zero_copy_impl(keys, ptrs, sizes)

        store.store.batch_get_into_multi_buffers = lambda *args: [8]
        with self.assertRaisesRegex(RuntimeError, "partial logical page"):
            store._get_batch_zero_copy_impl(keys, ptrs, sizes)

        store.store.batch_get_into_multi_buffers = lambda *args: [24]
        self.assertEqual(store._get_batch_zero_copy_impl(keys, ptrs, sizes), [24])

        with self.assertRaisesRegex(ValueError, "pointer and size counts differ"):
            store._pack_multi_buffer_meta(["page"], [1], [])

    def test_mooncake_group_identity_includes_sfa_layout(self):
        host = NPUSFAC8TokenToKVPoolHost(
            _device_pool(), 1, 0, 4, "page_first_kv_split", pin_memory=False
        )
        calls = []

        def put(keys, ptrs, sizes, config):
            calls.append((keys, ptrs, sizes, config))
            return [0] * len(keys)

        store = object.__new__(MooncakeStore)
        store.mem_pool_host = host
        store.extra_backend_tag = "model-release"
        store.mla_suffix = ""
        store.is_mla_backend = True
        store.should_split_heads = False
        store.enable_storage_metrics = False
        store._use_group_semantics = True
        store._replicate_config_cls = SimpleNamespace
        store.store = SimpleNamespace(
            batch_is_exist=lambda keys: [0] * len(keys),
            batch_put_from_multi_buffers=put,
        )

        self.assertEqual(store.batch_set_v1(["page"], torch.arange(4)), [True])
        keys, _, _, config = calls[0]
        self.assertEqual(keys, [f"model-release_page__k_{host.storage_key_suffix}"])
        self.assertEqual(config.group_ids, [f"sglang-hicache:{keys[0]}"])


if __name__ == "__main__":
    unittest.main()
