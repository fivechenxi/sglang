from __future__ import annotations

import torch

from sglang.srt.mem_cache.pool_host.base import HostKVCache
from sglang.srt.mem_cache.pool_host.common import ALLOC_MEMORY_FUNCS
from sglang.srt.utils import is_npu

_is_npu = is_npu()
if _is_npu:
    from sgl_kernel_npu.kvcacheio import TransferDirection, transfer_kv_dim_exchange


class NPUSFAC8TokenToKVPoolHost(HostKVCache):
    """Page-first Host mirror for the native Ascend SFA C8 layout.

    The packed SFA bytes and BF16 Lightning Indexer rows share one page
    allocator and one completion event in ``HiCacheController``.  No payload is
    converted to the ordinary MLA K/V representation.
    """

    @property
    def storage_key_suffix(self) -> str:
        index = f"i{self.indexer_layer_num}x{self.index_head_dim or 0}"
        suffix = (
            f"sfa_c8_logical_v1_p{self.page_size}_"
            f"m{self.layer_num}x{self.packed_head_dim}_{index}"
        )
        if self.draft_host_pool is not None:
            draft = self.draft_host_pool
            suffix += (
                f"_dm{draft.layer_num}x{draft.packed_head_dim}_"
                f"di{draft.indexer_layer_num}x{draft.index_head_dim or 0}"
            )
        return suffix

    def __init__(
        self,
        device_pool,
        host_to_device_ratio: float,
        host_size: int,
        page_size: int,
        layout: str,
        pin_memory: bool = True,
        device: str = "cpu",
        allocator_type: str = "default",
    ):
        if not getattr(device_pool, "sfa_c8_enabled", False):
            raise ValueError("NPUSFAC8TokenToKVPoolHost requires an SFA C8 pool")
        if getattr(device_pool, "layer_shard_enabled", False):
            raise ValueError("SFA C8 HiCache does not yet support layer-sharded pools")
        if layout != "page_first_kv_split":
            raise ValueError(
                "SFA C8 HiCache requires page_first_kv_split, " f"got {layout!r}"
            )
        payloads = device_pool.get_sfa_c8_page_payload_descriptor()
        if not payloads or payloads[0]["name"] != "sfa":
            raise ValueError("SFA C8 device payload descriptor has no SFA anchor")
        self.packed_head_dim = device_pool.sfa_c8_packed_head_dim
        self.index_head_dim = device_pool.index_head_dim
        self.indexer_layer_num = device_pool.indexer_layer_num
        self.indexer_layer_ids = tuple(device_pool.indexer_layer_ids)
        self.indexer_layer_id_to_slot = dict(device_pool.indexer_layer_id_to_slot)
        self.draft_host_pool = None
        self.logical_page_anchor = None
        super().__init__(
            device_pool,
            host_to_device_ratio,
            host_size,
            page_size,
            layout,
            pin_memory,
            device,
            allocator_type,
        )
        # HostKVCache expects a single anchor buffer.  SFA C8 deliberately has
        # two physical regions under the same logical page allocator.
        self.kv_buffer = self.packed_kv_buffer
        self.can_use_write_back_jit = False

    def get_size_per_token(self):
        self.layer_num = self._effective_host_layer_num()
        packed_bytes = self.layer_num * self.packed_head_dim
        indexer_bytes = 0
        if self.index_head_dim is not None:
            indexer_bytes = (
                self.indexer_layer_num
                * self.index_head_dim
                * self.device_pool.store_dtype.itemsize
            )
        return packed_bytes + indexer_bytes

    def get_ksize_per_token(self):
        total = self.size_per_token
        if self.draft_host_pool is not None:
            total += self.draft_host_pool.size_per_token
        return total

    def init_kv_buffer(self):
        alloc = ALLOC_MEMORY_FUNCS[self.device_pool.device]
        self.packed_kv_buffer = alloc(
            (
                self.page_num,
                self.layer_num,
                self.page_size,
                1,
                self.packed_head_dim,
            ),
            dtype=torch.int8,
            device=self.device,
            pin_memory=self.pin_memory,
            allocator=self.allocator,
        )
        self.index_k_buffer = None
        if self.index_head_dim is not None:
            self.index_k_buffer = alloc(
                (
                    self.page_num,
                    self.indexer_layer_num,
                    self.page_size,
                    1,
                    self.index_head_dim,
                ),
                dtype=self.device_pool.store_dtype,
                device=self.device,
                pin_memory=self.pin_memory,
                allocator=self.allocator,
            )
        return self.packed_kv_buffer

    def get_hybrid_pool_buffer(self):
        buffers = [self.packed_kv_buffer]
        if self.index_k_buffer is not None:
            buffers.append(self.index_k_buffer)
        if self.draft_host_pool is not None:
            buffers.extend(self.draft_host_pool.get_hybrid_pool_buffer())
        return buffers

    def attach_draft_host_pool(self, draft_host_pool) -> None:
        if not isinstance(draft_host_pool, NPUSFAC8TokenToKVPoolHost):
            raise TypeError("SFA C8 logical pages require an SFA C8 draft host pool")
        if draft_host_pool.page_size != self.page_size:
            raise ValueError("SFA C8 target and draft page sizes differ")
        if draft_host_pool.size != self.size:
            raise ValueError(
                "SFA C8 target and draft host capacities differ: "
                f"{self.size} != {draft_host_pool.size}"
            )
        draft_host_pool.logical_page_anchor = self
        self.draft_host_pool = draft_host_pool

    def _page_pairs(self, host_indices, device_indices):
        if host_indices.numel() != device_indices.numel():
            raise ValueError("SFA C8 host/device transfer lengths differ")
        if host_indices.numel() % self.page_size != 0:
            raise ValueError("SFA C8 HiCache transfer must be page-aligned")
        host = host_indices.reshape(-1, self.page_size)[:, 0].cpu().tolist()
        device = device_indices.reshape(-1, self.page_size)[:, 0].cpu().tolist()
        return [
            (int(h) // self.page_size, int(d) // self.page_size)
            for h, d in zip(host, device)
        ]

    def backup_from_device_all_layer(
        self, device_pool, host_indices, device_indices, io_backend
    ):
        if io_backend != "kernel_ascend":
            raise ValueError(f"SFA C8 HiCache requires kernel_ascend, got {io_backend}")
        if _is_npu:
            transfer_kv_dim_exchange(
                device_indices=device_indices,
                host_indices=host_indices,
                device_k=device_pool.packed_kv_buffer,
                host_k=self.packed_kv_buffer,
                device_v=torch.empty(0),
                host_v=torch.empty(0),
                device_index_k=device_pool.index_k_buffer,
                host_index_k=self.index_k_buffer,
                page_size=self.page_size,
                direction=TransferDirection.D2H,
            )
            return
        for host_page, device_page in self._page_pairs(host_indices, device_indices):
            self.packed_kv_buffer[host_page].copy_(
                device_pool.packed_kv_buffer[:, device_page], non_blocking=True
            )
            if self.index_k_buffer is not None:
                self.index_k_buffer[host_page].copy_(
                    device_pool.index_k_buffer[:, device_page], non_blocking=True
                )

    def load_to_device_per_layer(
        self, device_pool, host_indices, device_indices, layer_id, io_backend
    ):
        if io_backend != "kernel_ascend":
            raise ValueError(f"SFA C8 HiCache requires kernel_ascend, got {io_backend}")
        # Restore every physical region in one operation.  The controller calls
        # this method once per layer; doing the work at layer zero keeps the
        # existing layer-done protocol while avoiding repeated full-pool copies.
        if layer_id != 0:
            return
        if _is_npu:
            transfer_kv_dim_exchange(
                device_indices=device_indices,
                host_indices=host_indices,
                device_k=device_pool.packed_kv_buffer,
                host_k=self.packed_kv_buffer,
                device_v=torch.empty(0),
                host_v=torch.empty(0),
                device_index_k=device_pool.index_k_buffer,
                host_index_k=self.index_k_buffer,
                page_size=self.page_size,
                direction=TransferDirection.H2D,
            )
            return
        for host_page, device_page in self._page_pairs(host_indices, device_indices):
            device_pool.packed_kv_buffer[:, device_page].copy_(
                self.packed_kv_buffer[host_page], non_blocking=True
            )
            if self.index_k_buffer is not None:
                device_pool.index_k_buffer[:, device_page].copy_(
                    self.index_k_buffer[host_page], non_blocking=True
                )

    def get_data_page(self, index, flat: bool = True):
        raise RuntimeError(
            "SFA C8 L3 requires the logical-page multi-buffer storage API"
        )

    def get_dummy_flat_data_page(self):
        raise RuntimeError(
            "SFA C8 L3 requires the logical-page multi-buffer storage API"
        )

    def set_from_flat_data_page(self, index, data_page):
        raise RuntimeError(
            "SFA C8 L3 requires the logical-page multi-buffer storage API"
        )

    def get_logical_page_buffer_meta(self, indices):
        """Return one iovec list per logical page for Mooncake multi-buffer IO."""
        if len(indices) % self.page_size != 0:
            raise ValueError("SFA C8 L3 page metadata must be page-aligned")
        pages = indices.reshape(-1, self.page_size)[:, 0].cpu().tolist()
        region_buffers = self.get_hybrid_pool_buffer()
        result = []
        for token_index in pages:
            page = int(token_index) // self.page_size
            buffers = [
                (buf.data_ptr() + page * buf[0].nbytes, buf[0].nbytes)
                for buf in region_buffers
            ]
            result.append(buffers)
        return result

    def get_page_buffer_meta(self, indices):
        pages = self.get_logical_page_buffer_meta(indices)
        # Mooncake's MLA v1 path groups all entries for a page into one
        # batch_*_multi_buffers call and therefore one storage key.
        return (
            [ptr for page in pages for ptr, _ in page],
            [size for page in pages for _, size in page],
        )

    def is_stride_page_aligned(self, page_size_bytes: int = 4096) -> bool:
        buffers = self.get_hybrid_pool_buffer()
        return all(
            buf.data_ptr() % page_size_bytes == 0
            and buf[0].nbytes % page_size_bytes == 0
            for buf in buffers
        )
