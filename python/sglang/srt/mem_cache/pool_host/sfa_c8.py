from __future__ import annotations

import hashlib
import json

import torch

from sglang.srt.mem_cache.pool_host.base import HostKVCache
from sglang.srt.mem_cache.pool_host.common import ALLOC_MEMORY_FUNCS
from sglang.srt.utils import is_npu

_is_npu = is_npu()
if _is_npu:
    from sgl_kernel_npu.kvcacheio import TransferDirection, transfer_kv_dim_exchange


class NPUSFAC8TokenToKVPoolHost(HostKVCache):
    """Page-first Host mirror for the native Ascend SFA C8 layout.

    The target allocator is authoritative for the packed SFA bytes and BF16
    Lightning Indexer rows of both target and draft.  All regions share one
    completion event in ``HiCacheController``.  No payload is converted to the
    ordinary MLA K/V representation.
    """

    @property
    def storage_key_suffix(self) -> str:
        layout = {"target": self._own_storage_layout()}
        if self.draft_host_pool is not None:
            layout["draft"] = self.draft_host_pool._own_storage_layout()
        canonical = json.dumps(layout, sort_keys=True, separators=(",", ":"))
        fingerprint = hashlib.sha256(canonical.encode()).hexdigest()[:20]
        return f"sfa_c8_logical_v2_p{self.page_size}_{fingerprint}"

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
        if page_size != device_pool.page_size:
            raise ValueError(
                "SFA C8 host/device page sizes differ: "
                f"{page_size} != {device_pool.page_size}"
            )
        payloads = device_pool.get_sfa_c8_page_payload_descriptor()
        if not payloads or payloads[0]["name"] != "sfa":
            raise ValueError("SFA C8 device payload descriptor has no SFA anchor")
        self.packed_head_dim = device_pool.sfa_c8_packed_head_dim
        self.index_head_dim = device_pool.index_head_dim
        self.indexer_layer_num = device_pool.indexer_layer_num
        self.indexer_layer_ids = tuple(device_pool.indexer_layer_ids)
        self.indexer_layer_id_to_slot = dict(device_pool.indexer_layer_id_to_slot)
        # SFA C8 rejects layer sharding above, so the host payload owns every
        # layer in the device pool.  Set this before HostKVCache.__init__ calls
        # get_size_per_token() and init_kv_buffer(); keep the getter side-effect
        # free so later size queries cannot mutate the physical layout.
        self.layer_num = device_pool.layer_num
        self._validate_device_payloads(device_pool, payloads, page_size)
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

    @staticmethod
    def _dtype_name(dtype: torch.dtype) -> str:
        return str(dtype).removeprefix("torch.")

    def _own_storage_layout(self) -> dict:
        return {
            "page_size": self.page_size,
            "main_layers": self.layer_num,
            "main_layer_range": [self.start_layer, self.end_layer],
            "packed_dim": self.packed_head_dim,
            "packed_dtype": self._dtype_name(self.packed_kv_buffer.dtype),
            "indexer_layer_ids": list(self.indexer_layer_ids),
            "index_dim": self.index_head_dim,
            "index_dtype": (
                self._dtype_name(self.index_k_buffer.dtype)
                if self.index_k_buffer is not None
                else None
            ),
        }

    @staticmethod
    def _validate_device_payloads(device_pool, payloads, page_size: int) -> None:
        expected_names = ["sfa"]
        if device_pool.index_k_buffer is not None:
            expected_names.append("lightning_indexer")
        names = [payload.get("name") for payload in payloads]
        if names != expected_names:
            raise ValueError(
                "SFA C8 device payload order differs from the logical-page "
                f"layout: expected {expected_names}, got {names}"
            )

        expected_page_num = device_pool.size // page_size + 1
        specifications = [
            (
                "sfa",
                device_pool.packed_kv_buffer,
                device_pool.layer_num,
                device_pool.sfa_c8_packed_head_dim,
                torch.int8,
            )
        ]
        if device_pool.index_k_buffer is not None:
            specifications.append(
                (
                    "lightning_indexer",
                    device_pool.index_k_buffer,
                    device_pool.indexer_layer_num,
                    device_pool.index_head_dim,
                    device_pool.store_dtype,
                )
            )

        for payload, (name, buffer, layers, width, dtype) in zip(
            payloads, specifications
        ):
            expected_shape = (layers, expected_page_num, page_size, 1, width)
            if payload.get("buffer") is not buffer:
                raise ValueError(f"SFA C8 {name} descriptor references another buffer")
            if tuple(buffer.shape) != expected_shape:
                raise ValueError(
                    f"SFA C8 {name} buffer shape mismatch: "
                    f"expected {expected_shape}, got {tuple(buffer.shape)}"
                )
            if buffer.dtype != dtype or payload.get("dtype") != dtype:
                raise ValueError(
                    f"SFA C8 {name} dtype mismatch: expected {dtype}, "
                    f"got buffer={buffer.dtype}, descriptor={payload.get('dtype')}"
                )
            if not buffer.is_contiguous():
                raise ValueError(f"SFA C8 {name} buffer must be contiguous")
            if payload.get("layers") != layers:
                raise ValueError(
                    f"SFA C8 {name} layer count mismatch: "
                    f"expected {layers}, got {payload.get('layers')}"
                )
            expected_page_bytes = buffer[0, 0].nbytes
            if payload.get("page_bytes") != expected_page_bytes:
                raise ValueError(
                    f"SFA C8 {name} page bytes mismatch: "
                    f"expected {expected_page_bytes}, "
                    f"got {payload.get('page_bytes')}"
                )

        expected_mapping = {
            layer_id: slot
            for slot, layer_id in enumerate(device_pool.indexer_layer_ids)
        }
        if device_pool.indexer_layer_id_to_slot != expected_mapping:
            raise ValueError(
                "SFA C8 indexer layer mapping must be dense and ordered: "
                f"expected {expected_mapping}, "
                f"got {device_pool.indexer_layer_id_to_slot}"
            )

    def get_size_per_token(self):
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
        if draft_host_pool is self:
            raise ValueError("SFA C8 target cannot attach itself as its draft pool")
        if self.logical_page_anchor is not None:
            raise ValueError("An SFA C8 draft pool cannot own another draft pool")
        if self.draft_host_pool is not None:
            if self.draft_host_pool is draft_host_pool:
                return
            raise ValueError("SFA C8 target already has a different draft pool")
        if draft_host_pool.logical_page_anchor not in (None, self):
            raise ValueError("SFA C8 draft pool is already attached to another target")
        if draft_host_pool.page_size != self.page_size:
            raise ValueError("SFA C8 target and draft page sizes differ")
        target_device_pages = self.device_pool.packed_kv_buffer.shape[1]
        draft_device_pages = draft_host_pool.device_pool.packed_kv_buffer.shape[1]
        if target_device_pages != draft_device_pages:
            raise ValueError(
                "SFA C8 target and draft device capacities differ: "
                f"pages={target_device_pages}/{draft_device_pages}"
            )
        if (
            draft_host_pool.page_num != self.page_num
            or draft_host_pool.size != self.size
        ):
            raise ValueError(
                "SFA C8 target and draft host capacities differ: "
                f"pages={self.page_num}/{draft_host_pool.page_num}, "
                f"tokens={self.size}/{draft_host_pool.size}"
            )
        draft_host_pool.logical_page_anchor = self
        self.draft_host_pool = draft_host_pool

    @staticmethod
    def _page_numbers(
        indices,
        page_size: int,
        capacity: int,
        name: str,
        *,
        first_valid_index: int = 0,
    ) -> list[int]:
        if indices.dtype not in (torch.int32, torch.int64):
            raise TypeError(f"SFA C8 {name} indices must use an integer dtype")
        if indices.numel() % page_size != 0:
            raise ValueError(f"SFA C8 {name} indices must contain complete pages")
        rows = indices.reshape(-1, page_size).cpu()
        starts = rows[:, 0]
        offsets = torch.arange(page_size, dtype=rows.dtype).unsqueeze(0)
        if not torch.equal(rows, starts.unsqueeze(1) + offsets):
            raise ValueError(
                f"SFA C8 {name} indices must be contiguous within each page"
            )
        if bool(torch.any(starts.remainder(page_size) != 0)):
            raise ValueError(f"SFA C8 {name} indices must start on a page boundary")
        if bool(torch.any(starts < first_valid_index)) or bool(
            torch.any(starts + page_size > capacity)
        ):
            raise ValueError(f"SFA C8 {name} indices contain an out-of-range page")
        pages = starts.div(page_size, rounding_mode="floor")
        if pages.unique().numel() != pages.numel():
            raise ValueError(f"SFA C8 {name} indices contain duplicate pages")
        return pages.tolist()

    def _page_pairs(self, host_indices, device_indices, device_pool):
        if host_indices.numel() != device_indices.numel():
            raise ValueError("SFA C8 host/device transfer lengths differ")
        host_pages = self._page_numbers(host_indices, self.page_size, self.size, "host")
        device_capacity = device_pool.packed_kv_buffer.shape[1] * self.page_size
        device_pages = self._page_numbers(
            device_indices,
            self.page_size,
            device_capacity,
            "device",
            # Device page zero is the padding/dummy page and is never owned by
            # the paged allocator.  Persisting or restoring it would silently
            # turn padding state into a valid prefix page.
            first_valid_index=self.page_size,
        )
        return list(zip(host_pages, device_pages))

    def backup_from_device_all_layer(
        self, device_pool, host_indices, device_indices, io_backend
    ):
        if io_backend != "kernel_ascend":
            raise ValueError(f"SFA C8 HiCache requires kernel_ascend, got {io_backend}")
        page_pairs = self._page_pairs(host_indices, device_indices, device_pool)
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
        for host_page, device_page in page_pairs:
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
        # transfer_kv_dim_exchange launches both SFA and Indexer copies on the
        # controller's current load stream.  The per-layer events recorded after
        # this call (including no-op layers) are therefore ordered after all
        # four target/draft payload copies, so no layer can observe a partial page.
        if layer_id != 0:
            return
        page_pairs = self._page_pairs(host_indices, device_indices, device_pool)
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
        for host_page, device_page in page_pairs:
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
        pages = self._page_numbers(indices, self.page_size, self.size, "L3 host")
        region_buffers = self.get_hybrid_pool_buffer()
        result = []
        for page in pages:
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
