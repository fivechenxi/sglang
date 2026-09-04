import logging
from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.constants import GPU_MEMORY_TYPE_KV_CACHE
from sglang.srt.mem_cache.memory_pool import (
    MHATokenToKVPool,
    MLATokenToKVPool,
    get_tensor_size_bytes,
    unwrap_write_loc,
)
from sglang.srt.utils import get_bool_env_var
from sglang.srt.utils.common import is_npu

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention

if is_npu():
    import torch_npu

logger = logging.getLogger(__name__)


def _init_npu_conv_state(
    conv_state_in, conv_state_shape, speculative_num_draft_tokens: Optional[int] = None
):
    extra_conv_len = 0
    if speculative_num_draft_tokens is not None:
        extra_conv_len = speculative_num_draft_tokens - 1

    # conv_state shape (layers, pool_size, conv_wind + draft_step, dim) for conv1d ascendc ops require dim as last dim
    conv_state = [
        torch.zeros(
            size=(
                conv_state_in.shape[0],
                conv_state_in.shape[1],
                conv_shape[1] + extra_conv_len,
                conv_shape[0],
            ),
            dtype=conv_state_in.dtype,
            device=conv_state_in.device,
        )
        for conv_shape in conv_state_shape
    ]
    return conv_state


class NPUMHATokenToKVPool(MHATokenToKVPool):

    def __init__(
        self,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        head_num: int,
        head_dim: int,
        layer_num: int,
        device: str,
        enable_memory_saver: bool,
        v_head_dim: Optional[int] = None,
        swa_head_num: Optional[int] = None,
        swa_head_dim: Optional[int] = None,
        swa_v_head_dim: Optional[int] = None,
        start_layer: Optional[int] = None,
        end_layer: Optional[int] = None,
        enable_alt_stream: bool = True,
        enable_kv_cache_copy: bool = False,
        **kwargs,
    ):
        self.use_fia = get_bool_env_var("ASCEND_USE_FIA", "False")
        super().__init__(
            size=size,
            page_size=page_size,
            dtype=dtype,
            head_num=head_num,
            head_dim=head_dim,
            layer_num=layer_num,
            device=device,
            enable_memory_saver=enable_memory_saver,
            v_head_dim=v_head_dim,
            swa_head_num=swa_head_num,
            swa_head_dim=swa_head_dim,
            swa_v_head_dim=swa_v_head_dim,
            start_layer=start_layer,
            end_layer=end_layer,
            enable_alt_stream=enable_alt_stream,
            enable_kv_cache_copy=enable_kv_cache_copy,
            **kwargs,
        )

    def _create_buffers(self):
        with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            # [size, head_num, head_dim] for each layer
            # The padded slot 0 is used for writing dummy outputs from padded tokens.
            # Continuous memory improves the efficiency of Ascend`s transmission backend,
            # while other backends remain unchanged.
            self.k_buffer = torch.zeros(
                (
                    self.layer_num,
                    self.size // self.page_size + 1,
                    self.page_size,
                    self.head_num,
                    self.head_dim,
                ),
                dtype=self.store_dtype,
                device=self.device,
            )
            self.v_buffer = torch.zeros(
                (
                    self.layer_num,
                    self.size // self.page_size + 1,
                    self.page_size,
                    self.head_num,
                    self.v_head_dim,
                ),
                dtype=self.store_dtype,
                device=self.device,
            )

            if self.use_fia:
                # Use per-layer Python lists to avoid torch.compile capturing
                # the entire multi-layer tensor (OOM during graph capture).
                # Each layer view: [P*ps, 1, H, D], sharing the contiguous
                # storage allocated above.
                self.k_buffer = [
                    self.k_buffer[i].view(-1, 1, self.head_num, self.head_dim)
                    for i in range(self.layer_num)
                ]
                self.v_buffer = [
                    self.v_buffer[i].view(-1, 1, self.head_num, self.v_head_dim)
                    for i in range(self.layer_num)
                ]

    def _init_kv_copy_and_warmup(self):
        # implementation relies on self.data_strides / self.data_ptrs, which the
        # NPU paged buffer layout never builds.
        self._kv_copy_config = None

    # for disagg
    def get_contiguous_buf_infos(self):
        # layer_num x [seq_len, head_num, head_dim]
        # layer_num x [page_num, page_size, head_num, head_dim]
        kv_data_ptrs = [
            self.get_key_buffer(i).data_ptr()
            for i in range(self.start_layer, self.start_layer + self.layer_num)
        ] + [
            self.get_value_buffer(i).data_ptr()
            for i in range(self.start_layer, self.start_layer + self.layer_num)
        ]
        kv_data_lens = [
            self.get_key_buffer(i).nbytes
            for i in range(self.start_layer, self.start_layer + self.layer_num)
        ] + [
            self.get_value_buffer(i).nbytes
            for i in range(self.start_layer, self.start_layer + self.layer_num)
        ]
        if self.use_fia:
            kv_item_lens = [
                self.get_key_buffer(i)[0].nbytes * self.page_size
                for i in range(self.start_layer, self.start_layer + self.layer_num)
            ] + [
                self.get_value_buffer(i)[0].nbytes * self.page_size
                for i in range(self.start_layer, self.start_layer + self.layer_num)
            ]
        else:
            kv_item_lens = [
                self.get_key_buffer(i)[0].nbytes
                for i in range(self.start_layer, self.start_layer + self.layer_num)
            ] + [
                self.get_value_buffer(i)[0].nbytes
                for i in range(self.start_layer, self.start_layer + self.layer_num)
            ]
        return kv_data_ptrs, kv_data_lens, kv_item_lens

    def set_kv_buffer(
        self,
        layer: "RadixAttention",
        loc_info,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        k_scale: Optional[float] = None,
        v_scale: Optional[float] = None,
        layer_id_override: Optional[int] = None,
        dcp_kv_mask: Optional[torch.Tensor] = None,
    ):
        loc, _, _ = unwrap_write_loc(loc_info)
        if layer_id_override is not None:
            layer_id = layer_id_override
        else:
            layer_id = layer.layer_id
        if cache_k.dtype != self.dtype:
            if k_scale is not None:
                cache_k.div_(k_scale)
            if v_scale is not None:
                cache_v.div_(v_scale)
            cache_k = cache_k.to(self.dtype)
            cache_v = cache_v.to(self.dtype)

        if self.store_dtype != self.dtype:
            cache_k = cache_k.view(self.store_dtype)
            cache_v = cache_v.view(self.store_dtype)

        if self.use_fia:
            k_buffer_layer = self.k_buffer[layer_id - self.start_layer]
            v_buffer_layer = self.v_buffer[layer_id - self.start_layer]

            torch_npu.npu_scatter_nd_update_(
                k_buffer_layer,
                loc.view(-1, 1),
                cache_k.view(-1, 1, self.head_num, self.head_dim),
            )
            torch_npu.npu_scatter_nd_update_(
                v_buffer_layer,
                loc.view(-1, 1),
                cache_v.view(-1, 1, self.head_num, self.v_head_dim),
            )
        else:
            loc = loc.to(torch.int32)
            torch_npu._npu_reshape_and_cache(
                key=cache_k,
                value=cache_v,
                key_cache=self.k_buffer[layer_id - self.start_layer].view(
                    -1, self.page_size, self.head_num, self.head_dim
                ),
                value_cache=self.v_buffer[layer_id - self.start_layer].view(
                    -1, self.page_size, self.head_num, self.v_head_dim
                ),
                slot_indices=loc,
            )

    def _chunk_copy_npu_to_cpu(self, buf_of_layers, indices):
        chunk_size = self.cpu_offloading_chunk_size
        out = []
        for tensors_per_layer in buf_of_layers:  # [k_buf, v_buf]
            layer_chunks = []
            for i in range(0, len(indices), chunk_size):
                ci = indices[i : i + chunk_size]
                layer_chunks.append(
                    [
                        t[ci].to("cpu", non_blocking=True)
                        for t in tensors_per_layer
                        if t is not None
                    ]
                )
            out.append(layer_chunks)
        return out

    # Parent MHATokenToKVPool.get_cpu_copy / load_cpu_copy use
    # `self.k_buffer[layer_id][chunk_indices]` which indexes the first dim.
    # NPUMHATokenToKVPool stores buffers as
    #   (num_pages, page_size, head_num, head_dim)            # use_fia=False
    #   (num_pages*page_size, 1, head_num, head_dim)          # use_fia=True
    def get_cpu_copy(self, indices, mamba_indices=None):
        torch.npu.synchronize()
        buf_of_layers = []
        for local_layer_id in range(self.layer_num):
            k_layer = self.k_buffer[local_layer_id].view(
                -1, self.head_num, self.head_dim
            )
            v_layer = self.v_buffer[local_layer_id].view(
                -1, self.head_num, self.head_dim
            )
            buf_of_layers.append([k_layer, v_layer])
        kv_cache_cpu = self._chunk_copy_npu_to_cpu(buf_of_layers, indices)
        torch.npu.synchronize()
        return kv_cache_cpu

    def load_cpu_copy(self, kv_cache_cpu, indices, mamba_indices=None):
        torch.npu.synchronize()
        chunk_size = self.cpu_offloading_chunk_size
        for local_layer_id in range(self.layer_num):
            k_layer = self.k_buffer[local_layer_id].view(
                -1, self.head_num, self.head_dim
            )
            v_layer = self.v_buffer[local_layer_id].view(
                -1, self.head_num, self.head_dim
            )
            for i in range(0, len(indices), chunk_size):
                chunk_indices = indices[i : i + chunk_size]
                k_cpu, v_cpu = (
                    kv_cache_cpu[local_layer_id][i // chunk_size][0],
                    kv_cache_cpu[local_layer_id][i // chunk_size][1],
                )
                assert k_cpu.shape[0] == v_cpu.shape[0] == len(chunk_indices)
                k_layer[chunk_indices] = k_cpu.to(k_layer.device, non_blocking=True)
                v_layer[chunk_indices] = v_cpu.to(v_layer.device, non_blocking=True)
        torch.npu.synchronize()


class NPUMLATokenToKVPool(MLATokenToKVPool):
    @property
    def supports_cpu_offload(self) -> bool:
        # The packed C8 layout cannot use the inherited BF16 K/V host-copy
        # representation. Decode retraction must rebootstrap from Prefill.
        return not self.sfa_c8_enabled


    def __init__(
        self,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        kv_lora_rank: int,
        qk_rope_head_dim: int,
        index_head_dim: Optional[int],
        layer_num: int,
        device: str,
        enable_memory_saver: bool,
        start_layer: Optional[int] = None,
        end_layer: Optional[int] = None,
        indexer_layer_ids: Optional[list[int]] = None,
        sfa_c8_enabled: bool = False,
    ):
        super(MLATokenToKVPool, self).__init__(
            size=size,
            page_size=page_size,
            dtype=dtype,
            layer_num=layer_num,
            device=device,
            enable_memory_saver=enable_memory_saver,
            start_layer=start_layer,
            end_layer=end_layer,
        )

        self.kv_lora_rank = kv_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.index_head_dim = index_head_dim
        if self.index_head_dim is None:
            self.indexer_layer_ids = []
        elif indexer_layer_ids is None:
            self.indexer_layer_ids = list(range(self.start_layer, self.end_layer))
        else:
            self.indexer_layer_ids = list(indexer_layer_ids)
        if len(set(self.indexer_layer_ids)) != len(self.indexer_layer_ids):
            raise ValueError("indexer_layer_ids must be unique")
        if any(
            layer_id < self.start_layer or layer_id >= self.end_layer
            for layer_id in self.indexer_layer_ids
        ):
            raise ValueError(
                "indexer_layer_ids must be within the pool's layer range "
                f"[{self.start_layer}, {self.end_layer})"
            )
        self.indexer_layer_id_to_slot = {
            layer_id: slot for slot, layer_id in enumerate(self.indexer_layer_ids)
        }
        self.indexer_layer_num = len(self.indexer_layer_ids)
        self.sfa_c8_enabled = sfa_c8_enabled
        if self.sfa_c8_enabled:
            from sglang.srt.hardware_backend.npu.attention.sfa_c8 import (
                get_sfa_c8_packed_head_dim,
            )

            self.sfa_c8_packed_head_dim = get_sfa_c8_packed_head_dim(
                self.kv_lora_rank, self.qk_rope_head_dim
            )
        else:
            self.sfa_c8_packed_head_dim = None

        self.custom_mem_pool = None

        with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            # The padded slot 0 is used for writing dummy outputs from padded tokens.
            if self.sfa_c8_enabled:
                self.packed_kv_buffer = torch.zeros(
                    (
                        layer_num,
                        self.size // self.page_size + 1,
                        self.page_size,
                        1,
                        self.sfa_c8_packed_head_dim,
                    ),
                    dtype=torch.int8,
                    device=self.device,
                )
                self.k_buffer = None
                self.v_buffer = None
            else:
                self.packed_kv_buffer = None
                self.k_buffer = torch.zeros(
                    (
                        layer_num,
                        self.size // self.page_size + 1,
                        self.page_size,
                        1,
                        self.kv_lora_rank,
                    ),
                    dtype=self.store_dtype,
                    device=self.device,
                )
                self.v_buffer = torch.zeros(
                    (
                        layer_num,
                        self.size // self.page_size + 1,
                        self.page_size,
                        1,
                        self.qk_rope_head_dim,
                    ),
                    dtype=self.store_dtype,
                    device=self.device,
                )
            self.index_k_buffer = None
            if self.index_head_dim is not None:
                self.index_k_buffer = torch.zeros(
                    (
                        self.indexer_layer_num,
                        self.size // self.page_size + 1,
                        self.page_size,
                        1,
                        self.index_head_dim,
                    ),
                    dtype=self.store_dtype,
                    device=self.device,
                )

        self._finalize_allocation_log(size)
        if self.sfa_c8_enabled:
            logger.info(
                "NPU SFA C8 cache layout: main_dtype=%s, packed_head_dim=%d "
                "bytes, main_layers=%d, indexer_dtype=%s, indexer_layers=%d",
                self.packed_kv_buffer.dtype,
                self.sfa_c8_packed_head_dim,
                self.layer_num,
                self.index_k_buffer.dtype if self.index_k_buffer is not None else None,
                self.indexer_layer_num,
            )

    def get_kv_size_bytes(self):
        kv_size_bytes = 0
        if self.sfa_c8_enabled:
            for packed_cache in self.packed_kv_buffer:
                kv_size_bytes += get_tensor_size_bytes(packed_cache)
        else:
            for k_cache in self.k_buffer:
                kv_size_bytes += get_tensor_size_bytes(k_cache)
            for v_cache in self.v_buffer:
                kv_size_bytes += get_tensor_size_bytes(v_cache)
        if self.index_head_dim is not None:
            assert hasattr(self, "index_k_buffer")
            for index_k_cache in self.index_k_buffer:
                kv_size_bytes += get_tensor_size_bytes(index_k_cache)
        return kv_size_bytes

    def get_kv_buffer(self, layer_id: int):
        if self.sfa_c8_enabled:
            raise RuntimeError(
                "SFA C8 uses get_sfa_packed_kv_buffer(), not the BF16 K/V API"
            )
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)
        return (
            self.k_buffer[layer_id - self.start_layer],
            self.v_buffer[layer_id - self.start_layer],
        )

    def get_sfa_packed_kv_buffer(self, layer_id: int):
        if not self.sfa_c8_enabled:
            raise RuntimeError("SFA C8 packed KV cache is not enabled")
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)
        return self.packed_kv_buffer[layer_id - self.start_layer]

    def get_state_buf_infos(self):
        if self.index_head_dim is None:
            return [], [], []
        data_ptrs = [
            self.index_k_buffer[i].data_ptr() for i in range(self.indexer_layer_num)
        ]
        data_lens = [
            self.index_k_buffer[i].nbytes for i in range(self.indexer_layer_num)
        ]
        item_lens = [
            self.index_k_buffer[i][0].nbytes for i in range(self.indexer_layer_num)
        ]
        return data_ptrs, data_lens, item_lens

    def get_key_buffer(self, layer_id: int):
        if self.sfa_c8_enabled:
            raise RuntimeError("SFA C8 has no standalone key buffer")
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)

        if self.store_dtype != self.dtype:
            return self.k_buffer[layer_id - self.start_layer].view(self.dtype)
        return self.k_buffer[layer_id - self.start_layer]

    def get_value_buffer(self, layer_id: int):
        if self.sfa_c8_enabled:
            raise RuntimeError("SFA C8 has no standalone value buffer")
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)

        if self.store_dtype != self.dtype:
            return self.v_buffer[layer_id - self.start_layer].view(self.dtype)
        return self.v_buffer[layer_id - self.start_layer]

    def get_index_k_buffer(self, layer_id: int):
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)

        slot = self._get_indexer_slot(layer_id)
        if self.store_dtype != self.dtype:
            return self.index_k_buffer[slot].view(self.dtype)
        return self.index_k_buffer[slot]

    def _get_indexer_slot(self, layer_id: int) -> int:
        try:
            return self.indexer_layer_id_to_slot[layer_id]
        except KeyError as exc:
            raise ValueError(
                f"DSA layer {layer_id} does not own a physical indexer cache; "
                f"owners={self.indexer_layer_ids}"
            ) from exc

    # for disagg
    def get_contiguous_buf_infos(self):
        if self.sfa_c8_enabled:
            # PD transfer operates on opaque byte ranges.  Expose the native
            # packed cache directly: one pointer per layer and one page-sized
            # item per transfer index.  In particular, do not present separate
            # K/V views here; that would either double-transfer the same bytes
            # or silently expand C8 back to BF16 on the wire.
            kv_data_ptrs = [
                self.packed_kv_buffer[i].data_ptr() for i in range(self.layer_num)
            ]
            kv_data_lens = [
                self.packed_kv_buffer[i].nbytes for i in range(self.layer_num)
            ]
            kv_item_lens = [
                self.packed_kv_buffer[i][0].nbytes for i in range(self.layer_num)
            ]
            if self.index_head_dim is not None:
                # The Lightning Indexer remains BF16 in this phase.  It shares
                # the main cache page ids and therefore travels as additional
                # opaque page buffers in the same positional descriptor list.
                kv_data_ptrs += [
                    self.index_k_buffer[i].data_ptr()
                    for i in range(self.indexer_layer_num)
                ]
                kv_data_lens += [
                    self.index_k_buffer[i].nbytes
                    for i in range(self.indexer_layer_num)
                ]
                kv_item_lens += [
                    self.index_k_buffer[i][0].nbytes
                    for i in range(self.indexer_layer_num)
                ]
            logger.info(
                "NPU SFA C8 PD layout: packed_main_layers=%d, "
                "packed_main_page_bytes=%d, indexer_layers=%d, "
                "indexer_page_bytes=%d, descriptor_count=%d",
                self.layer_num,
                self.packed_kv_buffer[0][0].nbytes,
                self.indexer_layer_num,
                (
                    self.index_k_buffer[0][0].nbytes
                    if self.indexer_layer_num > 0
                    else 0
                ),
                len(kv_data_ptrs),
            )
            return kv_data_ptrs, kv_data_lens, kv_item_lens
        # MLA has only one kv_buffer, so only the information of this buffer needs to be returned.
        kv_data_ptrs = [self.k_buffer[i].data_ptr() for i in range(self.layer_num)] + [
            self.v_buffer[i].data_ptr() for i in range(self.layer_num)
        ]
        kv_data_lens = [self.k_buffer[i].nbytes for i in range(self.layer_num)] + [
            self.v_buffer[i].nbytes for i in range(self.layer_num)
        ]
        kv_item_lens = [self.k_buffer[i][0].nbytes for i in range(self.layer_num)] + [
            self.v_buffer[i][0].nbytes for i in range(self.layer_num)
        ]
        if self.index_head_dim is not None:
            kv_data_ptrs += [
                self.index_k_buffer[i].data_ptr() for i in range(self.indexer_layer_num)
            ]
            kv_data_lens += [
                self.index_k_buffer[i].nbytes for i in range(self.indexer_layer_num)
            ]
            kv_item_lens += [
                self.index_k_buffer[i][0].nbytes for i in range(self.indexer_layer_num)
            ]
        return kv_data_ptrs, kv_data_lens, kv_item_lens

    def set_kv_buffer(
        self,
        layer: "RadixAttention",
        loc_info,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
    ):
        if self.sfa_c8_enabled:
            raise RuntimeError(
                "SFA C8 uses set_sfa_packed_kv_buffer(), not the BF16 K/V API"
            )
        loc, _, _ = unwrap_write_loc(loc_info)
        layer_id = layer.layer_id
        if cache_k.dtype != self.dtype:
            cache_k = cache_k.to(self.dtype)
            cache_v = cache_v.to(self.dtype)

        if self.store_dtype != self.dtype:
            cache_k = cache_k.view(self.store_dtype)
            cache_v = cache_v.view(self.store_dtype)

        if cache_v is None:
            cache_k, cache_v = cache_k.split(
                [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
            )

        torch_npu.npu_scatter_nd_update_(
            self.k_buffer[layer_id - self.start_layer].view(-1, 1, self.kv_lora_rank),
            loc.view(-1, 1),
            cache_k.view(-1, 1, self.kv_lora_rank),
        )
        torch_npu.npu_scatter_nd_update_(
            self.v_buffer[layer_id - self.start_layer].view(
                -1, 1, self.qk_rope_head_dim
            ),
            loc.view(-1, 1),
            cache_v.view(-1, 1, self.qk_rope_head_dim),
        )

    def set_sfa_packed_kv_buffer(
        self,
        layer_id: int,
        slot_mapping_sfa,
        packed_kv: torch.Tensor,
    ):
        if not self.sfa_c8_enabled:
            raise RuntimeError("SFA C8 packed KV cache is not enabled")
        if packed_kv.dtype != torch.int8:
            raise TypeError(f"SFA C8 packed KV must be int8, got {packed_kv.dtype}")
        if packed_kv.shape[-1] != self.sfa_c8_packed_head_dim:
            raise ValueError(
                "SFA C8 packed KV has invalid head dim: "
                f"expected {self.sfa_c8_packed_head_dim}, got {packed_kv.shape[-1]}"
            )
        # Keep the SFA mapping explicit. In a future DCP path this must be the
        # DCP-sharded physical mapping, not the replicated LI/indexer mapping.
        loc, _, _ = unwrap_write_loc(slot_mapping_sfa)
        torch_npu.npu_scatter_nd_update_(
            self.packed_kv_buffer[layer_id - self.start_layer].view(
                -1, 1, self.sfa_c8_packed_head_dim
            ),
            loc.view(-1, 1),
            packed_kv.view(-1, 1, self.sfa_c8_packed_head_dim),
        )

    def set_index_k_buffer(
        self,
        layer_id: int,
        loc: torch.Tensor,
        index_k: torch.Tensor,
    ):
        if index_k.dtype != self.dtype:
            index_k = index_k.to(self.dtype)

        if self.store_dtype != self.dtype:
            index_k = index_k.view(self.store_dtype)

        slot = self._get_indexer_slot(layer_id)
        torch_npu.npu_scatter_nd_update_(
            self.index_k_buffer[slot].view(-1, 1, self.index_head_dim),
            loc.view(-1, 1),
            index_k.view(-1, 1, self.index_head_dim),
        )

    def _chunk_copy_npu_to_cpu(self, buf_of_layers, indices):
        chunk_size = self.cpu_offloading_chunk_size
        out = []
        for tensors_per_layer in buf_of_layers:  # [k_buf, v_buf, ik_buf/None]
            layer_chunks = []
            for i in range(0, len(indices), chunk_size):
                ci = indices[i : i + chunk_size]
                layer_chunks.append(
                    [
                        t[ci].to("cpu", non_blocking=True)
                        for t in tensors_per_layer
                        if t is not None
                    ]
                )
            out.append(layer_chunks)
        return out

    def get_cpu_copy(self, indices, mamba_indices=None):
        if self.sfa_c8_enabled:
            raise RuntimeError("SFA C8 CPU offload/HiCache is not supported yet")
        torch.npu.synchronize()
        buf_of_layers = []
        has_ik = self.index_head_dim is not None
        for local_layer_id in range(self.layer_num):
            k_layer = self.k_buffer[local_layer_id].view(-1, 1, self.kv_lora_rank)
            v_layer = self.v_buffer[local_layer_id].view(-1, 1, self.qk_rope_head_dim)
            indexer_slot = self.indexer_layer_id_to_slot.get(
                self.start_layer + local_layer_id
            )
            ik_layer = (
                self.index_k_buffer[indexer_slot].view(-1, 1, self.index_head_dim)
                if has_ik and indexer_slot is not None
                else None
            )
            buf_of_layers.append([k_layer, v_layer, ik_layer])

        kv_cache_cpu = self._chunk_copy_npu_to_cpu(buf_of_layers, indices)
        torch.npu.synchronize()
        return kv_cache_cpu

    def load_cpu_copy(self, kv_cache_cpu, indices, mamba_indices=None):
        if self.sfa_c8_enabled:
            raise RuntimeError("SFA C8 CPU offload/HiCache is not supported yet")
        torch.npu.synchronize()
        chunk_size = self.cpu_offloading_chunk_size
        has_ik = self.index_head_dim is not None
        for local_layer_id in range(self.layer_num):
            k_layer = self.k_buffer[local_layer_id].view(-1, 1, self.kv_lora_rank)
            v_layer = self.v_buffer[local_layer_id].view(-1, 1, self.qk_rope_head_dim)
            indexer_slot = self.indexer_layer_id_to_slot.get(
                self.start_layer + local_layer_id
            )
            ik_layer = (
                self.index_k_buffer[indexer_slot].view(-1, 1, self.index_head_dim)
                if has_ik and indexer_slot is not None
                else None
            )
            for i in range(0, len(indices), chunk_size):
                chunk_indices = indices[i : i + chunk_size]
                chunk = kv_cache_cpu[local_layer_id][i // chunk_size]
                k_cpu, v_cpu = chunk[0], chunk[1]
                assert k_cpu.shape[0] == len(chunk_indices)
                k_layer[chunk_indices] = k_cpu.to(k_layer.device, non_blocking=True)
                v_layer[chunk_indices] = v_cpu.to(v_layer.device, non_blocking=True)
                if ik_layer is not None:
                    ik_cpu = chunk[2]
                    ik_layer[chunk_indices] = ik_cpu.to(
                        ik_layer.device, non_blocking=True
                    )
        torch.npu.synchronize()
