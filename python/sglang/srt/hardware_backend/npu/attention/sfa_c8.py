from __future__ import annotations

import torch

from sglang.srt.environ import envs

SFA_C8_TILE_SIZE = 128
SFA_C8_QUANT_MODE = 2
SFA_C8_SCALE_REPO_MODE = 1


def is_sfa_c8_enabled() -> bool:
    return envs.SGLANG_NPU_ENABLE_SFA_C8.get()


def get_sfa_c8_incompatibilities(
    *,
    page_size: int,
    mlapo_enabled: bool,
    dcp_size: int,
    prefill_cp_enabled: bool,
    disaggregation_mode: str,
    hierarchical_cache_enabled: bool,
    cpu_offload_gb: float,
    speculative_algorithm: str,
    decode_graph_enabled: bool,
    prefill_graph_enabled: bool,
) -> list[str]:
    """Return features excluded from the standalone/MTP decode-graph phase.

    The production GLM/NEXTN path resolves NEXTN to EAGLE before this check.
    Its target verify, draft decode, and draft extend forwards are decode-graph
    workloads and use the same packed SFA cache API as eager execution.  General
    prefill graph capture remains excluded until it has its own A2 validation.
    """
    # decode_graph_enabled is deliberately accepted: the parameter documents
    # and test-protects the target-verify/draft-decode/draft-extend graph surface.
    incompatible = []
    if page_size != 128:
        incompatible.append("page_size other than 128")
    if mlapo_enabled:
        incompatible.append("SGLANG_NPU_USE_MLAPO")
    if dcp_size > 1:
        incompatible.append("decode context parallelism/DCP")
    if prefill_cp_enabled:
        incompatible.append("DSA prefill context parallelism")
    # PD transfer is byte-oriented.  NPUMLATokenToKVPool exposes each packed
    # SFA page as one contiguous item, so prefill and decode workers move the
    # native 656-byte representation without reconstructing BF16 K/V tensors.
    # Keep the argument in this validation surface so all three roles are
    # covered by the same phase-boundary tests.
    if hierarchical_cache_enabled:
        incompatible.append("hierarchical cache")
    if cpu_offload_gb > 0:
        incompatible.append("CPU offload")
    if speculative_algorithm not in ("NONE", "EAGLE"):
        incompatible.append(
            f"speculative algorithm {speculative_algorithm} "
            "(only NEXTN/EAGLE is supported)"
        )
    if prefill_graph_enabled:
        incompatible.append("prefill graph capture/replay")
    return incompatible


def get_sfa_c8_packed_head_dim(
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    tile_size: int = SFA_C8_TILE_SIZE,
) -> int:
    if kv_lora_rank % tile_size != 0:
        raise ValueError(
            "kv_lora_rank must be divisible by tile_size for SFA C8, "
            f"got kv_lora_rank={kv_lora_rank}, tile_size={tile_size}"
        )
    scale_bytes = kv_lora_rank // tile_size * torch.float32.itemsize
    rope_bytes = qk_rope_head_dim * torch.bfloat16.itemsize
    return kv_lora_rank + rope_bytes + scale_bytes


def pack_sfa_c8_kv(
    k_nope: torch.Tensor,
    k_rope: torch.Tensor,
    *,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    tile_size: int = SFA_C8_TILE_SIZE,
) -> torch.Tensor:
    """Pack INT8 K-nope, BF16 K-RoPE bytes, and FP32 tile scales."""
    import torch_npu

    if k_nope.shape[-1] != kv_lora_rank:
        raise ValueError(f"expected K-nope dim {kv_lora_rank}, got {k_nope.shape[-1]}")
    if k_rope.shape[-1] != qk_rope_head_dim:
        raise ValueError(
            f"expected K-RoPE dim {qk_rope_head_dim}, got {k_rope.shape[-1]}"
        )
    if k_nope.dtype != torch.bfloat16 or k_rope.dtype != torch.bfloat16:
        raise TypeError("SFA C8 pack expects BF16 K-nope and K-RoPE inputs")
    if k_nope.shape[:-1] != k_rope.shape[:-1]:
        raise ValueError(
            "K-nope and K-RoPE must have the same token/head dimensions, "
            f"got {k_nope.shape[:-1]} and {k_rope.shape[:-1]}"
        )

    prefix_shape = k_nope.shape[:-1]
    quantized, scales = torch_npu.npu_dynamic_block_quant(
        k_nope.contiguous().view(-1, 1, kv_lora_rank),
        dst_type=1,
        row_block_size=1,
        col_block_size=tile_size,
    )
    if quantized.dtype != torch.int8 or scales.dtype != torch.float32:
        raise TypeError(
            "SFA C8 dynamic block quant must return INT8 values and FP32 "
            f"scales, got {quantized.dtype} and {scales.dtype}"
        )
    quantized = quantized.view(*prefix_shape, kv_lora_rank)
    rope_bytes = k_rope.contiguous().view(torch.int8)
    scale_bytes = (
        scales.to(torch.float32).view(*prefix_shape, -1).contiguous().view(torch.int8)
    )
    packed = torch.cat([quantized, rope_bytes, scale_bytes], dim=-1)
    expected_dim = get_sfa_c8_packed_head_dim(kv_lora_rank, qk_rope_head_dim, tile_size)
    if packed.shape[-1] != expected_dim:
        raise RuntimeError(
            f"invalid SFA C8 packed dim: expected {expected_dim}, got {packed.shape[-1]}"
        )
    return packed


def run_sfa_c8_attention(
    q_nope: torch.Tensor,
    q_rope: torch.Tensor,
    packed_kv: torch.Tensor,
    sparse_indices: torch.Tensor,
    *,
    scale_value: float,
    block_table: torch.Tensor,
    actual_seq_lengths_query: torch.Tensor,
    actual_seq_lengths_kv: torch.Tensor,
    rope_head_dim: int,
    tile_size: int = SFA_C8_TILE_SIZE,
) -> torch.Tensor:
    """Run native TorchNPU QSFA using only keyword arguments."""
    import torch_npu

    if q_nope.dtype != torch.bfloat16 or q_rope.dtype != torch.bfloat16:
        raise TypeError("SFA C8 attention expects BF16 query tensors")
    if packed_kv.dtype != torch.int8:
        raise TypeError("SFA C8 attention expects an INT8 packed KV cache")
    if q_nope.shape[:-1] != q_rope.shape[:-1]:
        raise ValueError("Q-nope and Q-RoPE must have matching token/head dimensions")
    query = torch.cat([q_nope, q_rope], dim=-1).contiguous()
    return torch_npu.npu_kv_quant_sparse_flash_attention(
        query=query,
        key=packed_kv,
        value=packed_kv,
        sparse_indices=sparse_indices,
        scale_value=scale_value,
        key_quant_mode=SFA_C8_QUANT_MODE,
        value_quant_mode=SFA_C8_QUANT_MODE,
        block_table=block_table,
        actual_seq_lengths_query=actual_seq_lengths_query,
        actual_seq_lengths_kv=actual_seq_lengths_kv,
        sparse_block_size=1,
        layout_query="TND",
        layout_kv="PA_BSND",
        sparse_mode=3,
        attention_mode=2,
        quant_scale_repo_mode=SFA_C8_SCALE_REPO_MODE,
        tile_size=tile_size,
        rope_head_dim=rope_head_dim,
    )
