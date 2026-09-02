"""Single-card A2/A3 probe for the native GLM/DSA SFA C8 operators.

Run only on an isolated NPU. This allocates a second CANN runtime and must not
be executed beside a memory-tight production server process.
"""

from __future__ import annotations

import argparse
import json

import torch
import torch_npu

from sglang.srt.hardware_backend.npu.attention.sfa_c8 import pack_sfa_c8_kv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="npu:0")
    args = parser.parse_args()
    device = torch.device(args.device)
    torch.npu.set_device(device)
    torch.manual_seed(7)

    block_size = 128
    seq_len = 2048
    block_num = seq_len // block_size
    kv_lora_rank = 512
    rope_head_dim = 64
    tile_size = 128

    k_nope = torch.randn(
        block_num,
        block_size,
        1,
        kv_lora_rank,
        dtype=torch.bfloat16,
        device=device,
    )
    k_rope = torch.randn(
        block_num,
        block_size,
        1,
        rope_head_dim,
        dtype=torch.bfloat16,
        device=device,
    )
    quantized, scales = torch_npu.npu_dynamic_block_quant(
        k_nope.contiguous().view(-1, 1, kv_lora_rank),
        dst_type=1,
        row_block_size=1,
        col_block_size=tile_size,
    )
    restored = quantized.float() * scales.float().repeat_interleave(tile_size, dim=-1)
    quant_max_abs_error = (
        (restored.view_as(k_nope).float() - k_nope.float()).abs().max().item()
    )
    max_scale = scales.float().max().item()
    if quantized.dtype != torch.int8 or scales.dtype != torch.float32:
        raise AssertionError(
            f"unexpected quant output dtypes: {quantized.dtype}, {scales.dtype}"
        )
    if quant_max_abs_error > max_scale * 1.1 + 1e-6:
        raise AssertionError(
            f"dynamic block quant error {quant_max_abs_error} exceeds scale {max_scale}"
        )

    packed = pack_sfa_c8_kv(
        k_nope,
        k_rope,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=rope_head_dim,
    )
    if packed.shape != (block_num, block_size, 1, 656):
        raise AssertionError(f"unexpected packed shape: {packed.shape}")
    torch.testing.assert_close(
        packed[..., 512:640], k_rope.contiguous().view(torch.int8)
    )
    torch.testing.assert_close(
        packed[..., 640:],
        scales.view(block_num, block_size, 1, 4).contiguous().view(torch.int8),
    )

    q_nope = torch.randn(1, 1, 512, dtype=torch.bfloat16, device=device)
    q_rope = torch.randn(1, 1, 64, dtype=torch.bfloat16, device=device)
    topk = torch.arange(seq_len, dtype=torch.int32, device=device).view(1, 1, -1)
    block_table = torch.arange(block_num, dtype=torch.int32, device=device).view(1, -1)
    q_lens = torch.tensor([1], dtype=torch.int32, device=device)
    kv_lens = torch.tensor([seq_len], dtype=torch.int32, device=device)
    scale_value = 512**-0.5

    bf16_out, _, _ = torch_npu.npu_sparse_flash_attention(
        query=q_nope,
        key=k_nope,
        value=k_nope,
        query_rope=q_rope,
        key_rope=k_rope,
        sparse_indices=topk,
        scale_value=scale_value,
        block_table=block_table,
        actual_seq_lengths_query=q_lens,
        actual_seq_lengths_kv=kv_lens,
        sparse_block_size=1,
        layout_query="TND",
        layout_kv="PA_BSND",
        sparse_mode=3,
        attention_mode=2,
        return_softmax_lse=False,
    )
    c8_out = torch_npu.npu_kv_quant_sparse_flash_attention(
        query=torch.cat([q_nope, q_rope], dim=-1).contiguous(),
        key=packed,
        value=packed,
        sparse_indices=topk,
        scale_value=scale_value,
        key_quant_mode=2,
        value_quant_mode=2,
        block_table=block_table,
        actual_seq_lengths_query=q_lens,
        actual_seq_lengths_kv=kv_lens,
        sparse_block_size=1,
        layout_query="TND",
        layout_kv="PA_BSND",
        sparse_mode=3,
        attention_mode=2,
        quant_scale_repo_mode=1,
        tile_size=tile_size,
        rope_head_dim=rope_head_dim,
    )
    torch.npu.synchronize()
    qsfa_max_abs_error = (c8_out.float() - bf16_out.float()).abs().max().item()
    qsfa_mean_abs_error = (c8_out.float() - bf16_out.float()).abs().mean().item()

    print(
        json.dumps(
            {
                "dynamic_block_quant": {
                    "quant_dtype": str(quantized.dtype),
                    "scale_dtype": str(scales.dtype),
                    "scale_shape": list(scales.shape),
                    "max_abs_error": quant_max_abs_error,
                    "max_scale": max_scale,
                },
                "packed_kv": {"dtype": str(packed.dtype), "shape": list(packed.shape)},
                "qsfa_vs_bf16": {
                    "output_shape": list(c8_out.shape),
                    "max_abs_error": qsfa_max_abs_error,
                    "mean_abs_error": qsfa_mean_abs_error,
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
