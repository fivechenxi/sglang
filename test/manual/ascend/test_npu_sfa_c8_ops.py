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


def run_graph_probe(
    *,
    device: torch.device,
    packed_cache: torch.Tensor,
    block_table: torch.Tensor,
    topk: torch.Tensor,
    q_lens: torch.Tensor,
    kv_lens: torch.Tensor,
    scale_value: float,
) -> dict:
    """Capture and replay the exact C8 write + QSFA decode operator chain."""
    kv_lora_rank = 512
    rope_head_dim = 64
    static_k_nope = torch.randn(1, 1, kv_lora_rank, dtype=torch.bfloat16, device=device)
    static_k_rope = torch.randn(
        1, 1, rope_head_dim, dtype=torch.bfloat16, device=device
    )
    static_q_nope = torch.randn(1, 1, kv_lora_rank, dtype=torch.bfloat16, device=device)
    static_q_rope = torch.randn(
        1, 1, rope_head_dim, dtype=torch.bfloat16, device=device
    )
    static_slot = torch.tensor(
        [int(kv_lens[-1].item()) - 1], dtype=torch.int64, device=device
    )

    def graph_forward() -> torch.Tensor:
        packed = pack_sfa_c8_kv(
            static_k_nope,
            static_k_rope,
            kv_lora_rank=kv_lora_rank,
            qk_rope_head_dim=rope_head_dim,
        )
        torch_npu.npu_scatter_nd_update_(
            packed_cache.view(-1, 1, packed_cache.shape[-1]),
            static_slot.view(-1, 1),
            packed.view(-1, 1, packed_cache.shape[-1]),
        )
        return torch_npu.npu_kv_quant_sparse_flash_attention(
            query=torch.cat([static_q_nope, static_q_rope], dim=-1).contiguous(),
            key=packed_cache,
            value=packed_cache,
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
            tile_size=128,
            rope_head_dim=rope_head_dim,
        )

    for _ in range(2):
        graph_forward()
    torch.npu.synchronize()

    graph = torch.npu.NPUGraph()
    capture_stream = torch.npu.Stream()
    capture_stream.wait_stream(torch.npu.current_stream())
    with torch.npu.graph(
        graph,
        pool=torch.npu.graph_pool_handle(),
        stream=capture_stream,
        auto_dispatch_capture=True,
    ):
        graph_out = graph_forward()
    torch.npu.current_stream().wait_stream(capture_stream)
    torch.npu.synchronize()

    static_k_nope.copy_(torch.randn_like(static_k_nope))
    static_k_rope.copy_(torch.randn_like(static_k_rope))
    static_q_nope.copy_(torch.randn_like(static_q_nope))
    static_q_rope.copy_(torch.randn_like(static_q_rope))
    graph.replay()
    torch.npu.synchronize()
    replay_out = graph_out.clone()

    eager_out = graph_forward()
    torch.npu.synchronize()
    max_abs_error = (replay_out.float() - eager_out.float()).abs().max().item()
    mean_abs_error = (replay_out.float() - eager_out.float()).abs().mean().item()
    torch.testing.assert_close(replay_out, eager_out, rtol=1e-3, atol=1e-3)
    return {
        "captured": True,
        "replayed": True,
        "max_abs_error_vs_eager": max_abs_error,
        "mean_abs_error_vs_eager": mean_abs_error,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="npu:0")
    parser.add_argument(
        "--graph",
        action="store_true",
        help="also capture/replay quantize + cache write + QSFA in an NPU graph",
    )
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

    graph_probe = None
    if args.graph:
        graph_probe = run_graph_probe(
            device=device,
            packed_cache=packed.clone(),
            block_table=block_table,
            topk=topk,
            q_lens=q_lens,
            kv_lens=kv_lens,
            scale_value=scale_value,
        )

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
                "graph_probe": graph_probe,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
