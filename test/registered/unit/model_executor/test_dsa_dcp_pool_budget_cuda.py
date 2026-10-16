"""DSA DCP pool initialization and last-page kernel access on real CUDA memory."""

import pytest
import torch
from test_dsa_dcp_pool_budget import build_pool, pool_bytes
from test_dsa_dcp_pool_budget import runner as runner

from sglang.srt.model_executor.pool_configurator import DefaultPoolConfigurator
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b", runner_config="1-gpu-small")


def _write_and_read(pool, loc, keys, scales, pages):
    pool.set_index_k_scale_buffer(0, loc, keys, scales)
    return (
        pool.get_index_k_continuous(0, seq_len=64, page_indices=pages),
        pool.get_index_k_scale_continuous(0, seq_len=64, page_indices=pages),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dcp", [2, 4, 8])
@pytest.mark.parametrize("draft_layers", [0, 1])
def test_dcp_budget_and_last_page_kernel_graph(runner, dcp, draft_layers):
    runner.spec_algorithm.is_eagle.return_value = bool(draft_layers)
    runner.spec_aux_config.eagle_draft_num_layers = draft_layers
    budget = 32 * 1024**2
    with get_parallel().override(attn_dcp_size=dcp):
        cfg = DefaultPoolConfigurator(runner)
        tokens = cfg.calculate_pool_sizes(budget, 64).max_total_num_tokens
        pools = [build_pool(runner, tokens, device="cuda")]
        if draft_layers:
            pools.append(
                build_pool(runner, tokens, draft=True, layers=1, device="cuda")
            )
        assert sum(pool_bytes(pool) for pool in pools) <= budget

        logical_span = (tokens + 64) * dcp
        loc = torch.arange(logical_span - 64, logical_span, device="cuda")
        pages = torch.tensor([logical_span // 64 - 1], device="cuda", dtype=torch.int32)
        keys = torch.full((64, 128), 2, device="cuda", dtype=torch.float8_e4m3fn)
        scales = torch.arange(64, device="cuda", dtype=torch.float32) + 0.5
        for pool in pools:
            assert pool.page_size == 64
            if pool is not pools[0]:
                assert pool.kv_buffer[0].shape[0] == logical_span
            pool.kv_buffer[0][-1].fill_(17)
            assert bool((pool.kv_buffer[0][-1] == 17).all())

            # Compile the production Triton accessors before graph capture.
            got_keys, got_scales = _write_and_read(pool, loc, keys, scales, pages)
            torch.testing.assert_close(got_keys, keys.view(torch.uint8))
            torch.testing.assert_close(got_scales.view(torch.float32).flatten(), scales)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                got_keys, got_scales = _write_and_read(pool, loc, keys, scales, pages)
            keys.fill_(3)
            graph.replay()
            torch.cuda.synchronize()
            torch.testing.assert_close(got_keys, keys.view(torch.uint8))
            torch.testing.assert_close(got_scales.view(torch.float32).flatten(), scales)
