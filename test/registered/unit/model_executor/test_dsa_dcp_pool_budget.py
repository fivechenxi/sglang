"""Price the real DSA pools, including replicated index/draft and page zero.

CPU tensors exercise the allocation code without loading model weights.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from test_pool_configurator import (
    _configure_dsa_model,
    _make_model_runner,
)

from sglang.srt.mem_cache.kv_cache_configurator import KVCacheConfigurator
from sglang.srt.model_executor.pool_configurator import DefaultPoolConfigurator
from sglang.srt.runtime_context import get_disagg, get_memory, get_parallel
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


@pytest.fixture
def runner():
    cleanup = unittest.TestCase()
    mr = _make_model_runner(
        cleanup,
        num_layers=6,
        use_mla_backend=True,
        page_size=64,
        disaggregation_mode="decode",
    )
    _configure_dsa_model(mr)
    mr.kv_cache_dtype = torch.float8_e4m3fn
    mr.kv_cache_dtype_str = "fp8_e4m3"
    try:
        with get_parallel().override(attn_tp_size=1, attn_cp_size=1, attn_dcp_size=1):
            yield mr
    finally:
        cleanup.doCleanups()


def build_pool(runner, tokens, *, draft=False, layers=None, device="cpu"):
    """Use the production builder and its actual draft location/page policy."""
    layers = runner.layer_info.num_effective_layers if layers is None else layers
    kvc = SimpleNamespace(
        model_config=runner.model_config,
        is_draft_worker=draft,
        use_mla_backend=True,
        kv_cache_dtype=runner.kv_cache_dtype,
        device=device,
        layer_info=SimpleNamespace(
            start_layer=0, end_layer=layers, num_effective_layers=layers
        ),
    )
    kvc.loc_space_scale = KVCacheConfigurator.loc_space_scale.fget(kvc)
    kvc.pool_page_size = KVCacheConfigurator.pool_page_size.fget(kvc)
    return KVCacheConfigurator._build_dsa_kv_pool(
        kvc, max_total_num_tokens=tokens * kvc.loc_space_scale
    )


def pool_bytes(pool):
    buffers = pool.kv_buffer + pool.index_k_with_scale_buffer
    if pool.layer_shard_enabled:
        buffers += [pool.remote_kv_buffer, pool.index_key_cache.remote_buffer]
    return sum(b.numel() * b.element_size() for b in buffers)


@pytest.mark.parametrize("dcp", [2, 4])
@pytest.mark.parametrize("draft_layers", [0, 1, 3])
def test_dcp_budget_covers_real_target_and_draft(runner, dcp, draft_layers):
    available = 24 * 1024**2
    runner.spec_algorithm.is_eagle.return_value = bool(draft_layers)
    runner.spec_aux_config.eagle_draft_num_layers = draft_layers
    with get_parallel().override(attn_dcp_size=dcp):
        cfg = DefaultPoolConfigurator(runner)
        resolved = cfg.calculate_pool_sizes(available, 64)
        target = build_pool(runner, resolved.max_total_num_tokens)
        used = pool_bytes(target)
        if draft_layers:
            draft = build_pool(
                runner, resolved.max_total_num_tokens, draft=True, layers=draft_layers
            )
            used += pool_bytes(draft)
        assert used <= available
        # Do not solve under-budgeting with an arbitrary large safety factor.
        assert used + cfg._cell_size * 64 > available


@pytest.mark.parametrize("dcp", [2, 4])
def test_replicated_draft_is_not_scaled_twice(runner, dcp):
    with get_parallel().override(attn_dcp_size=dcp):
        draft = build_pool(runner, 256, draft=True, layers=1)
    logical_span = (256 + 64) * dcp
    assert draft.page_size == 64
    assert draft.kv_buffer[0].shape[0] == logical_span
    assert draft.index_k_with_scale_buffer[0].shape[0] * 64 == logical_span
    # The final allocator slot must be addressable in both buffers.
    draft.kv_buffer[0][logical_span - 1].fill_(7)
    draft.index_k_with_scale_buffer[0][logical_span // 64 - 1].fill_(9)


def test_dcp_reserved_page_can_exhaust_budget(runner):
    with get_parallel().override(attn_dcp_size=2):
        cfg = DefaultPoolConfigurator(runner)
        with pytest.raises(RuntimeError, match="Not enough memory"):
            cfg.calculate_pool_sizes(cfg._cell_size * 64, 64)


def test_dcp_one_capacity_is_unchanged(runner):
    with get_parallel().override(attn_dcp_size=1):
        cfg = DefaultPoolConfigurator(runner)
        budget = 24 * 1024**2
        assert cfg.calculate_pool_sizes(budget, 64).max_total_num_tokens == (
            budget // cfg._cell_size // 64 * 64
        )


def test_hisparse_ratio_is_not_multiplied_by_dcp(runner):
    with (
        get_parallel().override(attn_dcp_size=4),
        get_memory().override(enable_hisparse=True),
        patch(
            "sglang.srt.mem_cache.sparsity.parse_hisparse_config",
            return_value=SimpleNamespace(host_to_device_ratio=2),
        ),
    ):
        cfg = DefaultPoolConfigurator(runner)
        index_cost = cfg._compute_dsa_indexer_cell_size(kvc=runner, num_layers=6)
        assert index_cost == 132 * 6 * 2


@pytest.mark.parametrize("hierarchical_cache", [False, True])
@pytest.mark.parametrize("draft_layers", [0, 1])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
def test_index_layer_elision_matches_allocation(
    runner, hierarchical_cache, draft_layers, dtype
):
    runner.model_config.hf_config.index_topk_freq = 4
    runner.model_config.hf_config.index_skip_topk_offset = 3
    runner.kv_cache_dtype = dtype
    runner.spec_algorithm.is_eagle.return_value = bool(draft_layers)
    runner.spec_aux_config.eagle_draft_num_layers = draft_layers
    available = 24 * 1024**2
    with (
        get_parallel().override(attn_dcp_size=2),
        get_disagg().override(disaggregation_mode="null"),
        get_memory().override(enable_hierarchical_cache=hierarchical_cache),
    ):
        cfg = DefaultPoolConfigurator(runner)
        tokens = cfg.calculate_pool_sizes(available, 64).max_total_num_tokens
        target = build_pool(runner, tokens)
        assert any(b.numel() == 0 for b in target.index_k_with_scale_buffer) == (
            not hierarchical_cache
        )
        used = pool_bytes(target)
        if draft_layers:
            draft = build_pool(runner, tokens, draft=True, layers=draft_layers)
            assert all(b.numel() for b in draft.index_k_with_scale_buffer)
            used += pool_bytes(draft)
        assert used <= available
        assert used + cfg._cell_size * 64 > available


def test_explicit_token_cap_does_not_reserve_page_twice(runner):
    with get_parallel().override(attn_dcp_size=2):
        cfg = DefaultPoolConfigurator(runner)
        budget = 24 * 1024**2
        tokens = cfg.calculate_pool_sizes(budget, 64).max_total_num_tokens
        assert (
            cfg.calculate_pool_sizes_from_max_tokens(tokens, 64).max_total_num_tokens
            == tokens
        )
        assert (
            cfg.calculate_pool_sizes_from_max_tokens(1025, 64).max_total_num_tokens
            == 1024
        )
