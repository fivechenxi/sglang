import pytest

from sglang.srt.mem_cache.kv_cache_configurator import (
    _dsa_index_buffer_tokens_for_dcp,
    _dsa_index_pages_required,
)


@pytest.mark.parametrize(
    ("max_total_num_tokens", "page_size", "dcp_size"),
    [
        (138_048, 64, 2),
        (197_568, 64, 4),
        (138_049, 64, 2),
    ],
)
def test_dsa_index_pool_covers_last_dcp_allocator_page(
    max_total_num_tokens: int, page_size: int, dcp_size: int
):
    index_buffer_tokens = _dsa_index_buffer_tokens_for_dcp(
        max_total_num_tokens, page_size, dcp_size
    )
    # IndexKeyCache uses this page-count formula when allocating index buffers.
    allocated_pages = (index_buffer_tokens + page_size + 1) // page_size
    required_pages = _dsa_index_pages_required(
        max_total_num_tokens, page_size, dcp_size
    )
    assert allocated_pages >= required_pages
    if max_total_num_tokens % page_size == 0:
        assert allocated_pages == required_pages
