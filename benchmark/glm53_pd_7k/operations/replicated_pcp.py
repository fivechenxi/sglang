import os

def _research_replicated_pcp_allowed(info, *, dcp_size, is_mla_backend,
                                     is_hybrid_mla_backend,
                                     enable_all_cp_ranks_for_transfer):
    # Opt-in isolated validation only. Ordinary DSA PCP materializes full token
    # order on each CP rank; CP0 alone owns a complete source for DCP relayout.
    # Do NOT broaden this to hybrid/state-sharded or all-CP transfer layouts.
    return (
        os.getenv('SGLANG_RESEARCH_REPLICATED_PCP_DCP') == '1'
        and dcp_size == 2 and is_mla_backend and not is_hybrid_mla_backend
        and info.attn_cp_size == 4 and info.attn_tp_size == 2
        and not info.enable_dsa_cache_layer_split
        and not enable_all_cp_ranks_for_transfer
    )
