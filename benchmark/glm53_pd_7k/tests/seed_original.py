def _init_dsa_index_share_state(self) -> None:
    # Populate DSA index-share fields from the draft runner's hf_config.
    # Reused by the attention unit-test harnesses, which skip __init__.
    hf_config = self.draft_runner.model_config.hf_config
    # Reuse the first draft step's DSA indexer topk across the rest;
    # topk == 1 only (select_top_k_tokens reorders rows, desyncing indices).
    self.index_share_for_mtp_iteration = (
        getattr(hf_config, "index_share_for_mtp_iteration", False)
        and self.topk == 1
    )
    # GLM-5.2 MTP IndexShare: seed reused indexer top-k from draft-extend
    # (last verified token), not draft-decode step 0.
    self.dsa_index_topk = getattr(hf_config, "index_topk", None)
    self.seed_dsa_topk_from_draft_extend = (
        self.index_share_for_mtp_iteration and self.dsa_index_topk is not None
    )
