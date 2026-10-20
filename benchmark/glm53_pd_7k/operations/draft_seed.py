import os
import logging
logger = logging.getLogger(__name__)

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
    if os.environ.get("GLM53_RESEARCH_D_RECOMPUTE_SEED") == "1":
        # Research-only: recompute the first draft-decode indexer, then keep
        # normal within-iteration IndexShare. Target verification is unchanged.
        # This changes draft proposals/acceptance; it is not a bitexact shortcut.
        args = self.server_args
        if not (
            self.seed_dsa_topk_from_draft_extend
            and self.dsa_index_topk == 2048
            and args.disaggregation_mode == "decode"
            and args.tp_size == 8
            and args.dp_size == 4
            and args.dcp_size == 2
            and self.topk == 1
            and self.speculative_num_steps == 5
            and self.speculative_num_draft_tokens == 6
        ):
            raise RuntimeError("R56 seed recompute requires the pinned DCP2 MTP5/6 topology")
        self.seed_dsa_topk_from_draft_extend = False
        logger.warning(
            "GLM53_R56_DRAFT_SEED_RECOMPUTE gpu=%s within_iteration_index_share=True "
            "cross_phase_seed=False; target verification unchanged", self.gpu_id
        )
