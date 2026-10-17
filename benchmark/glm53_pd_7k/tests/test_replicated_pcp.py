"""Exercise the exact helper AST without importing GPU serving dependencies."""
import ast
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from operations.replicated_pcp import _research_replicated_pcp_allowed as allow


class PolicyTests(unittest.TestCase):
    def setUp(self):
        self.info=SimpleNamespace(attn_cp_size=4,attn_tp_size=2,enable_dsa_cache_layer_split=False)
        self.kw=dict(dcp_size=2,is_mla_backend=True,is_hybrid_mla_backend=False,enable_all_cp_ranks_for_transfer=False)

    def test_default_off(self):
        with patch.dict(os.environ,{},clear=True):self.assertFalse(allow(self.info,**self.kw))

    def test_exact_replicated_layout_optin(self):
        with patch.dict(os.environ,{'SGLANG_RESEARCH_REPLICATED_PCP_DCP':'1'}):
            self.assertTrue(allow(self.info,**self.kw))

    def test_sharded_or_hybrid_or_different_topology_rejected(self):
        with patch.dict(os.environ,{'SGLANG_RESEARCH_REPLICATED_PCP_DCP':'1'}):
            for key,value in [('dcp_size',4),('is_mla_backend',False),('is_hybrid_mla_backend',True),('enable_all_cp_ranks_for_transfer',True)]:
                self.assertFalse(allow(self.info,**dict(self.kw,**{key:value})))
            for key,value in [('attn_cp_size',8),('attn_tp_size',4),('enable_dsa_cache_layer_split',True)]:
                info=SimpleNamespace(**dict(vars(self.info),**{key:value}))
                self.assertFalse(allow(info,**self.kw))


if __name__=='__main__':unittest.main()
