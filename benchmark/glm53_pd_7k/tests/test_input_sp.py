"""Exercise the exact measured eligibility expression without importing torch."""
import ast
from pathlib import Path
from types import SimpleNamespace as NS
import unittest


SOURCE = Path(__file__).resolve().parents[1]/'runtime/input_sp_research_hook.py'
TREE = ast.parse(SOURCE.read_text())
CALL = next(n for n in TREE.body if isinstance(n, ast.FunctionDef) and n.name == 'call')
GUARD = next(n.value for n in CALL.body if isinstance(n, ast.Assign)
             and any(isinstance(t, ast.Name) and t.id == 'eligible' for t in n.targets))


def eligible(**changes):
    scope = dict(control={'target': 'input_sp'},
                 p=NS(tp_size=8, attn_tp_size=2, attn_cp_size=4, attn_dp_size=1),
                 cp=NS(dsa_use_prefill_cp=lambda _: True), forward_batch=None,
                 hidden_states=NS(ndim=2, shape=(4096, 6144), dtype='bf16'),
                 residual=NS(shape=(4096, 6144), dtype='bf16'), torch=NS(bfloat16='bf16'))
    scope.update(changes)
    return eval(compile(ast.Expression(GUARD), str(SOURCE), 'eval'), scope)


class InputSPTests(unittest.TestCase):
    def test_measured_layout(self):
        self.assertTrue(eligible())

    def test_topology_rejects_unmeasured_paths(self):
        fields = dict(tp_size=8, attn_tp_size=2, attn_cp_size=4, attn_dp_size=1)
        for name, value in [('tp_size', 4), ('attn_tp_size', 1), ('attn_cp_size', 2), ('attn_dp_size', 2)]:
            self.assertFalse(eligible(p=NS(**dict(fields, **{name: value}))))

    def test_decode_and_wrong_target_rejected(self):
        self.assertFalse(eligible(cp=NS(dsa_use_prefill_cp=lambda _: False)))
        self.assertFalse(eligible(control={'target': 'other'}))

    def test_residual_and_shape_guards(self):
        self.assertFalse(eligible(residual=None))
        self.assertFalse(eligible(residual=NS(shape=(4096, 6144), dtype='fp16')))
        for shape, dtype in [((127, 6144), 'bf16'), ((129, 6144), 'bf16'),
                             ((4096, 4096), 'bf16'), ((4096, 6144), 'fp16')]:
            self.assertFalse(eligible(hidden_states=NS(ndim=2, shape=shape, dtype=dtype)))

    def test_residual_copy_and_restore_are_retained(self):
        body = ast.unparse(next(n for n in TREE.body if isinstance(n, ast.FunctionDef) and n.name == 'transformed'))
        self.assertIn('.clone()', body)
        self.assertLess(body.index('attn_tp_reduce_scatter_tensor'), body.index('layernorm(shard, local_residual)'))
        self.assertLess(body.index('layernorm(shard, local_residual)'), body.index('all_gather_into_tensor(full'))
        self.assertIn('attn_tp_all_gather_into_tensor(restored, local_residual.contiguous())', body)


if __name__ == '__main__':
    unittest.main()
