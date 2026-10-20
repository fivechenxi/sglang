"""Exact candidate helper AST; byte copy oracle and pack/transfer ordering."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

from operations.bounded_pack import _research_send_bounded_dcp_chunks as send


def groups(src,dst):
    a=[];b=[]
    for x,y in zip(src,dst):
        if a and x==a[-1][-1]+1 and y==b[-1][-1]+1:
            a[-1].append(x);b[-1].append(y)
        else:a.append([x]);b.append([y])
    return a,b


class BoundedTests(unittest.TestCase):
    def run_case(self,src,dst,capacity_rows=3,fail_call=None):
        lengths=[2,5,7];sources=[1000,3000,5000];destinations=[7000,10000,13000]
        memory=bytearray([233]*20000);expect=memory[:];events=[];calls=0;awaiting=False
        for l,(base,unit) in enumerate(zip(sources,lengths)):
            for row in range(40):
                for j in range(unit):memory[base+row*unit+j]=(l*11+row*3+j)%233
        expect=memory[:]
        for a,b in zip(src,dst):
            for sp,dp,unit in zip(sources,destinations,lengths):expect[dp+b*unit:dp+(b+1)*unit]=memory[sp+a*unit:sp+(a+1)*unit]
        def pack(**kw):
            nonlocal awaiting
            self.assertFalse(awaiting,'buffer overwritten before transfer completed')
            awaiting=True;events.append('pack');offset=0;ptrs=[];idx=kw['src_token_indices']
            self.assertLessEqual(len(idx),capacity_rows)
            for sp,unit in zip(kw['kv_data_ptrs'],kw['token_item_lens']):
                ptrs.append(offset)
                for row in idx:memory[offset:offset+unit]=memory[sp+row*unit:sp+(row+1)*unit];offset+=unit
            return ptrs,list(range(len(idx)))
        def transfer(blocks):
            nonlocal calls,awaiting
            self.assertTrue(awaiting);events.append('transfer');calls+=1
            if calls==fail_call:return -7
            for sp,dp,n in blocks:memory[dp:dp+n]=memory[sp:sp+n]
            awaiting=False;return 0
        rc=send(pack_buffer=SimpleNamespace(get_size=lambda:sum(lengths)*capacity_rows),
                src_ptrs=sources,dst_ptrs=destinations,src_indices=src,dst_indices=dst,
                token_item_lens=lengths,pack_fn=pack,group_fn=groups,transfer_fn=transfer)
        if fail_call:self.assertEqual(rc,-7);self.assertEqual(calls,fail_call);self.assertTrue(awaiting)
        else:
            self.assertEqual(rc,0)
            for dp in destinations:self.assertEqual(memory[dp:dp+280],expect[dp:dp+280])
        self.assertEqual(events,['pack','transfer']*calls)
        return calls

    def test_fragmented_physical_rows_and_unequal_layer_bytes(self):
        self.assertEqual(self.run_case([8,2,15,3,9,0,22,1],[3,4,8,9,15,2,0,20]),3)

    def test_exact_boundary(self):self.assertEqual(self.run_case([0,1,2,3,4,5],[7,8,9,11,12,13]),2)
    def test_empty(self):self.assertEqual(self.run_case([],[]),0)
    def test_failure_never_reuses_buffer(self):self.run_case(list(range(9)),list(range(9)),fail_call=2)
    def test_tiny_buffer(self):
        with self.assertRaises(ValueError):self.run_case([0],[1],capacity_rows=0)
    def test_mismatched_rows_rejected(self):
        with self.assertRaises(ValueError):self.run_case([0,1],[1])


if __name__=='__main__':unittest.main()
