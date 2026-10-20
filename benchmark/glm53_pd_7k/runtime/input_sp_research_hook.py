"""Research-only lossless PCP4/TP2 pre-MoE sequence-parallel boundary.

Original model/layout/quantization preserved. Restore the complete CP residual
before returning. Verify both outputs at every real eligible layer first.
Do not enable SGLang's Qwen-only global layernorm-SP option.
"""
import functools
import json
import time
from pathlib import Path

import torch
from sglang.srt.distributed import get_tp_group
from sglang.srt.layers import communicator_dsa_cp as cp
from sglang.srt.layers.dp_attention import (
    attn_tp_all_gather_into_tensor,
    attn_tp_reduce_scatter_tensor,
)
from sglang.srt.runtime_context import get_parallel

root=Path('/research')
original=cp.DSACPCommunicateWithAllReduceAndLayerNormFn._gather_hidden_states_and_residual
control={};next_read=0.;checks={};acked=set()


def transformed(x,residual,layernorm):
    parallel=get_parallel()
    m=x.shape[0]//parallel.attn_tp_size
    shard=x.new_empty((m,*x.shape[1:]))
    attn_tp_reduce_scatter_tensor(shard,x.contiguous())
    # Norm mutates its residual input. Copy the local slice so the input ABI
    # does not acquire a surprising partial mutation on only half the rows.
    local_residual=residual.narrow(0,parallel.attn_tp_rank*m,m).clone()
    shard,local_residual=layernorm(shard,local_residual)
    full=shard.new_empty((m*parallel.tp_size,*shard.shape[1:]))
    get_tp_group().all_gather_into_tensor(full,shard.contiguous())
    restored=residual.new_empty(residual.shape)
    attn_tp_all_gather_into_tensor(restored,local_residual.contiguous())
    return full,restored


@functools.wraps(original)
def call(hidden_states,residual,forward_batch,layernorm,context,*,residual_input_mode):
    global next_read,control
    now=time.monotonic()
    if now>=next_read:
        control=json.loads((root/'control.json').read_text());next_read=now+1
    p=get_parallel()
    eligible=(control.get('target')=='input_sp' and p.tp_size==8 and
              p.attn_tp_size==2 and p.attn_cp_size==4 and p.attn_dp_size==1 and
              cp.dsa_use_prefill_cp(forward_batch) and
              hidden_states.ndim==2 and hidden_states.dtype==torch.bfloat16 and
              hidden_states.shape[1]==6144 and hidden_states.shape[0]>=128 and
              hidden_states.shape[0]%2==0 and residual is not None and
              residual.shape==hidden_states.shape and residual.dtype==hidden_states.dtype)
    if not eligible:
        return original(hidden_states,residual,forward_batch,layernorm,context,
                        residual_input_mode=residual_input_mode)
    rank=torch.cuda.current_device();run=control.get('run_id','unset')
    if control.get('verify_input_sp'):
        if control.get('mode')!='baseline':raise RuntimeError('verify from baseline only')
        x=hidden_states.clone();r=residual.clone()
        result=original(hidden_states,residual,forward_batch,layernorm,context,
                        residual_input_mode=residual_input_mode)
        candidate=transformed(x,r,layernorm)
        exact=[torch.equal(a,b) for a,b in zip(result,candidate)]
        finite=all(bool(torch.isfinite(t).all()) for t in candidate)
        rec=checks.setdefault(run,{'run_id':run,'rank':rank,'calls':0,'nonexact':0,
                                 'nonfinite':0,'max_abs':0.,'shapes':{}})
        rec['calls']+=1;rec['nonexact']+=int(not all(exact));rec['nonfinite']+=int(not finite)
        rec['shapes'][str(x.shape[0])]=rec['shapes'].get(str(x.shape[0]),0)+1
        if not all(exact):
            rec['max_abs']=max(rec['max_abs'],*(float((a-b).abs().max()) for a,b in zip(result,candidate)))
        tmp=root/f'input-sp-verify-r{rank}.next.json';tmp.write_text(json.dumps(rec))
        tmp.replace(root/f'input-sp-verify-r{rank}.json')
    elif control.get('mode')=='candidate':
        result=transformed(hidden_states,residual,layernorm)
    else:
        result=original(hidden_states,residual,forward_batch,layernorm,context,
                        residual_input_mode=residual_input_mode)
    if run not in acked:
        acked.add(run)
        ack={'run_id':run,'rank':rank,'mode':control['mode'],'shape':list(hidden_states.shape),
             'target':'input_sp','time':time.time()}
        tmp=root/f'input-sp-ack-r{rank}.next.json';tmp.write_text(json.dumps(ack))
        tmp.replace(root/f'input-sp-ack-r{rank}.json')
        print('INPUT_SP_RESEARCH_ACK '+json.dumps(ack),flush=True)
    return result


cp.DSACPCommunicateWithAllReduceAndLayerNormFn._gather_hidden_states_and_residual=staticmethod(call)
