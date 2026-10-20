"""CPU-only HTTP regression against stock or rebuilt Router, no GPU peers.

Controlled fake P/D workers hold response headers until all four requests have
been admitted. A second gate holds active streams. Check dispatch, streaming,
completion and cancellation load accounting; all raw snapshots are retained.
"""
import argparse
import asyncio
import json
import os
from pathlib import Path
import shutil
import sys
import time

import aiohttp
from aiohttp import web

ROOT=Path('/research')
PORT=18447


async def main(mode):
    out=ROOT/('mock-stream-load-r47-'+mode+'.json')
    if out.exists(): raise RuntimeError('Refuse overwriting regression evidence')
    report=dict(mode=mode, gpu_work=False, requests=[], snapshots=[], phase='startup')
    def save(): out.write_text(json.dumps(report,indent=2))
    save()
    header_gate=asyncio.Event(); stream_gate=asyncio.Event()
    runners=[]
    for n in range(5):
        role='prefill' if n<4 else 'decode'
        async def handler(request,n=n,role=role):
            if request.method=='GET':
                if 'health' in request.path: return web.Response(text='OK')
                if 'model_info' in request.path:
                    return web.json_response(dict(model_path='mock-model',is_generation=True))
                if request.path=='/v1/models':
                    return web.json_response(dict(data=[dict(id='mock-model',object='model',owned_by='test')]))
                if request.path=='/v1/loads':
                    return web.json_response(dict(loads=[dict(dp_rank=i,num_total_tokens=0) for i in range(4 if n==4 else 1)]))
                return web.json_response(dict(model_path='mock-model',served_model_name='mock-model',
                    tp_size=8,dp_size=4 if n==4 else 1,disaggregation_mode=role,
                    architectures=['GlmMoeDsaForCausalLM']))
            body=await request.json()
            report['requests'].append(dict(unix=time.time(),node=n,role=role,
                dp_rank=body.get('data_parallel_rank'),prompt=body.get('prompt'),
                stream=body.get('stream'))); save()
            await header_gate.wait()
            if role=='prefill': return web.json_response(dict(text='',meta_info={}))
            response=web.StreamResponse(headers={'Content-Type':'text/event-stream'})
            await response.prepare(request)
            await response.write(b'data: {"choices":[{"text":"x"}]}\n\n')
            await stream_gate.wait()
            await response.write(b'data: [DONE]\n\n')
            await response.write_eof()
            return response
        app=web.Application()
        app.router.add_route('*','/{path:.*}',handler)
        runner=web.AppRunner(app); await runner.setup(); runners.append(runner)
        await web.TCPSite(runner,'127.0.0.1',30541+n).start()
    env=os.environ.copy()
    if mode=='candidate':
        lib=ROOT/'sgl-model-gateway/target/release/libsglang_router_rs.so'
        dest=ROOT/'sgl-model-gateway/bindings/python/src'
        if not lib.exists(): raise RuntimeError('Candidate is not built')
        shutil.copy2(lib,dest/'sglang_router/sglang_router_rs.abi3.so')
        env['PYTHONPATH']=str(dest)
    cmd=[sys.executable,'-m','sglang_router.launch_router','--host','127.0.0.1',
         '--port',str(PORT),'--prometheus-port','29447','--pd-disaggregation',
         '--prefill-policy','cache_aware','--decode-policy','cache_aware','--dp-aware',
         '--disable-retries']
    for n in range(4): cmd+=['--prefill','http://127.0.0.1:'+str(30541+n),str(31541+n)]
    cmd+=['--decode','http://127.0.0.1:30545']
    log=(ROOT/('mock-stream-load-r47-'+mode+'.log')).open('x')
    proc=await asyncio.create_subprocess_exec(*cmd,env=env,stdout=log,stderr=log)
    tasks=[]
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
            async def snapshot(label):
                async with session.get(f'http://127.0.0.1:{PORT}/workers') as response:
                    response.raise_for_status(); data=await response.json()
                snap=dict(label=label,unix=time.time(),workers=[dict(url=w['url'],
                    role=w['worker_type'],load=w['load'],healthy=w['is_healthy']) for w in data['workers']])
                report['snapshots'].append(snap); save(); return snap
            deadline=time.monotonic()+60
            while True:
                if proc.returncode is not None: raise RuntimeError('Router startup failed')
                try:
                    first=await snapshot('readiness')
                    if len(first['workers'])==8 and all(w['healthy'] for w in first['workers']): break
                except aiohttp.ClientError: pass
                if time.monotonic()>deadline: raise RuntimeError('Router readiness timed out')
                await asyncio.sleep(.5)
            async def request(i,tail=''):
                async with session.post(f'http://127.0.0.1:{PORT}/v1/completions',json={
                    'model':'mock-model','prompt':chr(65+i)*2048+tail,'max_tokens':4,'stream':True}) as response:
                    text=await response.text()
                    return dict(id=i,status=response.status,done='data: [DONE]' in text)
            for i in range(4):
                tasks.append(asyncio.create_task(request(i)))
                await asyncio.sleep(.5)
                await snapshot('preheaders_'+str(i+1))
            header_gate.set(); await asyncio.sleep(.5)
            await snapshot('active_streams')
            stream_gate.set()
            report['responses']=await asyncio.gather(*tasks)
            await asyncio.sleep(.3); await snapshot('drained')
            initial=list(report['requests'])
            report['initial_P_nodes']=sorted({r['node'] for r in initial if r['role']=='prefill'})
            report['initial_D_ranks']=sorted({r['dp_rank'] for r in initial if r['role']=='decode'},key=str)
            report['placement_pass']=len(report['initial_P_nodes'])==4 and len(report['initial_D_ranks'])==4
            report['warm_responses']=await asyncio.gather(*(request(i,' distinct-tail') for i in range(4)))
            warmed=report['requests'][len(initial):]
            def mapping(rows):
                return {(r['role'],r['prompt'][0]):(r['node'],r['dp_rank']) for r in rows}
            report['affinity_pass']=len(warmed)==8 and mapping(initial)==mapping(warmed)
            header_gate.clear(); stream_gate.clear()
            cancelled=asyncio.create_task(request(4)); tasks.append(cancelled)
            await asyncio.sleep(.5); await snapshot('before_cancel')
            cancelled.cancel(); await asyncio.gather(cancelled,return_exceptions=True)
            await asyncio.sleep(.5); await snapshot('after_cancel')
            # Release fake handlers too, without changing the captured cancel result.
            header_gate.set(); stream_gate.set()
            for snapshot_row in report['snapshots']:
                snapshot_row['P_load']=sum(w['load'] for w in snapshot_row['workers'] if w['role']=='prefill')
                snapshot_row['D_load']=sum(w['load'] for w in snapshot_row['workers'] if w['role']=='decode')
            snaps={s['label']:s for s in report['snapshots']}
            report['dispatch_counter_pass']=all((snaps['preheaders_'+str(i)]['P_load'],
                snaps['preheaders_'+str(i)]['D_load'])==(i,i) for i in range(1,5))
            report['stream_counter_pass']=(snaps['active_streams']['P_load'],snaps['active_streams']['D_load'])==(4,4)
            report['drain_counter_pass']=(snaps['drained']['P_load'],snaps['drained']['D_load'])==(0,0)
            report['cancel_counter_pass']=(snaps['after_cancel']['P_load'],snaps['after_cancel']['D_load'])==(0,0)
            report['phase']='complete'; save()
            print(json.dumps({k:v for k,v in report.items() if k not in ('requests','snapshots')}),flush=True)
            expected_dispatch=(mode=='candidate')
            if report['dispatch_counter_pass'] != expected_dispatch or not all(
                    report[k] for k in ('stream_counter_pass','drain_counter_pass','cancel_counter_pass')) \
                    or not all(r['status']==200 and r['done'] for r in report['responses']+report['warm_responses']) \
                    or (mode=='candidate' and not (report['placement_pass'] and report['affinity_pass'])):
                raise RuntimeError('Router regression did not meet expected stock/candidate behavior')
    finally:
        header_gate.set();stream_gate.set()
        for task in tasks:
            if not task.done():task.cancel()
        await asyncio.gather(*tasks,return_exceptions=True)
        if proc.returncode is None:
            proc.terminate()
            try:await asyncio.wait_for(proc.wait(),10)
            except asyncio.TimeoutError:proc.kill();await proc.wait()
        for runner in runners:await runner.cleanup()
        log.close();save()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=['stock','candidate'],required=True)
    asyncio.run(main(parser.parse_args().mode))
