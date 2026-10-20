"""Real HTTP router regression with CPU-only fake P/D workers; no GPU traffic."""
import argparse
import asyncio
import collections
import json
import os
from pathlib import Path
import sys
import time
import aiohttp
from aiohttp import web

ROOT = None
SOURCE = None
PORT=18459

async def main(mode):
    report={'mode':mode,'gpu':False,'cases':[],'seen':[],'cancelled':[]}
    output=ROOT/f'cpu-{mode}.json'
    assert not output.exists(), 'Preserve old evidence'
    def save(): output.write_text(json.dumps(report,indent=2))
    hold=asyncio.Event()
    runners=[]
    async def closed(request,label,role):
        for _ in range(160):
            if request.transport is None or request.transport.is_closing():
                report['cancelled'].append([label,role]); save();return
            await asyncio.sleep(.05)
    for role,port in [('prefill',30559),('decode',30560)]:
        async def handler(request,role=role):
            if request.method=='GET':
                if 'health' in request.path:return web.Response(text='OK')
                if request.path=='/v1/models':return web.json_response({'data':[{'id':'mock-model'}]})
                if request.path=='/v1/loads':return web.json_response({'loads':[]})
                return web.json_response(dict(model_path='mock-model',served_model_name='mock-model',
                    is_generation=True,tp_size=8,dp_size=4 if role=='decode' else 1,
                    disaggregation_mode=role,architectures=['GlmMoeDsaForCausalLM']))
            body=await request.json()
            prompt=body.get('prompt') or body.get('messages',[{}])[0].get('content','')
            label=prompt.rsplit('|',1)[-1]
            report['seen'].append(dict(role=role,label=label,stream=body.get('stream'),
                stream_options=body.get('stream_options'),rank=body.get('data_parallel_rank')));save()
            if role=='prefill':
                if label in ('p500','p400'):
                    await asyncio.sleep(.12)
                    return web.json_response({'error':{'message':label}},status=int(label[1:]))
                if label=='app_error':return web.json_response({'error':{'message':'bootstrap failed'}})
                if label=='abort_native':return web.json_response({'meta_info':{'finish_reason':{'type':'abort'}}})
                if label=='abort_chat':return web.json_response({'choices':[{'finish_reason':'abort'}]})
                if label=='bad_json':return web.Response(text='broken JSON')
                if label=='sse_error':
                    response=web.StreamResponse(headers={'Content-Type':'text/event-stream'})
                    await response.prepare(request)
                    await response.write(b'data: {"error":{"message":"bootstrap failed"}}\n\n')
                    await closed(request,label,role);return response
                if label=='body_disconnect':
                    response=web.StreamResponse(headers={'Content-Type':'application/json','Content-Length':'9999'})
                    await response.prepare(request);await response.write(b'{"text":')
                    request.transport.close();return response
                if label=='cancel':await closed(request,label,role);return web.json_response({})
                return web.json_response({'text':'','meta_info':{'input_token_logprobs':[[0.1,1,'x']]}})
            if label in ('p500','p400','app_error','abort_native','abort_chat','bad_json','sse_error','body_disconnect','cancel'):
                await closed(request,label,role)
                return web.json_response({'error':'cancelled'},status=499)
            if label.startswith('load'):
                await hold.wait()
            if not body.get('stream'):
                return web.json_response({'choices':[{'text':'ok','finish_reason':'stop'}]})
            response=web.StreamResponse(headers={'Content-Type':'text/event-stream'})
            await response.prepare(request)
            await response.write(b'data: {"choices":[{"text":"ok","finish_reason":null}]}\n\n')
            await asyncio.sleep(.1)
            await response.write(b'data: {"choices":[{"text":"!","finish_reason":"stop"}]}\n\ndata: [DONE]\n\n')
            return response
        app=web.Application();app.router.add_route('*','/{path:.*}',handler)
        runner=web.AppRunner(app);await runner.setup();runners.append(runner)
        await web.TCPSite(runner,'127.0.0.1',port).start()
    env=os.environ.copy()
    source=SOURCE
    env['PYTHONPATH']=str(source/'bindings/python/src')
    cmd=[sys.executable,'-m','sglang_router.launch_router','--host','127.0.0.1','--port',str(PORT),
        '--prometheus-port','29459','--pd-disaggregation','--prefill','http://127.0.0.1:30559','31559',
        '--decode','http://127.0.0.1:30560','--dp-aware','--disable-retries','--disable-circuit-breaker',
        '--prefill-policy','cache_aware','--decode-policy','cache_aware',
        '--request-timeout-secs','20','--log-level','warn']
    if mode=='candidate':
        cmd += ['--balance-abs-threshold','0','--balance-rel-threshold','1.0']
    log=(ROOT/f'cpu-{mode}.log').open('x')
    proc=await asyncio.create_subprocess_exec(*cmd,env=env,stdout=log,stderr=log)
    tasks=[]
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as session:
            async def workers():
                async with session.get(f'http://127.0.0.1:{PORT}/workers') as r:return (await r.json())['workers']
            for _ in range(100):
                try:
                    ws=await workers()
                    if len(ws)==5 and all(w['is_healthy'] for w in ws):break
                except (aiohttp.ClientError,KeyError):pass
                if proc.returncode is not None:raise RuntimeError('Router exited')
                await asyncio.sleep(.2)
            else:raise RuntimeError('Readiness timeout')
            async def call(label,stream=True,chat=False,logprobs=False):
                prompt='common GPQA or MMLU prefix '*200+'|'+label
                body={'model':'mock-model','max_tokens':4,'stream':stream}
                if stream:body['stream_options']={'include_usage':True}
                if chat:body['messages']=[{'role':'user','content':prompt}]
                else:body['prompt']=prompt
                if logprobs:body['logprobs']=1
                start=time.monotonic()
                try:
                    async with session.post(f'http://127.0.0.1:{PORT}/v1/'+('chat/completions' if chat else 'completions'),json=body) as r:
                        text=await r.text()
                        result=dict(label=label,status=r.status,elapsed=time.monotonic()-start,text=text[:1500],done='[DONE]' in text)
                except Exception as e:result=dict(label=label,error=type(e).__name__,elapsed=time.monotonic()-start)
                report['cases'].append(result);save();return result
            if mode=='stock':
                await call('sse_error',chat=True)
            else:
                for label in ('p500','p400','app_error','abort_native','abort_chat','bad_json','sse_error','body_disconnect'):
                    result=await call(label,chat=label in ('p500','sse_error','abort_chat'))
                    assert result.get('status',0)>=400 and result['elapsed']<2,result
                    for _ in range(50):
                        if [label,'decode'] in report['cancelled']:break
                        await asyncio.sleep(.05)
                    assert [label,'decode'] in report['cancelled'],label
                for stream,chat,logprobs in [(True,False,False),(True,True,False),(False,False,False),(True,False,True)]:
                    result=await call('normal',stream,chat,logprobs)
                    assert result.get('status')==200 and (not stream or result['done']),result
                task=asyncio.create_task(call('cancel'));tasks.append(task)
                await asyncio.sleep(.3);task.cancel();await asyncio.gather(task,return_exceptions=True)
                for _ in range(60):
                    if all(['cancel',role] in report['cancelled'] for role in ('prefill','decode')):break
                    await asyncio.sleep(.05)
                assert all(['cancel',role] in report['cancelled'] for role in ('prefill','decode'))
            # Warm a common-prefix affinity entry, then hold 32 concurrent requests.
            await call('warm')
            for i in range(32):
                tasks.append(asyncio.create_task(call('load'+str(i))))
                await asyncio.sleep(.025)
            for _ in range(50):
                counts=collections.Counter(r['rank'] for r in report['seen'] if r['role']=='decode' and r['label'].startswith('load'))
                if sum(counts.values())==32:break
                await asyncio.sleep(.05)
            report['held_decode_distribution']=dict(counts);report['held_workers']=await workers();save()
            hold.set();await asyncio.gather(*tasks,return_exceptions=True)
            if mode=='candidate':
                assert len(counts)==4 and max(counts.values())<=12,counts
                assert all(r['stream'] is False and r['stream_options'] is None for r in report['seen'] if r['role']=='prefill')
                assert all(r.get('status')==200 and r['done'] for r in report['cases'] if r['label'].startswith('load'))
            else:
                assert report['cases'][0].get('error')=='TimeoutError'
                assert max(counts.values())==32,counts
            await asyncio.sleep(.4)
            report['final_workers']=await workers()
            assert all(w['load']==0 for w in report['final_workers'])
            report['passed']=True;save()
            print(json.dumps({k:v for k,v in report.items() if k not in ('seen','held_workers','final_workers','cases')}),flush=True)
    finally:
        hold.set()
        for t in tasks:
            if not t.done():t.cancel()
        await asyncio.gather(*tasks,return_exceptions=True)
        if proc.returncode is None:
            proc.terminate()
            try:await asyncio.wait_for(proc.wait(),10)
            except asyncio.TimeoutError:proc.kill();await proc.wait()
        for runner in runners:await runner.cleanup()
        log.close();save()

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--mode',choices=['stock','candidate'],required=True)
    p.add_argument('--source-root',type=Path,required=True,
                   help='Built sgl-model-gateway root with the Python extension installed')
    p.add_argument('--output-dir',type=Path,required=True)
    args=p.parse_args()
    ROOT=args.output_dir.resolve()
    ROOT.mkdir(parents=True,exist_ok=True)
    SOURCE=args.source_root.resolve()
    asyncio.run(main(args.mode))
