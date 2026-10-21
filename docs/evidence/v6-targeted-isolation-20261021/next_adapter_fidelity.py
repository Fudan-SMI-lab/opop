from pathlib import Path
import hashlib,json,time,traceback
from kernel_optimizer.runtime.execution import ExecutionRequest,LocalExecutor,write_json

def audit(context):
    import torch
    import torch.distributed as dist
    from kernel_optimizer.project.evaluation import load_adapter,compare
    p=context['parameters'];request=p['request']
    adapter=load_adapter(Path(p['adapter']));state=adapter.prepare(context,request)
    if hasattr(adapter,'set_mode'):adapter.set_mode(state,'quality')
    refs=torch.load(Path(p['controls'])/f"reference-rank-{context['rank']}.pt",map_location='cpu',weights_only=True,mmap=True)
    rows=[]
    adapter.reset(state)
    for step in range(request['continuous_steps']+1):
        expected_step=step
        if step==request['continuous_steps']:
            adapter.reset(state);expected_step=0
        if hasattr(adapter,'before_run'):adapter.before_run(state)
        result=adapter.run(state);observed=adapter.observe(state,result)
        fields=['loss','gradients','parameters']
        check=compare({k:observed[k] for k in fields},{k:refs[expected_step][k] for k in fields},request['quality'])
        micro=None
        if 'microbatch_losses' in observed:
            value=observed['microbatch_losses']
            if isinstance(value,(list,tuple)):value=torch.stack(value)
            micro=compare(value,refs[expected_step]['microbatch_losses'],request['quality'])
        raw=state['raw_model']  # manually audited state mapping in both generated adapters
        equal=True;elements=0
        for parameter in raw.parameters():
            gradient=parameter.grad
            if gradient is None:raise ValueError('missing full gradient')
            peer=gradient.detach().clone();dist.broadcast(peer,src=0)
            equal=bool(torch.equal(gradient,peer)) and equal;elements+=gradient.numel()
        rows.append({'step':step,'reference_step':expected_step,'reset_before':step in (0,request['continuous_steps']),
                     'full_observation':check,'extra_microbatch_loss':micro,
                     'exact_replica_gradients':equal,'gradient_elements':elements})
        del observed,result
    return {'valid':all(r['full_observation']['passed'] and r['exact_replica_gradients'] and (r['extra_microbatch_loss'] is None or r['extra_microbatch_loss']['passed']) for r in rows),
            'checks':rows,'scope':'Developer manual observer-schema alignment: unchanged generated adapter compared with original independently source-audited full references; two continuous updates plus reset replay and exact cross-rank gradients. Not performance measurement.',
            'adapter_sha256':hashlib.sha256(Path(p['adapter']).read_bytes()).hexdigest()}

def main():
    old=Path('/root/autodl-tmp/dev/v6-work/evidence-followup-20261021/fresh/ddp-communication')
    root=Path('/root/autodl-tmp/dev/v6-work/targeted-isolation-20261021')
    request=json.loads((old/'request.json').read_text());prepared=json.loads((old/'prepared.json').read_text())
    rows=[];done=set()
    while len(done)<2:
        for arm in ['control','treatment']:
            if arm in done:continue
            path=root/'ddp-contract-ab'/arm/'result.json'
            if not path.exists():continue
            done.add(arm)
            try:
                record=json.loads(path.read_text());assert record['status']=='generated',record
                generated=Path(record['receipt']['sandbox'])/'adapter.py'
                out=root/'ddp-adapter-fidelity'/arm;out.mkdir(parents=True)
                adapter=out/'adapter.py';adapter.write_bytes(generated.read_bytes())
                result=LocalExecutor().run(ExecutionRequest(entry=Path(__file__).resolve(),function='audit',cwd=out,
                    output=out/'run',devices=request['devices'],distributed_backend=request['distributed_backend'],
                    lease_dir=Path(request['lease_dir']),wait_for_devices=True,worker_timeout_s=request['worker_timeout_s'],
                    parameters={'request':request,'adapter':str(adapter),'controls':prepared['controls_path']}))
                write_json(out/'result.json',result)
                row={'arm':arm,'status':result['status'],'valid':bool(result.get('ranks')) and all(r.get('result',{}).get('valid') for r in result['ranks']),
                     'adapter_sha256':hashlib.sha256(adapter.read_bytes()).hexdigest(),'reference':prepared['controls_path']}
            except Exception as e:row={'arm':arm,'status':'error','error':str(e),'traceback':traceback.format_exc()}
            rows.append(row);write_json(root/'ddp-adapter-fidelity-result.json',{'runs':rows});print(json.dumps(row),flush=True)
        if len(done)<2:time.sleep(5)
    write_json(root/'ddp-adapter-fidelity-completed.json',{'completed':True,'arms':len(rows)})

if __name__=='__main__':main()
