from pathlib import Path
import json,time,hashlib
from kernel_optimizer.runtime.execution import ExecutionRequest,LocalExecutor,write_json
r=Path('/root/autodl-tmp/dev/v6-work/evidence-followup-20261021')
b=r/'fresh/ddp-communication'

while not (b/'prepared.json').exists(): time.sleep(5)
p=json.loads((b/'prepared.json').read_text());req=json.loads((b/'request.json').read_text());c=Path(p['controls_path'])
assert p['adapter_sha256']=='6d6d5799353e2e8bedd6892b971416246a67b32122c6f38ab24552fa1a17c603', 'Adapter changed after manual source inspection; repeat review'

params={'case':'qwen','request':req,'source':str(c.parent/'reference-source'),'controls':str(c)}
if req['assets']:params['assets']=req['assets']['assets']
result=LocalExecutor().run(ExecutionRequest(entry=r/'ddp-source-audit-worker.py',function='audit',cwd=c.parent/'reference-source',output=r/'source-audit/ddp-communication',devices=req['devices'][:1],lease_dir=req['lease_dir'],wait_for_devices=True,worker_timeout_s=req['worker_timeout_s'],parameters=params))
write_json(r/'source-audit/ddp-communication.json',result)
print(json.dumps({'status':result['status'],'valid':bool(result.get('ranks')) and all(x.get('result',{}).get('valid') for x in result.get('ranks',[]))}),flush=True)
