from pathlib import Path
import json
from kernel_optimizer.runtime.execution import ExecutionRequest,LocalExecutor,write_json
r=Path('/root/autodl-tmp/dev/v6-work/evidence-followup-20261021');b=r/'fresh/resnet-latency'
p=json.loads((b/'prepared.json').read_text());req=json.loads((b/'request.json').read_text());c=Path(p['controls_path'])
params={'case':'resnet','request':req,'source':str(c.parent/'reference-source'),'controls':str(c),'assets':req['assets']['assets_dir'],'top1_key':'predicted_class_ids'}
result=LocalExecutor().run(ExecutionRequest(entry=r/'source-audit-worker.py',function='inference_reference',cwd=c.parent/'reference-source',output=r/'source-audit/resnet-latency',devices=req['devices'],lease_dir=req['lease_dir'],wait_for_devices=True,worker_timeout_s=req['worker_timeout_s'],parameters=params))
write_json(r/'source-audit/resnet-latency.json',result)
print(json.dumps({'status':result['status'],'valid':all(x.get('result',{}).get('valid') for x in result.get('ranks',[]))}),flush=True)
