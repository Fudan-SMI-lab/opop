from pathlib import Path
import json
from kernel_optimizer.runtime.execution import ExecutionRequest,LocalExecutor,write_json
root=Path('/root/autodl-tmp/dev/v6-work/agent-repair-20261021')
rows=[]
for case in ['ddp-communication','resnet-latency']:
 b=root/'fresh-e2e'/case;prepared=json.loads((b/'prepared.json').read_text());request=json.loads((b/'request.json').read_text());controls=Path(prepared['controls_path']);source=controls.parent/'reference-source'
 params={'case':'ddp' if case.startswith('ddp') else 'resnet','request':request,'source':str(source),'controls':str(controls)}
 if case.startswith('resnet'):params.update(assets=request['assets']['assets_dir'],top1_key='predicted_label_ids',top1_representation='list')
 result=LocalExecutor().run(ExecutionRequest(entry=root/'source-audit-worker.py',function='ddp_reference' if case.startswith('ddp') else 'inference_reference',cwd=source,output=root/'source-audit'/case,devices=request['devices'][:1],lease_dir=request['lease_dir'],wait_for_devices=True,worker_timeout_s=request['worker_timeout_s'],parameters=params))
 write_json(root/'source-audit'/(case+'.json'),result)
 row={'case':case,'status':result['status'],'valid':all(r.get('result',{}).get('valid') for r in result.get('ranks',[])),'method':'Developer original-source comparison; ResNet labels represented as a Python list to match evaluator observation schema; logits unchanged'}
 rows.append(row);write_json(root/'source-audit-result.json',{'runs':rows});print(json.dumps(row),flush=True)
