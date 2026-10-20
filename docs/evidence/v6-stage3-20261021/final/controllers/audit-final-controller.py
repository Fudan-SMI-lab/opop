from pathlib import Path
import json
from kernel_optimizer.runtime.execution import ExecutionRequest,LocalExecutor,write_json
r=Path('/root/opop-v6-stage3');root=r/'final-e2e-187617c';results=[]
for name in ['resnet-memory','resnet-latency','qwen-ttft','qwen-throughput','graph-latency','ddp-step','ddp-communication']:
 p=root/name;request=json.loads((p/'request.json').read_text());prepared=json.loads((p/'prepared.json').read_text());controls=Path(prepared['controls_path']);source=controls.parent/'reference-source'
 case='graph' if name.startswith('graph') else name.split('-')[0]
 parameters={'case':case,'request':request,'source':str(source),'controls':str(controls)}
 if case=='resnet':parameters.update(assets='/root/autodl-tmp/dev/v6-work/validation-assets/resnet18',top1_key='predicted_classes' if name=='resnet-memory' else 'predicted_class_ids')
 if case=='qwen':parameters['assets']='/root/autodl-tmp/c3-qwen3-4b-operator-goals/assets/Qwen3-4B/1cfa9a7208912126459214e8b04321603b3df60c'
 destination=r/'final-source-audit'/name
 actual=LocalExecutor().run(ExecutionRequest(entry=r/'stage3_audit_worker.py',function='ddp_reference' if case=='ddp' else 'inference_reference',cwd=source,output=destination,devices=request['devices'][:1],lease_dir=Path(request['lease_dir']),wait_for_devices=True,worker_timeout_s=600,parameters=parameters))
 write_json(destination/'result.json',actual);results.append({'case':name,'result':actual});write_json(r/'final-source-audit/result.json',{'scope':'Independent developer audit; no modifications or messages to active pipelines','runs':results});print(name,actual['status'],flush=True)
 if case=='graph':
  heldout={**request,'workload':{'id':'unseen_graph','nodes':7168,'edges':143360}}
  actual=LocalExecutor().run(ExecutionRequest(entry=r/'stage3_audit_worker.py',function='graph_heldout_probe',cwd=source,output=r/'graph-frozen-heldout-probe',devices=request['devices'],lease_dir=Path(request['lease_dir']),wait_for_devices=True,worker_timeout_s=600,parameters={'request':heldout,'source':str(source),'adapter':str(source/'opop_adapter.py')}))
  write_json(r/'graph-frozen-heldout-probe/result.json',actual);print('graph heldout',actual['status'],flush=True)
