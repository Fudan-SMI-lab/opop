from pathlib import Path
import json,traceback
from kernel_optimizer.runtime.execution import ExecutionRequest,LocalExecutor,write_json
from kernel_optimizer.project.request import ProjectRequest
from kernel_optimizer.project.variants import prepare_variant
from kernel_optimizer.project.search import ProjectSearch
from kernel_optimizer.config import load_config
r=Path('/root/opop-v6-stage3');base=r/'graph-followup-8fd2123';request=ProjectRequest.model_validate_json((base/'request.json').read_bytes());prep=json.loads((base/'prepared.json').read_text());source=Path(prep['controls_path']).parent/'reference-source';records=[]
original_config=json.loads((source/'workloads.json').read_text());heldout=original_config['held_out_workloads'][0]
for workload in [request.workload,heldout]:
 output=r/'graph-followup-source-audit'/workload['id'];actual=LocalExecutor().run(ExecutionRequest(entry=r/'stage3_audit_worker.py',function='graph_heldout_probe',cwd=source,output=output,devices=request.devices,lease_dir=request.lease_dir,wait_for_devices=True,worker_timeout_s=600,parameters={'request':request.model_copy(update={'workload':workload}).portable(),'source':str(source),'adapter':str(source/'opop_adapter.py')}));records.append({'workload':workload,'result':actual});write_json(r/'graph-followup-source-audit/result.json',{'runs':records});print(workload['id'],[x.get('result',{}).get('valid') for x in actual['ranks']],flush=True)
if all(x['result']['status']=='passed' and all(y.get('result',{}).get('valid') for y in x['result']['ranks']) for x in records):
 output=Path('/root/autodl-tmp/dev/v6-work/stage3-variants/graph-followup-heldout');cfg=load_config(r/'deployment.yaml',[]);cfg.opencode.port=15545;variant=request.model_copy(update={'workload':heldout});record={'parent':str(base),'output':str(output),'workload':heldout,'scope':'Frozen followup evaluator variant; actual contract-only Agent call; original-source check already passed'}
 try:
  prepared=prepare_variant(base,variant,output,config=cfg);record['prepared_status']=prepared['status']
  if prepared['status']!='prepared':raise ValueError('variant controls failed')
  ProjectSearch(cfg,output).discover();record['ready_for_frozen_bundle']=True
 except Exception as e:record.update(error=str(e),traceback=traceback.format_exc())
 write_json(output.parent/'graph-followup-variant-result.json',record);print(record,flush=True)
