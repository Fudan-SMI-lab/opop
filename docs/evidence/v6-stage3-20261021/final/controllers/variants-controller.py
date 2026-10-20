from pathlib import Path
import json,traceback
from kernel_optimizer.config import load_config
from kernel_optimizer.project.request import ProjectRequest
from kernel_optimizer.project.variants import prepare_variant
from kernel_optimizer.project.search import ProjectSearch
from kernel_optimizer.runtime.execution import ExecutionRequest,LocalExecutor,write_json
r=Path('/root/opop-v6-stage3');root=r/'final-e2e-187617c'
destination=Path('/root/autodl-tmp/dev/v6-work/stage3-variants');destination.mkdir(exist_ok=True)
jobs=[('resnet-memory','resnet-latency'),('qwen-ttft','qwen-throughput'),('ddp-step','ddp-communication')]
records=[]
for index,(case,opposite) in enumerate(jobs):
 parent=root/case;original=ProjectRequest.model_validate_json((parent/'request.json').read_bytes())
 source=Path(json.loads((parent/'prepared.json').read_text())['controls_path']).parent/'reference-source'
 configdata=json.loads((source/'workloads.json').read_text())
 kind=case.split('-')[0];section={'resnet':'resnet_inference','qwen':'qwen_inference','ddp':'gpt2_ddp_training'}[kind]
 other=json.loads((root/opposite/'request.json').read_text())
 for mode in ['heldout','cross-goal']:
  output=destination/(case+'-'+mode)
  config=load_config(r/'deployment.yaml',[]);config.opencode.port=15530+index*2+(mode=='cross-goal')
  update={'workload':configdata[section]['held_out_workloads'][0]} if mode=='heldout' else {'goal':other['goal'],'objective':other['objective']}
  request=original.model_copy(update=update)
  record={'parent':str(parent),'output':str(output),'mode':mode,'workload':request.workload,'objective':request.objective,
    'scope':'Developer-started frozen-evaluator variant; contract adaptation uses real Agent; no candidate code/parameter edits'}
  try:
   prepared=prepare_variant(parent,request,output,config=config);record['prepared_status']=prepared['status']
   if prepared['status']!='prepared':raise ValueError('variant controls did not pass')
   controls=Path(prepared['controls_path']);actual_source=controls.parent/'reference-source'
   parameters={'case':kind,'request':request.portable(),'source':str(actual_source),'controls':str(controls)}
   if kind=='resnet':parameters.update(assets='/root/autodl-tmp/dev/v6-work/validation-assets/resnet18',top1_key='predicted_classes')
   if kind=='qwen':parameters['assets']='/root/autodl-tmp/c3-qwen3-4b-operator-goals/assets/Qwen3-4B/1cfa9a7208912126459214e8b04321603b3df60c'
   actual=LocalExecutor().run(ExecutionRequest(entry=r/'stage3_audit_worker.py',function='ddp_reference' if kind=='ddp' else 'inference_reference',cwd=actual_source,output=output/'source-audit',devices=request.devices[:1],lease_dir=request.lease_dir,wait_for_devices=True,worker_timeout_s=600,parameters=parameters))
   write_json(output/'source-audit.json',actual)
   valid=actual['status']=='passed' and all(x.get('result',{}).get('valid',False) for x in actual['ranks'])
   record['source_valid']=valid
   if not valid:raise ValueError('independent original-source comparison failed; no frozen candidate replay')
   search=ProjectSearch(config,output);search.discover();record['ready_for_frozen_bundle']=True
  except Exception as e:
   record.update(error=str(e),traceback=traceback.format_exc())
  records.append(record);write_json(destination/'preparation-result.json',{'runs':records})
  print(case,mode,record.get('error','ready'),flush=True)
