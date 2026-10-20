from pathlib import Path
import json,time,traceback
from kernel_optimizer.config import load_config
from kernel_optimizer.project.branches import fork_run
from kernel_optimizer.project.search import ProjectSearch
from kernel_optimizer.project.replay import transfer_bundle
from kernel_optimizer.project.variants import prepare_variant
from kernel_optimizer.runtime.execution import write_json
root=Path('/root/autodl-tmp/dev/v6-work/agent-repair-20261021')
cases=['qwen-ttft','qwen-throughput','ddp-step','ddp-communication']
rows=[];done=set()
config=load_config('/root/opop-v6-stage3/deployment.yaml',[])
config.opencode.port=15745
while len(done)<len(cases)*2:
 for case in cases:
  for arm in ['control','treatment']:
   key=case+'-'+arm
   if key in done:continue
   result_file=root/case/'result.json'
   if not result_file.exists():continue
   match=next((r for r in json.loads(result_file.read_text())['runs'] if r['arm']==arm),None)
   if match is None:continue
   done.add(key)
   if not match.get('decision',{}).get('accepted'):
    rows.append({'case':case,'arm':arm,'status':'not_accepted','original':match})
    write_json(root/'independent-validation-result.json',{'runs':rows});continue
   source=Path(match['output']);original=json.loads((source/'trials'/match['trial']/'result.json').read_text())
   for mode in ['repeat','heldout']:
    destination=root/'independent-validation'/(key+'-'+mode)
    row={'case':case,'arm':arm,'mode':mode,'source':str(source),'output':str(destination),'scope':'Developer-started frozen-code validation; no new candidate generation'}
    try:
     if mode=='repeat':
      fork_run(source,destination,revision_id=original['bundle']['parent_revision'])
     else:
      origin=ProjectSearch(config,source)
      reference=Path(origin.prepared['controls_path']).parent/'reference-source'
      definitions=json.loads((reference/'workloads.json').read_text())
      section='qwen_inference' if case.startswith('qwen') else 'gpt2_ddp_training'
      workload=definitions[section]['held_out_workloads'][0]
      request=origin.request.model_copy(update={'workload':workload})
      prepared=prepare_variant(source,request,destination,config=config)
      if prepared['status']!='prepared':raise ValueError('frozen adapter failed heldout controls')
     search=ProjectSearch(config,destination)
     discovery=search.current_discovery()
     bundle,metadata=transfer_bundle(source,original['bundle']['id'],search,discovery=discovery)
     result=search.validate_bundle(bundle,metadata,discovery=discovery,publish=False)
     row.update(decision=result['decision'],evaluation=result.get('evaluation'),measurement=result.get('measurement'),trial=bundle.id)
    except Exception as exc:
     row.update(error=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc())
    rows.append(row);write_json(root/'independent-validation-result.json',{'runs':rows});print(json.dumps({k:v for k,v in row.items() if k not in ['evaluation','measurement','traceback']}),flush=True)
 time.sleep(5)
write_json(root/'independent-validation-completed.json',{'completed':True,'runs':len(rows)})
