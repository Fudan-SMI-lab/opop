from pathlib import Path
import json,time,traceback
from kernel_optimizer.runtime.execution import ExecutionRequest,LocalExecutor,write_json
root=Path('/root/autodl-tmp/dev/v6-work/agent-repair-20261021')
rows=[]
for case in ['qwen-ttft','resnet-latency']:
 b=root/'fresh-final'/case
 while not (b/'prepared.json').exists():time.sleep(5)
 try:
  prepared=json.loads((b/'prepared.json').read_text());request=json.loads((b/'request.json').read_text())
  if prepared['status']!='prepared':raise ValueError('adapter controls failed')
  controls=Path(prepared['controls_path']);source=controls.parent/'reference-source'
  params={'case':'qwen' if case.startswith('qwen') else 'resnet','request':request,'source':str(source),'controls':str(controls)}
  if case.startswith('qwen'):params['assets']=request['assets']['qwen3_4b']
  else:params.update(assets=str(Path(request['assets']['config.json']).parent),top1_key='predicted_class_ids')
  result=LocalExecutor().run(ExecutionRequest(entry=root/'source-audit-worker.py',function='inference_reference',cwd=source,output=root/'fresh-source-audit'/case,devices=request['devices'][:1],lease_dir=request['lease_dir'],wait_for_devices=True,worker_timeout_s=request['worker_timeout_s'],parameters=params))
  write_json(root/'fresh-source-audit'/(case+'.json'),result)
  row={'case':case,'status':result['status'],'valid':bool(result.get('ranks')) and all(r.get('result',{}).get('valid') for r in result.get('ranks',[])),'method':'Manually reviewed observation mapping; original source model/input construction and full output comparison'}
 except Exception as exc:row={'case':case,'error':str(exc),'traceback':traceback.format_exc()}
 rows.append(row);write_json(root/'fresh-source-audit-result.json',{'runs':rows});print(json.dumps(row),flush=True)
