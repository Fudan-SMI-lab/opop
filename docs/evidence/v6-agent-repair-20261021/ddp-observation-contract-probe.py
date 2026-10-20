from pathlib import Path
import json
from kernel_optimizer.config import load_config
from kernel_optimizer.project.branches import fork_run
from kernel_optimizer.project.search import ProjectSearch
from kernel_optimizer.project.communication import available_fields
from kernel_optimizer.runtime.execution import write_json
root=Path('/root/autodl-tmp/dev/v6-work/agent-repair-20261021');source=root/'fresh-final/ddp-communication';out=root/'ddp-observation-contract-probe'
prepared=json.loads((source/'prepared.json').read_text());parent=fork_run(source,out,revision_id=prepared['revision'])
search=ProjectSearch(load_config(root/'ddp-communication-fresh-final-private.yaml',[]),out)
run=search.execute(parent,'same-adapter-request-discovery','profile')
observations=[]
for row in run.get('ranks',[]):
 if row.get('result',{}).get('discovery'):
  data=json.loads(Path(row['result']['discovery']).read_text())['communication']
  observations.append({'rank':row['rank'],'status':data['status'],'communication_kernel_count':data['communication_kernel_count'],'communication_kernel_sum_ms':data['communication_kernel_sum_ms'],'available_fields':sorted(available_fields(data))})
write_json(out/'result.json',{'scope':'Same frozen request and adapter; only framework availability interface changed; profiling values are diagnostic, not clean performance','required_observations':search.request.required_observations,'run':run,'observations':observations})
print(json.dumps({'status':run['status'],'observations':observations}),flush=True)
