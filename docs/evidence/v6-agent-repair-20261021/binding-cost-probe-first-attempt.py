from pathlib import Path
import json
from kernel_optimizer.config import load_config
from kernel_optimizer.project.branches import fork_run
from kernel_optimizer.project.search import ProjectSearch
from kernel_optimizer.models.project import CandidateBundle
from kernel_optimizer.runtime.execution import write_json
root=Path('/root/autodl-tmp/dev/v6-work/agent-repair-20261021')
base=Path('/root/opop-v6-stage3/final-e2e-187617c')
case='ddp-communication';trial='trial-d17bd1eb8f'
saved=json.loads((base/case/'trials'/trial/'generated.json').read_text())
output=root/'binding-cost-probe'
parent=fork_run(base/case,output,revision_id=saved['bundle']['parent_revision'])
search=ProjectSearch(load_config('/root/opop-v6-stage3/deployment.yaml',[]),output)
candidate=search.revisions.stage(CandidateBundle.model_validate(saved['bundle']))
result=search.collect_extra(candidate,['binding_cost'])
write_json(output/'result.json',result)
assert result['binding_cost']['status']=='observed',result['binding_cost'].get('reason')
assert search.revisions.current().id==parent.id
print(json.dumps({'status':result['binding_cost']['status'],'pairs':result['binding_cost']['elapsed_pairs']}),flush=True)
