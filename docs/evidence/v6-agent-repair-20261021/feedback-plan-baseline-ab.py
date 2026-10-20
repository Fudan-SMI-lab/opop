from pathlib import Path
import json
from kernel_optimizer.config import load_config
from kernel_optimizer.project.agents import ProjectAgents
from kernel_optimizer.project.generated import CollectionPlan
from kernel_optimizer.project.generation_snapshot import load
from kernel_optimizer.project.runtime_config import project_agent_config
from kernel_optimizer.store.run_store import RunStore
from kernel_optimizer.runtime.execution import write_json
from kernel_optimizer.wiring import Runtime
root=Path('/root/autodl-tmp/dev/v6-work/agent-repair-20261021')
original=json.loads((root/'feedback-plan-probe/plan.json').read_text())
snapshot=Path(original['agent']['generation_snapshots'][0]);_,_,files=load(snapshot)
before=files['inputs/evaluation.json'].decode();value=json.loads(before)
prepared=json.loads((root/'feedback-plan-probe/prepared.json').read_text())
value['original_reference']={'revision':prepared['revision'],
 'ranks':[{'rank':r['rank'],'metrics':r.get('result',{}).get('metrics',{})} for r in prepared['evaluation']['ranks']],
 'interpretation':"Original independently measured reference, not necessarily the candidate's current parent. Parent discovery in observations.json is intrusive profiling and must not be substituted for this clean score. Paired parent values, when available, are reported separately in evaluation.measurement."}
edit={'path':'inputs/evaluation.json','old':before,'new':json.dumps(value,indent=2)}
output=root/'feedback-plan-baseline-ab';store=RunStore.create(output.parent,output.name,{'scope':'Single factual input addition, original clean baseline; fresh independent collection-agent session; no optimization-action rules'})
config=project_agent_config(load_config(root/'feedback-plan-baseline-deployment.yaml',[]));config.usage.mode='observe_only'
choices={r['revision'] for r in json.loads(files['inputs/revisions.json'])['candidates']}
def validate(path):
 p=CollectionPlan.model_validate_json((path/'collection.json').read_bytes())
 if p.collectors and p.revision not in choices:raise ValueError('select a supplied actual revision')
with Runtime(config,log_dir=output) as runtime:
 directory,receipt=ProjectAgents(runtime,store).replay_snapshot(snapshot,edit=edit,validator=validate)
plan=json.loads((directory/'collection.json').read_text());write_json(output/'result.json',{'control':original,'treatment':{'plan':plan,'agent':receipt},'edit':edit,'scope':'Generation comparison, not evidence that a chosen optimization improves performance'})
print(json.dumps(plan),flush=True)
