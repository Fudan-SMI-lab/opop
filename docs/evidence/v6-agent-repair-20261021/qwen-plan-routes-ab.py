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
root=Path('/root/autodl-tmp/dev/v6-work/agent-repair-20261021');source=root/'fresh-final/qwen-ttft'
original=next(d for p in (source/'collector-plans').glob('*/result.json') if (d:=json.loads(p.read_text()))['plan'].get('revision'))
snapshot=Path(original['agent']['generation_snapshots'][0]);_,_,files=load(snapshot)
before=files['inputs/evaluation.json'].decode();value=json.loads(before)
measured=[r['metrics'] for r in value['endpoint']['ranks']]
run=None;receipt=None
for path in (source/'jobs').glob('*pair-*-candidate/run/receipt.json'):
 d=json.loads(path.read_text())
 if [r.get('result',{}).get('metrics') for r in d.get('ranks',[])]==measured:run=d;receipt=str(path);break
assert run is not None
for compact,actual in zip(value['endpoint']['ranks'],run['ranks'],strict=True):
 assert compact['rank']==actual['rank']
 for bundle,coverage in compact['coverage'].items():
  routes=actual['result']['coverage'][bundle].get('candidate_declared_routes',[])
  if routes:coverage['candidate_declared_routes']=routes
value['endpoint']['interpretation']='Reference invocations may implement fallback, verification or composition. Route labels are candidate-reported telemetry, not independent native proof; corroborate them with traces or actual exception evidence.'
edit={'path':'inputs/evaluation.json','old':before,'new':json.dumps(value,indent=2)}
output=root/'qwen-plan-routes-ab';store=RunStore.create(output.parent,output.name,{'scope':'Only restore existing candidate-reported route telemetry and its provenance warning; same frozen generation inputs/model/prompt; no failure-policy routing'})
config=project_agent_config(load_config(root/'qwen-route-plan-private.yaml',[]));config.usage.mode='observe_only'
choices={r['revision'] for r in json.loads(files['inputs/revisions.json'])['candidates']};names=set(json.loads(files['inputs/available-collectors.json']))
def validate(path):
 p=CollectionPlan.model_validate_json((path/'collection.json').read_bytes())
 if p.collectors and p.revision not in choices:raise ValueError('select a supplied actual revision')
 if set(p.collectors)-names:raise ValueError('collector is not registered')
with Runtime(config,log_dir=output) as runtime:
 directory,receipt_agent=ProjectAgents(runtime,store).replay_snapshot(snapshot,edit=edit,validator=validate)
plan=json.loads((directory/'collection.json').read_text())
write_json(output/'result.json',{'control':original,'treatment':{'plan':plan,'agent':receipt_agent},'edit':edit,'source_receipt':receipt,'scope':'Single generation comparison; telemetry is a candidate claim, no optimization outcome or independent mechanism proof inferred'})
print(json.dumps(plan),flush=True)
