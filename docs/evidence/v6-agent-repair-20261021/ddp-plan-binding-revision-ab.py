from pathlib import Path
import json
from kernel_optimizer.config import load_config
from kernel_optimizer.project.agents import ProjectAgents
from kernel_optimizer.project.generated import CollectionPlan
from kernel_optimizer.project.generation_snapshot import load
from kernel_optimizer.project.runtime_config import project_agent_config
from kernel_optimizer.store.run_store import RunStore
from kernel_optimizer.project.transactions import RevisionStore
from kernel_optimizer.runtime.execution import write_json
from kernel_optimizer.wiring import Runtime
root=Path('/root/autodl-tmp/dev/v6-work/agent-repair-20261021');source=root/'fresh-fixed/ddp-communication'
original=next(d for p in (source/'collector-plans').glob('*/result.json') if (d:=json.loads(p.read_text()))['plan'].get('revision'))
snapshot=Path(original['agent']['generation_snapshots'][0]);_,_,files=load(snapshot)
before=files['inputs/evaluation.json'].decode();value=json.loads(before)
prepared=json.loads((source/'prepared.json').read_text())
assert value['original_reference']['revision']==prepared['revision']
value['original_reference']['bundles']=list(RevisionStore(RunStore.open(source)).revision(prepared['revision']).bundles)
edit={'path':'inputs/evaluation.json','old':before,'new':json.dumps(value,indent=2)}
output=root/'ddp-plan-binding-revision-ab'
store=RunStore.create(output.parent,output.name,{'scope':'Only add actual installed bundle identifiers from the original-reference revision. Same frozen prompt/model/inputs; no binding-cost diagnosis or prescribed collector added.'})
config=project_agent_config(load_config(root/'ddp-binding-revision-private.yaml',[]));config.usage.mode='observe_only'
choices={r['revision'] for r in json.loads(files['inputs/revisions.json'])['candidates']};names=set(json.loads(files['inputs/available-collectors.json']))
def validate(path):
 p=CollectionPlan.model_validate_json((path/'collection.json').read_bytes())
 if p.collectors and p.revision not in choices:raise ValueError('select a supplied actual revision')
 if set(p.collectors)-names:raise ValueError('collector is not registered')
with Runtime(config,log_dir=output) as runtime:
 directory,agent=ProjectAgents(runtime,store).replay_snapshot(snapshot,edit=edit,validator=validate)
plan=json.loads((directory/'collection.json').read_text())
write_json(output/'result.json',{'control':original,'treatment':{'plan':plan,'agent':agent},'edit':edit,'source_reference':str(source/'prepared.json'),'scope':'Single frozen-input planning comparison; no optimization outcome inferred.'})
print(json.dumps(plan),flush=True)
