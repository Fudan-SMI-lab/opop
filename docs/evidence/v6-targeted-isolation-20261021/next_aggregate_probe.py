from pathlib import Path
import json,hashlib
from kernel_optimizer.config import load_config
from kernel_optimizer.project.agents import ProjectAgents
from kernel_optimizer.project.generation_snapshot import load
from kernel_optimizer.project.generated import CollectionPlan
from kernel_optimizer.project.measurement import metric
from kernel_optimizer.project.runtime_config import project_agent_config
from kernel_optimizer.runtime.execution import write_json
from kernel_optimizer.store.run_store import RunStore
from kernel_optimizer.wiring import Runtime

old=Path('/root/autodl-tmp/dev/v6-work/evidence-followup-20261021')
root=Path('/root/autodl-tmp/dev/v6-work/targeted-isolation-20261021')
prior=json.loads((root/'planner-clarity-check/ddp-communication/result.json').read_text())
snapshot=Path(prior['receipt']['generation_snapshots'][0]);manifest,prompt,files=load(snapshot)
value=json.loads(files['inputs/evaluation.json']);request=json.loads(files['inputs/request.json'])
prepared=json.loads((old/'fresh/ddp-communication/prepared.json').read_text())
objective=request['objective'];reducer=objective.get('rank_aggregation','max' if objective['direction']=='minimize' else 'min')
aggregate={'metric':objective['name'],'unit':objective['unit'],'rank_aggregation':reducer,'value':metric(prepared['evaluation'],objective['name'],reducer),
           'scope':'Original reference objective reduced from the per-rank metrics above using the frozen request; not a paired improvement or candidate score.'}
value['original_reference']['objective_aggregate']=aggregate
out=root/'ddp-aggregate-probe';store=RunStore.create(out.parent,out.name,{'kind':'adaptive_objective_aggregate_probe'})
write_json(out/'protocol.json',{'snapshot':str(snapshot),'control':str(root/'planner-clarity-check/ddp-communication'),
 'scope':'One field addition to original_reference in evaluation.json; every original JSON value retained, same prompt/model/tool settings. Reuses preceding fixed planner as comparator, no statistical generality claim.',
 'addition':aggregate,'original_input_sha256':hashlib.sha256(files['inputs/evaluation.json']).hexdigest()})
config=project_agent_config(load_config(old/'ddp-communication-fresh-private.yaml',[]));config.opencode.port=16845;config.usage.mode='observe_only'
def validate(directory):
    p=CollectionPlan.model_validate_json((directory/'collection.json').read_bytes())
    assert set(p.collectors)<=set(json.loads(files['inputs/available-collectors.json']))
    choices={r['revision'] for r in json.loads(files['inputs/revisions.json'])['candidates']}
    assert not p.collectors or p.revision in choices
with Runtime(config,log_dir=out) as runtime:
    directory,receipt=ProjectAgents(runtime,store).replay_snapshot(snapshot,
      edit={'path':'inputs/evaluation.json','old':files['inputs/evaluation.json'].decode(),'new':json.dumps(value,indent=2)},validator=validate)
result={'status':'generated','receipt':receipt,'plan':json.loads((directory/'collection.json').read_text())}
write_json(out/'result.json',result);write_json(out/'completed.json',{'completed':True});print(json.dumps(result['plan']),flush=True)
