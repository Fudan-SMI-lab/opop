from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
import hashlib,json,traceback
from kernel_optimizer.config import load_config
from kernel_optimizer.project.agents import ProjectAgents
from kernel_optimizer.project.generation_snapshot import load
from kernel_optimizer.project.generated import CollectionPlan
from kernel_optimizer.project.runtime_config import project_agent_config
from kernel_optimizer.runtime.execution import write_json
from kernel_optimizer.store.run_store import RunStore
from kernel_optimizer.wiring import Runtime

old=Path('/root/autodl-tmp/dev/v6-work/evidence-followup-20261021')
root=Path('/root/autodl-tmp/dev/v6-work/targeted-isolation-20261021')
capability='candidate_diagnostics records executed candidate Python calls, handled exceptions with propagated library traceback locations, and limited scalar returns in a separate instrumented run. It does not compare intermediate tensors, measure per-site numerical errors, or locate the first numerical divergence. Those observations remain unavailable unless another actual collector supplies them.'
binding='For binding_cost, additive means the candidate retains all installed parent bundles and adds new bundles; it does not mean the replacement must add mathematical work. Direct module/project bindings are eligible under this condition, and no functional replacements does not imply zero binding overhead. Removing/replacing an installed parent bundle makes this passthrough comparison inapplicable.'
addition='\n'+capability+'\n'+binding+'\n'
def run(case,snapshot_name,offset):
    snapshot=old/'fresh'/case/'generation-snapshots'/snapshot_name
    manifest,prompt,files=load(snapshot);out=root/'planner-clarity-check'/case
    store=RunStore.create(out.parent,out.name,{'kind':'cross_task_clarity_check'})
    write_json(out/'protocol.json',{'snapshot':str(snapshot),'addition':addition,'scope':'Adaptive clarity check. Qwen comparator is the preceding capability-only probe, adding only the binding-cost eligibility explanation; DDP is a cross-task real-call check against its original saved planner. Not an independently randomized new A/B.',
        'input_sha256':{n:hashlib.sha256(v).hexdigest() for n,v in files.items()}})
    config=project_agent_config(load_config(old/(case+'-fresh-private.yaml'),[]));config.opencode.port=16842+offset;config.usage.mode='observe_only'
    def validate(directory):
        p=CollectionPlan.model_validate_json((directory/'collection.json').read_bytes())
        assert set(p.collectors)<=set(json.loads(files['inputs/available-collectors.json']))
        candidates={r['revision'] for r in json.loads(files['inputs/revisions.json'])['candidates']}
        assert not p.collectors or p.revision in candidates
    try:
        with Runtime(config,log_dir=out) as runtime:
            directory,receipt=ProjectAgents(runtime,store).replay_snapshot(snapshot,edit={'path':'prompt','old':prompt,'new':prompt+addition},validator=validate)
        row={'case':case,'status':'generated','receipt':receipt,'plan':json.loads((directory/'collection.json').read_text())}
    except Exception as e:row={'case':case,'status':'error','error':str(e),'traceback':traceback.format_exc()}
    write_json(out/'result.json',row);print(json.dumps({k:v for k,v in row.items() if k not in ['receipt','plan']}),flush=True)
    return row
rows=[]
with ThreadPoolExecutor(max_workers=2) as pool:
    tasks=[pool.submit(run,*args) for args in [('qwen-ttft','generation-d0eef7e5923b.json',0),('ddp-communication','generation-62257ce9b6dc.json',1)]]
    for task in as_completed(tasks):rows.append(task.result());write_json(root/'planner-clarity-result.json',{'runs':rows})
write_json(root/'planner-clarity-completed.json',{'completed':True,'runs':len(rows)})
