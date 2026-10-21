from pathlib import Path
import json,hashlib
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
snapshot=old/'fresh/qwen-ttft/generation-snapshots/generation-d0eef7e5923b.json'
manifest,prompt,files=load(snapshot)
addition='\ncandidate_diagnostics records executed candidate Python calls, handled exceptions with propagated library traceback locations, and limited scalar returns in a separate instrumented run. It does not compare intermediate tensors, measure per-site numerical errors, or locate the first numerical divergence. Those observations remain unavailable unless another actual collector supplies them.\n'
out=root/'qwen-capability-probe';store=RunStore.create(out.parent,out.name,{'kind':'adaptive_frozen_prompt_probe'})
write_json(out/'protocol.json',{'snapshot':str(snapshot),'treatment':addition,
 'input_sha256':{n:hashlib.sha256(v).hexdigest() for n,v in files.items()},
 'scope':'Adaptive third prompt arm after observing collector capability hallucination in the original and fresh control. Reuses qwen-planner-ab/control as unchanged comparator; no new independent control or statistical claim. Only capability description added, no binding clarification or prescribed collector choice.'})
config=project_agent_config(load_config(old/'qwen-ttft-fresh-private.yaml',[]));config.opencode.port=16840;config.usage.mode='observe_only'
def validate(directory):
    p=CollectionPlan.model_validate_json((directory/'collection.json').read_bytes())
    assert set(p.collectors)<=set(json.loads(files['inputs/available-collectors.json']))
    candidates={r['revision'] for r in json.loads(files['inputs/revisions.json'])['candidates']}
    assert not p.collectors or p.revision in candidates
with Runtime(config,log_dir=out) as runtime:
    directory,receipt=ProjectAgents(runtime,store).replay_snapshot(snapshot,
        edit={'path':'prompt','old':prompt,'new':prompt+addition},validator=validate)
result={'status':'generated','receipt':receipt,'plan':json.loads((directory/'collection.json').read_text())}
write_json(out/'result.json',result);write_json(out/'completed.json',{'completed':True})
print(json.dumps(result['plan']),flush=True)
