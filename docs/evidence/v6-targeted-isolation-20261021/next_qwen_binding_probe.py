from pathlib import Path
import json
from kernel_optimizer.config import load_config
from kernel_optimizer.project.branches import fork_run
from kernel_optimizer.project.search import ProjectSearch
from kernel_optimizer.runtime.execution import write_json

old=Path('/root/autodl-tmp/dev/v6-work/evidence-followup-20261021')
root=Path('/root/autodl-tmp/dev/v6-work/targeted-isolation-20261021')
source=old/'fresh/qwen-ttft';out=root/'qwen-binding-eligibility'
parent=fork_run(source,out,revision_id='20b69c256741a0d1d90fc6ccaf090347c2dbb16ef9007bc7486f57c5337e1df6')
search=ProjectSearch(load_config(old/'qwen-ttft-fresh-private.yaml',[]),out)
candidate=search.revisions.revision('f5a797fab6e3796855819951a220124f9d6305b3152f9de9304b60b3a17d4030')
write_json(out/'protocol.json',{'scope':'Developer executes only the newly selected binding_cost collector from planner-clarity-check/qwen-ttft on the original rejected third-round revision. No candidate change, no optimization acceptance, no injection into running repair A/B.',
 'parent':parent.id,'candidate':candidate.id,'new_collectors':['binding_cost']})
result=search.collect_extra(candidate,['binding_cost'])
write_json(out/'result.json',result);write_json(out/'completed.json',{'completed':True,'status':result['binding_cost']['status']})
print(json.dumps({k:result['binding_cost'].get(k) for k in ['status','reason','elapsed_pairs']}),flush=True)
