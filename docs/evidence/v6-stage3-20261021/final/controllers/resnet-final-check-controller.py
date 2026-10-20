from pathlib import Path
import json,traceback
from kernel_optimizer.config import load_config
from kernel_optimizer.project.search import ProjectSearch
from kernel_optimizer.project.replay import transfer_bundle
from kernel_optimizer.project.branches import fork_run
from kernel_optimizer.project.versioning import promote_version
from kernel_optimizer.runtime.execution import write_json
r=Path('/root/opop-v6-stage3');origin=r/'final-e2e-187617c/resnet-memory';root=Path('/root/autodl-tmp/dev/v6-work/stage3-variants');cfg=load_config(r/'deployment.yaml',[])
search=ProjectSearch(cfg,root/'resnet-memory-heldout')
try:
 bundle,meta=transfer_bundle(origin,'trial-c0affa75db-param-bfc570b5',search)
 proof={'status':'supported','bundle':bundle.id}
except Exception as e:proof={'status':'blocked_by_replay_contract','error':str(e),'traceback':traceback.format_exc(),'source_bundle':'trial-c0affa75db-param-bfc570b5','source_replaces':['trial-3c4e540078-param-d377cbf2'],'target_active_bundles':list(search.revisions.current().bundles)}
write_json(root/'resnet-final-replay-preflight.json',proof);print(proof,flush=True)
output=root/'resnet-final-independent';prepared=json.loads((origin/'prepared.json').read_text());fork_run(origin,output,revision_id=prepared['revision']);result=promote_version(ProjectSearch(cfg,output),origin,publish=False);write_json(root/'resnet-final-independent-result.json',{'scope':'Independent complete final version AB/BA against original baseline, no publication to original run','trial':result['trial'],'decision':result['decision'],'measurement':result['measurement']});print('independent',result['decision'],result['measurement'],flush=True)
