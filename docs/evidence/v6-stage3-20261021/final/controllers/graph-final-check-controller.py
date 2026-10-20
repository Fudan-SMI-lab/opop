from pathlib import Path
import json,traceback
from kernel_optimizer.config import load_config
from kernel_optimizer.project.search import ProjectSearch
from kernel_optimizer.project.replay import transfer_bundle
from kernel_optimizer.project.branches import fork_run
from kernel_optimizer.project.versioning import promote_version
from kernel_optimizer.runtime.execution import write_json
r=Path('/root/opop-v6-stage3');origin=r/'graph-followup-8fd2123';root=Path('/root/autodl-tmp/dev/v6-work/stage3-variants');cfg=load_config(r/'deployment.yaml',[])
source_id='trial-c671461bdd-param-412ad883';record={'source_bundle':source_id,'source_output':str(origin),'scope':'Developer independent final candidate replay; code and parameters frozen; explicit source-only replacement omission'}
try:
 search=ProjectSearch(cfg,root/'graph-followup-heldout');bundle,meta=transfer_bundle(origin,source_id,search,allow_absent_replacements=True);result=search.validate_bundle(bundle,meta,publish=False);record.update(trial=result['trial'],decision=result['decision'],measurement=result.get('measurement'))
except Exception as e:record.update(error=str(e),traceback=traceback.format_exc())
write_json(root/'graph-final-replay-result.json',record);print('heldout',record.get('decision',record.get('error')),flush=True)
output=root/'graph-final-independent';prepared=json.loads((origin/'prepared.json').read_text());fork_run(origin,output,revision_id=prepared['revision']);result=promote_version(ProjectSearch(cfg,output),origin,publish=False);write_json(root/'graph-final-independent-result.json',{'scope':'Independent complete final version AB/BA against original baseline; no publication to original run','trial':result['trial'],'decision':result['decision'],'measurement':result['measurement']});print('independent',result['decision'],result['measurement'],flush=True)
