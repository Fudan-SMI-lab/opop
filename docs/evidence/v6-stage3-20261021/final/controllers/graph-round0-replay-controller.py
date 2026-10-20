from pathlib import Path
import json,traceback
from kernel_optimizer.config import load_config
from kernel_optimizer.project.search import ProjectSearch
from kernel_optimizer.project.replay import transfer_bundle
from kernel_optimizer.runtime.execution import write_json
r=Path('/root/opop-v6-stage3');root=Path('/root/autodl-tmp/dev/v6-work/stage3-variants');origin=r/'graph-followup-8fd2123';source_id='trial-678b0307f5-param-9fc9967a';output=root/'graph-followup-heldout';record={'source_bundle':source_id,'source_output':str(origin),'destination':str(output),'scope':'Developer independent frozen round-0 graph followup candidate replay; no code or parameter changes'}
try:
 search=ProjectSearch(load_config(r/'deployment.yaml',[]),output);bundle,meta=transfer_bundle(origin,source_id,search)
 result=search.validate_bundle(bundle,meta,publish=False);record.update(trial=result['trial'],decision=result['decision'],measurement=result.get('measurement'))
except Exception as e:record.update(error=str(e),traceback=traceback.format_exc())
write_json(root/'graph-round0-replay-result.json',record);print(record.get('decision',record.get('error')),flush=True)
