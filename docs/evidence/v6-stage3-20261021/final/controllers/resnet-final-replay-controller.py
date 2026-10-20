from pathlib import Path
import json,traceback
from kernel_optimizer.config import load_config
from kernel_optimizer.project.search import ProjectSearch
from kernel_optimizer.project.replay import transfer_bundle
from kernel_optimizer.runtime.execution import write_json
r=Path('/root/opop-v6-stage3');root=Path('/root/autodl-tmp/dev/v6-work/stage3-variants');origin=r/'final-e2e-187617c/resnet-memory';source_id='trial-c0affa75db-param-bfc570b5';rows=[]
for mode in ['heldout','cross-goal']:
 output=root/('resnet-memory-'+mode);record={'mode':mode,'source_bundle':source_id,'source_output':str(origin),'destination':str(output),'scope':'Developer independent frozen final candidate replay with explicit omission of source-only replacement history; no code or parameter changes'}
 try:
  search=ProjectSearch(load_config(r/'deployment.yaml',[]),output);bundle,meta=transfer_bundle(origin,source_id,search,allow_absent_replacements=True)
  result=search.validate_bundle(bundle,meta,publish=False);record.update(trial=result['trial'],decision=result['decision'],measurement=result.get('measurement'))
 except Exception as e:record.update(error=str(e),traceback=traceback.format_exc())
 rows.append(record);write_json(root/'resnet-final-replay-result.json',{'runs':rows});print(mode,record.get('decision',record.get('error')),flush=True)
