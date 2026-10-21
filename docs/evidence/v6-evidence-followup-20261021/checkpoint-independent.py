from pathlib import Path
import json, traceback
from kernel_optimizer.config import load_config
from kernel_optimizer.project.branches import fork_run
from kernel_optimizer.project.search import ProjectSearch
from kernel_optimizer.project.versioning import promote_version
from kernel_optimizer.runtime.execution import write_json
root=Path('/root/autodl-tmp/dev/v6-work/evidence-followup-20261021')
protocol={'scope':'Developer independent complete-version validation of immutable round-two checkpoints while the autonomous third round waits for provider recovery. Not a completed three-round cohort. No candidate edits, new Agent input, source publication or retry-until-pass.',
 'framework':'a6f2b6e88ec31f169acd888601fcbb32cc2c1f9c',
 'cases':[
  {'case':'qwen-ttft','revision':'9fb6e370922706c28b91ff0bb2273a32d29150c9376855c3880b21d819684b42'},
  {'case':'resnet-latency','revision':'f12635aeab45ee38bb106d310a6ceff09f2faf366dde6d423cf56a72d015f7d1'}]}
write_json(root/'checkpoint-independent-protocol.json',protocol)
rows=[]
for item in protocol['cases']:
 case=item['case'];source=root/'fresh'/case;out=root/'checkpoint-independent'/case
 row={**item,'scope':protocol['scope'],'source':str(source),'output':str(out)}
 try:
  config=load_config(root/(case+'-fresh-private.yaml'),[])
  origin=ProjectSearch(config,source)
  fork_run(source,out,revision_id=origin.prepared['revision'])
  search=ProjectSearch(config,out)
  result=promote_version(search,source,revision_id=item['revision'],publish=False)
  row.update(status='evaluated',decision=result['decision'],measurement=result['measurement'],quality=result['quality'],coverage_errors=result['coverage_errors'],bundles=result['bundles'],trial=result['trial'])
 except Exception as exc:row.update(status='error',error=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc())
 rows.append(row);write_json(root/'checkpoint-independent-result.json',{'runs':rows});print(json.dumps(row),flush=True)
write_json(root/'checkpoint-independent-completed.json',{'completed':True,'runs':len(rows)})
