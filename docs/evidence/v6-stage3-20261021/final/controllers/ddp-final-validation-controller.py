from pathlib import Path
import json,time,traceback
from kernel_optimizer.config import load_config
from kernel_optimizer.project.search import ProjectSearch
from kernel_optimizer.project.replay import transfer_bundle
from kernel_optimizer.runtime.execution import write_json
r=Path('/root/opop-v6-stage3');origin=r/'final-e2e-187617c/ddp-step';root=Path('/root/autodl-tmp/dev/v6-work/stage3-variants');record={'scope':'Developer independent frozen final candidate cross-goal/held-out tests; no changes to original pipeline','runs':[]}
while True:
 line=(r/'final-ddp-step.log').read_text().splitlines()[-1]
 try:returned=json.loads(line)
 except json.JSONDecodeError:returned={}
 if returned.get('status')=='completed_experiment':break
 if not Path('/proc/927915').exists():
  record.update(status='original_pipeline_did_not_complete',last_log_line=line);write_json(root/'ddp-final-replay-result.json',record);raise SystemExit(1)
 time.sleep(5)
revision=json.loads((origin/'project/current.json').read_text())['revision'];bundles=json.loads((origin/'project/revisions'/(revision+'.json')).read_text())['bundles'];record['source_revision']=revision
if not bundles:
 record.update(status='not_applicable_no_accepted_candidate',reason='Both variant baselines were prepared and source-audited, but the complete original run accepted no candidate to transfer.');write_json(root/'ddp-final-replay-result.json',record);print(record['status'],flush=True);raise SystemExit()
assert len(bundles)==1,bundles
for mode in ['heldout','cross-goal']:
 output=root/('ddp-step-'+mode);row={'mode':mode,'source_bundle':bundles[0],'destination':str(output)}
 try:
  search=ProjectSearch(load_config(r/'deployment.yaml',[]),output);bundle,meta=transfer_bundle(origin,bundles[0],search);result=search.validate_bundle(bundle,meta,publish=False);row.update(trial=result['trial'],decision=result['decision'],measurement=result.get('measurement'))
 except Exception as e:row.update(error=str(e),traceback=traceback.format_exc())
 record['runs'].append(row);record['status']='completed' if len(record['runs'])==2 else 'running';write_json(root/'ddp-final-replay-result.json',record);print(mode,row.get('decision',row.get('error')),flush=True)
