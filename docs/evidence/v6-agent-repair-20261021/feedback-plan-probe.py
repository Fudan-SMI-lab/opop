from pathlib import Path
import json,time,shutil
from kernel_optimizer.config import load_config
from kernel_optimizer.project.branches import fork_run
from kernel_optimizer.project.search import ProjectSearch
from kernel_optimizer.project.diagnostic_planning import endpoint_summary
from kernel_optimizer.runtime.execution import write_json
root=Path('/root/autodl-tmp/dev/v6-work/agent-repair-20261021')
source=root/'ddp-communication/0-control';original=json.loads((source/'trials/trial-8a27998780/result.json').read_text())
output=root/'feedback-plan-probe';parent=fork_run(source,output,revision_id=original['bundle']['parent_revision'])
search=ProjectSearch(load_config(root/'feedback-plan-probe-deployment.yaml',[]),output)
observations=original['parameter_search']['observations']
evidence={'stage':'parameter_screening','decision':original['decision'],'observations':[
 {'revision':o['revision'],'parameters':o['bundle']['parameters'],'constraints':o['constraints'],
  'endpoint':endpoint_summary(o['run']),'receipt':str(source/'jobs'/(o['bundle']['id']+'-screen')/'run'/'receipt.json')}
 for o in observations]}
discovery=json.loads((output/'discovery.json').read_text())
plan=search.plan_collectors(discovery,evaluation=evidence,revisions=[search.revisions.revision(o['revision']) for o in observations])
write_json(output/'plan.json',plan);print(json.dumps(plan['plan']),flush=True)
while shutil.disk_usage(root).free<10*1024**3:time.sleep(5)
extra=search.collect_extra(search.revisions.revision(plan['plan']['revision']),plan['plan']['collectors']) if plan['plan']['collectors'] else {}
write_json(output/'result.json',{'plan':plan,'observations':extra,'scope':'Real collection-agent probe on unmodified measured screening failures; no prescribed collector or optimization action'})
assert search.revisions.current().id==parent.id
print(json.dumps({'completed':True,'collectors':{k:v['status'] for k,v in extra.items()}}),flush=True)
