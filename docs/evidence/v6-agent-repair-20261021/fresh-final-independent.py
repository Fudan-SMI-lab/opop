from pathlib import Path
import json,time,traceback
from kernel_optimizer.config import load_config
from kernel_optimizer.project.branches import fork_run
from kernel_optimizer.project.search import ProjectSearch
from kernel_optimizer.project.versioning import promote_version
from kernel_optimizer.runtime.execution import write_json
root=Path('/root/autodl-tmp/dev/v6-work/agent-repair-20261021')
rows=[];done=set()
config=load_config('/root/opop-v6-stage3/deployment.yaml',[])
while len(done)<3:
 for launch in json.loads((root/'fresh-final-launch.json').read_text()):
  case=launch['case']
  if case in done:continue
  log=Path(launch['log']).read_text(errors='replace')
  completed='"status": "completed_experiment"' in log
  if not completed and Path('/proc/'+str(launch['pid'])).exists():continue
  done.add(case)
  row={'case':case,'source':launch['output'],'scope':'Developer-started independent AB/BA of the complete final frozen version against its own original baseline; no new candidate or publication'}
  try:
   if not completed:raise ValueError('fresh entry did not finish its two autonomous rounds')
   source=Path(launch['output']);origin=ProjectSearch(config,source)
   if not origin.revisions.current().bundles:
    row['status']='no_accepted_candidate'
   else:
    out=root/'fresh-final-independent'/case
    fork_run(source,out,revision_id=origin.prepared['revision'])
    search=ProjectSearch(config,out)
    result=promote_version(search,source,publish=False)
    row.update(status='evaluated',decision=result['decision'],measurement=result['measurement'],quality=result['quality'],coverage_errors=result['coverage_errors'],bundles=result['bundles'],output=str(out),trial=result['trial'])
  except Exception as exc:row.update(error=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc())
  rows.append(row);write_json(root/'fresh-final-independent-result.json',{'runs':rows});print(json.dumps(row),flush=True)
 time.sleep(5)
write_json(root/'fresh-final-independent-completed.json',{'completed':True,'runs':len(rows)})
