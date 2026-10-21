from pathlib import Path
import json,time,traceback
from kernel_optimizer.config import load_config
from kernel_optimizer.project.branches import fork_run
from kernel_optimizer.project.search import ProjectSearch
from kernel_optimizer.project.versioning import promote_version
from kernel_optimizer.runtime.execution import write_json
r=Path('/root/autodl-tmp/dev/v6-work/evidence-followup-20261021');source=r/'ddp-stack-repair';out=r/'ddp-feedback-continuation'
while not (r/'ddp-stack-repair-completed.json').exists():
 if not Path('/proc/490408').exists():raise RuntimeError('Initial assisted repair terminated without completion; inspect log before continuing')
 time.sleep(5)
feedback=(source/'latest-feedback.json').read_text()
fork_run(source,out)
write_json(out/'continuation-protocol.json',{'scope':'One further normal Agent round in the assisted-repair lineage, using only the prior framework-produced feedback. No manual diagnosis, prescribed fix, candidate edit or new failure-routing policy. Frozen a6f2b6e includes generic traceback metadata. This is not a fresh raw-entry experiment.','feedback_source':str(source/'latest-feedback.json')})
search=ProjectSearch(load_config(r/'ddp-feedback-continuation-private.yaml',[]),out)
result=search.trial(feedback=feedback)
row={k:result.get(k) for k in ['trial','decision','measurement']};write_json(r/'ddp-feedback-continuation-result.json',row)
if result['decision'].get('accepted'):
 independent=r/'ddp-feedback-continuation-independent';fork_run(out,independent,revision_id=search.prepared['revision']);v=promote_version(ProjectSearch(search.config,independent),out,publish=False);write_json(r/'ddp-feedback-continuation-independent-result.json',{k:v.get(k) for k in ['trial','decision','measurement','quality','coverage_errors']})
write_json(r/'ddp-feedback-continuation-completed.json',{'completed':True});print(json.dumps(row),flush=True)
