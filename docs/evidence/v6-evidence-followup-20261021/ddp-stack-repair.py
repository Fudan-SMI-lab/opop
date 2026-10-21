from pathlib import Path
import json,hashlib
from kernel_optimizer.config import load_config
from kernel_optimizer.project.branches import fork_run
from kernel_optimizer.project.search import ProjectSearch
from kernel_optimizer.project.versioning import promote_version
from kernel_optimizer.runtime.execution import write_json
r=Path('/root/autodl-tmp/dev/v6-work/evidence-followup-20261021');old=Path('/root/autodl-tmp/dev/v6-work/agent-repair-20261021')
source=old/'ddp-automatic-continuation';out=r/'ddp-stack-repair'
fork_run(source,out)
feedback=json.loads((source/'latest-feedback.json').read_text())
traces=[]
for p in (r/'ddp-capture-trace/jobs/capture-trace/run').glob('rank-*.log'):
 s=p.read_text();start=s.index('DEVELOPER_CAPTURE_STACK')
 traces.append({'source':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'traceback':s[start:]})
feedback['supplemental_evidence']={'scope':'Developer-started diagnostic: only printed the exception traceback, preserving original computation and fallback. No repair prescribed. Both ranks identify the same source operation.', 'capture_exception':traces, 'limitation':'This identifies the first observed capture failure, not successful graph execution, correctness or performance. Existing returned-output checks ran the fallback parent. Steady replay correctness is not established.'}
write_json(out/'experimental-feedback.json',feedback)
search=ProjectSearch(load_config(r/'ddp-stack-repair-private.yaml',[]),out)
result=search.trial(feedback=json.dumps(feedback))
record={'scope':'One developer-assisted autonomous selection/candidate round using original feedback plus actual traceback only.',**{k:result.get(k) for k in ['trial','decision','measurement']}}
write_json(r/'ddp-stack-repair-result.json',record)
if result['decision'].get('accepted'):
 other=r/'ddp-stack-repair-independent'
 fork_run(out,other,revision_id=search.prepared['revision'])
 validated=promote_version(ProjectSearch(search.config,other),out,publish=False)
 write_json(r/'ddp-stack-repair-independent-result.json',{k:validated.get(k) for k in ['trial','decision','measurement','quality','coverage_errors']})
write_json(r/'ddp-stack-repair-completed.json',{'completed':True})
print(json.dumps(record),flush=True)
