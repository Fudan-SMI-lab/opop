from pathlib import Path
import hashlib,json,time
from kernel_optimizer.config import load_config
from kernel_optimizer.project.branches import fork_run
from kernel_optimizer.project.search import ProjectSearch
from kernel_optimizer.project.versioning import promote_version
from kernel_optimizer.runtime.execution import write_json
root=Path('/root/autodl-tmp/dev/v6-work/agent-repair-20261021')
while not (root/'fresh-fixed-independent-completed.json').exists():time.sleep(5)
source=root/'fresh-fixed/ddp-communication'
assert '"status": "completed_experiment"' in (root/'ddp-communication-fresh-fixed.log').read_text()
feedback=(source/'latest-feedback.json').read_bytes()
data=json.loads(feedback)
assert 'binding_cost' in data.get('resources',{}).get('extra_collectors',{}).get('after',{})
out=root/'ddp-automatic-continuation'
fork_run(source,out)
(out/'source-feedback.json').write_bytes(feedback)
config=load_config(root/'ddp-automatic-continuation-private.yaml',[])
search=ProjectSearch(config,out)
result=search.trial(feedback=feedback.decode())
record={'scope':'Developer-started one-round isolated continuation of the completed two-round raw-entry DDP run. Uses the exact untouched framework latest-feedback, including diagnostics autonomously chosen and executed after round two. No manual diagnosis, chosen group, candidate or parameter added. This is a continuation, not a new uninterrupted raw-entry run.',
    'source':str(source),'output':str(out),'source_feedback_sha256':hashlib.sha256(feedback).hexdigest(),
    'trial':result['trial'],'decision':result['decision'],'measurement':result.get('measurement')}
write_json(root/'ddp-automatic-continuation-result.json',record)
if result['decision'].get('accepted'):
    independent=root/'ddp-automatic-continuation-independent'
    fork_run(out,independent,revision_id=search.prepared['revision'])
    validation=promote_version(ProjectSearch(config,independent),out,publish=False)
    write_json(root/'ddp-automatic-continuation-independent-result.json',{
        'scope':'Developer-started complete-version independent AB/BA against original; unchanged code/parameters and no publication.',
        'output':str(independent),**{k:validation.get(k) for k in ['trial','decision','measurement','quality','coverage_errors','bundles']}})
else:
    write_json(root/'ddp-automatic-continuation-independent-result.json',{'status':'no_accepted_candidate'})
write_json(root/'ddp-automatic-continuation-completed.json',{'completed':True})
print(json.dumps(record),flush=True)
