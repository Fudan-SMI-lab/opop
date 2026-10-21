from pathlib import Path
import json
from kernel_optimizer.config import load_config
from kernel_optimizer.project.branches import fork_run
from kernel_optimizer.project.search import ProjectSearch
from kernel_optimizer.runtime.execution import write_json
r=Path('/root/autodl-tmp/dev/v6-work/evidence-followup-20261021');old=Path('/root/autodl-tmp/dev/v6-work/agent-repair-20261021');source=old/'ddp-automatic-continuation';out=r/'framework-traceback-probe'
fork_run(source,out)
s=ProjectSearch(load_config(old/'ddp-automatic-continuation-private.yaml',[]),out)
original=json.loads((old/'ddp-automatic-continuation-result.json').read_text())
run=s.execute(s.revisions.revision(original['decision']['revision']),'original-candidate','candidate_diagnostics')
rows=[{'rank':x['rank'],'events':x.get('result',{}).get('candidate_diagnostics',{}).get('events',[])} for x in run['ranks']]
assert run['status']=='passed'
for row in rows:
 assert any(frame['file'].endswith('transformers/masking_utils.py') and 'torch.tensor(0.0' in frame['source'] for e in row['events'] for frame in e.get('traceback',[])), row['rank']
write_json(out/'result.json',{'scope':'Actual double-GPU new-collector check using unchanged original candidate, fallback and workload; no developer candidate instrumentation.','passed':True,'run':run,'events':rows})
print(json.dumps({'passed':True,'ranks':len(rows)}),flush=True)
