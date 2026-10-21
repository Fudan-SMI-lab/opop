from pathlib import Path
import json,hashlib
from kernel_optimizer.config import load_config
from kernel_optimizer.models.project import CandidateBundle
from kernel_optimizer.project.branches import fork_run
from kernel_optimizer.project.search import ProjectSearch
from kernel_optimizer.runtime.execution import write_json
r=Path('/root/autodl-tmp/dev/v6-work/evidence-followup-20261021');old=Path('/root/autodl-tmp/dev/v6-work/agent-repair-20261021');source=old/'fresh-final/qwen-ttft'
d=json.loads((source/'trials/trial-e262267559/parameter-search.json').read_text());bundle=CandidateBundle.model_validate_json(json.dumps(d['observations'][0]['bundle']))
out=r/'qwen-fp-fusion-probe';fork_run(source,out,revision_id=bundle.parent_revision)
search=ProjectSearch(load_config(old/'qwen-ttft-fresh-final-private.yaml',[]),out)
edits=[]
for edit in bundle.edits:
 if edit.path==bundle.installer:
  raw=search.store.get_artifact(edit.content_ref);code=raw.decode();old_text='HEAD_DIM=head_dim,'
  assert code.count(old_text)==2
  code=code.replace(old_text,'HEAD_DIM=head_dim, enable_fp_fusion=False,')
  edit=edit.model_copy(update={'content_ref':search.store.put_artifact(code)})
  (out/'instrumented-candidate.py').write_text(code)
  write_json(out/'protocol.json',{'scope':'Developer causal intervention: only disable compiler FP fusion at the two norm/rotary kernel launches. Same workload, implementation otherwise, internal/formal tolerance and fallback. Not autonomous optimization, not AB/BA performance evidence.', 'original_sha256':hashlib.sha256(raw).hexdigest(),'modified_sha256':hashlib.sha256(code.encode()).hexdigest()})
 edits.append(edit)
bundle=bundle.model_copy(update={'id':'qwen-fp-fusion-intervention','edits':tuple(edits)})
revision=search.revisions.stage(bundle)
run=search.execute(revision,'fp-fusion','evaluate')
write_json(out/'result.json',{'scope':'One intervention endpoint; no publication or paired performance claim','run':run})
print(json.dumps({'status':run['status'],'ranks':[{'checks':x.get('result',{}).get('checks'),'coverage':x.get('result',{}).get('coverage')} for x in run.get('ranks',[])]}),flush=True)
