from pathlib import Path
import json,hashlib
from kernel_optimizer.config import load_config
from kernel_optimizer.models.project import CandidateBundle
from kernel_optimizer.project.branches import fork_run
from kernel_optimizer.project.search import ProjectSearch
from kernel_optimizer.runtime.execution import write_json

old=Path('/root/autodl-tmp/dev/v6-work/evidence-followup-20261021')
root=Path('/root/autodl-tmp/dev/v6-work/targeted-isolation-20261021')
source=old/'fresh/qwen-ttft';out=root/'qwen-rounding-intervention'
saved=json.loads((source/'trials/trial-recovered-19ccfe875f/parameter-search.json').read_text())
bundle=CandidateBundle.model_validate_json(json.dumps(saved['observations'][0]['bundle']))
fork_run(source,out,revision_id=bundle.parent_revision)
search=ProjectSearch(load_config(old/'qwen-ttft-fresh-private.yaml',[]),out)
edits=[]
for edit in bundle.edits:
    if edit.path==bundle.installer:
        raw=search.store.get_artifact(edit.content_ref);code=raw.decode()
        old_expression='_fused_rms_norm(hidden_states, normalized_shape, weight, eps)[0]'
        new_expression='(_fused_rms_norm(hidden_states, normalized_shape, None, eps)[0] * weight)'
        assert code.count(old_expression)==2
        code=code.replace(old_expression,new_expression)
        compile(code,'candidate.py','exec')
        write_json(out/'protocol.json',{'scope':'Developer arithmetic intervention only, not Agent-generated repair. Replace weighted fused RMSNorm by unweighted fused BF16 output then BF16 weight multiplication at both gate and steady invocation. All thresholds, guards, selected sites and parent bundles unchanged. No publication.',
           'original_sha256':hashlib.sha256(raw).hexdigest(),'changed_sha256':hashlib.sha256(code.encode()).hexdigest(),
           'old_expression':old_expression,'new_expression':new_expression})
        (out/'instrumented-candidate.py').write_text(code)
        edit=edit.model_copy(update={'content_ref':search.store.put_artifact(code)})
    edits.append(edit)
bundle=bundle.model_copy(update={'id':'qwen-rounding-intervention','edits':tuple(edits)})
revision=search.revisions.stage(bundle)
run=search.execute(revision,'evaluate','evaluate')
write_json(out/'result.json',{'run':run,'revision':revision.id,'scope':'Single diagnostic intervention; even quality passing is not independent performance acceptance.'})
write_json(out/'completed.json',{'completed':True,'worker_status':run['status']})
print(json.dumps({'status':run['status'],'checks':[r.get('result',{}).get('checks') for r in run.get('ranks',[])],
                 'metrics':[r.get('result',{}).get('metrics') for r in run.get('ranks',[])]}),flush=True)
