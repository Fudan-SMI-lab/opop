from pathlib import Path
import json,hashlib
from kernel_optimizer.config import load_config
from kernel_optimizer.models.project import CandidateBundle
from kernel_optimizer.project.branches import fork_run
from kernel_optimizer.project.search import ProjectSearch
from kernel_optimizer.runtime.execution import write_json

old=Path('/root/autodl-tmp/dev/v6-work/evidence-followup-20261021')
root=Path('/root/autodl-tmp/dev/v6-work/targeted-isolation-20261021')
source=old/'fresh/qwen-ttft';out=root/'qwen-norm-shadow'
saved=json.loads((source/'trials/trial-recovered-19ccfe875f/parameter-search.json').read_text())
bundle=CandidateBundle.model_validate_json(json.dumps(saved['observations'][0]['bundle']))
fork_run(source,out,revision_id=bundle.parent_revision)
search=ProjectSearch(load_config(old/'qwen-ttft-fresh-private.yaml',[]),out)
helper='''
def _shadow_stats(actual, expected):
    a=actual.detach().float(); b=expected.detach().float()
    d=(a-b).abs()
    return {"max_abs_error":float(d.max()), "unequal":int((d!=0).sum()),
            "elements":a.numel(),"violations_002":int((d > 0.02+0.02*b.abs()).sum())}

def _shadow_norm(call_id, x, weight, eps, fused, reference):
    import json
    xf=x.float(); inv=torch.rsqrt(xf.pow(2).mean(-1,keepdim=True)+eps)
    native_unweighted=_fused_rms_norm(x,[weight.numel()],None,eps)[0]
    native_fp32=_fused_rms_norm(xf,[weight.numel()],None,eps)[0]
    expressions={
        "fused_weighted":fused,
        "source_expression":weight*(xf*inv).to(x.dtype),
        "weight_before_round":(xf*inv*weight.float()).to(x.dtype),
        "native_unweighted_then_weight":native_unweighted*weight,
        "native_fp32_then_round_weight":native_fp32.to(x.dtype)*weight,
    }
    record={"call":call_id,"shape":list(x.shape),"weight_shape":list(weight.shape),
            "weight_min":float(weight.min()),"weight_max":float(weight.max()),
            "comparisons_to_parent":{name:_shadow_stats(v,reference) for name,v in expressions.items()},
            "fused_vs_weight_before_round":_shadow_stats(fused,expressions["weight_before_round"]),
            "native_fp32_vs_source_unweighted":_shadow_stats(native_fp32,xf*inv)}
    print("RMS_SHADOW "+json.dumps(record),flush=True)

'''
edits=[]
for edit in bundle.edits:
    if edit.path==bundle.installer:
        raw=search.store.get_artifact(edit.content_ref);code=raw.decode()
        needle='            reference = parent(hidden_states)\n'
        assert code.count(needle)==1
        code=code.replace(needle,needle+'            _shadow_norm(call_id, hidden_states, weight, eps, fused, reference)\n')
        code=code.replace('def _no_route(',helper+'def _no_route(',1)
        compile(code,'instrumented-candidate.py','exec')
        write_json(out/'protocol.json',{'scope':'Developer shadow-only arithmetic on each original first live norm gate. Candidate returned output, comparisons, threshold, routes and fallback unchanged. Additional operations perturb execution; no timing claims.',
                  'source':str(source),'trial':'trial-recovered-19ccfe875f','original_sha256':hashlib.sha256(raw).hexdigest(),
                  'instrumented_sha256':hashlib.sha256(code.encode()).hexdigest(),
                  'interpretation':'First local mismatch is measured in execution order on current candidate inputs. Later sites do not share the original full-model inputs; shadow results alone do not isolate global error causality.'})
        (out/'instrumented-candidate.py').write_text(code)
        edit=edit.model_copy(update={'content_ref':search.store.put_artifact(code)})
    edits.append(edit)
bundle=bundle.model_copy(update={'id':'qwen-rms-shadow','edits':tuple(edits)})
revision=search.revisions.stage(bundle)
run=search.execute(revision,'shadow','evaluate')
rows=[]
for p in (out/'jobs/shadow/run').glob('rank-*.log'):
    rows.extend(json.loads(line.removeprefix('RMS_SHADOW ')) for line in p.read_text().splitlines() if line.startswith('RMS_SHADOW '))
write_json(out/'result.json',{'run':run,'rows':rows,'revision':revision.id,'scope':'Intrusive diagnostic; quality comparisons retained, timing not performance evidence.'})
write_json(out/'completed.json',{'completed':True,'sites':len(rows),'worker_status':run['status']})
print(json.dumps({'status':run['status'],'sites':len(rows),'first':rows[:1],
                 'checks':[r.get('result',{}).get('checks') for r in run.get('ranks',[])]}),flush=True)
