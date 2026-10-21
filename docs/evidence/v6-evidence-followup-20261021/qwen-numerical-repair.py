from pathlib import Path
import json
from kernel_optimizer.config import load_config
from kernel_optimizer.project.branches import fork_run
from kernel_optimizer.project.search import ProjectSearch
from kernel_optimizer.project.versioning import promote_version
from kernel_optimizer.runtime.execution import write_json
r=Path('/root/autodl-tmp/dev/v6-work/evidence-followup-20261021');old=Path('/root/autodl-tmp/dev/v6-work/agent-repair-20261021');source=old/'fresh-final/qwen-ttft';out=r/'qwen-numerical-repair'
feedback=json.loads((source/'trials/trial-e262267559/result.json').read_text())
fork_run(source,out,revision_id=feedback['bundle']['parent_revision'])
boundaries=json.loads((r/'qwen-boundaries/result.json').read_text())['boundaries']
nofusion=json.loads((r/'qwen-boundaries-no-fusion/result.json').read_text())['boundaries']
shadow0=json.loads((r/'qwen-rotary-isolation/result.json').read_text())['boundaries']
shadow2=json.loads((r/'qwen-third-layer-isolation/result.json').read_text())['boundaries']
independent=next(x for x in json.loads((old/'fresh-final-independent-result.json').read_text())['runs'] if x.get('case')=='qwen-ttft')
feedback['supplemental_evidence']={
 'scope':'Developer isolated CUDA diagnostics on the exact failed candidate, task and worker environment; data below are observed, not prescribed repair actions. Historical accepted flag is retained, but actual candidate path and independent performance did not validate the claimed fusion. This branch begins at the original empty parent, not that accepted fallback bundle.',
 'original_candidate':{'first_boundaries':boundaries[:7],'final_logits':boundaries[-1],'source':str(r/'qwen-boundaries/result.json')},
 'first_layer_shadow_comparisons':{'rows':[x for x in shadow0 if x['name'].startswith('shadow:')],'source':str(r/'qwen-rotary-isolation/result.json')},
 'only_disable_fp_fusion_at_two_norm_rotary_launches':{'first_unequal':[x for x in nofusion if x.get('unequal',0)>0][:3],'final_logits':nofusion[-1],'source':str(r/'qwen-boundaries-no-fusion/result.json'),'limitation':'This change alone still violates the unchanged formal tolerance; returned-output checks pass only through internal parent fallback.'},
 'third_layer_shadow_comparisons':{'rows':[x for x in shadow2 if x['name'].startswith('shadow:')],'source':str(r/'qwen-third-layer-isolation/result.json')},
 'independent_original_version':{k:independent.get(k) for k in ['decision','measurement']},
 'interpretation':'The unmodified first-layer input norm and Q/K/V projections agree exactly; the first observed mismatch is fused Q/K norm plus rotary. Disabling compiler FP fusion fixes that first-layer discrepancy, but at the third layer an isolated norm comparison still differs before downstream propagation. No specific reduction/rsqrt instruction has been proven causal. Original/fused code, source and full observations remain accessible. Internal/full-project tolerance is unchanged; CPU simulation is not CUDA verification.'}
write_json(out/'experimental-feedback.json',feedback)
search=ProjectSearch(load_config(r/'qwen-numerical-repair-private.yaml',[]),out)
result=search.trial(feedback=json.dumps(feedback))
record={'scope':'One developer-assisted selection/candidate round based on measured CUDA facts, preserving the goal and all quality thresholds.',**{k:result.get(k) for k in ['trial','decision','measurement']}}
write_json(r/'qwen-numerical-repair-result.json',record)
if result['decision'].get('accepted'):
 other=r/'qwen-numerical-repair-independent';fork_run(out,other,revision_id=search.prepared['revision'])
 validated=promote_version(ProjectSearch(search.config,other),out,publish=False)
 write_json(r/'qwen-numerical-repair-independent-result.json',{k:validated.get(k) for k in ['trial','decision','measurement','quality','coverage_errors']})
write_json(r/'qwen-numerical-repair-completed.json',{'completed':True})
print(json.dumps(record),flush=True)
