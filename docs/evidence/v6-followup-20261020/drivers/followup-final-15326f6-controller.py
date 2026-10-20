from pathlib import Path
import subprocess,sys,json
from kernel_optimizer.config import load_config
from kernel_optimizer.project.search import ProjectSearch
from kernel_optimizer.project.agents import ProjectAgents
from kernel_optimizer.wiring import Runtime
from kernel_optimizer.runtime.execution import write_json
from kernel_optimizer.runtime.provenance import framework_source
root=Path('/root/autodl-tmp/dev/v6-work');out=Path(__file__).parent
cmd=[sys.executable,'-m','pytest','-q','tests/test_v6_timeline.py','tests/test_v6_functional_binding.py','tests/test_v6_project_quality.py','tests/test_v6_replacement_trace.py','tests/test_v6_project_measurement.py','tests/test_v6_candidate_diagnostics.py']
r=subprocess.run(cmd,capture_output=True,text=True);write_json(out/'tests.json',{'command':cmd,'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr,'source':framework_source()});print('tests',r.returncode,flush=True)
if r.returncode:raise SystemExit(r.returncode)
timeline=root/'followup-ddp-timeline-15326f6'
cmd=[sys.executable,'-u','-m','examples.v6_validation.run_timeline_comparison','--config',str(root/'configs/box42-v6.yaml'),'--run',str(root/'p6-ddp-feedback-composition-b'),'--parent','888033e6ea90b3bb72588f4cbf63637a9842b5a7c988ea729b8937dea052d172','--candidate','af2f908004204353e68a5a091f52127963bfbd689573ba7ef91750f67acd6f59','--output',str(timeline)]
with (out/'timeline.log').open('wb') as log:r=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
print('timeline',r.returncode,flush=True)
if r.returncode:raise SystemExit(r.returncode)
s=ProjectSearch(load_config(root/'configs/box42-v6.yaml',[]),root/'p6-ddp-feedback-composition-b')
run=s.execute(s.revisions.current(),out.name+'-gpu-markers','execution_timeline',collector_options={'gradient_readiness':True,'gpu_gradient_markers':True})
write_json(out/'gpu-markers.json',run);print('gpu-markers',run['status'],flush=True)
if run['status']!='passed':raise SystemExit(1)
reports=[json.loads(Path(p).read_text()) for r in run['ranks'] for p in r['result']['reports']]
marker_summary=[{'rank':r['rank'],'clock':r['clock'],'profiled_wall_ms':r['profiled_wall_ms'],'events':len(r['gradient_stream_markers']['events']),'buckets':len(r['gradient_readiness']['buckets']),'scope':r['gradient_stream_markers']['scope'],'workload_window':r['workload_window']} for r in reports]
raw=json.loads((root/'followup-ddp-binding-df33b74/results.json').read_text())
ablation={'protocol':raw['protocol'],'runs':[{'arm':r['arm'],'pair':r['pair'],'comparison':r['comparison'],'ranks':[{'valid':x['valid'],'checks':x['checks'],'metrics':x['metrics']} for x in r['ranks']]} for r in raw['runs']]}
bundle=s.revisions.bundle('trial-bee8d599aa-param-47de78f8')
inputs={'timeline.json':(timeline/'summary.json').read_text(),'binding-ablation.json':json.dumps(ablation),'gpu-markers.json':json.dumps(marker_summary),'request.json':s.request.model_dump_json(indent=2),'timeline-collector.py':Path('src/kernel_optimizer/project/timeline.py').read_text()}
for edit in bundle.edits:
 if edit.path.endswith('.py'):inputs['communication-candidate.py']=s.store.get_artifact(edit.content_ref)
(out/'review-runtime').mkdir(exist_ok=True)
def validate(d):
 j=json.loads((d/'review.json').read_text())
 for key in ('observations','hypotheses','limitations','next_measurements'):
  if key not in j:raise ValueError('missing '+key)
with Runtime(s.config,log_dir=out/'review-runtime') as runtime:
 directory,receipt=ProjectAgents(runtime,s.store).generate('diagnostic-review',inputs,
 'Review the actual communication and binding measurements. This is a developer-launched diagnostic review, not an optimization or policy-routing task. Read the files using tools and distinguish clean whole-step measurements from separate profiled runs. Identify what native traces and bounded CPU clocks establish, and what remains unknown about GPU readiness, consumer dependencies and transfer vs waiting. Parent and candidate include inherited compute replacements; passthrough adds the child binding but skips child computation. Do not call this pure Python overhead without accounting for dispatch-induced native execution changes. Do not prescribe model-name or failure-type routing rules, modify framework, create an optimization candidate, or run GPU work. Write review.json with observations, hypotheses, limitations, next_measurements and evidence references. Do not infer throughput gain from communication kernel sums.',('review.json',),validator=validate)
 write_json(out/'communication-review.json',{'agent':receipt,'review':json.loads((directory/'review.json').read_text())})
print('review completed',receipt['session'],flush=True)
