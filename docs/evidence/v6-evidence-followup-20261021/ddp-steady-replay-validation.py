from pathlib import Path
import hashlib,json
from kernel_optimizer.config import load_config
from kernel_optimizer.project.branches import fork_run
from kernel_optimizer.project.search import ProjectSearch
from kernel_optimizer.project.versioning import promote_version
from kernel_optimizer.project.measurement import valid_run
from kernel_optimizer.runtime.execution import write_json
r=Path('/root/autodl-tmp/dev/v6-work/evidence-followup-20261021');source=r/'ddp-feedback-continuation';out=r/'ddp-steady-replay-validation';revision='66d9e2e385cd599c064fb9c3cca933baa7037b087b68bd27742c941b5f00990e'
config=load_config(r/'ddp-feedback-continuation-private.yaml',[]);origin=ProjectSearch(config,source);original_hash=hashlib.sha256((source/'request.json').read_bytes()).hexdigest()
fork_run(source,out,revision_id=origin.prepared['revision']);search=ProjectSearch(config,out)
controls=Path('/dev/shm/opop-ddp-four-step-controls-20261021');controls.mkdir(exist_ok=False)
search.request=search.request.model_copy(update={'continuous_steps':4})
search.prepared={**search.prepared,'controls_path':str(controls)}
protocol={'scope':'Developer supplemental four-update robustness test, not formal candidate acceptance. Only this isolated worker request extends continuous quality checks from 2 to 4; workload per invocation, quality thresholds, adapter, candidate bytes and parameters unchanged. The original frozen request is never changed. Generated baseline reference tensors are in temporary shared memory to avoid filling the data disk.','source':str(source),'revision':revision,'original_request_sha256':original_hash,'extended_worker_request':search.request.portable(),'controls':str(controls)}
write_json(out/'protocol.json',protocol)
control=search.execute(search.revisions.revision(origin.prepared['revision']),'four-step-controls','controls');write_json(out/'controls-result.json',control)
if not valid_run(control):raise RuntimeError('Four-step baseline controls did not pass')
records=[]
for p in controls.glob('reference-rank-*.pt'):
 h=hashlib.sha256()
 with p.open('rb') as stream:
  for b in iter(lambda:stream.read(4*1024*1024),b''):h.update(b)
 records.append({'path':str(p),'bytes':p.stat().st_size,'sha256':h.hexdigest()})
write_json(out/'reference-manifest.json',{'references':records,'scope':'Regenerated unmodified baseline, four continuous updates; temporary shared-memory tensors, no model/reference tensor in public export.'})
actual=search.execute(search.revisions.revision(revision),'four-step-candidate','evaluate');write_json(out/'result.json',{'protocol':protocol,'controls':control,'candidate':actual,'valid':valid_run(actual)})
print(json.dumps({'four_step_valid':valid_run(actual),'checks':[x.get('result',{}).get('checks') for x in actual.get('ranks',[])]}),flush=True)
assert hashlib.sha256((source/'request.json').read_bytes()).hexdigest()==original_hash
# One fixed independent repeat, retaining the original communication goal and two-step contract.
independent=r/'ddp-feedback-independent-fixed';fork_run(source,independent,revision_id=origin.prepared['revision']);check=promote_version(ProjectSearch(config,independent),source,revision_id=revision,publish=False)
write_json(r/'ddp-feedback-independent-fixed-result.json',{'scope':'One independent full-version AB/BA of the same candidate despite its original communication-noise rejection, to check reproducibility and the observed auxiliary step reduction. No new generation, parameter change or publication. Both original and repeat decisions are retained.',**{k:check.get(k) for k in ['trial','decision','measurement','quality','coverage_errors','bundles']}})
write_json(r/'ddp-steady-replay-validation-completed.json',{'completed':True,'four_step_valid':valid_run(actual),'independent_decision':check['decision']});print(json.dumps(check['decision']),flush=True)
