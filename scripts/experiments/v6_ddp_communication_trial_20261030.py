"""One ordinary DDP trial from the audited current-device empty baseline."""
import argparse,json,os,subprocess,sys,traceback
from pathlib import Path
from datetime import datetime,timezone
from kernel_optimizer.config import load_config
from kernel_optimizer.project.branches import fork_run
from kernel_optimizer.project.search import ProjectSearch
from kernel_optimizer.runtime.execution import resolve_devices,write_json
from kernel_optimizer.runtime.provenance import framework_source
from scripts.experiments.v6_audit_fsdp_20261028 import read,sha
from scripts.experiments.v6_ddp_semantics_20261027 import disk_observation

BASELINE='7a49c6370b52d881a52550766b279df22684adc16cf3c91be56275f8fe2e1df9'
CORE='ed24af7acc8caa15578e0b859c85ef8a901be51354448ed39094606ed9e10a2f'
AUDIT='b6f94dbb2651438ccd66253559fa516ac6a79ad38ccc1ca1e92c098d0f8b0553'
SOURCE='cbf30d10062ab7a4bce98e8003d0829ed91268e2f4ea0c9685e385bad268611c'
NEGATIVE='3f72d15a1a2bf83733ae9a5b39c0e3596b227aae7e0257213a60645be85023d8'

def now():return datetime.now(timezone.utc).isoformat()
def preflight(a,cfg):
 assert sha(a.parent/'terminal-audit.json')==AUDIT and sha(a.parent/'source-check.json')==SOURCE and sha(a.parent/'rank-negative-check.json')==NEGATIVE
 audit=read(a.parent/'terminal-audit.json');assert read(a.parent/'followup-completed.json')['phase']=='completed'and audit['all_frozen_unchanged']
 s=ProjectSearch(cfg,a.parent/'ddp-current');assert s.revisions.current().id==s.prepared['revision']==BASELINE and not s.revisions.current().bundles
 assert not(s.output/'discovery.json').exists()and not(s.output/'latest-feedback.json').exists()
 assert s.prepared['adapter_sha256']=='b276a7ed5524fc0c47d6080a6349a0a65b9e14bda8270cbaf87d6cd81bceb193'
 assert resolve_devices(s.request.devices)==s.request.devices and s.request.devices==['GPU-20e8003c-f531-d50c-084c-a9a97fe94004','GPU-a24af2e6-99bb-9729-5d14-50fa10a2b321']
 assert s.request.continuous_steps==2 and s.request.warmup==2 and s.request.repeats==5 and len(s.request.diagnostic_metrics)==1
 assert s.request.objective['aggregation_order']=='ranks_then_samples'and s.request.objective['rank_aggregation']=='max'
 assert audit['aggregates']['step_ms']['median']==133.379621 and audit['constraints'][0]['limit']==140.04860205 and audit['constraints'][0]['passed']
 assert audit['source_result']['valid']and audit['negative_results'][1]['detected']
 assert framework_source()['sha256']==CORE
 frozen=dict(read(a.parent/'protocol.json')['frozen_identities'])
 for p in a.parent.rglob('*'):
  if p.is_file()and not p.is_symlink()and '__pycache__'not in p.parts and 'framework'not in p.parts:frozen[str(p)]=sha(p)
 assert all(sha(p)==h for p,h in frozen.items())
 assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
 free=int(subprocess.check_output(['df','-B1','--output=avail','/root/autodl-tmp'],text=True).splitlines()[-1]);assert free>=10*1024**3
 return s,frozen,free

def main(a):
 os.umask(0o077);cfg=load_config(a.config,[]);cfg.usage.mode='observe_only';s,frozen,free=preflight(a,cfg)
 info={'core':framework_source(),'source':str(s.output),'parent':BASELINE,'devices':s.request.devices,'frozen_files':len(frozen),'rounds':1,'baseline_step_ms':133.379621,'step_limit_ms':140.04860205,'baseline_audit_sha256':AUDIT,'source_check_sha256':SOURCE,'negative_check_sha256':NEGATIVE,'free_bytes':free,'planned_growth_bytes':2*1024**3,'warning_free_bytes':8*1024**3}
 if a.preflight:print(json.dumps({'status':'passed',**info}));return
 a.output.mkdir(parents=True,exist_ok=True);assert not(a.output/'protocol.json').exists()
 write_json(a.output/'protocol.json',{'at':now(),**info,'frozen_identities':frozen,
  'scope':'One ordinary existing-CLI resume rounds1/default scopes/default parameter exploration from the fresh current-device source-validated empty DDP baseline. Fresh baseline discovery, no historical Graph bundle or old discovery installed. No developer selection/group/bucket/algorithm/candidate/feedback table or implementation supplied. No source/control rerun or reference copy. Existing source-grounded diagnostic metric and reviewed serial-reference mapping are assistance; not unaided original-entry. Core/prompt/sampler/acceptance unchanged; API observe-only. No independent/native stage scheduled by this controller; actual planner can use existing collectors. Failure retained, no retry until pass or further Qwen/FSDP chain.',
  'disk':disk_observation(Path('/root/autodl-tmp'))})
 results={}
 try:
  branch=a.output/'ddp-search';parent=fork_run(s.output,branch);search=ProjectSearch(cfg,branch)
  assert parent.id==BASELINE and not parent.bundles and search.request==s.request and search.prepared==s.prepared
  assert not(branch/'discovery.json').exists()and not(branch/'latest-feedback.json').exists()
  write_json(a.output/'preparation-reuse.json',{'source':str(s.output),'parent':BASELINE,'request_sha256':sha(branch/'request.json'),'prepared_sha256':sha(branch/'prepared.json'),'controls_path':search.prepared['controls_path'],'source_check_sha256':SOURCE,'negative_check_sha256':NEGATIVE,'initial_bundles':[],'fresh_baseline_discovery_required':True,'scope':'Exact new controls/contract/source shared read-only; no old timing/Graph, no new source comparison in this trial. Generic request unmodified.'})
  argv=[sys.executable,'-B','-m','kernel_optimizer.cli','--config',str(a.config),'optimize-project','--request',str(branch/'request.json'),'--output',str(branch),'--resume','--rounds','1']
  write_json(a.output/'search-protocol.json',{'at':now(),'argv':argv,'scope':'Ordinary discovery/planner/selector/candidate/default parameter exploration and original acceptance rules. No manual optimizer answer or feedback injection.'})
  write_json(a.output/'state.json',{'at':now(),'phase':'running','stage':'fresh-discovery-and-one-ordinary-trial'})
  with(a.output/'search-controller.log').open('wb')as log:r=subprocess.run(argv,stdout=log,stderr=subprocess.STDOUT)
  current=ProjectSearch(cfg,branch).revisions.current();trials=[{'path':str(p),'sha256':sha(p),'decision':read(p).get('decision')}for p in sorted((branch/'trials').glob('*/result.json'))]
  results['search']={'status':'completed'if r.returncode==0 else'error','returncode':r.returncode,'current':current.id,'bundles':list(current.bundles),'trials':trials,'independent_acceptance':'not_run_in_this_batch'};assert len(trials)<=1
 except Exception:results['controller']={'status':'error','traceback':traceback.format_exc()}
 same=all(sha(p)==h for p,h in frozen.items())and s.revisions.current().id==BASELINE
 results['preservation']={'status':'passed'if same else'failed','files':len(frozen),'unchanged':same}
 terminal={'phase':'completed'if same and results.get('search',{}).get('status')=='completed'and 'controller'not in results else'completed_with_errors','finished_at':now(),'results':results,'disk':disk_observation(Path('/root/autodl-tmp')),'next':'Audit complete actual role inputs/tools/probes/parameters/quality/constraints/noise/native communication. Only genuinely new accepted version can justify declared independent whole-version B/C/C/B and necessary native evidence. No retry until pass or automatic Qwen/FSDP extension.'}
 write_json(a.output/'followup-results.json',results);write_json(a.output/'state.json',terminal);write_json(a.output/'followup-completed.json',terminal)

if __name__=='__main__':
 p=argparse.ArgumentParser(__doc__)
 for n in ['config','parent','output']:p.add_argument('--'+n,type=Path,required=True)
 p.add_argument('--preflight',action='store_true');main(p.parse_args())
