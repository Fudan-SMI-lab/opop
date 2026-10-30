"""Fresh DDP placement/diagnostic mapping, complete controls and source audit only."""
import argparse,json,os,subprocess,traceback
from datetime import datetime,timezone
from pathlib import Path
from kernel_optimizer.config import load_config
from kernel_optimizer.project.search import ProjectSearch
from kernel_optimizer.project.request import ProjectRequest
from kernel_optimizer.project.variants import prepare_variant
from kernel_optimizer.project.constraints import constraints_for_runs
from kernel_optimizer.runtime.execution import ExecutionRequest,LocalExecutor,resolve_devices,write_json
from kernel_optimizer.runtime.provenance import framework_source
from scripts.experiments.v6_audit_fsdp_20261028 import read,sha
from scripts.experiments.v6_ddp_semantics_20261027 import disk_observation
from scripts.experiments.v6_post_batch_audit_20261027 import execute_probe
from scripts.experiments.v6_box43_followup_20261027 import ddp_reference

CORE='ed24af7acc8caa15578e0b859c85ef8a901be51354448ed39094606ed9e10a2f'
BASELINE='7a49c6370b52d881a52550766b279df22684adc16cf3c91be56275f8fe2e1df9'
ADAPTER='b276a7ed5524fc0c47d6080a6349a0a65b9e14bda8270cbaf87d6cd81bceb193'
CURRENT='09b717664af4cfdce3978a3001b1e69d4f0d7a9fda43cba20a99e6c08653b772'

def now():return datetime.now(timezone.utc).isoformat()

def preflight(a,config):
 old=ProjectSearch(config,a.parent);base=old.revisions.revision(old.prepared['revision'])
 assert base.id==BASELINE and not base.bundles and old.revisions.current().id==CURRENT
 assert old.prepared['adapter_sha256']==ADAPTER and not old.request.diagnostic_metrics
 devices=resolve_devices(a.devices);assert len(set(devices))==2
 inventory=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,driver_version','--format=csv,noheader'],text=True)
 assert all(x in inventory for x in devices) and all(x not in inventory for x in old.request.devices)
 q=old.request.portable();q['devices']=devices
 q['diagnostic_metrics']={'diagnostic.max_rank_nccl_kernel_sum_ms':{
  'source':{'collector':'torch_communication','field':'communication_kernel_sum_ms'},'unit':'ms',
  'rank_aggregation':'max','aggregation_order':'ranks_then_samples'}}
 request=ProjectRequest.model_validate(q);before=old.request.portable();after=request.portable()
 assert {k for k in set(before)|set(after)if before.get(k)!=after.get(k)}=={'devices','diagnostic_metrics'}
 assert request.warmup==2 and request.repeats==5 and request.continuous_steps==2 and request.workload['world_size']==2
 assert request.objective['rank_aggregation']=='max'and request.objective['aggregation_order']=='ranks_then_samples'
 assert framework_source()['sha256']==CORE
 frozen={str(p):sha(p)for p in [a.parent/'request.json',a.parent/'prepared.json',old.revisions.pointer]}
 for f in base.files:
  p=old.store.artifacts_dir/f.sha256;assert sha(p)==f.sha256;frozen[str(p)]=f.sha256
 for p in Path(old.prepared['controls_path']).parent.rglob('*'):
  if p.is_file()and not p.is_symlink()and '__pycache__'not in p.parts:frozen[str(p)]=sha(p)
 prior=a.parent.parent.parent/'fsdp-communication-trial-20261030'
 for n in ['followup-completed.json','terminal-audit.json','public-terminal-audit.json']:
  p=prior/n;frozen[str(p)]=sha(p)
 assert sha(prior/'terminal-audit.json')=='e4b798cab65826efff5e3a57b2235f93b34a11e9ed56a5193d43b270d6f828c6'
 assert read(prior/'terminal-audit.json')['decision']['accepted'] is False
 size=sum((Path(old.prepared['controls_path'])/f'reference-rank-{r}.pt').stat().st_size for r in range(2))
 return old,request,frozen,inventory,size

def main(a):
 os.umask(0o077);config=load_config(a.config,[]);config.usage.mode='observe_only'
 old,request,frozen,inventory,size=preflight(a,config)
 metadata={'at':now(),'core':framework_source(),'parent':str(a.parent),'original_empty_baseline':BASELINE,
  'historical_current':CURRENT,'adapter_sha256':ADAPTER,'old_devices':old.request.devices,'new_devices':request.devices,
  'request_changed_fields':['devices','diagnostic_metrics'],'inventory':inventory,'frozen_identities':frozen,
  'space_budget':{'fresh_controls_bytes':size,'profiles_metadata_bytes':1024**3,'warning_floor_bytes':8*1024**3},
  'scope':'Fresh current-device original empty DDP baseline only. Existing frozen source/adapter and full training workload/quality/completion retained; old Graph bundle not installed. Explicit source-grounded communication metric mapping through existing collector interface is assistance, not a production change or unaided original entry. 2-rank controls then serial one-GPU original-source global-batch comparison and one isolated 2-rank exact-gradient negative probe, sequential. No optimization Agent/search/acceptance, old timing reuse, reference copy or hand-picked candidate/group/implementation.'}
 if a.preflight:print(json.dumps({'status':'passed',**metadata}));return
 a.output.mkdir(parents=True,exist_ok=True);assert not(a.output/'protocol.json').exists()
 assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
 free=int(subprocess.check_output(['df','-B1','--output=avail','/root/autodl-tmp'],text=True).splitlines()[-1]);assert free>8*1024**3+size+1024**3
 write_json(a.output/'protocol.json',{**metadata,'disk':disk_observation(Path('/root/autodl-tmp'))})
 write_json(a.output/'variant-request.json',request.portable());results={}
 try:
  branch=a.output/'ddp-current';write_json(a.output/'state.json',{'at':now(),'phase':'running','stage':'fresh_controls'})
  prepared=prepare_variant(old.output,request,branch,config=config)
  write_json(a.output/'preparation-reuse.json',{k:prepared[k]for k in ['status','revision','contract_ref','adapter_sha256','controls_path','reuse']})
  results['prepare']={'status':prepared['status'],'controls':prepared['controls_path']}
  assert prepared['status']=='prepared','Preserve failed baseline, no repair or search.'
  assert prepared['adapter_sha256']==ADAPTER and prepared['reuse']['contract_adaptation'] is None
  s=ProjectSearch(config,branch);assert not s.revisions.current().bundles and s.revisions.current().id==prepared['revision']
  source=Path(prepared['controls_path']).parent/'reference-source';assert sha(source/'opop_adapter.py')==ADAPTER
  write_json(a.output/'state.json',{'at':now(),'phase':'running','stage':'fresh_serial_source_validation'})
  source_run=LocalExecutor().run(ExecutionRequest(entry=Path(__file__).resolve(),function='ddp_reference',cwd=source,
   output=a.output/'source-worker',devices=request.devices[:1],lease_dir=request.lease_dir,
   worker_timeout_s=request.worker_timeout_s,wait_for_devices=True,
   parameters={'source':str(source),'controls':prepared['controls_path'],'request':request.portable()}))
  source_result={'status':'passed'if source_run['status']=='passed'and source_run['ranks'][0].get('result',{}).get('valid')else'failed',
   'run':source_run,'adapter_sha256':ADAPTER,
   'scope':'Actual fresh one-GPU original-source serial global batch of 8 sequences, two continuous SGD updates; full gradients/parameters per rank and averaged loss compared to NEW controls, plus exact cross-rank gradient equality. Existing reviewed manual mapping retained. No old score/control reused and no source tensors copied or persisted as new shared references.'}
  write_json(a.output/'source-check.json',source_result);results['source']={'status':source_result['status'],'sha256':sha(a.output/'source-check.json')}
  assert source_result['status']=='passed','Preserve source mismatch; no retries or search.'
  write_json(a.output/'state.json',{'at':now(),'phase':'running','stage':'exact_rank_relation_negative'})
  negative=execute_probe(source,request.portable(),a.output/'rank-negative-worker')
  write_json(a.output/'rank-negative-check.json',negative)
  results['negative']={'status':negative['status'],'detected':negative['detected'],'sha256':sha(a.output/'rank-negative-check.json')}
  assert negative['probe_valid'] and negative['detected'],'Exact rank-replica violation must be detected; preserve failure.'
  results['baseline']={'current':s.revisions.current().id,'bundles':[],
   'constraints':constraints_for_runs([prepared['evaluation']],request.objective,prepared['evaluation']),
   'reference_hashes':{str(Path(prepared['controls_path'])/f'reference-rank-{r}.pt'):sha(Path(prepared['controls_path'])/f'reference-rank-{r}.pt')for r in range(2)}}
 except Exception:results['error']={'status':'error','traceback':traceback.format_exc()}
 same=all(sha(p)==h for p,h in frozen.items())and old.revisions.current().id==CURRENT
 results['preservation']={'status':'passed'if same else'failed','files':len(frozen),'unchanged':same}
 terminal={'phase':'completed'if same and 'error'not in results else'completed_with_errors','finished_at':now(),'results':results,
  'disk':disk_observation(Path('/root/autodl-tmp')),
  'next':'Audit full source/controls/exact-rank negative and six separate communication traces and space. Only if suitable, declare one ordinary real communication-priority DDP trial with default scopes/parameters and no manual candidate; no Qwen/FSDP retry.'}
 write_json(a.output/'followup-results.json',results);write_json(a.output/'state.json',terminal);write_json(a.output/'followup-completed.json',terminal);print(json.dumps(terminal),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser(__doc__)
 for n in ['config','parent','output']:p.add_argument('--'+n,type=Path,required=True)
 p.add_argument('--devices',nargs=2,required=True);p.add_argument('--preflight',action='store_true');main(p.parse_args())
