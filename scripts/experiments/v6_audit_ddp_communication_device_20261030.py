"""Read-only audit of fresh DDP controls, source and exact-rank negative."""
import argparse,json,statistics,subprocess
from pathlib import Path
from collections import Counter
from datetime import datetime,timezone
from kernel_optimizer.config import load_config
from kernel_optimizer.project.search import ProjectSearch
from kernel_optimizer.project.communication import normalize,trace_window
from kernel_optimizer.runtime.execution import write_json
from kernel_optimizer.runtime.provenance import framework_source
from scripts.experiments.v6_audit_fsdp_20261028 import read,sha

def main(root):
 p=read(root/'protocol.json');done=read(root/'followup-completed.json');assert done['phase']=='completed'
 assert framework_source()==p['core'] and all(sha(n)==h for n,h in p['frozen_identities'].items())
 cfg=load_config(root/'private-config.yaml',[]);s=ProjectSearch(cfg,root/'ddp-current');old=ProjectSearch(cfg,Path(p['parent']))
 assert old.revisions.current().id==p['historical_current'] and s.revisions.current().id==s.prepared['revision']==p['original_empty_baseline']
 assert not s.revisions.current().bundles and s.prepared['reuse']['contract_adaptation'] is None
 before,after=old.request.portable(),s.request.portable();assert {k for k in set(before)|set(after)if before.get(k)!=after.get(k)}=={'devices','diagnostic_metrics'}
 folder=Path(s.prepared['controls_path']);rq=read(folder/'request.json');receipt=read(folder/'receipt.json')
 assert rq['parameters']['request']==after and rq['parameters']['action']=='controls' and not rq['parameters'].get('bundles')
 assert sha(rq['parameters']['adapter'])==p['adapter_sha256'];assert sha(rq['entry'])==receipt['entry_sha256']
 manifest=read(Path(rq['cwd'])/'.opop-revision.json');assert all(sha(Path(rq['cwd'])/n)==h for n,h in manifest['files'].items())
 assert receipt['status']=='passed'and receipt['worker_pids_reaped']and len(receipt['ranks'])==2
 rows=[];traces=[];workers=[]
 for r in receipt['ranks']:
  rank=r['rank'];x=r['result'];st=read(folder/f'rank-{rank}-started.json')
  assert st['parameters_ref']['request_sha256']==sha(folder/'request.json')and r['framework_source']==p['core']and r['binding']['torch_uuid']==p['new_devices'][rank]
  assert all(st['revision']['files'][n]==h for n,h in manifest['files'].items())
  assert r['status']=='passed'and x['valid']and len(x['samples'])==5
  c=x['controls'];assert len(c['reset'])==2 and all(t['passed']and t['max_abs_error']==0 and t['compared_elements']==248879620 for t in c['reset'])
  assert c['identity']['passed']and c['identity']['max_abs_error']==0 and c['identity_covered']and c['wrong_output_rejected']and c['missing_coverage_rejected']
  for n,v in x['metrics'].items():
   vals=x['metric_samples'].get(n)or [z[n]for z in x['samples']];assert abs(statistics.median(vals)-v)<1e-7
  for n,d in x['diagnostic_metrics'].items():
   assert d['status']=='observed'and len(d['values'])==len(d['evidence'])==3
   for path,v in zip(d['evidence'],d['values'],strict=True):
    raw=read(path);windowed,w=trace_window(raw,'opop.communication_metric_workload');assert w
    calc=normalize(windowed,{'kernels':[]},rank);field=d['specification']['source']['field'];assert abs(calc[field]-v)<1e-8
    names=Counter(e['name']for e in windowed['traceEvents']if e.get('cat')=='kernel'and 'nccl'in e.get('name','').lower())
    traces.append({'rank':rank,'path':path,'sha256':sha(path),'window':w,'value':v,'kernel_names':dict(names),'communication_kernel_count':calc['communication_kernel_count'],'logical_input_bytes':calc['logical_input_bytes'],'scope':d['scope']})
  log=(folder/f'rank-{rank}.log').read_text();assert 'Traceback (most recent call last)'not in log
  rows.append({'rank':rank,'samples':x['samples'],'controls':c,'metrics':x['metrics'],'metric_samples':x['metric_samples'],'result_sha256':sha(folder/f'rank-{rank}.json'),'reference_sha256':sha(folder/f'reference-rank-{rank}.pt'),'reference_bytes':(folder/f'reference-rank-{rank}.pt').stat().st_size,'log_sha256':sha(folder/f'rank-{rank}.log')})
 assert len(traces)==6
 aggregate={}
 for n in receipt['ranks'][0]['result']['metrics']:
  vals=[max(t)for t in zip(*(r['result']['metric_samples'].get(n)or [z[n]for z in r['result']['samples']]for r in receipt['ranks']),strict=True)]
  mid=statistics.median(vals);aggregate[n]={'aligned_rank_max':vals,'median':mid,'mad':statistics.median(abs(v-mid)for v in vals)}
 assert aggregate['step_ms']['median']==done['results']['baseline']['constraints'][0]['original_reference']
 assert abs(aggregate['step_ms']['median']*1.05-done['results']['baseline']['constraints'][0]['limit'])<1e-8
 for path,count in [(root/'source-worker',1),(root/'rank-negative-worker',2)]:
  q=read(path/'request.json');rec=read(path/'receipt.json');assert rec['status']=='passed'and rec['worker_pids_reaped']and len(rec['ranks'])==count
  assert q['parameters']['request']==after and Path(q['cwd'])==Path(rq['cwd'])and sha(q['entry'])==rec['entry_sha256']
  for r in rec['ranks']:
   assert r['framework_source']==p['core']and r['binding']['torch_uuid']==p['new_devices'][r['rank']]
   workers.append({'kind':path.name,'rank':r['rank'],'request_sha256':sha(path/'request.json'),'entry_sha256':rec['entry_sha256'],'result_sha256':sha(path/f"rank-{r['rank']}.json"),'log_sha256':sha(path/f"rank-{r['rank']}.log")})
 source=read(root/'source-check.json');x=source['run']['ranks'][0]['result'];assert source['status']=='passed'and x['valid']and len(x['checks'])==6
 for row in x['checks']:
  for n,c in row.items():
   if n in ['gradients','parameters','global_loss']:assert c['passed']and c['compared_elements']==(1 if n=='global_loss' else 124439808)
 assert len(x['exact_replica_gradients'])==2 and all(c['passed']and c['max_abs_error']==0 for c in x['exact_replica_gradients'])
 negative=read(root/'rank-negative-check.json');assert negative['probe_valid']and negative['detected']
 n0,n1=[r['result']for r in negative['run']['ranks']]
 assert n0['actual_abs_change']==0 and n0['exact_replica_equal_after']and not n0['detected']
 assert n1['actual_abs_change']>0 and n1['scalar_within_tolerance']and not n1['exact_replica_equal_after']and n1['detected']and n1['detection']=='observe_assertion'
 assert all(sha(n)==h for n,h in p['frozen_identities'].items())
 report={'at':datetime.now(timezone.utc).isoformat(),'finished_at':done['finished_at'],'core':p['core'],'frozen_files':len(p['frozen_identities']),'all_frozen_unchanged':True,'request_changed_fields':['devices','diagnostic_metrics'],
  'source_current':old.revisions.current().id,'current':s.revisions.current().id,'bundles':[],'adapter_sha256':p['adapter_sha256'],'entry_sha256':receipt['entry_sha256'],'request_sha256':sha(s.output/'request.json'),'source_manifest_sha256':sha(Path(rq['cwd'])/'.opop-revision.json'),
  'ranks':rows,'aggregates':aggregate,'traces':traces,'source_result':x,'source_check_sha256':sha(root/'source-check.json'),'negative_results':[n0,n1],'negative_check_sha256':sha(root/'rank-negative-check.json'),'source_negative_workers':workers,
  'constraints':done['results']['baseline']['constraints'],'counts':{'control_ranks':2,'clean_rank_samples':10,'reset_checks':4,'identity_checks':2,'raw_metric_traces':6,'fresh_serial_source_workers':1,'full_gradient_parameter_rank_step_pairs':4,'source_loss_checks':2,'exact_replica_full_checks':2,'negative_ranks':2},
  'scope':'Fresh empty current-device baseline, not optimization acceptance. Existing source-grounded metric mapping and reviewed serial-reference mapping are explicit assistance. Source values have small within-tolerance differences, not claimed zero. DDP gradient replicas are bitwise equal.',
  'audit_reader_error':'Attempt1 had a Python missing-space SyntaxError in the read-only auditor before any file comparison; retained and corrected reader only, no experiment/source/negative rerun.',
  'limits':['NCCL kernel sum includes waiting; six separate metric profiles are not clean step wall or physical wire/critical-path cost. Kernel names alone do not prove transport.',
   'No new Agent/candidate/search/old Graph installed. All tails retained, no old timing reused or anchor changed.',
   'Source serial full tensors were compared during the worker but not persisted as new raw source files; auditing saved source-result summaries/code is not independently recomputing those source tensor statistics again.',
   'Original control outputs are persisted in full; source/negative runs use separate leases and completion boundaries. The exact-rank mutation is isolated outside timing, not a candidate change.',
   'Baseline worker controls do not expose evaluate-only math_policy_unchanged. Precision policy is checked through unchanged source/adapter/request/worker core, not an invented result field.'],
  'df':subprocess.check_output(['df','-B1','--output=size,used,avail','/root/autodl-tmp'],text=True),'gpu_processes':subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True)}
 write_json(root/'terminal-audit.json',report);public={**report,'terminal_audit_sha256':sha(root/'terminal-audit.json')};write_json(root/'public-terminal-audit.json',public)
 print(json.dumps({'audit':public['terminal_audit_sha256'],'public':sha(root/'public-terminal-audit.json'),'bytes':(root/'public-terminal-audit.json').stat().st_size,'step':aggregate['step_ms'],'communication':aggregate['diagnostic.max_rank_nccl_kernel_sum_ms'],'kernel_names':[t['kernel_names']for t in traces],'counts':report['counts']}))

if __name__=='__main__':
 p=argparse.ArgumentParser(__doc__);p.add_argument('--root',type=Path,required=True);main(p.parse_args().root)
