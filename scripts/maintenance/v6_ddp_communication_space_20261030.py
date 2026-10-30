"""Bounded server-only cold pass before fresh DDP controls; no deleted evidence."""
import argparse,gzip,hashlib,json,os,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from v6_server_storage import compress_group,df,digest,now,save,sha,signature
from v6_cache_constraint_space_20261030 import idle,versions
from v6_communication_space_20261030 import current_versions
from kernel_optimizer.store.run_store import RunStore
from kernel_optimizer.project.transactions import RevisionStore

BASE=Path('/root/autodl-tmp');W=BASE/'dev/v6-work'
RUN_CASES=['fsdp-continuation-followup-20261028','diagnostic-index-independent-20261029',
 'diagnostic-index-read-fixed-20261029','diagnostic-index-fixed-20261029',
 'fsdp-diagnostic-validation-20261029','qwen-end-to-end-20261028',
 'metric-mapping-decode-20261028','qwen-joint-feedback-routed-20261029',
 'qwen-end-to-end-original-followup-20261029','qwen-constraint-continuation-20261029']
INPUT_CASES=['qwen-device-feedback-20261030','qwen-setup-feedback-20261030',
 'qwen-setup-cpu-feedback-20261030','qwen-cache-interface-candidate-20261030',
 'qwen-cache-constraint-feedback-20261030','qwen-accepted-continuation-20261029']

def read(p):
 p=Path(p)
 return json.loads(p.read_text() if p.exists() else gzip.decompress(p.with_suffix(p.suffix+'.gz').read_bytes()))

def context():
 idle();protected=dict(read(W/'qwen-cache-independent-20261030/protocol.json')['frozen_identities'])
 for case in ['fsdp-communication-device-20261030','fsdp-communication-trial-20261030']:
  root=W/case;assert read(root/'followup-completed.json')['phase']=='completed'
  protected.update(read(root/'protocol.json')['frozen_identities'])
  for p in root.rglob('*'):
   if p.is_file() and not p.is_symlink():protected[str(p)]=sha(p)
 for case in ['qwen-cache-constraint-feedback-20261030','qwen-cache-independent-20261030']:
  for n in ['terminal-audit.json','public-terminal-audit.json','followup-completed.json']:
   p=W/case/n;protected[str(p)]=sha(p)
 ddp=W/'ddp-step-followup-20261028/ddp-step-original';store=RunStore.open(ddp);revs=RevisionStore(store)
 prepared=read(ddp/'prepared.json');base=revs.revision(prepared['revision']);assert not base.bundles
 assert base.id=='7a49c6370b52d881a52550766b279df22684adc16cf3c91be56275f8fe2e1df9'
 for p in [ddp/'prepared.json',ddp/'request.json',revs.pointer]:protected[str(p)]=sha(p)
 for f in base.files:
  assert hashlib.sha256(store.get_artifact(f.content_ref)).hexdigest()==f.sha256
  protected[str(store.artifacts_dir/f.sha256)]=f.sha256
 for p in Path(prepared['controls_path']).parent.rglob('*'):
  if p.is_file() and not p.is_symlink():protected[str(p)]=sha(p)
 assert all(sha(p)==h for p,h in protected.items())
 prior=read(BASE/'dev/storage-audit-20261029/qwen-accepted-continuation/postcheck.json')
 refs=prior['full_fsdp_references_verified'];assert all(sha(r['path'])==r['sha256']for r in refs)
 return protected,prior,versions(prior),current_versions(),refs,revs.current().id

def main(a):
 os.umask(0o077);protect,prior,v,nv,refs,ddp_current=context()
 if not a.apply:
  assert not a.output.exists();a.output.mkdir(parents=True)
  records={};terminals={};retained={};manifests={};folders=set()
  for case in RUN_CASES+INPUT_CASES:
   root=W/case;done=root/'followup-completed.json'
   if case=='qwen-accepted-continuation-20261029':
    # Original controller died at shutdown; recovery completed separately.
    # Its stale state is preserved, never rewritten as original completion.
    assert (root/'power-interruption-20261030.json').exists()
    terminals[str(root/'power-interruption-20261030.json')]=sha(root/'power-interruption-20261030.json')
    done=W/'qwen-accepted-recovery-20261030/followup-completed.json'
   d=read(done)
   assert d.get('phase')in [None,'completed','completed_with_errors'] and d.get('results')
   terminals[str(done)]=sha(done)
   if case in RUN_CASES:
    for receipt in root.glob('**/run/receipt.json*'):
     if 'framework'in receipt.parts or 'sandboxes'in receipt.parts:continue
     if receipt.suffix=='.gz' and receipt.with_suffix('').exists():continue
     assert read(receipt)['worker_pids_reaped'];run=receipt.parent
     for p in run.rglob('*.json'):
      if p.is_symlink() or not p.is_file() or str(p)in protect:continue
      sig=signature(p)
      if sig['size']<256*1024 or sig['nlink']!=1:continue
      assert p.resolve().is_relative_to(run.resolve()) and p.resolve().is_relative_to(root.resolve())
      assert not {'source','reference-source','controls','artifacts','private-history'}&set(p.relative_to(run).parts)
      json.loads(p.read_bytes())
      records[str(p)]={'files':[str(p)],'sha256':sha(p),'original':sig,'kind':'Closed worker RUN JSON/raw trace; whole original evidence retained losslessly.'};folders.add(run)
   if case in INPUT_CASES:
    for p in root.glob('**/generation-snapshots/*.json'):
     if 'framework'in p.parts:continue
     m=read(p)
     if m.get('schema_version')!='project-generation-snapshot.v1':continue
     sandbox=Path(m['sandbox']);store=RunStore.open(Path(m['store']))
     for name,e in m['files'].items():
      f=sandbox/name
      if not name.startswith('inputs/')or f.suffix not in ['.json','.txt']or e['bytes']<1024**2:continue
      if any(x in name for x in ['environment','config'])or str(f)in protect or f.is_symlink()or not f.is_file():continue
      sig=signature(f)
      if sig['nlink']!=1:continue
      assert f.resolve().is_relative_to(root.resolve())
      raw=store.get_artifact(e['ref']);assert len(raw)==e['bytes']==sig['size'] and hashlib.sha256(raw).hexdigest()==e['sha256']==sha(f);json.loads(raw)
      records[str(f)]={'files':[str(f)],'sha256':e['sha256'],'original':sig,'kind':'Redundant closed input materialization; immutable payload remains online.',
       'snapshot':str(p),'store':str(store.run_dir),'artifact_ref':e['ref']};manifests[str(p)]=sha(p);folders.add(sandbox)
  # Old standalone workers predate the jobs/*/run layout. Only their explicit
  # completed trace payloads, not source/reference/results, are eligible.
  for rel in ['goal-switch-audit-20261027/tp2-dependency-retry','goal-switch-20261027/qwen-tp1-readiness']:
   folder=W/rel;receipt=folder/'receipt.json';assert read(receipt)['worker_pids_reaped']
   terminals[str(receipt)]=sha(receipt)
   for p in folder.glob('*trace.json'):
    assert not p.is_symlink()and str(p)not in protect and p.stat().st_nlink==1
    json.loads(p.read_bytes())
    records[str(p)]={'files':[str(p)],'sha256':sha(p),'original':signature(p),'kind':'Explicit old completed standalone worker raw trace; source/result/reference unchanged.'}
   folders.add(folder)
  for folder in folders:
   for p in folder.rglob('*'):
    if p.is_file()and not p.is_symlink()and str(p)not in records:retained[str(p)]=sha(p)
  plan={'at':now(),'records':list(records.values()),'protected':protect,'terminals':terminals,'retained':retained,'manifests':manifests,
   'historical_versions':v,'current_versions':nv,'refs':refs,'ddp_current':ddp_current,
   'scope':'Server-only exact gzip of explicitly closed RUN JSON/traces and redundant input materializations. No current FSDP evidence, DDP preparation dependency, accepted source/bundle/decision, immutable store/snapshot payload, reference, model, provider or private history removed. Restore individual old physical paths from new journal.'}
  assert records and not set(records)&set(protect);save(a.output/'plan.json',plan)
  print(json.dumps({'phase':'planned','files':len(records),'raw_bytes':sum(r['original']['size']for r in records.values()),'protected':len(protect),'retained':len(retained),'manifests':len(manifests),'plan_sha256':sha(a.output/'plan.json'),
   'by_case':{c:sum(r['original']['size']for p,r in records.items()if '/'+c+'/'in p)for c in RUN_CASES+INPUT_CASES}}));return
 plan=read(a.output/'plan.json');assert plan['protected']==protect and plan['historical_versions']==v and plan['current_versions']==nv and plan['ddp_current']==ddp_current
 for k in ['protected','terminals','retained','manifests']:assert all(sha(p)==h for p,h in plan[k].items())
 run=a.output/'run';run.mkdir(exist_ok=False);before=df(BASE);rows=[];save(run/'state.json',{'phase':'running','at':now()})
 with(run/'cold-json-journal.jsonl').open('x')as journal:
  for i,r in enumerate(plan['records'],1):
   idle();p=Path(r['files'][0]);assert signature(p)==r['original'] and sha(p)==r['sha256']
   assert df(BASE)['available']>r['original']['size']+512*1024**2
   rows.append(compress_group(r,journal))
   if i%50==0:print(json.dumps({'compressed':i,'total':len(plan['records']),'at':now()}),flush=True)
 for r,row in zip(plan['records'],rows,strict=True):
  assert sha(row['archive'])==row['archive_sha256']and digest(row['archive'],compressed=True)==r['sha256']and not Path(r['files'][0]).exists()
  if 'artifact_ref'in r:assert hashlib.sha256(RunStore.open(Path(r['store'])).get_artifact(r['artifact_ref'])).hexdigest()==r['sha256']
 for k in ['protected','terminals','retained','manifests']:assert all(sha(p)==h for p,h in plan[k].items())
 assert versions(prior)==v and current_versions()==nv;assert all(sha(r['path'])==r['sha256']for r in refs);idle();after=df(BASE)
 (run/'RESTORE.md').write_text('# Exact server-only restoration\n\nOld physical paths are cold, not lost. Immutable snapshot artifacts stay directly loadable. Restore one path:\n\n```sh\n/root/autodl-tmp/dev-venv/bin/python /root/autodl-tmp/dev/opop-v6/scripts/maintenance/v6_server_storage.py restore-json --journal '+str(run/'cold-json-journal.jsonl')+' --path /absolute/original/path\n```\n\nCompressed/decompressed SHA and metadata are verified. Older journals are separate. No raw download.\n')
 result={'phase':'completed','finished_at':now(),'before':before,'after':after,'files':len(rows),'raw_bytes':sum(x['bytes']for x in rows),'compressed_bytes':sum(x['archive_bytes']for x in rows),'net_reclaimed_bytes':after['available']-before['available'],
  'protected_files':len(protect),'retained_files':len(plan['retained']),'manifests':len(plan['manifests']),'immutable_payloads_unchanged':True,'full_fsdp_refs':len(refs),'historical_versions':len(v),'current_versions':nv,'ddp_current':ddp_current,'journal_sha256':sha(run/'cold-json-journal.jsonl'),'plan_sha256':sha(a.output/'plan.json'),'scope':plan['scope']}
 save(run/'state.json',result);save(run/'followup-completed.json',result);print(json.dumps(result))

if __name__=='__main__':
 p=argparse.ArgumentParser(__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--apply',action='store_true');main(p.parse_args())
