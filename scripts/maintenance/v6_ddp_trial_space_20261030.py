"""Exact small cold-evidence list to reserve the ordinary DDP trial budget."""
import argparse,json,os,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from v6_server_storage import compress_group,df,digest,now,save,sha,signature
from v6_cache_constraint_space_20261030 import idle,versions
from v6_communication_space_20261030 import current_versions
BASE=Path('/root/autodl-tmp');W=BASE/'dev/v6-work'
FILES=[
 'validation-runs/readiness-20261020-a/qwen_inference/interactive-trace.json',
 'validation-runs/readiness-20261020-a/qwen_inference/offline-trace.json',
 'autonomous-cohort-20261027/qwen-ttft/discovery.json',
 'qwen-end-to-end-20261028/qwen-end-to-end/discovery.json',
 'qwen-joint-feedback-routed-20261029/qwen-continuation/discovery.json',
 'qwen-joint-feedback-20261029/qwen-continuation/discovery.json',
 'qwen-constraint-continuation-20261029/qwen-continuation/discovery.json',
 'qwen-constraint-continuation-20261029/fixed-native.json',
 'next-stage-20261027/qwen-fixed/native.json',
 'diagnostic-index-independent-20261029/native-qwen-candidate.json',
 'qwen-end-to-end-20261028/native-candidate.json',
 'qwen-end-to-end-20261028/native-baseline.json',
 'qwen-routed-independent-20261029/native-baseline.json']

def read(p):return json.loads(Path(p).read_text())
def main(a):
 os.umask(0o077);idle();oldplan=read(BASE/'dev/storage-audit-20261030/ddp-communication-expanded/plan.json');protect=dict(oldplan['protected'])
 fresh=W/'ddp-communication-device-20261030';assert read(fresh/'followup-completed.json')['phase']=='completed'
 assert sha(fresh/'terminal-audit.json')=='b6f94dbb2651438ccd66253559fa516ac6a79ad38ccc1ca1e92c098d0f8b0553'
 for p in fresh.rglob('*'):
  if p.is_file()and not p.is_symlink():protect[str(p)]=sha(p)
 prior=read(BASE/'dev/storage-audit-20261029/qwen-accepted-continuation/postcheck.json');v=versions(prior);nv=current_versions()
 if not a.apply:
  assert not a.output.exists();a.output.mkdir(parents=True);records=[];retained={};closed={}
  for name in FILES:
   p=W/name;assert p.is_file()and not p.is_symlink()and p.stat().st_nlink==1 and str(p)not in protect
   assert p.resolve().is_relative_to(W.resolve());json.loads(p.read_bytes())
   if name.startswith('validation-runs/'):
    done=W/'validation-runs/readiness-20261020-a/suite-receipt.json';rows=read(done);q=[r for r in rows if r['case']=='qwen_inference'];assert len(q)==1 and q[0]['exit_code']==0 and not q[0]['timed_out']
   else:
    done=W/name.split('/')[0]/'followup-completed.json';d=read(done);assert d.get('phase')in [None,'completed','completed_with_errors']and d.get('results')
   closed[str(done)]=sha(done);records.append({'files':[str(p)],'original':signature(p),'sha256':sha(p),'kind':'Old completed derived discovery/native summary or readiness raw trace. No result decision, source, controls or immutable store modified.'})
  selected={r['files'][0]for r in records}
  for folder in {Path(n).parent for n in selected}:
   for p in folder.iterdir():
    if p.is_file()and not p.is_symlink()and str(p)not in selected:retained[str(p)]=sha(p)
  assert all(sha(p)==h for p,h in protect.items())
  plan={'at':now(),'records':records,'protected':protect,'retained':retained,'closed':closed,'versions':v,'current_versions':nv,'scope':'Only explicit thirteen old derived discovery/native-summary JSON and closed readiness trace payloads, gzip with exact restore. Full decisions/controls/raw native SQLite/source/accepted implementations/immutable store/private history remain online.'}
  save(a.output/'plan.json',plan);print(json.dumps({'files':len(records),'raw_bytes':sum(r['original']['size']for r in records),'protected':len(protect),'retained':len(retained),'plan_sha256':sha(a.output/'plan.json')}));return
 plan=read(a.output/'plan.json');assert plan['protected']==protect and plan['versions']==v and plan['current_versions']==nv
 for k in ['protected','retained','closed']:assert all(sha(p)==h for p,h in plan[k].items())
 run=a.output/'run';run.mkdir(exist_ok=False);before=df(BASE);save(run/'state.json',{'phase':'running','at':now()});rows=[]
 with(run/'cold-json-journal.jsonl').open('x')as journal:
  for r in plan['records']:
   idle();p=Path(r['files'][0]);assert sha(p)==r['sha256']and signature(p)==r['original'];rows.append(compress_group(r,journal))
 for r in rows:assert sha(r['archive'])==r['archive_sha256']and digest(r['archive'],compressed=True)==r['sha256']
 for k in ['protected','retained','closed']:assert all(sha(p)==h for p,h in plan[k].items())
 assert versions(prior)==v and current_versions()==nv;idle();after=df(BASE)
 (run/'RESTORE.md').write_text('# Server-only restoration\n\n```sh\n/root/autodl-tmp/dev-venv/bin/python /root/autodl-tmp/dev/opop-v6/scripts/maintenance/v6_server_storage.py restore-json --journal '+str(run/'cold-json-journal.jsonl')+' --path /absolute/original/path\n```\n\nEach compressed/decompressed hash and metadata verified. Older journals separate. Missing plain files are cold, not lost.\n')
 result={'phase':'completed','finished_at':now(),'files':len(rows),'raw_bytes':sum(x['bytes']for x in rows),'compressed_bytes':sum(x['archive_bytes']for x in rows),'before':before,'after':after,'net_reclaimed_bytes':after['available']-before['available'],'protected':len(protect),'retained':len(plan['retained']),'historical_versions':len(v),'current_versions':nv,'journal_sha256':sha(run/'cold-json-journal.jsonl'),'scope':plan['scope']}
 save(run/'state.json',result);save(run/'followup-completed.json',result);print(json.dumps(result))
if __name__=='__main__':
 p=argparse.ArgumentParser(__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--apply',action='store_true');main(p.parse_args())
