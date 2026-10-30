"""One bounded old rejected-result/RUN cold pass before the ordinary trial."""
import argparse,gzip,json,os
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from v6_server_storage import compress_group,df,digest,now,save,sha,signature
from v6_cache_constraint_space_20261030 import idle,versions
from v6_communication_space_20261030 import current_versions
BASE=Path('/root/autodl-tmp');W=BASE/'dev/v6-work'
def read(p):return json.loads(Path(p).read_text())
def main(a):
    os.umask(0o077);idle();fresh=W/'fsdp-communication-device-20261030'
    assert read(fresh/'followup-completed.json')['phase']=='completed'
    protect=dict(read(W/'qwen-cache-independent-20261030/protocol.json')['frozen_identities'])
    protect.update(read(fresh/'protocol.json')['frozen_identities'])
    for root in [fresh,W/'qwen-cache-independent-20261030',W/'qwen-cache-constraint-feedback-20261030']:
        for name in ['terminal-audit.json','followup-completed.json']:protect[str(root/name)]=sha(root/name)
    for p in (fresh/'fsdp-current').rglob('*'):
        if p.is_file() and not p.is_symlink():protect[str(p)]=sha(p)
    assert all(sha(p)==h for p,h in protect.items())
    prior=read(BASE/'dev/storage-audit-20261029/qwen-accepted-continuation/postcheck.json');v=versions(prior);nv=current_versions()
    if not a.apply:
        assert not a.output.exists();a.output.mkdir(parents=True);records=[];terminals={};retained={}
        preview=read(W/'communication-priority-20261030/rejected-record-space-preview.json')
        for row in preview:
            p=Path(row['path']);assert p.name=='result.json' and p.parent.parent.name=='trials'
            root=W/p.relative_to(W).parts[0];done=root/'followup-completed.json';assert read(done).get('phase')in [None,'completed','completed_with_errors']
            assert read(p)['decision']['accepted'] is False and str(p)not in protect and not p.is_symlink() and p.stat().st_nlink==1
            records.append({'files':[str(p)],'original':signature(p),'sha256':sha(p),'kind':'closed rejected trial result; no acceptance changed'})
            terminals[str(done)]=sha(done)
        for case in ['qwen-device-feedback-20261030','qwen-setup-feedback-20261030']:
            root=W/case;done=root/'followup-completed.json';assert read(done)['phase']=='completed';terminals[str(done)]=sha(done)
            for p in root.glob('qwen-continuation/jobs/*/run/*.json'):
                if p.is_symlink()or str(p)in protect or p.stat().st_nlink!=1 or p.stat().st_size<256*1024:continue
                receipt=p.parent/'receipt.json';assert read(receipt)['worker_pids_reaped']
                records.append({'files':[str(p)],'original':signature(p),'sha256':sha(p),'kind':'completed old RUN JSON'})
        selected={p for r in records for p in r['files']}
        for folder in {Path(p).parent for p in selected}:
            for p in folder.iterdir():
                if p.is_file() and not p.is_symlink() and str(p)not in selected:retained[str(p)]=sha(p)
        plan={'at':now(),'records':records,'terminals':terminals,'retained':retained,'protected':protect,'historical_versions':v,'new_versions':nv,
              'scope':'Only individually selected old rejected trial result JSON and completed old RUN JSON. Whole content/samples/errors preserved by verified gzip and exact restore journal. No current source/controls/snapshot/provider/accepted version or decision modified.'}
        save(a.output/'plan.json',plan);print(json.dumps({'files':len(records),'raw_bytes':sum(x['original']['size']for x in records),'protected':len(protect),'retained':len(retained),'plan_sha256':sha(a.output/'plan.json')}));return
    plan=read(a.output/'plan.json');assert plan['protected']==protect and plan['historical_versions']==v and plan['new_versions']==nv
    for key in ['protected','retained','terminals']:assert all(sha(p)==h for p,h in plan[key].items())
    run=a.output/'run';run.mkdir(exist_ok=False);before=df(BASE);rows=[];save(run/'state.json',{'phase':'running','at':now()})
    with(run/'cold-json-journal.jsonl').open('x')as journal:
        for row in plan['records']:
            idle();assert sha(row['files'][0])==row['sha256'] and signature(Path(row['files'][0]))==row['original'];rows.append(compress_group(row,journal))
    for row in rows:assert sha(row['archive'])==row['archive_sha256'] and digest(row['archive'],compressed=True)==row['sha256']
    for key in ['protected','retained','terminals']:assert all(sha(p)==h for p,h in plan[key].items())
    assert versions(prior)==v and current_versions()==nv;idle();after=df(BASE)
    (run/'RESTORE.md').write_text('# Server-only exact restoration\n\nUse the original path in cold-json-journal.jsonl:\n\n```sh\n/root/autodl-tmp/dev-venv/bin/python /root/autodl-tmp/dev/opop-v6/scripts/maintenance/v6_server_storage.py restore-json --journal '+str(run/'cold-json-journal.jsonl')+' --path /absolute/original/path\n```\n\nAll original samples/errors/decisions retained losslessly; older journals separate.\n')
    result={'phase':'completed','finished_at':now(),'before':before,'after':after,'payloads':len(rows),'raw_bytes':sum(x['bytes']for x in rows),
        'compressed_bytes':sum(x['archive_bytes']for x in rows),'net_reclaimed_bytes':after['available']-before['available'],
        'protected_files':len(protect),'retained_files':len(plan['retained']),'historical_versions':len(v),'new_versions':nv,
        'journal_sha256':sha(run/'cold-json-journal.jsonl'),'scope':plan['scope']}
    save(run/'state.json',result);save(run/'followup-completed.json',result);print(json.dumps(result))
if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--apply',action='store_true');main(p.parse_args())
