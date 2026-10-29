"""New explicit cold RUN-JSON maintenance; no old controller restart or transfer."""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from v6_server_storage import compress_group, df, digest, now, save, sha, signature
from kernel_optimizer.project.transactions import RevisionStore
from kernel_optimizer.store.run_store import RunStore

BASE = Path('/root/autodl-tmp'); WORK = BASE/'dev/v6-work'
CASES = ['autonomous-cohort-20261027','ddp-step-followup-20261028',
         'fsdp-readiness-followup-20261028','fsdp-continuation-20261028',
         'goal-switch-20261027','metric-mapping-decode-20261028','semantics-lifetime-20261027',
         'qwen-joint-feedback-routed-20261029','qwen-constraint-continuation-20261029',
         'diagnostic-index-independent-20261029','qwen-routed-independent-20261029']


def read(path): return json.loads(Path(path).read_text())


def idle():
    assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    for line in subprocess.check_output(['ps','-eo','pid,args'],text=True).splitlines():
        assert 'kernel_optimizer.runtime.worker' not in line


def versions(prior):
    checked = []
    for row in prior['retained_current_versions']:
        store = RunStore.open(Path(row['path'])); current = RevisionStore(store).current()
        assert current.id == row['revision'] and list(current.bundles) == row['bundles']
        for file in current.files:
            assert hashlib.sha256(store.get_artifact(file.content_ref)).hexdigest() == file.sha256
        checked.append({'revision':current.id,'files':len(current.files),'bundles':list(current.bundles)})
    store = RunStore.open(WORK/'diagnostic-index-independent-20261029/qwen-independent')
    revision = RevisionStore(store).revision(prior['new_qwen_exploration_verified']['revision'])
    for file in revision.files: assert hashlib.sha256(store.get_artifact(file.content_ref)).hexdigest() == file.sha256
    assert RevisionStore(store).current().id == prior['new_qwen_exploration_verified']['source_current']
    return checked


def main(args):
    os.umask(0o077); idle()
    protocol = read(WORK/'qwen-cache-interface-fixed-20261030/protocol.json')
    protected = protocol['frozen_identities']
    assert len(protected) == 500 and all(sha(p)==h for p,h in protected.items())
    prior_path = BASE/'dev/storage-audit-20261029/qwen-accepted-continuation/postcheck.json'
    prior = read(prior_path); checked = versions(prior)
    refs = prior['full_fsdp_references_verified']
    assert all(sha(x['path'])==x['sha256'] for x in refs)
    if not args.apply:
        assert not args.output.exists(); args.output.mkdir(parents=True)
        records=[]; terminals=[]; retained={}
        for case in CASES:
            root=WORK/case; terminal=root/'followup-completed.json'; done=read(terminal)
            assert done.get('phase') in [None,'completed','completed_with_errors'] and done['results']
            assert done.get('finished_at',done.get('at'))
            terminals.append({'path':str(terminal),'sha256':sha(terminal)})
            for folder,dirs,files in os.walk(root):
                dirs[:]=[d for d in dirs if d not in ['framework','sandboxes','artifacts','source','controls','private-history','terminal-private-history','generation-snapshots']]
                run=Path(folder)
                if run.name!='run':continue
                receipt=run/'receipt.json'
                if receipt.exists(): value=read(receipt)
                elif receipt.with_suffix('.json.gz').exists():value=json.loads(gzip.decompress(receipt.with_suffix('.json.gz').read_bytes()))
                else:continue
                if not value.get('worker_pids_reaped'):continue
                for name in files:
                    p=run/name
                    if p.is_symlink() or not p.is_file():continue
                    s=signature(p)
                    eligible=(p.suffix=='.json' and s['size']>=256*1024 and s['nlink']==1 and str(p) not in protected)
                    if eligible:
                        assert p.resolve().is_relative_to(root.resolve())
                        records.append({'files':[str(p)],'original':s,'sha256':sha(p)})
                    else:retained[str(p)]=sha(p)
        assert records and not ({p for row in records for p in row['files']} & set(protected))
        plan={'at':now(),'records':records,'terminal_records':terminals,'retained':retained,
              'protected_current_frozen':protected,'prior_postcheck_sha256':sha(prior_path),
              'scope':'Only explicit completed old RUN regular JSON; symlinks/inode aliases/source/stores/controls/models/provider/current inputs excluded. Verified gzip before removal, exact server restore journal. No current-device RUNs selected.'}
        save(args.output/'plan.json',plan)
        print(json.dumps({'status':'prepared','payloads':len(records),'bytes':sum(x['original']['size']for x in records),
            'retained':len(retained),'frozen':len(protected),'versions':len(checked),'full_fsdp_refs':len(refs),
            'by_case':{c:sum(x['original']['size'] for x in records if ('/'+c+'/') in x['files'][0]) for c in CASES}}));return
    plan=read(args.output/'plan.json');assert plan['protected_current_frozen']==protected
    assert all(sha(x['path'])==x['sha256'] for x in plan['terminal_records'])
    assert all(sha(p)==h for p,h in plan['retained'].items())
    for row in plan['records']:
        assert sha(row['files'][0])==row['sha256'] and signature(Path(row['files'][0]))==row['original']
    run=args.output/'run';run.mkdir(exist_ok=False);before=df(BASE);rows=[]
    save(run/'state.json',{'phase':'running','at':now(),'before':before})
    with (run/'cold-json-journal.jsonl').open('x') as journal:
        for row in plan['records']:
            idle();assert df(BASE)['available']>row['original']['size']+512*1024**2
            rows.append(compress_group(row,journal))
    for row in rows:
        assert sha(row['archive'])==row['archive_sha256']
        assert digest(row['archive'],compressed=True)==row['sha256']
        assert not Path(row['files'][0]).exists()
    assert all(sha(p)==h for p,h in plan['retained'].items())
    assert all(sha(p)==h for p,h in protected.items())
    assert all(sha(x['path'])==x['sha256'] for x in refs)
    assert versions(prior)==checked;idle();after=df(BASE)
    (run/'RESTORE.md').write_text('# Server-only lossless restoration\n\nOriginal content is retained in verified gzip payloads. Use the exact path in cold-json-journal.jsonl:\n\n```sh\n/root/autodl-tmp/dev-venv/bin/python /root/autodl-tmp/dev/opop-v6/scripts/maintenance/v6_server_storage.py restore-json --journal '+str(run/'cold-json-journal.jsonl')+' --path /absolute/original/path\n```\n\nRestore verifies compressed/decompressed hashes and original metadata. Older journals remain separate; no raw downloads required.\n')
    result={'phase':'completed','finished_at':now(),'before':before,'after':after,'payloads':len(rows),
        'raw_bytes':sum(r['bytes']for r in rows),'compressed_bytes':sum(r['archive_bytes']for r in rows),
        'actual_df_reclaimed_bytes':after['available']-before['available'],'retained_files':len(plan['retained']),
        'current_frozen_verified':len(protected),'versions':checked,'full_fsdp_refs':6,
        'old_qwen_exploration_preserved':True,'plan_sha256':sha(args.output/'plan.json'),
        'journal_sha256':sha(run/'cold-json-journal.jsonl'),'scope':plan['scope']}
    save(run/'state.json',result);save(run/'followup-completed.json',result);print(json.dumps(result))


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--apply',action='store_true');main(p.parse_args())
