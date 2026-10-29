"""Cold redundant input materializations; immutable snapshot payloads stay live."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from v6_server_storage import compress_group,df,digest,now,save,sha,signature
from v6_cache_constraint_space_20261030 import versions,idle
from kernel_optimizer.project.transactions import RevisionStore
from kernel_optimizer.store.run_store import RunStore

BASE=Path('/root/autodl-tmp');WORK=BASE/'dev/v6-work'
CASES=['autonomous-cohort-20261027','qwen-end-to-end-20261028','metric-mapping-decode-20261028',
       'qwen-end-to-end-original-followup-20261029','ddp-step-goal-20261028','qwen-relation-feedback-replay-20261029',
       'fsdp-continuation-followup-20261028','qwen-joint-feedback-routed-20261029','qwen-joint-feedback-20261029',
       'qwen-constraint-continuation-20261029','diagnostic-index-read-20261029','diagnostic-index-candidate-20261029',
       'ddp-step-followup-20261028','fsdp-search-20261028','qwen-attention-evidence-20261029',
       'accepted-feedback-replay-20261028','feedback-execution-validation-20261028','ddp-semantics-20261027']


def read(p):return json.loads(Path(p).read_text())


def current_versions():
    rows=[]
    for root in [WORK/'qwen-cache-constraint-feedback-20261030/qwen-continuation',WORK/'qwen-cache-independent-20261030/qwen-independent']:
        store=RunStore.open(root);rev=RevisionStore(store);current=rev.current()
        for version in [current,*([rev.revision('52668e91d57b29cc1c1b8738e7e135ae278c184564e039b6d84db7eb6a00419e')] if 'independent' in root.name else [])]:
            for f in version.files:assert hashlib.sha256(store.get_artifact(f.content_ref)).hexdigest()==f.sha256
            for b in version.bundles:
                bundle=rev.bundle(b)
                # RevisionStore validation resolves bundle bytes without changing pointers.
                assert bundle.id==b
        rows.append({'path':str(root),'current':current.id,'bundles':list(current.bundles)})
    return rows


def main(args):
    os.umask(0o077);idle()
    active=WORK/'qwen-cache-independent-20261030';assert read(active/'followup-completed.json')['phase']=='completed'
    protected=dict(read(active/'protocol.json')['frozen_identities'])
    for root in [active,WORK/'qwen-cache-constraint-feedback-20261030']:
        for n in ['terminal-audit.json','public-terminal-audit.json','followup-completed.json']:protected[str(root/n)]=sha(root/n)
    assert all(sha(p)==h for p,h in protected.items())
    old=read(BASE/'dev/storage-audit-20261029/qwen-accepted-continuation/postcheck.json')
    prior_versions=versions(old);new_versions=current_versions();refs=old['full_fsdp_references_verified']
    assert all(sha(x['path'])==x['sha256'] for x in refs)
    if not args.apply:
        assert not args.output.exists();args.output.mkdir(parents=True)
        records={};immutable={};terminals={};retained={};sandboxes=set()
        for case in CASES:
            root=WORK/case;terminal=root/'followup-completed.json';done=read(terminal)
            assert done.get('phase') in [None,'completed','completed_with_errors'] and done.get('results')
            terminals[str(terminal)]=sha(terminal)
            for path in root.glob('**/generation-snapshots/*.json'):
                if 'framework' in path.parts:continue
                m=read(path)
                if m.get('schema_version')!='project-generation-snapshot.v1':continue
                sandbox=Path(m['sandbox']);store=RunStore.open(Path(m['store']))
                for name,e in m['files'].items():
                    p=sandbox/name
                    if not name.startswith('inputs/') or p.suffix not in ['.json','.txt'] or e['bytes']<1024**2:continue
                    if any(x in name for x in ['environment','config']) or str(p) in protected or p.is_symlink() or not p.exists():continue
                    sig=signature(p)
                    if sig['nlink']!=1:continue
                    assert p.resolve().is_relative_to((sandbox/'inputs').resolve()) and p.resolve().is_relative_to(root.resolve())
                    raw=store.get_artifact(e['ref']);assert hashlib.sha256(raw).hexdigest()==e['sha256']==sha(p)
                    assert len(raw)==sig['size']==e['bytes'];json.loads(raw)
                    if str(p) in records:assert records[str(p)]['sha256']==e['sha256'];continue
                    records[str(p)]={'files':[str(p)],'original':sig,'sha256':e['sha256'],'snapshot':str(path),
                        'store':str(store.run_dir),'artifact_ref':e['ref'],'scope':'Redundant closed sandbox input; immutable snapshot artifact remains uncompressed and directly loadable.'}
                    immutable[str(path)]=sha(path);sandboxes.add(sandbox)
        for sandbox in sandboxes:
            for p in sandbox.rglob('*'):
                if p.is_file() and not p.is_symlink() and str(p) not in records:retained[str(p)]=sha(p)
        assert records and not(set(records)&set(protected))
        plan={'at':now(),'records':list(records.values()),'immutable_snapshot_manifests':immutable,'retained_other_files':retained,
            'terminals':terminals,'protected':protected,'prior_versions':prior_versions,'new_versions':new_versions,
            'full_fsdp_refs':refs,'scope':'Server-only lossless cold compression of closed redundant JSON/text-JSON input materializations, verified against immutable snapshot artifacts before/after. No snapshot/store payload, provider config, model, shared reference, source, current frozen input, accepted implementation or pointer is changed. Restore physical input paths from journal when needed.'}
        save(args.output/'plan.json',plan)
        print(json.dumps({'status':'prepared','files':len(records),'raw_bytes':sum(r['original']['size'] for r in records.values()),
            'snapshots_kept':len(immutable),'other_files_kept':len(retained),'protected':len(protected),
            'df':df(BASE),'plan_sha256':sha(args.output/'plan.json')}));return
    plan=read(args.output/'plan.json');assert plan['protected']==protected
    for inventory in ['immutable_snapshot_manifests','retained_other_files','terminals','protected']:
        assert all(sha(p)==h for p,h in plan[inventory].items())
    assert new_versions==plan['new_versions'] and prior_versions==plan['prior_versions']
    run=args.output/'run';run.mkdir(exist_ok=False);before=df(BASE);rows=[]
    save(run/'state.json',{'phase':'running','at':now(),'before':before})
    with(run/'cold-json-journal.jsonl').open('x')as journal:
        for i,r in enumerate(plan['records'],1):
            idle();p=Path(r['files'][0]);assert signature(p)==r['original'] and sha(p)==r['sha256']
            assert df(BASE)['available']>r['original']['size']+512*1024**2
            rows.append(compress_group(r,journal))
            if i%50==0:print(json.dumps({'compressed':i,'total':len(plan['records']),'at':now()}),flush=True)
    for r,row in zip(plan['records'],rows,strict=True):
        assert sha(row['archive'])==row['archive_sha256'] and digest(row['archive'],compressed=True)==r['sha256']
        assert not Path(r['files'][0]).exists()
        assert hashlib.sha256(RunStore.open(Path(r['store'])).get_artifact(r['artifact_ref'])).hexdigest()==r['sha256']
    for inventory in ['immutable_snapshot_manifests','retained_other_files','terminals','protected']:
        assert all(sha(p)==h for p,h in plan[inventory].items())
    assert versions(old)==prior_versions and current_versions()==new_versions
    assert all(sha(x['path'])==x['sha256'] for x in refs);idle();after=df(BASE)
    (run/'RESTORE.md').write_text('# Server-only restoration\n\nImmutable generation snapshot artifacts remain available through generation_snapshot.load. Physical old sandbox input paths are cold, not lost. To restore one path, use:\n\n```sh\n/root/autodl-tmp/dev-venv/bin/python /root/autodl-tmp/dev/opop-v6/scripts/maintenance/v6_server_storage.py restore-json --journal '+str(run/'cold-json-journal.jsonl')+' --path /absolute/original/path\n```\n\nCompressed/decompressed SHA and original metadata are checked. No raw download. Older journals remain separate.\n')
    result={'phase':'completed','finished_at':now(),'before':before,'after':after,'payloads':len(rows),
        'raw_bytes':sum(r['bytes'] for r in rows),'compressed_bytes':sum(r['archive_bytes'] for r in rows),
        'actual_df_reclaimed_bytes':after['available']-before['available'],'snapshots_kept':len(plan['immutable_snapshot_manifests']),
        'retained_files':len(plan['retained_other_files']),'protected_files':len(protected),'historical_versions':prior_versions,
        'current_versions':new_versions,'fsdp_full_refs_checked':len(refs),'immutable_payloads_byteequal_after':True,
        'plan_sha256':sha(args.output/'plan.json'),'journal_sha256':sha(run/'cold-json-journal.jsonl'),'scope':plan['scope']}
    save(run/'state.json',result);save(run/'followup-completed.json',result);print(json.dumps(result))


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--apply',action='store_true');main(p.parse_args())
