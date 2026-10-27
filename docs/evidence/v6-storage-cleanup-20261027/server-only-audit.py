from pathlib import Path
from datetime import datetime, timezone
import collections, gzip, hashlib, json, os, stat, subprocess, sys, tarfile, time

B=Path('/root/autodl-tmp'); R=B/'dev/opop-v6'; A=B/'dev/storage-audit-20261027/server-only-run'
sys.path.insert(0,str(R/'scripts/maintenance'))
import v6_server_storage as m
start=time.time()
result={'started_at':m.now(),'scope':'Server-only post-completion verification; no bulk transfer and no new experiment.'}
state=json.loads((A/'state.json').read_text());completed=json.loads((A/'followup-completed.json').read_text())
result['terminal']={'phase':state['phase'],'finished_at':state.get('finished_at'),
                    'state_matches_completion':state==completed,'controller_log_bytes':(A/'controller.log').stat().st_size}
assert state['phase']=='complete' and state==completed

with gzip.open(A/'protected-before.json.gz','rt') as f: before=json.load(f)
missing=[];changed=[]
for row in before:
    p=Path(row['path'])
    try:s=p.lstat()
    except FileNotFoundError:missing.append(row['path']);continue
    actual={'size':s.st_size,'mode':s.st_mode,'mtime_ns':s.st_mtime_ns,'inode':s.st_ino}
    differences={k:{'before':row[k],'after':v} for k,v in actual.items() if row[k]!=v}
    if differences:changed.append({'path':str(p),'differences':differences})
result['protected']={'count':len(before),'missing':missing,'changed':changed,
    'snapshot_scope':'Taken during cold_json stage after environment/freeze stages; not a pre-maintenance content hash census.'}
print(json.dumps({'stage':'protected','files':len(before),'missing':len(missing),'changed':len(changed)}),flush=True)

with gzip.open(A/'environment-plan.json.gz','rt') as f: envplan=json.load(f)
envjournal=[json.loads(x) for x in (A/'environment-journal.jsonl').read_text().splitlines()]
assert envjournal==envplan['rows']
env_errors=[]
for row in envjournal:
    p,q=Path(row['path']),Path(row['retained'])
    old,new=m.signature(p),m.signature(q)
    if (old['device'],old['inode'])!=(new['device'],new['inode']) or not m.same_metadata(old,row['original']) or not m.same_metadata(new,row['retained_before']) or m.sha(p)!=row['sha256']:
        env_errors.append(row['path'])
result['environment']={'journal_rows':len(envjournal),'plan_equal':True,'all_linked_file_sha_verified':True,
    'errors':env_errors,'untouched_paths_present':all(Path(p).exists() for p in envplan['untouched']),
    'untouched_paths':len(envplan['untouched'])}
print(json.dumps({'stage':'environment','rows':len(envjournal),'errors':len(env_errors),'elapsed_s':time.time()-start}),flush=True)

freezes=json.loads((A/'freeze-plan.json').read_text());fj=[json.loads(x) for x in (A/'freeze-journal.jsonl').read_text().splitlines()]
assert freezes==fj
heads=set();ferrors=[];members=0
for row in freezes:
    marker=Path(row['path'])/'RESTORE_FROM_GIT.json'
    if json.loads(marker.read_text())!=row or m.sha(Path(row['metadata_archive']))!=row['metadata_sha256']:
        ferrors.append(row['path'])
    heads.add(row['head']);heads.update(x.split()[0] for x in row['refs'])
    with tarfile.open(row['metadata_archive'],'r:gz') as t:
        for member in t:
            if member.isfile():
                stream=t.extractfile(member)
                while stream.read(1024**2):pass
            members+=1
objects=subprocess.check_output(['git','-C',str(R),'rev-list','--objects','--missing=print',*sorted(heads)],text=True)
missing_objects=[x for x in objects.splitlines() if x.startswith('?')]
result['freezes']={'checkouts':len(freezes),'metadata_tar_members_read':members,'errors':ferrors,
    'unique_ref_targets':len(heads),'reachable_objects_checked':len(objects.splitlines()),'missing_git_objects':missing_objects}
print(json.dumps({'stage':'freezes','checkouts':len(freezes),'errors':len(ferrors),'missing_objects':len(missing_objects)}),flush=True)

with gzip.open(A/'cold-json-plan.json.gz','rt') as f: plan=json.load(f)
journal=[json.loads(x) for x in (A/'cold-json-journal.jsonl').read_text().splitlines()]
expected={tuple(x['files']):x for x in plan['records']}; actual={tuple(x['files']):x for x in journal}
assert len(expected)==len(plan['records']) and len(actual)==len(journal) and set(expected)==set(actual)
errors=[];removed=0;raw=compressed=0
for row in journal:
    original=expected[tuple(row['files'])]
    if row['original']!=original['original']:errors.append({'record':row['path'],'error':'source metadata differs'})
    p=Path(row['archive']);h=hashlib.sha256();n=0
    try:
        if m.sha(p)!=row['archive_sha256'] or p.stat().st_size!=row['archive_bytes']:
            raise ValueError('compressed SHA/size mismatch')
        with gzip.open(p,'rb') as f:
            while data:=f.read(4*1024**2):h.update(data);n+=len(data)
        if h.hexdigest()!=row['sha256'] or n!=row['bytes']:
            raise ValueError('restored SHA/length mismatch')
        for name in row['files']:
            if os.path.lexists(name):raise ValueError('uncompressed original unexpectedly present: '+name)
            removed+=1
    except Exception as exc:errors.append({'archive':str(p),'error':repr(exc)})
    raw+=row['bytes'];compressed+=row['archive_bytes']
shared_errors=[]
for row in plan['shared_kept']:
    for name in row['files']:
        p=Path(name)
        if not p.exists() or any(m.signature(p)[k]!=row['original'][k] for k in ['size','mode','inode','mtime_ns']):
            shared_errors.append(name)
marker_errors=[]
for root in plan['roots']:
    p=B/root/'COLD_JSON_ARCHIVE.json'
    if not p.exists() or json.loads(p.read_text()).get('journal')!=str(A/'cold-json-journal.jsonl'):
        marker_errors.append(str(p))
result['cold_json']={'planned_payloads':len(expected),'journal_payloads':len(journal),'every_compressed_and_restored_sha_verified':not errors,
    'removed_uncompressed_paths':removed,'uncompressed_bytes_verified':raw,'compressed_bytes_verified':compressed,
    'errors':errors,'retained_shared_inodes':len(plan['shared_kept']),'retained_shared_path_errors':shared_errors,
    'restore_index_errors':marker_errors,'journal_sha256':m.sha(A/'cold-json-journal.jsonl')}
print(json.dumps({'stage':'cold_json','payloads':len(journal),'errors':len(errors),'elapsed_s':time.time()-start}),flush=True)

# Restore one actual archived multialias sample into isolated audit paths, not frozen originals.
chosen=min((r for r in journal if len(r['files'])>1),key=lambda r:r['bytes'])
sample=A/'actual-restore-sample';sample.mkdir(exist_ok=False)
copy=dict(chosen,files=[str(sample/('alias-'+str(i)+'.json')) for i in range(len(chosen['files']))])
m.restore_row(copy)
inodes=set()
for name in copy['files']:
    p=Path(name);s=p.stat();inodes.add(s.st_ino)
    assert m.sha(p)==chosen['sha256'] and stat.S_IMODE(s.st_mode)==chosen['mode'] and s.st_mtime_ns==chosen['mtime_ns']
    assert s.st_uid==chosen['original']['uid'] and s.st_gid==chosen['original']['gid']
assert len(inodes)==1
result['real_restore_sample']={'archive':chosen['archive'],'source_original_path':chosen['path'],
    'bytes':chosen['bytes'],'aliases':len(copy['files']),'sha256':chosen['sha256'],
    'sha_mode_owner_mtime_verified':True,'hardlink_identity_restored':True,'isolated_restore_root':str(sample)}
result['df_after_audit']=m.df(B)
result['net_reclaimed_bytes']=result['df_after_audit']['available']-state['before']['available']
result['passed']=not any([missing,changed,env_errors,ferrors,missing_objects,errors,shared_errors,marker_errors]) and result['environment']['untouched_paths_present']
result['finished_at']=m.now();result['elapsed_s']=time.time()-start
m.save(A/'verification-result.json',result)
print(json.dumps({'stage':'complete','passed':result['passed'],'net_reclaimed_bytes':result['net_reclaimed_bytes'],
    'available_bytes':result['df_after_audit']['available'],'elapsed_s':result['elapsed_s']}),flush=True)
raise SystemExit(0 if result['passed'] else 1)
