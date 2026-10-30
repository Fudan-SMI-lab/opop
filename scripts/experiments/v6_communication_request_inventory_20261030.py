"""Bounded read-only inventory; no provider, worker, model, collective or restore.

Inspects existing top-level/one-child request manifests only. A negative result is
limited to these locations and is not a claim that no other project exists.
"""
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path('/root/autodl-tmp/dev/v6-work')
OUT = ROOT/'communication-request-inventory-20261030'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    OUT.mkdir(exist_ok=False)
    actual = subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines()
    rows = []; skipped_workers = 0
    paths = sorted(set(ROOT.glob('*/request.json')) | set(ROOT.glob('*/*/request.json')))
    for path in paths:
        d = json.loads(path.read_text())
        if len(d.get('devices',[])) < 2: continue
        if not all(k in d for k in ['source_root','goal','objective','quality','workload']):
            skipped_workers += 1; continue
        entry = Path(d['source_root'])/d['entry']
        prepared = path.with_name('prepared.json')
        status = json.loads(prepared.read_text()).get('status') if prepared.is_file() else None
        rows.append({'path':str(path),'request_sha256':sha(path),'entry':d['entry'],
            'source_entry_exists':entry.is_file(),'source_entry_sha256':sha(entry) if entry.is_file() else None,
            'communication':d.get('communication'),'objective_name':d['objective'].get('name'),
            'devices':d['devices'],'matches_current_uuid_set':set(d['devices']) == set(actual),
            'prepared_status':status,'prepared_sha256':sha(prepared) if prepared.is_file() else None})
    by_entry = defaultdict(list)
    for row in rows: by_entry[row['entry']].append(row)
    public = {'at':datetime.now(timezone.utc).isoformat(),
        'scope':'Existing v6-work/*/request.json and v6-work/*/*/request.json only. Read-only metadata/source-entry hashes; no missing cold-file restore or search outside this scope.',
        'project_requests':len(rows),'excluded_worker_requests':skipped_workers,
        'entries':[{'entry':entry,'requests':len(items),'source_hashes':sorted({r['source_entry_sha256'] for r in items if r['source_entry_sha256']}),
            'current_device_prepared':sum(r['matches_current_uuid_set'] and r['prepared_status']=='prepared' for r in items),
            'communication_modes':sorted({r['communication'] for r in items if r['communication']})} for entry,items in sorted(by_entry.items())],
        'decision':'Existing inspected prepared real project families are DDP/FSDP variants, not a newly supplied communication task. Latest FSDP rejection and Qwen closure remain; repaired DDP77ac is not GPU-eligible under this audit. Close this repair chain. Prefer intake/preflight of a new real distributed request (source, workload, objective and original quality/constraints), without choosing its algorithm or implementation. Alternative existing-DDP direction requires a separately declared evidence decision, not another repair appended here.',
        'pending_input':'User preference for new real task versus separately reconsidered existing DDP was requested asynchronously. No new task/source/workload has been invented; no new experiment is armed.',
        'new_models':0,'new_gpu_workers':0,'new_collectives':0,'new_agent_roles':0,
        'df':subprocess.check_output(['df','-B1','--output=size,used,avail',str(ROOT)],text=True),
        'gpu_workers':subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_gpu_memory','--format=csv,noheader'],text=True)}
    (OUT/'request-inventory.json').write_text(json.dumps(rows,indent=2)+'\n')
    public['inventory_sha256'] = sha(OUT/'request-inventory.json')
    (OUT/'public-inventory.json').write_text(json.dumps(public,indent=2)+'\n')
    print(json.dumps(public)); print('PUBLIC_SHA',sha(OUT/'public-inventory.json'))


if __name__ == '__main__': main()
