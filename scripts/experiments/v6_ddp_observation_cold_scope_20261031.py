"""Server-only lossless cold storage of closed FSDP metric traces, not reruns."""
import json
from pathlib import Path
import subprocess
from scripts.experiments.v6_audit_fsdp_20261028 import read, sha
from scripts.maintenance.v6_server_storage import compress_group
from scripts.maintenance.v6_cold_archive import signature, df, save, now

WORK = Path('/root/autodl-tmp/dev/v6-work')
OUT = Path('/root/autodl-tmp/dev/storage-audit-20261031/ddp-observation-fixed')


def main():
    assert not OUT.exists()
    assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    latest = WORK/'ddp-observation-mechanics-repair-20261031'
    assert read(latest/'terminal-audit.json')['all_frozen_unchanged']
    old = WORK/'fsdp-communication-trial-20261030'
    assert read(old/'followup-completed.json')['phase'] == 'completed'
    paths = sorted(old.glob('fsdp-search/jobs/*/run/secondary-metrics/*/metric-rank-*.json'))
    assert len(paths) == 216
    protected = read(latest/'protocol.json')['frozen_identities'].copy()
    for p in latest.rglob('*'):
        if p.is_file() and not p.is_symlink(): protected[str(p)] = sha(p)
    selected = set(paths)
    other = {str(p):sha(p) for p in old.rglob('*') if p.is_file() and not p.is_symlink() and p not in selected}
    assert all(sha(p) == h for p,h in protected.items())
    rows = []
    for p in paths:
        assert p.resolve().is_relative_to(old.resolve()) and not p.is_symlink()
        assert str(p) not in protected and p.stat().st_nlink == 1
        assert not Path(str(p)+'.gz').exists()
        rows.append({'files':[str(p)],'original':signature(p),'sha256':sha(p)})
    OUT.mkdir(parents=True)
    save(OUT/'plan.json',{'at':now(),'scope':'216 raw metric JSON from the closed rejected FSDP trial only; all decisions, sources, controls, accepted versions and immutable snapshots unchanged. No DDP/current observation files compressed. Gzip on server, no raw transfer.',
         'rows':rows,'protected':protected,'other':other})
    run = OUT/'run'; run.mkdir()
    before = df(Path('/root/autodl-tmp'))
    with (run/'cold-json-journal.jsonl').open('x') as stream:
        for row in rows:
            assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
            assert sha(row['files'][0]) == row['sha256']
            result = compress_group(row,stream)
            assert result['sha256'] == row['sha256']
    assert all(sha(p) == h for p,h in protected.items())
    assert all(sha(p) == h for p,h in other.items())
    journal = [json.loads(line) for line in (run/'cold-json-journal.jsonl').read_text().splitlines()]
    after = df(Path('/root/autodl-tmp'))
    result = {'at':now(),'before':before,'after':after,'files':len(rows),
        'raw_bytes':sum(r['original']['size'] for r in rows),
        'gzip_bytes':sum(r['archive_bytes'] for r in journal),'protected_verified':len(protected),
        'other_files_verified':len(other),'journal_sha256':sha(run/'cold-json-journal.jsonl'),
        'plan_sha256':sha(OUT/'plan.json'),'all_compressed_and_decompressed_hashes_verified':True}
    (run/'RESTORE.md').write_text('# Server-only restoration\n\n'
        'Closed FSDP metric JSON are cold, not lost. All original decisions remain unchanged.\n\n'
        '```sh\n/root/autodl-tmp/dev-venv/bin/python /root/autodl-tmp/dev/opop-v6/scripts/maintenance/v6_server_storage.py restore-json '
        '--journal '+str(run/'cold-json-journal.jsonl')+' --path /absolute/original/path\n```\n')
    save(run/'completed.json',result); print(json.dumps(result))


if __name__ == '__main__': main()
