"""Explicit server-only maintenance. No network transfer or model calls.

Keep current workloads online. Deduplicate an unused environment, retire reviewed
Git checkouts, and losslessly compress large cold JSON with an exact restore index.
"""
from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import tarfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from v6_archive_worker_records import archive, digest
from v6_cold_archive import checked, df, no_users, now, save, sha, signature, within


BASE = Path('/root/autodl-tmp')
REPO = BASE/'dev/opop-v6'


def journal_write(stream, row):
    stream.write(json.dumps(row) + '\n')
    stream.flush()
    os.fsync(stream.fileno())


def same_metadata(a, b):
    return all(a[k] == b[k] for k in ['size', 'mode', 'uid', 'gid', 'mtime_ns'])


def link_equal(original, retained, expected_sha):
    old, new = signature(original), signature(retained)
    assert stat.S_ISREG(old['mode']) and stat.S_ISREG(new['mode'])
    assert same_metadata(old, new)
    assert sha(original) == expected_sha == sha(retained)
    temporary = original.with_name(original.name + '.v6-dedup-tmp')
    assert not temporary.exists()
    os.link(retained, temporary)
    assert signature(original) == old
    os.replace(temporary, original)
    assert original.stat().st_ino == retained.stat().st_ino


def environment(output):
    source, retained = BASE/'kernel-opt-venv', BASE/'dev-venv'
    no_users(BASE, ['kernel-opt-venv'])
    before = df(BASE)
    rows, untouched = [], []
    for directory, dirs, files in os.walk(source, followlinks=False):
        for name in files:
            p = Path(directory)/name
            old = signature(p)
            if not stat.S_ISREG(old['mode']):
                continue
            q = retained/p.relative_to(source)
            if not q.is_file() or q.is_symlink():
                untouched.append(str(p)); continue
            new = signature(q)
            if old['inode'] == new['inode'] and old['device'] == new['device']:
                continue
            if not same_metadata(old, new):
                untouched.append(str(p)); continue
            content = sha(p)
            if sha(q) != content:
                untouched.append(str(p)); continue
            rows.append({'path': str(p), 'retained': str(q), 'sha256': content,
                         'original': old, 'retained_before': new})
    with gzip.open(output/'environment-plan.json.gz', 'wt') as f:
        json.dump({'rows': rows, 'untouched': untouched}, f)
    # Only byte/metadata-identical files are shared; different files remain in place.
    with (output/'environment-journal.jsonl').open('x') as journal:
        for row in rows:
            journal_write(journal, row)
            link_equal(Path(row['path']), Path(row['retained']), row['sha256'])
    return {'before': before, 'after': df(BASE), 'linked_files': len(rows),
            'original_blocks': sum(r['original']['blocks'] for r in rows),
            'untouched_files': len(untouched), 'paths_retained': True,
            'note': 'Both environment paths remain usable. Shared files must be copied apart before an intentional in-place edit.'}


def git(root, *args):
    return subprocess.check_output(['git', '-C', str(root), *args])


def freezes(output):
    parent = BASE/'dev/opop-v6-freezes'
    no_users(BASE, ['dev/opop-v6-freezes'])
    rows = []
    for root in sorted(parent.iterdir()):
        assert root.is_dir() and not root.is_symlink() and root.resolve().parent == parent
        head = git(root, 'rev-parse', 'HEAD').decode().strip()
        assert not git(root, 'status', '--porcelain').strip(), root
        refs = git(root, 'show-ref').decode().splitlines()
        for commit in {head, *(line.split()[0] for line in refs)}:
            git(REPO, 'cat-file', '-e', commit)
        ignored = git(root, 'ls-files', '--others', '--ignored', '--exclude-standard', '-z')
        for raw in ignored.split(b'\0'):
            if raw:
                p = Path(os.fsdecode(raw))
                assert p.suffix in ['.pyc', '.pyo'] or '.pytest_cache' in p.parts, (root, p)
        metadata = output/(root.name+'-git-metadata.tar.gz')
        with tarfile.open(metadata, 'w:gz') as archive_file:
            for directory, dirs, files in os.walk(root/'.git'):
                if Path(directory) == root/'.git':
                    dirs[:] = [n for n in dirs if n != 'objects']
                for name in files:
                    p = Path(directory)/name
                    archive_file.add(p, arcname=p.relative_to(root).as_posix(), recursive=False)
        # Git holds the trees/blobs; the small archive preserves refs/config/reflog metadata.
        rows.append({'path': str(root), 'head': head, 'refs': refs,
                     'metadata_archive': str(metadata), 'metadata_sha256': sha(metadata),
                     'restore': 'git clone --shared /root/autodl-tmp/dev/opop-v6 <restored-path>; git -C <restored-path> checkout --detach <head>. Preserve the marker separately before recreating the directory.'})
    save(output/'freeze-plan.json', rows)
    before = df(BASE)
    with (output/'freeze-journal.jsonl').open('x') as journal:
        for row in rows:
            root = Path(row['path'])
            assert root.resolve().parent == parent and not root.is_symlink()
            assert git(root, 'rev-parse', 'HEAD').decode().strip() == row['head']
            assert not git(root, 'status', '--porcelain').strip()
            journal_write(journal, row)
            shutil.rmtree(root)
            root.mkdir()
            save(root/'RESTORE_FROM_GIT.json', row)
    return {'before': before, 'after': df(BASE), 'retired_checkouts': len(rows),
            'source_objects_retained_in': str(REPO)}


def json_plan(settings, output):
    # These distinct environments/tools are outside this authorized, proven set.
    excluded = {'kernel-opt-environment', 'orch-environment', 'old-nodejs'}
    names = [r for g in settings['groups'] if g['name'] not in excluded
             for r in g['roots'] if r != 'dev/opop-v6-freezes']
    roots = [checked(BASE, name) for name in names]
    protected = [checked(BASE, n) for n in settings['protected']]
    for root in roots:
        assert not root.is_symlink()
        assert not any(within(root, p) or within(p, root) for p in protected)
    no_users(BASE, names)
    records = {}
    for root in roots:
        for directory, dirs, files in os.walk(root, followlinks=False):
            for name in files:
                p = Path(directory)/name
                s = signature(p)
                if not stat.S_ISREG(s['mode']):
                    continue
                key = (s['device'], s['inode'])
                r = records.setdefault(key, {'files': [], 'eligible': False, 'original': s})
                r['files'].append(str(p))
                if p.suffix in ['.json', '.jsonl'] and s['size'] >= 256*1024:
                    r['eligible'] = True
    selected, shared = [], []
    for r in records.values():
        if not r.pop('eligible'):
            continue
        if len(r['files']) != r['original']['nlink']:
            shared.append(r); continue
        r['files'].sort(key=lambda p: (Path(p).suffix not in ['.json', '.jsonl'], p))
        selected.append(r)
    selected.sort(key=lambda r: r['original']['size'])
    with gzip.open(output/'cold-json-plan.json.gz', 'wt') as f:
        json.dump({'roots': names, 'records': selected, 'shared_kept': shared}, f)
    return {'records': selected, 'roots': names, 'shared_kept': len(shared)}


def compress_group(row, journal):
    paths = [Path(p) for p in row['files']]
    for p in paths:
        assert signature(p) == row['original'], p
    result = archive(paths[0])  # Existing verified gzip round-trip implementation.
    result.update(files=row['files'], original=row['original'], at=now())
    # All hardlink aliases stay in one record and share one compressed payload.
    assert sha(paths[0]) == result['sha256']
    for p in paths:
        assert signature(p) == row['original'], p
    journal_write(journal, result)
    for p in paths:
        s = p.stat()
        assert s.st_ino == row['original']['inode'] and s.st_mtime_ns == row['original']['mtime_ns']
        p.unlink()
    return result


def compress_json(settings, output):
    plan = json_plan(settings, output)
    before = df(BASE)
    count = paths = raw = compressed = 0
    with (output/'cold-json-journal.jsonl').open('x') as journal:
        for r in plan['records']:
            # One payload at a time; enough room even for an incompressible file.
            assert df(BASE)['available'] > r['original']['size'] + 512*1024**2
            v = compress_group(r, journal)
            count += 1; paths += len(r['files'])
            raw += r['original']['size']; compressed += v['archive_bytes']
            if count % 50 == 0:
                save(output/'json-progress.json', {'at': now(), 'completed': count,
                     'planned': len(plan['records']), 'original_bytes': raw,
                     'compressed_bytes': compressed, 'file_paths': paths})
    for name in plan['roots']:
        save(checked(BASE, name)/'COLD_JSON_ARCHIVE.json', {
            'journal': str(output/'cold-json-journal.jsonl'), 'source_root': str(checked(BASE, name)),
            'note': 'Some large cold JSON and their recorded hardlink aliases are losslessly compressed on this server. Results unchanged; restore a selected original path with v6_server_storage.py restore-json.'})
    return {'before': before, 'after': df(BASE), 'compressed_unique_payloads': count,
            'original_paths': paths, 'original_bytes': raw, 'compressed_bytes': compressed,
            'shared_inodes_left_online': plan['shared_kept'],
            'journal': str(output/'cold-json-journal.jsonl')}


def restore_row(row):
    zipped = Path(row['archive'])
    assert sha(zipped) == row['archive_sha256']
    assert digest(zipped, compressed=True) == row['sha256']
    first = None
    for name in row['files']:
        path = Path(name)
        assert path.is_absolute() and within(path, BASE)
        if path.exists():
            assert sha(path) == row['sha256']
        elif first is None:
            path.parent.mkdir(parents=True, exist_ok=True)
            temp = path.with_name(path.name+'.restore-tmp')
            with gzip.open(zipped, 'rb') as src, temp.open('xb') as dst:
                shutil.copyfileobj(src, dst, 1024*1024)
                dst.flush(); os.fsync(dst.fileno())
            assert sha(temp) == row['sha256']
            os.chmod(temp, stat.S_IMODE(row['original']['mode']))
            os.chown(temp, row['original']['uid'], row['original']['gid'])
            os.utime(temp, ns=(row['original']['mtime_ns'], row['original']['mtime_ns']))
            os.replace(temp, path)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            os.link(first, path)
        if first is None:
            first = path
    return {'restored_paths': row['files'], 'sha256': row['sha256']}


def run(settings, output):
    output.mkdir(parents=True, exist_ok=True)
    assert not (output/'state.json').exists(), 'Existing run; review its journals before resuming.'
    state = {'at': now(), 'pid': os.getpid(), 'phase': 'running', 'before': df(BASE), 'results': {}}
    save(output/'state.json', state)
    try:
        for name, task in [('environment', lambda: environment(output)),
                           ('freezes', lambda: freezes(output)),
                           ('cold_json', lambda: compress_json(settings, output))]:
            state['stage'] = name; save(output/'state.json', state)
            result = task()
            state['results'][name] = result
            save(output/(name+'-result.json'), result)
            save(output/'state.json', state)
        state.update(phase='complete', finished_at=now(), after=df(BASE))
    except Exception as exc:
        state.update(phase='error', error=repr(exc), finished_at=now(), after=df(BASE))
        raise
    finally:
        save(output/'state.json', state)
        save(output/'followup-completed.json', state)


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('action', choices=['run', 'restore-json'])
    p.add_argument('--plan', type=Path)
    p.add_argument('--output', type=Path)
    p.add_argument('--journal', type=Path)
    p.add_argument('--path')
    args = p.parse_args()
    if args.action == 'run':
        run(json.loads(args.plan.read_text()), args.output)
    else:
        assert args.path and args.journal
        for line in args.journal.read_text().splitlines():
            row = json.loads(line)
            if args.path in row['files']:
                print(json.dumps(restore_row(row))); return
        raise ValueError('Path not present in restore journal')


if __name__ == '__main__':
    main()
