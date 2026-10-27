"""Explicit maintenance only: stream cold server files, verify locally, then retire.

This is not a framework retention policy. The caller supplies reviewed exact roots
and protected paths. Large archives and private history stay outside the Git repo.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shlex
import shutil
import stat
import subprocess
import sys
import tarfile


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(stream):
    h = hashlib.sha256()
    while data := stream.read(4 * 1024 * 1024):
        h.update(data)
    return h.hexdigest()


def sha(path):
    with Path(path).open('rb') as f:
        return digest(f)


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    with temp.open('w', encoding='utf-8') as f:
        json.dump(value, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)


def within(path, root):
    return path == root or root in path.parents


def checked(base, name):
    part = PurePosixPath(name)
    assert not part.is_absolute() and '..' not in part.parts and part.parts
    path = base.joinpath(*part.parts)
    assert path != base and within(path.parent.resolve(), base)
    return path


def df(base):
    text = subprocess.check_output(['df', '-B1', '--output=size,used,avail', str(base)], text=True)
    size, used, available = map(int, text.splitlines()[-1].split())
    return dict(size=size, used=used, available=available, at=now())


def no_users(base, roots):
    gpu = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True)
    assert not gpu.strip(), 'GPU workers still active'
    users = []
    prefixes = [str(checked(base, r)) for r in roots]
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        try:
            command = (proc/'cmdline').read_bytes().decode(errors='replace')
            if any(p in command for p in prefixes):
                users.append(proc.name)
                continue
            links = [proc/'cwd', proc/'exe', *list((proc/'fd').iterdir())]
            if any(any(str(p.resolve()) == s or str(p.resolve()).startswith(s + '/') for s in prefixes) for p in links):
                users.append(proc.name)
                continue
            maps = (proc/'maps').read_text(errors='replace')
            if any(p in maps for p in prefixes):
                users.append(proc.name)
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
    assert not users, {'active_users': users}


def signature(p):
    s = p.lstat()
    return dict(size=s.st_size, mtime_ns=s.st_mtime_ns, mode=s.st_mode,
                device=s.st_dev, inode=s.st_ino, nlink=s.st_nlink,
                blocks=s.st_blocks * 512, uid=s.st_uid, gid=s.st_gid)


def load_manifest(path):
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        return json.load(f)


def prepare(plan, group, manifest_path):
    base = Path(plan['remote_base']).resolve()
    roots = group['roots']
    protected = [checked(base, p) for p in plan['protected']]
    paths = set()
    for name in roots:
        p = checked(base, name)
        assert p.exists() and not p.is_symlink(), name
        assert not any(within(p, q) or within(q, p) for q in protected), name
        paths.add(p)
        for directory, dirs, files in os.walk(p, followlinks=False):
            for n in dirs + files:
                paths.add(Path(directory)/n)
    no_users(base, roots)
    rows, hashes = [], {}
    for p in sorted(paths, key=lambda p: p.as_posix()):
        name = p.relative_to(base).as_posix()
        checked(base, name)
        s = signature(p)
        row = dict(path=name, **s)
        if stat.S_ISREG(s['mode']):
            key = (s['device'], s['inode'], s['size'], s['mtime_ns'])
            if key not in hashes:
                hashes[key] = sha(p)
            row.update(kind='file', sha256=hashes[key])
        elif stat.S_ISLNK(s['mode']):
            row.update(kind='symlink', target=os.readlink(p))
        elif stat.S_ISDIR(s['mode']):
            row.update(kind='directory')
        else:
            raise ValueError(f'Unsupported special file: {p}')
        assert signature(p) == s, f'Changed during inventory: {p}'
        rows.append(row)
    manifest = dict(at=now(), group=group, plan_sha256=plan['_sha256'],
                    remote_base=str(base), protected=plan['protected'],
                    before=df(base), files=rows)
    with gzip.open(manifest_path, 'wt', encoding='utf-8', compresslevel=1) as f:
        json.dump(manifest, f)
    unique = {(r['device'], r['inode']): r['size'] for r in rows if r['kind'] == 'file'}
    return {'group': group['name'], 'entries': len(rows), 'manifest_sha256': sha(manifest_path),
            'unique_bytes': sum(unique.values())}


def stream(manifest_path):
    m = load_manifest(manifest_path)
    base = Path(m['remote_base'])
    with tarfile.open(fileobj=sys.stdout.buffer, mode='w|gz', compresslevel=1) as archive:
        archive.dereference = False
        for r in m['files']:
            p = checked(base, r['path'])
            assert signature(p) == {k: r[k] for k in signature(p)}, r['path']
            archive.add(p, arcname=r['path'], recursive=False)


def verify(archive_path, manifest_path):
    m = load_manifest(manifest_path)
    expected = {r['path']: r for r in m['files']}
    seen = {}
    with tarfile.open(archive_path, 'r|gz') as archive:
        for member in archive:
            name = member.name.rstrip('/')
            assert name in expected and name not in seen, name
            r = expected[name]
            assert member.mode == stat.S_IMODE(r['mode']), name
            assert member.uid == r['uid'] and member.gid == r['gid'], name
            if member.isfile():
                assert r['kind'] == 'file' and member.size == r['size'], name
                content_sha = digest(archive.extractfile(member))
                assert content_sha == r['sha256'], name
                seen[name] = content_sha
            elif member.islnk():
                assert r['kind'] == 'file' and seen.get(member.linkname) == r['sha256'], name
                seen[name] = r['sha256']
            elif member.issym():
                assert r['kind'] == 'symlink' and member.linkname == r['target'], name
                seen[name] = None
            else:
                assert member.isdir() and r['kind'] == 'directory', name
                seen[name] = None
    assert set(seen) == set(expected), 'Archive membership mismatch'
    return dict(verified_at=now(), archive=str(archive_path), archive_bytes=archive_path.stat().st_size,
                archive_sha256=sha(archive_path), manifest_sha256=sha(manifest_path),
                verified_entries=len(seen), group=m['group']['name'],
                scope='Every member, file SHA256, symlink target, hardlink content and mode/owner checked.')


def retire(manifest_path, receipt_path):
    m = load_manifest(manifest_path)
    receipt = json.loads(Path(receipt_path).read_text())
    assert receipt['manifest_sha256'] == sha(manifest_path)
    assert receipt['verified_entries'] == len(m['files'])
    assert receipt['group'] == m['group']['name']
    base = Path(m['remote_base']).resolve()
    protected = [checked(base, p) for p in m['protected']]
    roots = [checked(base, p) for p in m['group']['roots']]
    no_users(base, m['group']['roots'])
    hashes = {}
    for r in m['files']:
        p = checked(base, r['path'])
        assert any(within(p, root) for root in roots)
        assert not any(within(p, q) or within(q, p) for q in protected)
        s = signature(p)
        assert all(s[k] == r[k] for k in s), f'Changed since archive: {p}'
        if r['kind'] == 'file':
            key = (s['device'], s['inode'])
            if key not in hashes:
                hashes[key] = sha(p)
            assert hashes[key] == r['sha256'], p
        elif r['kind'] == 'symlink':
            assert os.readlink(p) == r['target'], p
    before = df(base)
    # Only exact archived entries are unlinked. Unexpected/new files keep dirs nonempty.
    count = 0
    for r in m['files']:
        if r['kind'] != 'directory':
            p = checked(base, r['path'])
            s = p.lstat()
            assert (s.st_ino, s.st_size, s.st_mtime_ns, s.st_mode) == (r['inode'], r['size'], r['mtime_ns'], r['mode'])
            p.unlink()
            count += 1
    for r in sorted(m['files'], key=lambda r: len(PurePosixPath(r['path']).parts), reverse=True):
        if r['kind'] == 'directory':
            try:
                checked(base, r['path']).rmdir()
            except OSError:
                pass
    result = dict(at=now(), group=m['group']['name'], receipt=receipt, deleted_entries=count,
                  before=before, after=df(base), roots=m['group']['roots'],
                  restore='Verify archive and manifest hashes locally; stream tar.gz over SSH to tar -xzf - -C /root/autodl-tmp. Restore original paths before replay. See local RESTORE.md.')
    for root in roots:
        root.mkdir(parents=True, exist_ok=True)
        save(root/'ARCHIVED.json', result)
    return result


def ssh(plan, arguments, **kwargs):
    remote = [plan['remote_python'], plan['remote_script'], *map(str, arguments)]
    command = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', '-o', 'ServerAliveInterval=20',
               '-o', 'ServerAliveCountMax=3', '-p', str(plan['port']), plan['host'], shlex.join(remote)]
    return subprocess.run(command, **kwargs)


def copy_remote(plan, remote, local, upload=False):
    pair = [str(local), f"{plan['host']}:{remote}"]
    if not upload:
        pair.reverse()
    subprocess.run(['scp', '-q', '-P', str(plan['port']), *pair], check=True)


def execute(plan_path):
    plan = json.loads(plan_path.read_text(encoding='utf-8'))
    dest = Path(plan['local_destination']); dest.mkdir(parents=True, exist_ok=True)
    copy_remote(plan, plan['remote_plan'], plan_path, upload=True)
    state = dict(started_at=now(), pid=os.getpid(), plan_sha256=sha(plan_path), phase='running', groups=[])
    save(dest/'state.json', state)
    try:
        for group in plan['groups']:
            name = group['name']
            local_manifest = dest/(name+'.manifest.json.gz')
            remote_manifest = plan['remote_metadata']+'/'+local_manifest.name
            archive = dest/(name+'.tar.gz')
            receipt_path = dest/(name+'.verified.json')
            if (dest/(name+'.retired.json')).exists():
                raise RuntimeError('Already retired; review/resume explicitly instead of overwriting receipts: '+name)
            state.update(current_group=name, step='manifest'); save(dest/'state.json', state)
            result = ssh(plan, ['prepare', '--plan', plan['remote_plan'], '--group', name,
                                '--manifest', remote_manifest], capture_output=True, check=True)
            summary = json.loads(result.stdout)
            copy_remote(plan, remote_manifest, local_manifest)
            assert sha(local_manifest) == summary['manifest_sha256']
            assert shutil.disk_usage(dest).free > summary['unique_bytes'] + 2*1024**3, 'Insufficient local disk reserve'
            state.update(step='stream', manifest_summary=summary); save(dest/'state.json', state)
            partial = archive.with_suffix(archive.suffix+'.partial')
            with partial.open('wb') as output, (dest/(name+'.stream.log')).open('wb') as error:
                ssh(plan, ['stream', '--manifest', remote_manifest], stdout=output, stderr=error, check=True)
                output.flush(); os.fsync(output.fileno())
            state.update(step='verify'); save(dest/'state.json', state)
            receipt = verify(partial, local_manifest)
            os.replace(partial, archive)
            receipt['archive'] = str(archive)
            save(receipt_path, receipt)
            remote_receipt = plan['remote_metadata']+'/'+receipt_path.name
            copy_remote(plan, remote_receipt, receipt_path, upload=True)
            state.update(step='retire'); save(dest/'state.json', state)
            result = ssh(plan, ['retire', '--manifest', remote_manifest, '--receipt', remote_receipt],
                         capture_output=True, check=True)
            retired = json.loads(result.stdout)
            save(dest/(name+'.retired.json'), retired)
            state['groups'].append(retired); save(dest/'state.json', state)
        state.update(phase='complete', finished_at=now())
    except Exception as exc:
        state.update(phase='error', finished_at=now(), error=repr(exc))
        if isinstance(exc, subprocess.CalledProcessError):
            state['stderr'] = (exc.stderr or b'').decode(errors='replace')[-5000:]
        raise
    finally:
        save(dest/'state.json', state)
        save(dest/'completion.json', state)


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('action', choices=['prepare', 'stream', 'retire', 'execute', 'verify'])
    p.add_argument('--plan', type=Path); p.add_argument('--group')
    p.add_argument('--manifest', type=Path); p.add_argument('--receipt', type=Path)
    p.add_argument('--archive', type=Path)
    a = p.parse_args()
    if a.action == 'execute':
        execute(a.plan); return
    if a.action == 'prepare':
        plan = json.loads(a.plan.read_text()); plan['_sha256'] = sha(a.plan)
        group = next(g for g in plan['groups'] if g['name'] == a.group)
        result = prepare(plan, group, a.manifest)
    elif a.action == 'stream':
        stream(a.manifest); return
    elif a.action == 'retire':
        result = retire(a.manifest, a.receipt)
    else:
        result = verify(a.archive, a.manifest)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
