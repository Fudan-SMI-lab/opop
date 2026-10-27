"""Wait without model calls, then queue one continuation in an existing Codex thread.

This is experiment orchestration, not part of the optimization framework. It
uses existing SSH authentication and the local Codex daemon's current session.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import subprocess
import time


def now():
    return datetime.now(timezone.utc).isoformat()


def save(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(temporary, path)


def probe_script(root):
    return '''import json, os
from pathlib import Path
root = Path(ROOT_LITERAL)
def alive(pid):
    try:
        os.kill(pid, 0)
        stat = Path('/proc') / str(pid) / 'stat'
        return not stat.exists() or stat.read_text().rsplit(')', 1)[1].split()[0] != 'Z'
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
def probe():
    final = root / 'followup-completed.json'
    if final.exists():
        try:
            value = json.loads(final.read_text())
        except json.JSONDecodeError:
            return {'terminal': False, 'reason': 'completion_record_being_written'}
        return {'terminal': True, 'reason': 'batch_finished', 'record': str(final),
                'completed_at': value.get('at'), 'result_keys': list(value.get('results', {}))}
    controllers = []
    for launch, completion in [('launch.json', 'controller-completed.json'),
                               ('followup-launch.json', 'followup-completed.json')]:
        path = root / launch
        if not path.exists():
            return {'terminal': False, 'reason': 'launch_record_missing', 'record': str(path)}
        pid = json.loads(path.read_text())['pid']
        running = alive(pid)
        controllers.append({'launch': launch, 'pid': pid, 'running': running})
        if not running and not (root / completion).exists():
            return {'terminal': True, 'reason': 'controller_exited_without_completion',
                    'controllers': controllers, 'missing_record': str(root / completion)}
    return {'terminal': False, 'reason': 'waiting', 'controllers': controllers}
print(json.dumps(probe()))
'''.replace('ROOT_LITERAL', repr(root))


def probe(args):
    if getattr(args, 'local_root', None):
        final = args.local_root / 'completion.json'
        if final.exists():
            value = json.loads(final.read_text(encoding='utf-8'))
            return {'terminal': True, 'reason': 'maintenance_' + value['phase'],
                    'record': str(final), 'completed_at': value.get('finished_at'),
                    'completed_groups': [g['group'] for g in value.get('groups', [])]}
        launch = args.local_root / 'launch.json'
        if not launch.exists():
            return {'terminal': False, 'reason': 'launch_record_missing'}
        pid = json.loads(launch.read_text(encoding='utf-8'))['pid']
        if os.name == 'nt':
            import ctypes
            from ctypes import wintypes
            kernel = ctypes.WinDLL('kernel32', use_last_error=True)
            kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
            kernel.OpenProcess.restype = wintypes.HANDLE
            kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
            kernel.WaitForSingleObject.restype = wintypes.DWORD
            kernel.CloseHandle.argtypes = [wintypes.HANDLE]
            handle = kernel.OpenProcess(0x00100000, False, pid)
            if not handle and ctypes.get_last_error() == 5:
                raise PermissionError('Cannot inspect maintenance controller')
            wait_result = kernel.WaitForSingleObject(handle, 0) if handle else 0
            if handle:
                kernel.CloseHandle(handle)
            if wait_result == 0xFFFFFFFF:
                raise OSError('Cannot query maintenance controller state')
            running = bool(handle) and wait_result == 258
        else:
            try:
                os.kill(pid, 0); running = True
            except ProcessLookupError:
                running = False
        return {'terminal': not running, 'reason': 'waiting' if running else 'controller_exited_without_completion',
                'pid': pid}
    command = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=12',
               '-o', 'ServerAliveInterval=10', '-o', 'ServerAliveCountMax=2',
               '-p', str(args.port), args.host, shlex.quote(args.remote_python) + ' -']
    result = subprocess.run(command, input=probe_script(args.remote_root),
                            text=True, encoding='utf-8', capture_output=True, timeout=45)
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or f'SSH exited {result.returncode}')
    return json.loads(result.stdout)


def queue_continuation(args, state, observation, runner=subprocess.run):
    prompt = args.prompt_file.read_text(encoding='utf-8')
    prompt += '\nCompletion watcher event: ' + json.dumps(observation, ensure_ascii=False)
    state.update(phase='queueing', observed_at=now(), observation=observation)
    save(args.state, state)
    command = [args.codex, 'queue', '--thread', args.thread, '--message', prompt,
               '--cd', str(args.cwd)]
    # No model/permission override: keep the existing task configuration.
    try:
        result = runner(command, capture_output=True, timeout=90)
        state.update(phase='queued' if result.returncode == 0 else 'delivery_uncertain',
                     queue_returncode=result.returncode,
                     queue_stdout=result.stdout.decode('utf-8', errors='replace'),
                     queue_stderr=result.stderr.decode('utf-8', errors='replace'), finished_at=now())
    except Exception as exc:
        state.update(phase='delivery_uncertain', error=f'{type(exc).__name__}: {exc}', finished_at=now())
    save(args.state, state)
    return 0 if state['phase'] == 'queued' else 1


def lock_process(path):
    stream = path.open('a+b')
    if stream.tell() == 0:
        stream.write(b'0'); stream.flush()
    stream.seek(0)
    if os.name == 'nt':
        import msvcrt
        msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
    else:
        import fcntl
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    return stream


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--host', required=True)
    parser.add_argument('--port', type=int, required=True)
    parser.add_argument('--remote-root', required=True)
    parser.add_argument('--local-root', type=Path, help='Watch a local archive controller instead of SSH experiments.')
    parser.add_argument('--remote-python', default='python3')
    parser.add_argument('--thread', required=True)
    parser.add_argument('--cwd', type=Path, required=True)
    parser.add_argument('--codex', default='codex')
    parser.add_argument('--prompt-file', type=Path, required=True)
    parser.add_argument('--state', type=Path, required=True)
    parser.add_argument('--interval', type=float, default=30)
    parser.add_argument('--probe-once', action='store_true')
    args = parser.parse_args()
    if args.probe_once:
        print(json.dumps(probe(args))); return 0
    args.state.parent.mkdir(parents=True, exist_ok=True)
    lock = lock_process(args.state.with_suffix('.lock'))
    previous = json.loads(args.state.read_text(encoding='utf-8')) if args.state.exists() else {}
    if previous.get('phase') in {'queued', 'queueing', 'delivery_uncertain'}:
        print('Existing delivery record; inspect it before submitting another continuation.'); return 0
    state = {'pid': os.getpid(), 'started_at': now(), 'thread': args.thread,
             'remote_root': args.remote_root, 'phase': 'waiting', 'poll_interval_s': args.interval,
             'scope': 'SSH/file checks only; no model calls until a terminal event is queued.'}
    save(args.state, state)
    while not args.state.with_suffix('.stop').exists():
        try:
            observation = probe(args)
        except Exception as exc:
            observation = {'terminal': False, 'reason': 'probe_unavailable',
                           'error': f'{type(exc).__name__}: {exc}'}
        state.update(last_checked_at=now(), observation=observation)
        save(args.state, state)
        if observation['terminal']:
            return queue_continuation(args, state, observation)
        time.sleep(args.interval)
    state.update(phase='stopped', finished_at=now()); save(args.state, state)
    lock.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
