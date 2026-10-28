"""Compress an explicit list of completed diagnostic directories on the server."""
import argparse
import gzip
import json
import os
from pathlib import Path
import traceback

from v6_server_storage import BASE, checked, compress_json, df, digest, json_plan, no_users, now, save, sha


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); settings = json.loads(args.plan.read_text()); output = args.output
    os.umask(0o077); output.mkdir(parents=True, exist_ok=False)
    state = {'at': now(), 'phase': 'running', 'before': df(BASE), 'plan_sha256': sha(args.plan)}
    save(output / 'state.json', state)
    try:
        roots = [r for g in settings['groups'] for r in g['roots']]
        for name in roots:
            root = checked(BASE, name)
            final = json.loads((root / 'followup-completed.json').read_text())
            assert final['phase'] in {'completed', 'completed_with_errors', 'complete'}
        no_users(BASE, roots)
        plan = json_plan(settings, output)
        selected = {path for row in plan['records'] for path in row['files']}
        retained = {}
        for name in roots:
            for path in checked(BASE, name).rglob('*'):
                if path.is_file() and not path.is_symlink() and str(path) not in selected:
                    retained[str(path)] = sha(path)
        with gzip.open(output / 'retained-before.json.gz', 'wt') as stream:
            json.dump(retained, stream)
        result = compress_json(settings, output)
        rows = [json.loads(line) for line in (output / 'cold-json-journal.jsonl').read_text().splitlines()]
        assert len(rows) == len(plan['records'])
        for row in rows:
            assert sha(row['archive']) == row['archive_sha256']
            assert digest(row['archive'], compressed=True) == row['sha256']
            assert all(not Path(path).exists() for path in row['files'])
        for path, expected in retained.items():
            assert sha(path) == expected, path
        after = df(BASE)
        state.update(phase='completed', finished_at=now(), after=after, result=result,
                     verified_payloads=len(rows), retained_files_verified=len(retained),
                     actual_df_reclaimed_bytes=after['available']-state['before']['available'])
        (output / 'RESTORE.md').write_text(
            '# Server-only diagnostic restoration\n\n'
            'All original content remains in verified gzip payloads on this server. No raw download is required.\n'
            'Look up the original path in cold-json-journal.jsonl, then run:\n\n'
            '```sh\n/root/autodl-tmp/dev-venv/bin/python /root/autodl-tmp/dev/opop-v6/scripts/maintenance/v6_server_storage.py '
            f'restore-json --journal {output}/cold-json-journal.jsonl --path /absolute/original/path\n```\n'
            'The restore verifies both archive and decompressed hashes and restores all recorded inode aliases.\n')
    except Exception:
        state.update(phase='error', finished_at=now(), error=traceback.format_exc(), after=df(BASE))
        raise
    finally:
        save(output / 'state.json', state); save(output / 'followup-completed.json', state)
        print(json.dumps(state), flush=True)


if __name__ == '__main__':
    main()
