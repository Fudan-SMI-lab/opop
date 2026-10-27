"""Read-only storage inventory for an explicitly requested maintenance window.

No content hashing, compression of artifacts, deletion, process changes or
framework behavior changes. Per-file records contain metadata, not file bytes.
"""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import gzip
import json
import os
from pathlib import Path
import subprocess


def category(path):
    name = path.name
    if 'framework' in path.parts or 'opop-v6-freezes' in path.parts:
        return 'frozen_checkouts'
    if path.suffix == '.pt':
        return 'tensor_pt'
    if name.endswith(('.nsys-rep', '.sqlite', '.qdrep')):
        return 'native_profiles'
    if name.endswith('.json'):
        if any(t in name.lower() for t in ['trace', 'timeline']) or name.startswith('metric-rank-'):
            return 'raw_trace_and_metric_json'
        if name.startswith('discovery'):
            return 'discovery_json'
        if name in ['result.json', 'rank-0-result.json', 'rank-1-result.json']:
            return 'result_json'
        return 'other_json'
    if name.endswith(('.tar.gz', '.tgz', '.tar', '.gz', '.zip')):
        return 'archives'
    return 'other'


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('roots', nargs='+', type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    unique, per_category, per_root = set(), defaultdict(lambda: defaultdict(int)), defaultdict(lambda: defaultdict(int))
    errors, large, links = [], [], []
    records = args.output/'files.jsonl.gz'
    with gzip.open(records, 'wt', encoding='utf-8', compresslevel=1) as stream:
        for root in args.roots:
            for directory, dirs, files in os.walk(root, followlinks=False):
                # Include directory symlinks in the link inventory; never traverse.
                names = files+[name for name in dirs if (Path(directory)/name).is_symlink()]
                for name in names:
                    path = Path(directory)/name
                    try:
                        stat = path.lstat()
                        row = {'path': str(path), 'bytes': stat.st_size, 'blocks': stat.st_blocks*512,
                               'device': stat.st_dev, 'inode': stat.st_ino, 'nlink': stat.st_nlink,
                               'mtime_ns': stat.st_mtime_ns, 'mode': stat.st_mode}
                        if path.is_symlink():
                            row.update(kind='symlink', target=os.readlink(path), resolved=str(path.resolve()))
                            links.append(row)
                        else:
                            row.update(kind='file', category=category(path))
                            key = (stat.st_dev, stat.st_ino)
                            row['first_inode_occurrence'] = key not in unique
                            if key not in unique:
                                unique.add(key)
                                for counter in [per_category[row['category']], per_root[str(root)]]:
                                    counter['files'] += 1
                                    counter['bytes'] += stat.st_size
                                    counter['blocks'] += stat.st_blocks*512
                                if stat.st_size >= 100*1024**2:
                                    large.append(row)
                        stream.write(json.dumps(row)+'\n')
                    except (FileNotFoundError, PermissionError, OSError) as error:
                        errors.append({'path': str(path), 'error': str(error)})
    df = subprocess.run(['df', '-B1', '--output=size,used,avail,pcent,target', '/root/autodl-tmp', '/'], capture_output=True, text=True)
    du = subprocess.run(['du', '-x', '-B1', '--max-depth=2', '/root/autodl-tmp'], capture_output=True, text=True)
    summary = {'at': datetime.now(timezone.utc).isoformat(), 'roots': list(map(str, args.roots)),
        'unique_files': len(unique), 'categories': dict(per_category), 'per_root': dict(per_root),
        'largest': sorted(large, key=lambda row: row['bytes'], reverse=True), 'symlinks': links, 'errors': errors,
        'df': {'returncode': df.returncode, 'stdout': df.stdout, 'stderr': df.stderr},
        'du': {'returncode': du.returncode, 'stdout': du.stdout, 'stderr': du.stderr},
        'scope': 'Read-only metadata census, globally deduplicated by device/inode across these roots. Running files may change. No candidate is deletion-approved merely from this census; hash/coverage/dependency verification follows after the batch ends.'}
    (args.output/'inventory.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps({'output': str(args.output), 'files': len(unique), 'categories': dict(per_category), 'errors': len(errors)}))


if __name__ == '__main__':
    main()
