"""Losslessly archive terminal worker request/start records; explicitly restore on demand.

Only jobs/*/run/request.json and rank-*-started.json are eligible. Receipts,
results, reference tensors, traces, candidate source and provider state stay live.
Dry-run by default. Use --apply to replace originals with verified .json.gz files.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil


def digest(path, *, compressed=False):
    h = hashlib.sha256()
    opener = gzip.open if compressed else open
    with opener(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def eligible(root):
    for receipt in root.rglob("receipt.json"):
        run = receipt.parent
        if run.name != "run" or run.parent.parent.name != "jobs":
            continue
        value = json.loads(receipt.read_text())
        if value.get("worker_pids_reaped") is not True:
            continue
        pids = [row.get("pid") for row in value.get("ranks", [])]
        if any(pid and Path(f"/proc/{pid}").exists() for pid in pids):
            continue
        for path in [run / "request.json", *run.glob("rank-*-started.json")]:
            if path.is_file() and not path.is_symlink():
                yield path


def archive(path):
    before = path.stat()
    target = path.with_suffix(path.suffix + ".gz")
    original_sha = digest(path)
    if target.exists():
        if digest(target, compressed=True) != original_sha:
            raise ValueError(f"Existing archive differs: {target}")
    else:
        temporary = target.with_name(target.name + ".tmp")
        with path.open("rb") as source, temporary.open("wb") as destination:
            with gzip.GzipFile(filename="", mode="wb", fileobj=destination, compresslevel=3, mtime=0) as zipped:
                shutil.copyfileobj(source, zipped, 1024 * 1024)
            destination.flush()
            os.fsync(destination.fileno())
        if digest(temporary, compressed=True) != original_sha:
            raise ValueError(f"Archive round-trip mismatch: {path}")
        os.chmod(temporary, before.st_mode & 0o777)
        temporary.replace(target)
    after = path.stat()
    if (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
        raise ValueError(f"Original changed while archiving: {path}")
    row = {"path": str(path), "archive": str(target), "bytes": before.st_size,
           "sha256": original_sha, "archive_bytes": target.stat().st_size,
           "archive_sha256": digest(target), "mode": before.st_mode & 0o777,
           "mtime_ns": before.st_mtime_ns}
    # Journal the verified compressed copy before removing the redundant original.
    return row


def restore(manifest):
    count = 0
    for line in manifest.read_text().splitlines():
        row = json.loads(line)
        path, compressed = Path(row["path"]), Path(row["archive"])
        if path.exists():
            if digest(path) != row["sha256"]:
                raise ValueError(f"Refusing to overwrite changed file: {path}")
            continue
        if digest(compressed) != row["archive_sha256"]:
            raise ValueError(f"Archive hash mismatch: {compressed}")
        temporary = path.with_name(path.name + ".restore-tmp")
        with gzip.open(compressed, "rb") as source, temporary.open("wb") as destination:
            shutil.copyfileobj(source, destination, 1024 * 1024)
        if digest(temporary) != row["sha256"]:
            raise ValueError(f"Restore hash mismatch: {path}")
        os.chmod(temporary, row["mode"])
        os.utime(temporary, ns=(row["mtime_ns"], row["mtime_ns"]))
        temporary.replace(path)
        count += 1
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="*", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--restore", type=Path)
    args = parser.parse_args()
    if args.restore:
        print(json.dumps({"restored": restore(args.restore)}))
        return
    if not args.roots or args.apply and args.manifest is None:
        parser.error("Supply explicit roots, and --manifest with --apply")
    if not Path("/proc/self").exists():
        parser.error("Archiving requires Linux /proc to exclude live worker processes")
    roots = [p.resolve(strict=True) for p in args.roots]
    paths = sorted(set(p for root in roots for p in eligible(root)))
    print(json.dumps({"eligible_files": len(paths), "bytes": sum(p.stat().st_size for p in paths),
                      "apply": args.apply}), flush=True)
    if not args.apply:
        return
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    raw = compressed = 0
    with args.manifest.open("a", encoding="utf-8") as journal:
        for number, path in enumerate(paths, 1):
            assert any(path.resolve().is_relative_to(root) for root in roots)
            row = archive(path)
            journal.write(json.dumps(row) + "\n"); journal.flush(); os.fsync(journal.fileno())
            path.unlink()
            raw += row["bytes"]; compressed += row["archive_bytes"]
            if number % 100 == 0:
                print(json.dumps({"archived": number, "original_bytes": raw, "compressed_bytes": compressed}), flush=True)
    print(json.dumps({"completed": True, "archived": len(paths), "original_bytes": raw,
                      "compressed_bytes": compressed, "manifest": str(args.manifest)}))


if __name__ == "__main__":
    main()
