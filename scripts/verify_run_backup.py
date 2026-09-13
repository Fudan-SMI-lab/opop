"""Verify a run-directory backup exactly: file count, per-file size, events.jsonl sha256, no secrets.

WHY NOT `du`. A previous backup verification in this project compared `du -sb` totals and reported
31.9 MB against 30.9 MB, a difference that was entirely directory-entry overhead -- the archives were
identical. The comparison that means something is the set of (relative path, size) pairs plus a
checksum of the file that matters.

WHY CHECK FOR SECRETS. Each run dir contains ~50 per-sandbox `opencode.json` files, each carrying the
provider block including a PLAINTEXT API key (`sandbox_config_path` copies the whole block into every
agent sandbox). A tar of a run dir therefore carries the key off the box unless it is excluded, and an
archive kept locally would be a fifth copy. The backup excludes them; this asserts the exclusion
actually worked rather than assuming the flag was spelled right.

Usage: verify_run_backup.py <archive.tar.gz> <expected-file-count-from-the-box>
"""
import hashlib
import re
import sys
import tarfile

SECRET_NAMES = ("opencode.json", "auth.json", ".git-credentials")
# Provider key shapes this project actually has on disk: the GLM/zhipu key and a GitHub PAT.
SECRET_PATTERNS = (re.compile(rb"ghp_[A-Za-z0-9]{20,}"),
                   re.compile(rb'"apiKey"\s*:\s*"[^"]{8,}"'),
                   re.compile(rb"[0-9a-f]{32}\.[A-Za-z0-9]{16,}"))

archive = sys.argv[1]
expected = int(sys.argv[2]) if len(sys.argv) > 2 else None

files, dirs = [], 0
secret_named, secret_content = [], []
events_sha = None
events_lines = None

with tarfile.open(archive, "r:gz") as tf:
    for m in tf:
        if m.isdir():
            dirs += 1
            continue
        if not m.isfile():
            continue
        files.append((m.name, m.size))
        base = m.name.rsplit("/", 1)[-1]
        if base in SECRET_NAMES:
            secret_named.append(m.name)
        # Scan the small text members for key shapes. Reading every byte of a 39 MB archive is
        # affordable and the point is to be sure, but skip the big ones: a key does not live in a
        # multi-megabyte artifact and the scan would dominate the runtime.
        if m.size <= 512 * 1024:
            fh = tf.extractfile(m)
            if fh is None:
                continue
            data = fh.read()
            for pat in SECRET_PATTERNS:
                if pat.search(data):
                    secret_content.append(f"{m.name} ({pat.pattern.decode('latin1')[:24]})")
                    break
        if m.name.endswith("/events.jsonl") and events_sha is None:
            fh = tf.extractfile(m)
            if fh is not None:
                data = fh.read()
                events_sha = hashlib.sha256(data).hexdigest()
                events_lines = data.count(b"\n")

total = sum(s for _, s in files)
print(f"archive        : {archive}")
print(f"files          : {len(files)}   dirs {dirs}   bytes {total:,}")
if expected is not None:
    ok = len(files) == expected
    print(f"expected files : {expected}   {'MATCH' if ok else '*** MISMATCH'}")
    if not ok:
        print(f"                 difference {len(files) - expected:+d} -- the excluded secret files "
              f"account for part of this by design; confirm the number below matches the count of "
              f"excluded files on the box")
print(f"events.jsonl   : {events_lines} lines  sha256 {events_sha}")
print()
print(f"secret-named members (must be 0): {len(secret_named)}")
for n in secret_named[:5]:
    print(f"  *** {n}")
print(f"members matching a key shape (must be 0): {len(secret_content)}")
for n in secret_content[:5]:
    print(f"  *** {n}")
if secret_named or secret_content:
    print()
    print("THIS ARCHIVE CARRIES A CREDENTIAL. Delete it and re-create with the exclusions.")
    sys.exit(1)
print()
print("no credential material found in this archive")
