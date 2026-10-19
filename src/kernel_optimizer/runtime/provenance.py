"""Record the actual framework bytes imported by a fresh experimental worker."""
import hashlib
from pathlib import Path


def framework_source():
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    files = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix in {".py", ".md"})
    for path in files:
        digest.update(path.relative_to(root).as_posix().encode() + b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return {"sha256": digest.hexdigest(), "files": len(files),
            "scope": "kernel_optimizer Python sources and role prompts at worker entry"}
