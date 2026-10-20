"""Actual NCCL tuning decisions, kept separate from kernel-name guesses."""
import hashlib
from pathlib import Path
import re


_DECISION = re.compile(
    r"(\w+): (\d+) Bytes -> Algo (\S+) proto (\S+) channel\{Lo\.\.Hi\}=\{(\d+)\.\.(\d+)\}")


def read_logs(paths):
    rows, evidence = [], []
    for path in sorted(map(Path, paths)):
        raw = path.read_bytes()
        evidence.append({"file": str(path), "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)})
        found = {}
        for line_number, line in enumerate(raw.decode(errors="replace").splitlines(), 1):
            match = _DECISION.search(line)
            if not match:
                continue
            values = match.groups()
            key = tuple(values)
            if key not in found:
                operation, size, algorithm, protocol, low, high = values
                found[key] = {"operation": operation, "logical_bytes": int(size), "algorithm": algorithm,
                    "protocol": protocol, "channel_low": int(low), "channel_high": int(high),
                    "occurrences": 0, "evidence": {"file": str(path), "line": line_number, "text": line}}
            found[key]["occurrences"] += 1
        rows.extend(found.values())
    return {"status": "observed" if rows else "unavailable", "decisions": rows, "evidence": evidence,
        "scope": "Separate diagnostic worker including communicator setup and warmed workload calls. Counts are log records, not scoped invocation counts. No event-to-collective or operator-group join is claimed.",
        "interpretation": "Kernel entry names do not identify the selected algorithm/protocol. These are actual NCCL tuning log decisions; payload-size matches alone do not establish per-group identity.",
        "reason": None if rows else "No supported tuning-decision records found; raw logs retained. Log format and availability are version-dependent."}
