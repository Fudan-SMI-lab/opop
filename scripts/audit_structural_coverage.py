"""Which structural classes did a run actually MEASURE, and which did it only generate?

Motivated by run-l3-48-20260905-010737, where the agent generated eight tensor-core kernels
and the harness rejected all eight while publishing all seven scalar-FMA ones -- a perfect
0/8 vs 7/0 partition that no single event or report surfaced. A run can look healthy (12
published spaces, a real 8.90x speedup) while an entire structural class was generated and
never timed.

Reads candidate sources off disk rather than trusting event labels, because the interesting
property (does this kernel use `tl.dot`?) is not in any event.

Usage:  python scripts/audit_structural_coverage.py [run-dir ...]
        python scripts/audit_structural_coverage.py            # every runs/run-l3-*
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def host_path(p: str) -> Path:
    """Translate a WSL /mnt/<drive>/... path to its Windows form."""
    if p.startswith("/mnt/") and len(p) > 6:
        return Path(p[5].upper() + ":/" + p[7:])
    return Path(p)


def classify(source: str) -> dict:
    return {
        "dots": len(re.findall(r"tl\.dot\(", source)),
        "lowp_cast": bool(re.search(r"\.to\(tl\.(?:float16|bfloat16)\)", source)),
        "dtype_knob": bool(re.search(r"PARAMS\s*=\s*\{[^}]*(?:DTYPE|PRECISION)", source,
                                     re.S | re.I)),
    }


def audit(run_dir: Path) -> int:
    events = run_dir / "events.jsonl"
    if not events.exists():
        print(f"{run_dir.name}: no events.jsonl")
        return 0
    ev = [json.loads(line) for line in events.open(encoding="utf-8")]
    published = {e["payload"]["space"]["candidate_id"] for e in ev
                 if e["type"] == "SPACE_PUBLISHED"}
    # A candidate's best latency, if it was ever tuned -- "published" is necessary but not
    # sufficient for "measured".
    tuned = {e["payload"]["candidate_id"]: e["payload"]["best_ms"] for e in ev
             if e["type"] == "TUNING_DONE"}

    seen: dict[str, dict] = {}
    for f in sorted((f for f in (run_dir / "jobs").glob("*-wit-*-eval-*.json")
                     if not f.name.endswith(".out.json")),
                    key=lambda p: p.stat().st_mtime):
        cid = f.name.split("-wit-")[0]
        if cid in seen:
            continue  # first witness = the a=0 source, before any repair rewrote it
        job = json.loads(f.read_text(encoding="utf-8"))
        src = host_path(job.get("kernel_src_path", ""))
        if not src.exists():
            continue
        seen[cid] = classify(src.read_text(encoding="utf-8"))

    if not seen:
        print(f"{run_dir.name}: no candidate sources found on disk")
        return 0

    tally = {True: [0, 0], False: [0, 0]}  # uses_dot -> [published, rejected]
    print(f"\n=== {run_dir.name} ===")
    print(f"{'candidate':17}{'tl.dot':>7}{'lowp':>6}{'dtype_knob':>11}"
          f"{'published':>10}{'tuned_ms':>10}")
    for cid, d in sorted(seen.items(), key=lambda kv: (-kv[1]["dots"], kv[0])):
        pub = cid in published
        tally[bool(d["dots"])][0 if pub else 1] += 1
        ms = tuned.get(cid)
        print(f"{cid:17}{d['dots']:>7}{str(d['lowp_cast']):>6}"
              f"{str(d['dtype_knob']):>11}{str(pub):>10}"
              f"{(f'{ms:.2f}' if ms else '-'):>10}")

    tc, sc = tally[True], tally[False]
    print(f"  uses tl.dot (tensor core):  {tc[0]} published, {tc[1]} rejected")
    print(f"  no tl.dot (scalar FMA):     {sc[0]} published, {sc[1]} rejected")

    # The signal worth flagging: a whole class generated and never accepted.
    problems = 0
    for label, (pub, rej) in (("tensor-core", tc), ("scalar-FMA", sc)):
        if rej and not pub:
            print(f"  WARNING: every {label} candidate was rejected ({rej} of {rej}). "
                  f"That class was generated but never measured.")
            problems += 1
    return problems


def main(argv: list[str]) -> int:
    targets = [Path(a) for a in argv[1:]] or sorted((REPO / "runs").glob("run-l3-*"))
    total = sum(audit(t if t.is_absolute() else REPO / t) for t in targets)
    print(f"\n{total} run(s) with a wholly-rejected structural class")
    return 0  # informational: never fail a build over this


if __name__ == "__main__":
    sys.exit(main(sys.argv))
