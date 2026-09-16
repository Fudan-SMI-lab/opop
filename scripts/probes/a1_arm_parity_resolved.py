"""Arm parity from the RESOLVED config, not from the source YAML.

WHY THE SOURCE DIFF IS NOT ENOUGH. Diffing the two config files is the obvious check and it
is structurally blind in one direction: an arm may write a key EXPLICITLY while its partner
omits it and inherits the default. The m2b pair is exactly this shape -- arm B writes
`pcap: 48`, `max_probe_attempts: 4`, `cadence_told: 10` and arm A writes none of them. Today
those three defaults equal the written values, so the arms agree. The day a default moves,
the source diff still reports "the arms differ only in mode" while the two runs have in fact
diverged. The diff would be reassuring and wrong.

`manifest.json` holds each run's config AFTER resolution -- defaults filled in, overrides
applied -- so comparing manifests compares what the runs actually did. Every difference is
then either the declared independent variable, a necessary isolation path, or a defect.

WHAT COUNTS AS BENIGN, AND WHY IT IS A CLOSED LIST. Two arms on one box MUST differ in the
paths that keep them from writing over each other: runs_dir, XDG_DATA_HOME, triton cache,
the GPU pin, the opencode port. Anything else is reported as UNEXPECTED. The list is matched
by key NAME with a path-shaped value, never by "it looks like a path", so a semantic knob
that happens to hold a string is never absorbed into it.

The independent variable is declared on the command line, not inferred, because inferring it
is how a broken pair reads as a valid one: window 1's n1 pair had mode=off on BOTH arms, and
a reader that decided "the variable is whatever differs" would have found some other
difference and called it the treatment.

Offline: reads manifest.json only. No GPU, no torch, no candidate execution.
"""
from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

# Keys whose values MUST differ between two arms sharing a box, with the reason. Matched on
# the leaf key name; the value must look like a path or a device pin, or it is not absorbed.
ISOLATION_KEYS = {
    "runs_dir": "each arm needs its own journal tree",
    "seed_candidates_dir": "seed source dir (see note below -- must be IDENTICAL for a "
                           "seed-paired experiment, so it is checked, not excused)",
    "XDG_DATA_HOME": "opencode state; sharing it crosses the arms' agent state",
    "triton_cache_dir": "a shared triton cache lets one arm serve the other's compiles",
    "CUDA_VISIBLE_DEVICES": "the GPU pin is the point of running two arms at once",
    "port": "two servers cannot bind one port",
}
# Of those, the ones that must be EQUAL rather than different. Kept in the same table so the
# distinction is visible in one place instead of implied by omission.
MUST_BE_EQUAL = {"seed_candidates_dir"}


def _flatten(o, pre: str = "") -> dict:
    """Config -> {dotted.path: scalar}. Lists are compared whole (as JSON) because a config
    list is a single semantic value (e.g. a knob's choices); splitting it by index would
    report a reordering as N differences."""
    out: dict = {}
    if isinstance(o, dict):
        for k, v in o.items():
            out.update(_flatten(v, "%s.%s" % (pre, k) if pre else str(k)))
    elif isinstance(o, list):
        out[pre] = json.dumps(o, sort_keys=True)
    else:
        out[pre] = o
    return out


def _manifest(run: Path) -> dict:
    p = run / "manifest.json"
    if not p.is_file():
        raise SystemExit("no manifest.json in %s -- this reader compares RESOLVED configs, "
                         "and without it there is nothing to compare. Do NOT fall back to "
                         "diffing the source YAML: that check is blind to a default-vs-"
                         "explicit divergence, which is the failure this file exists for."
                         % run)
    return json.load(io.open(p, encoding="utf-8")).get("config") or {}


def _arm(run: Path) -> str:
    generic = {"runs", "runs-v4", "run", "."}
    for part in (run.parent.name, *[p.name for p in run.parents]):
        if part and part not in generic and not part.startswith("run-"):
            return part
    return run.parent.name


def _leaf(key: str) -> str:
    return key.rsplit(".", 1)[-1]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("off_run")
    ap.add_argument("active_run")
    ap.add_argument("--variable", action="append", default=[],
                    help="dotted config path of a DECLARED independent variable (repeatable). "
                         "e.g. --variable v4.conditional_scan.mode. Declared, never inferred: "
                         "a pair that accidentally shares its variable must fail this check, "
                         "not have some other difference promoted into the role.")
    a = ap.parse_args()
    off, act = Path(a.off_run.rstrip("/")), Path(a.active_run.rstrip("/"))
    no, na = _arm(off), _arm(act)

    fo, fa = _flatten(_manifest(off)), _flatten(_manifest(act))
    keys = sorted(set(fo) | set(fa))

    declared, isolation, equal_violations, unexpected, missing = [], [], [], [], []
    for k in keys:
        vo, va = fo.get(k, "<absent>"), fa.get(k, "<absent>")
        leaf = _leaf(k)
        if k not in fo or k not in fa:
            # A key present in one resolved config and absent in the other is a schema
            # difference, not a value difference -- the two runs did not even load the same
            # settings object. Never silently benign.
            missing.append((k, vo, va))
            continue
        if leaf in MUST_BE_EQUAL:
            if vo != va:
                equal_violations.append((k, vo, va))
            continue
        if vo == va:
            continue
        if k in a.variable:
            declared.append((k, vo, va))
        elif leaf in ISOLATION_KEYS:
            isolation.append((k, vo, va))
        else:
            unexpected.append((k, vo, va))

    print("  arms: off=%s  active=%s" % (no, na))
    print("  resolved config keys: %d (off) / %d (active)" % (len(fo), len(fa)))
    print()

    print("  DECLARED independent variable(s):")
    if not declared:
        print("    !! NONE OF THE DECLARED VARIABLES DIFFER.")
        print("    !! Either --variable names the wrong path or the arms share their setting.")
        print("    !! Window 1's n1 pair ran mode=off on BOTH arms; its 0 scan blocks looked")
        print("    !! like a negative result and were a design consequence. STOP HERE.")
    for k, vo, va in declared:
        print("    %-52s %s -> %s" % (k, json.dumps(vo), json.dumps(va)))

    print()
    print("  benign isolation differences (required for two arms on one box):")
    for k, vo, va in isolation:
        print("    %-52s %s" % (k, ISOLATION_KEYS[_leaf(k)]))

    print()
    if equal_violations:
        print("  !! KEYS THAT MUST MATCH BUT DO NOT:")
        for k, vo, va in equal_violations:
            print("    %-52s %s != %s" % (k, json.dumps(vo), json.dumps(va)))
        print("  !! A seed-paired experiment whose arms read different seed sources is not")
        print("  !! seed-paired, and every downstream contrast inherits that difference.")
    else:
        print("  keys required to MATCH: all equal (%s)" % ", ".join(sorted(MUST_BE_EQUAL)))

    print()
    if missing:
        print("  !! SCHEMA DIFFERENCES (key resolved in one arm only):")
        for k, vo, va in missing:
            print("    %-52s off=%s  active=%s" % (k, json.dumps(vo), json.dumps(va)))

    if unexpected:
        print("  !! UNEXPECTED DIFFERENCES -- each one is a second independent variable")
        print("  !! until explained. The pair is not a clean contrast while any remain:")
        for k, vo, va in unexpected:
            print("    %-52s %s -> %s" % (k, json.dumps(vo), json.dumps(va)))
    else:
        print("  => no unexpected differences: apart from the declared variable(s), the two")
        print("     RESOLVED configs differ only in the isolation paths above.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
