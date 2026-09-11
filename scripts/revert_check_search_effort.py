"""Revert-check for `check_wrapup.check_search_effort`.

This check exists because of a FALSE ALARM I nearly filed. Five candidates on each live arm show 80
trials against `trials_per_space: 40`, which reads as a 2x budget overrun -- and a budget overrun would
break arm parity on search effort, which is the one thing a paired latency comparison cannot lose. The
grouping resolves it: every space is at exactly 40, and a K-expansion publishes a SECOND space for the
same candidate. The budget field is named `trials_per_space`.

So the harness has to defend BOTH directions, and they fail differently:

  * grouping removed  -> a correct run is reported as an overrun (a false alarm that would have me
                         restarting healthy runs)
  * threshold removed -> a real overrun is reported as PASS (silent, and it makes the arms'
                         latencies incomparable)

EACH VARIANT IS PROBED WHERE IT ACTS, on a fixture that reaches its branch. Recorded twice in this
project: a single fixed probe reported real variants as SHAM, and separately an `if` patched to `False`
left an `elif` chain to recompute the same answer so six tests "passed on the defect".

    python scripts/revert_check_search_effort.py
"""

from __future__ import annotations

import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(os.environ.get("KOPT_REPO_ROOT") or Path(__file__).resolve().parents[1])
TARGET = ROOT / "scripts" / "check_wrapup.py"
TESTS = "tests/test_check_wrapup.py"

VARIANTS = {
    "grouping_by_space_removed": (
        ("        per_space[sid] = per_space.get(sid, 0) + 1",
         "        per_space[cid or sid] = per_space.get(cid or sid, 0) + 1"),
        "counts per CANDIDATE instead of per space -- the false alarm this check exists to prevent. "
        "On the live pair it turns five healthy candidates per arm into 'a 2x budget overrun', and "
        "acting on that would mean restarting runs that are behaving exactly as designed",
        ["two_spaces_of_40_on_one_candidate_is_NOT_a_budget_overrun"], "two_spaces"),
    "overrun_threshold_dropped": (
        ("    over = {s: n for s, n in per_space.items() if budget and n > budget}",
         "    over = {}"),
        "stops reporting a real overrun. Silent in the worst way: both arms finish, both print "
        "latencies, and the comparison is between unequal search efforts -- the recorded "
        "`equal-configs-do-not-imply-equal-search` failure, at the budget level",
        ["a_single_space_over_the_budget_IS_reported",
         "search_effort_reads_the_budget_from_the_manifest_not_a_default"], "overrun"),
    "budget_hardcoded": (
        ('        budget = ((man.get("config") or {}).get("budgets") or {}).get("trials_per_space")',
         "        budget = 40"),
        "ignores the run's own configured budget and assumes 40. A run configured for 20 then passes "
        "at 30 trials per space. The recorded rule is that a threshold must come from the artefact, "
        "not from what I remember the config saying",
        ["search_effort_reads_the_budget_from_the_manifest_not_a_default"], "budget20"),
}


def _trial(space: str, cand: str, i: int) -> dict:
    return {"seq": i, "ts": 1789000000.0 + i, "type": "TRIAL_DONE", "payload": {"trial": {
        "trial_id": "t%d" % i, "candidate_id": cand, "space_id": space, "status": "complete",
        "params": {"values": {}},
        "latency_ms": {"mean": 1.0, "median": 1.0, "std": 0.0, "min": 1.0, "max": 1.0,
                       "n_samples": 20}}}}


def _fixture(kind: str, tmp: Path) -> Path:
    """One run directory shaped so the named branch is reached."""
    d = tmp / kind
    d.mkdir(parents=True)
    if kind == "two_spaces":          # correct run: 80 trials, 2 spaces of 40
        evs = ([_trial("sp-a", "cand-1", i) for i in range(40)]
               + [_trial("sp-b", "cand-1", 100 + i) for i in range(40)])
        budget = 40
    elif kind == "overrun":           # real overrun: 41 in ONE space
        evs = [_trial("sp-a", "cand-1", i) for i in range(41)]
        budget = 40
    elif kind == "budget20":          # configured for 20, ran 30 -- only a manifest read catches it
        evs = [_trial("sp-a", "cand-1", i) for i in range(30)]
        budget = 20
    else:
        raise AssertionError("unknown probe kind %r" % kind)
    (d / "manifest.json").write_text(
        json.dumps({"config": {"budgets": {"trials_per_space": budget}}}), encoding="utf-8")
    with (d / "events.jsonl").open("w", encoding="utf-8") as fh:
        for e in evs:
            fh.write(json.dumps(e) + "\n")
    return d


_CODE = ("import sys; sys.path.insert(0, 'scripts'); import check_wrapup\n"
         "from pathlib import Path\n"
         "o = check_wrapup.check_search_effort(Path(sys.argv[1]))\n"
         "print('spaces=%s max=%s over=%s budget=%s | %s'\n"
         "      % (o['spaces'], o['max_per_space'], o['spaces_over_budget'], o['budget'],\n"
         "         o['verdict'][:70]))\n")


def _probe(check_dir: Path, kind: str) -> str:
    tmp = Path(tempfile.mkdtemp(prefix="se-probe-"))
    try:
        d = _fixture(kind, tmp)
        r = subprocess.run(
            [sys.executable, "-B", "-c", _CODE, str(d)], cwd=check_dir, capture_output=True,
            env={**os.environ, "PYTHONPATH": "src", "PYTHONDONTWRITEBYTECODE": "1"})
        return (r.stdout.decode("utf-8", "replace").strip()
                or "ERR: " + r.stderr.decode("utf-8", "replace").strip()[-220:])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main() -> int:
    src = io.open(TARGET, encoding="utf-8").read()
    kinds = ("two_spaces", "overrun", "budget20")
    baselines = {k: _probe(ROOT, k) for k in kinds}
    print("BASELINE probe readings (one per branch, so each variant is probed where it acts):")
    for k in kinds:
        print("  %-11s %s" % (k, baselines[k][:140]))
    print()
    bad = 0
    for name, ((old, new), why, claimed, kind) in VARIANTS.items():
        if src.count(old) != 1:
            print("  %-26s SKIPPED -- anchor occurs %d times" % (name, src.count(old)))
            print("      %s" % why)
            bad += 1
            continue
        io.open(TARGET, "w", encoding="utf-8", newline="\n").write(src.replace(old, new))
        try:
            probe = _probe(ROOT, kind)
            if probe == baselines[kind]:
                print("  %-26s SHAM -- reading identical to the baseline on the %s probe, so it "
                      "patches nothing reachable" % (name, kind))
                print("      %s" % why)
                bad += 1
                continue
            print("  %-26s changes the %s probe:" % (name, kind))
            print("      was: %s" % baselines[kind][:120])
            print("      now: %s" % probe[:120])
            flipped = []
            for t in claimed:
                r = subprocess.run(
                    [sys.executable, "-B", "-m", "pytest", TESTS, "-q", "--no-header",
                     "-p", "no:cacheprovider", "-k", t],
                    cwd=ROOT, capture_output=True,
                    env={**os.environ, "PYTHONPATH": "src", "PYTHONDONTWRITEBYTECODE": "1"})
                if r.returncode != 0:
                    flipped.append(t)
            if flipped:
                print("      CAUGHT by %s" % ", ".join(x[:58] for x in flipped))
            else:
                print("      *** NOT CAUGHT *** (claimed %s)" % claimed)
                bad += 1
            print("      %s" % why)
        finally:
            io.open(TARGET, "w", encoding="utf-8", newline="\n").write(src)
    r = subprocess.run([sys.executable, "-B", "-m", "pytest", TESTS, "-q", "--no-header",
                        "-p", "no:cacheprovider"], cwd=ROOT, capture_output=True,
                       env={**os.environ, "PYTHONPATH": "src", "PYTHONDONTWRITEBYTECODE": "1"})
    print("\nrestored: %s" % r.stdout.decode("utf-8", "replace").strip().splitlines()[-1])
    if bad:
        print("%d variant(s) sham, skipped, or not caught." % bad)
        return 1
    print("Every variant changes behaviour AND is caught by a case it names.")
    return 0 if r.returncode == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
