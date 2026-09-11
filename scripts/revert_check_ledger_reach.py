"""Revert-check for `check_wrapup.check_ledger_reach`.

The check answers one question: was S2d(c) ever ADMINISTERED? Its outlet is a single line --
`ledger_entries=self.ledger.get(family_id, [])`, keyed per family -- so a reconciled ledger reaches a
rewriter prompt only when the SAME family is called again afterwards. Measured: no family in any run
ever was, so the argument has been `[]` on every rewriter call the project has made, in both arms.

The failure this guards against is not a crash, it is a WRITE-UP. `check_reconciliation` reports PASS
on exactly this state -- entries exist, declarations reconcile, counts look healthy -- and PASS there is
about journalling. Reading it as delivery would publish "the ledger did not help" about a treatment that
was never given, which is worse than publishing nothing.

Both directions have to hold, and they fail differently:

  * round-counting broken   -> a first round reads as a revisit, and S2d(c) gets a fabricated result
  * ordering dropped        -> a reconciliation emitted at round 2's CLOSE counts as if round 2's
                               prompt had carried it
  * reach test inverted     -> a genuinely reached ledger is reported as never administered, which
                               would discard the only S2d(c) evidence a longer run could produce

EACH VARIANT IS PROBED WHERE IT ACTS, on a fixture that reaches its branch. Recorded twice in this
project: a single fixed probe reported real variants as SHAM, and separately an `if` patched to `False`
left an `elif` chain to recompute the same answer so six tests "passed on the defect".

    python scripts/revert_check_ledger_reach.py
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
    "round_counted_per_event": (
        ("            rounds.setdefault(fid, {}).setdefault(key, reconciled[fid] >= 1)",
         "            rounds.setdefault(fid, {})[len(rounds.get(fid, {}))] = reconciled[fid] >= 1"),
        "counts each REWRITE_PRODUCED as its own round. Every round in every completed L3 run "
        "produced exactly TWO candidates (9/9 measured), so this doubles every round: a family's "
        "FIRST round reads as two, and a run that never administered S2d(c) reports a revisit. That "
        "is not a missed detection, it is a fabricated positive result in the paper",
        ["two_candidates_in_one_round_are_ONE_round_not_two"], "one_round"),
    "ordering_dropped": (
        ("    revisited = sorted(f for f, ks in rounds.items() if any(ks.values()))",
         "    revisited = sorted(f for f, ks in rounds.items()\n"
         "                       if len(ks) >= 2 and reconciled[f] >= 1)"),
        "restores the pair-of-counts test. A family with 2 rounds and 1 reconciliation looks reached "
        "-- but if the reconciliation was emitted at round 2's close it came AFTER the last prompt "
        "that could have carried it. The reconciliation for round N is emitted at round N's close, so "
        "the flag has to be read at each call's own position",
        ["two_rounds_whose_only_reconciliation_lands_LAST_is_not_reach"], "recon_last"),
    "reach_never_fires": (
        ("            rounds.setdefault(fid, {}).setdefault(key, reconciled[fid] >= 1)",
         "            rounds.setdefault(fid, {}).setdefault(key, False)"),
        "hardwires 'never administered'. The check would then be unfalsifiable -- it would print the "
        "conclusion I already believe on a run that actually did administer the treatment, and the "
        "one piece of S2d(c) evidence a longer run could produce would be discarded on arrival",
        ["a_second_round_after_a_reconciliation_IS_reach"], "reached"),
    "unreconciled_revisit_counts_as_reach": (
        ("            rounds.setdefault(fid, {}).setdefault(key, reconciled[fid] >= 1)",
         "            rounds.setdefault(fid, {}).setdefault(key, len(rounds.get(fid, {})) >= 1)"),
        "treats any second round as reach, whether or not anything was reconciled. `self.ledger` is "
        "populated only by EXPECTATIONS_RECONCILED, so a second round after a FAILED reconciliation "
        "still receives an empty list -- and RECONCILE_FAILED is an event this harness emits",
        ["a_reconciled_family_that_is_never_revisited_is_still_UNADMINISTERED",
         "two_rounds_whose_only_reconciliation_lands_LAST_is_not_reach"], "recon_last"),
}


def _rewrite(fam: str, cand: str, ts: float, seq: int) -> dict:
    return {"seq": seq, "ts": ts, "type": "REWRITE_PRODUCED", "payload": {
        "candidate_id": cand, "family_id": fam, "hypothesis_id": "H1", "change_summary": "x",
        "expectations": [{"dimension": "shared_bytes", "expect": "decrease", "why": "w"}]}}


def _recon(fam: str, cand: str, ts: float, seq: int) -> dict:
    return {"seq": seq, "ts": ts, "type": "EXPECTATIONS_RECONCILED", "payload": {
        "family_id": fam, "candidate_id": cand, "round": 1, "n_declared": 1,
        "reconciliation": {"hypothesis_id": "H1", "per_dimension": [], "hits": 1, "misses": 0,
                           "vacuous": 0, "dimensions_unpredicted": [],
                           "dimensions_unmeasured": [], "caveat": "c"}}}


def _fixture(kind: str, tmp: Path) -> Path:
    """One run directory shaped so the named branch is reached.

    A single fixture cannot serve all four: `reach_never_fires` only shows on a run that DID reach,
    and `round_counted_per_event` only shows where two candidates share one call.
    """
    d = tmp / kind
    d.mkdir(parents=True)
    if kind == "one_round":          # two candidates, ONE call -- must read as 1 round
        evs = [_rewrite("fam-a", "c1", 100.0, 0), _rewrite("fam-a", "c2", 100.0, 1)]
    elif kind == "reached":          # round 2 issued AFTER round 1's reconciliation -> reach
        evs = [_rewrite("fam-a", "c1", 100.0, 0), _rewrite("fam-a", "c2", 100.0, 1),
               _recon("fam-a", "c1", 110.0, 2),
               _rewrite("fam-a", "c5", 300.0, 3), _rewrite("fam-a", "c6", 300.0, 4)]
    elif kind == "recon_last":       # 2 rounds, 1 reconciliation, emitted after BOTH calls
        evs = [_rewrite("fam-a", "c1", 100.0, 0), _rewrite("fam-a", "c2", 100.0, 1),
               _rewrite("fam-a", "c5", 300.0, 2), _rewrite("fam-a", "c6", 300.0, 3),
               _recon("fam-a", "c5", 310.0, 4)]
    else:
        raise AssertionError("unknown probe kind %r" % kind)
    with (d / "events.jsonl").open("w", encoding="utf-8") as fh:
        for e in evs:
            fh.write(json.dumps(e) + "\n")
    return d


_CODE = ("import sys; sys.path.insert(0, 'scripts'); import check_wrapup\n"
         "from pathlib import Path\n"
         "o = check_wrapup.check_ledger_reach(Path(sys.argv[1]))\n"
         "print('rounds=%s recon=%s reached=%s | %s'\n"
         "      % (o['rounds_per_family'], o['reconciled_per_family'],\n"
         "         o['families_revisited'], o['verdict'][:60]))\n")


def _probe(check_dir: Path, kind: str) -> str:
    tmp = Path(tempfile.mkdtemp(prefix="lr-probe-"))
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
    kinds = ("one_round", "reached", "recon_last")
    baselines = {k: _probe(ROOT, k) for k in kinds}
    print("BASELINE probe readings (one per branch, so each variant is probed where it acts):")
    for k in kinds:
        print("  %-11s %s" % (k, baselines[k][:150]))
    print()
    bad = 0
    for name, ((old, new), why, claimed, kind) in VARIANTS.items():
        if src.count(old) != 1:
            print("  %-38s SKIPPED -- anchor occurs %d times" % (name, src.count(old)))
            print("      %s" % why)
            bad += 1
            continue
        io.open(TARGET, "w", encoding="utf-8", newline="\n").write(src.replace(old, new))
        try:
            probe = _probe(ROOT, kind)
            if probe == baselines[kind]:
                print("  %-38s SHAM -- reading identical to the baseline on the %s probe, so it "
                      "patches nothing reachable" % (name, kind))
                print("      %s" % why)
                bad += 1
                continue
            print("  %-38s changes the %s probe:" % (name, kind))
            print("      was: %s" % baselines[kind][:130])
            print("      now: %s" % probe[:130])
            flipped = []
            skipped = []
            for t in claimed:
                r = subprocess.run(
                    [sys.executable, "-B", "-m", "pytest", TESTS, "-q", "--no-header",
                     "-p", "no:cacheprovider", "-k", t],
                    cwd=ROOT, capture_output=True,
                    env={**os.environ, "PYTHONPATH": "src", "PYTHONDONTWRITEBYTECODE": "1"})
                out = r.stdout.decode("utf-8", "replace")
                # A SKIP is not a verdict: recorded twice in this project that a locally-skipped
                # revert-check variant was a real FAIL on the A800. Report it as unverified, never
                # as caught and never as not caught.
                ran = " no tests ran" not in out
                if not ran:
                    skipped.append(t)
                elif r.returncode != 0:
                    flipped.append(t)
            if flipped:
                print("      CAUGHT by %s" % ", ".join(x[:58] for x in flipped))
            elif skipped:
                print("      UNVERIFIED -- every claimed test SKIPPED here (%s); a skip is not a "
                      "verdict, re-run where they execute" % ", ".join(x[:40] for x in skipped))
                bad += 1
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
