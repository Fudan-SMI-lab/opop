"""Do the new tests actually FAIL on the pooled implementation they name?

A test that passes on the bug is worse than no test. Three variants, each a minimal reversion of one
part of the fix, run against the real test file in an isolated tree.
"""
import os, re, shutil, subprocess, sys, tempfile
from pathlib import Path

# The repo root. `parents[1]` is right when this lives in scripts/ or .tmp-arms/, but the A800
# copy sits AT the repo root, where parents[1] points above it and copytree copied a tree with
# no src/. KOPT_REPO_ROOT lets the caller say, and the fallback probes for src/.
ROOT = Path(os.environ.get("KOPT_REPO_ROOT") or (
    Path(__file__).resolve().parent
    if (Path(__file__).resolve().parent / "src").is_dir()
    else Path(__file__).resolve().parents[1]))
ORCH = "src/kernel_optimizer/control/orchestrator.py"
RECON = "src/kernel_optimizer/evaluation/reconcile.py"

VARIANTS = {
    # The original defect: one entry per ROUND, every declaration against the round's delta map.
    "pooled_again": (ORCH, [(
        """            for cand_id, group in by_cand.items():""",
        """            for cand_id, group in [("", declared)] if declared else []:"""), (
        """                own = self._candidate_conversion(family_id, cand_id, conversion, _cv,
                                                 profile_before)""",
        """                own = conversion"""),
    ], ["test_two_candidates_in_one_round_are_reconciled_SEPARATELY"]),
    # Per-candidate grouping kept, but an unmeasured candidate inherits the round's deltas.
    "unmeasured_inherits_round": (ORCH, [(
        """        crun = self.runs.get(cand_id)
        if crun is None or crun.best_ms is None:
            return {}""",
        """        crun = self.runs.get(cand_id)
        if crun is None or crun.best_ms is None:
            return round_conversion"""),
    ], ["test_a_candidates_entry_is_never_scored_against_another_candidates_deltas",
        "test_two_candidates_in_one_round_are_reconciled_SEPARATELY"]),
    # The heading stops naming the candidate.
    "heading_drops_candidate": (RECON, [(
        """        out.append("## Round %s — %s%s" % (rnd if rnd is not None else "?", hid,
                                          " (%s)" % cid if cid else ""))""",
        """        out.append("## Round %s — %s" % (rnd if rnd is not None else "?", hid))"""),
    ], ["test_the_ledger_heading_names_the_candidate_so_two_sections_are_distinguishable"]),
}

def run(tree: Path, names: list[str]) -> dict[str, bool]:
    """True == the test PASSED."""
    r = subprocess.run(
        [sys.executable, "-B", "-m", "pytest", "tests/test_s2d_wiring.py", "-q", "--no-header",
         "-p", "no:cacheprovider"] + sum([["-k", n] for n in names[:1]], []),
        cwd=tree, capture_output=True,
        env={**os.environ, "PYTHONPATH": "src", "PYTHONDONTWRITEBYTECODE": "1"})
    out = r.stdout.decode("utf-8", "replace") + r.stderr.decode("utf-8", "replace")
    return {"rc": r.returncode, "out": out}

def main() -> int:
    bad = 0
    for name, (relpath, subs, claimed) in VARIANTS.items():
        tmp = Path(tempfile.mkdtemp(prefix="rp-"))
        try:
            tree = tmp / "t"
            shutil.copytree(ROOT, tree, ignore=shutil.ignore_patterns(
                "runs*", ".git", "__pycache__", "*.pyc", ".tmp-arms", "external*"))
            p = tree / relpath
            src = p.read_text(encoding="utf-8")
            for old, new in subs:
                if src.count(old) != 1:
                    print("  %-28s SKIPPED -- anchor occurs %d times" % (name, src.count(old)))
                    bad += 1
                    src = None
                    break
                src = src.replace(old, new)
            if src is None:
                continue
            p.write_text(src, encoding="utf-8")
            flipped = []
            for t in claimed:
                res = run(tree, [t])
                if res["rc"] != 0:
                    flipped.append(t)
            if flipped:
                print("  %-28s CAUGHT by %s" % (name, ", ".join(x[5:60] for x in flipped)))
            else:
                print("  %-28s *** NOT CAUGHT *** (claimed %s)" % (name, claimed))
                bad += 1
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    return 1 if bad else 0

if __name__ == "__main__":
    print("VARIANTS of the per-candidate reconciliation fix:")
    raise SystemExit(main())
