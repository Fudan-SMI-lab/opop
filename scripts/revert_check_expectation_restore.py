"""Do the S2d in-flight-restore tests fail on the version that could not restore?

The defect they guard is silent in the worst way: the resumed round reports n_declared=0, which is
indistinguishable in the log from an agent that declared nothing.
"""
import os, shutil, subprocess, sys, tempfile
from pathlib import Path

ROOT = Path(os.environ.get("KOPT_REPO_ROOT") or (
    Path(__file__).resolve().parent if (Path(__file__).resolve().parent / "src").is_dir()
    else Path(__file__).resolve().parents[1]))
ORCH = "src/kernel_optimizer/control/orchestrator.py"

VARIANTS = {
    # The state before this fix: REWRITE_PRODUCED is not read at all on resume.
    "no_inflight_restore": ([(
        """            elif ev.type == "REWRITE_PRODUCED":""",
        """            elif ev.type == "REWRITE_PRODUCED" and False:"""),
    ], ["test_declarations_from_an_unreconciled_round_are_restored"]),
    # Restores everything, including candidates already reconciled -- which puts the NEXT round back
    # into "scored against the previous round's predictions", through the resume path.
    "restore_even_the_reconciled": ([(
            """            if cand_id in reconciled_cands or not family_id:""",
            """            if not family_id:"""),
    ], ["test_an_already_reconciled_candidate_is_not_restored",
        "test_a_pre_fix_pooled_entry_retires_its_familys_declarations"]),
    # Loses the candidate attribution, dropping the round straight back into the pooled defect.
    "restore_without_candidate_id": ([(
            """                                          expectation=e, candidate_id=cand_id))""",
            """                                          expectation=e))"""),
    ], ["test_declarations_from_an_unreconciled_round_are_restored"]),
    # A rewrite that declared nothing still fires the restore event, making the event meaningless.
    "event_even_when_nothing_restored": ([(
            """        if n_inflight:""",
            """        if True:"""),
    ], ["test_a_rewrite_that_declared_nothing_restores_nothing"]),
}

def main() -> int:
    bad = 0
    for name, (subs, claimed) in VARIANTS.items():
        tmp = Path(tempfile.mkdtemp(prefix="rr-"))
        try:
            tree = tmp / "t"
            shutil.copytree(ROOT, tree, ignore=shutil.ignore_patterns(
                "runs*", ".git", "__pycache__", "*.pyc", ".tmp-arms", "external*"))
            p = tree / ORCH
            src = p.read_text(encoding="utf-8")
            ok = True
            for old, new in subs:
                if src.count(old) != 1:
                    print("  %-34s SKIPPED -- anchor occurs %d times" % (name, src.count(old)))
                    bad += 1; ok = False; break
                src = src.replace(old, new)
            if not ok:
                continue
            p.write_text(src, encoding="utf-8")
            flipped = []
            for t in claimed:
                r = subprocess.run(
                    [sys.executable, "-B", "-m", "pytest", "tests/test_resume_restore.py", "-q",
                     "--no-header", "-p", "no:cacheprovider", "-k", t[5:45]],
                    cwd=tree, capture_output=True,
                    env={**os.environ, "PYTHONPATH": "src", "PYTHONDONTWRITEBYTECODE": "1"})
                if r.returncode != 0:
                    flipped.append(t)
            if flipped:
                print("  %-34s CAUGHT by %s" % (name, ", ".join(x[5:55] for x in flipped)))
            else:
                print("  %-34s *** NOT CAUGHT *** (claimed %s)" % (name, claimed))
                bad += 1
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    return 1 if bad else 0

if __name__ == "__main__":
    print("VARIANTS of the S2d in-flight declaration restore:")
    raise SystemExit(main())
