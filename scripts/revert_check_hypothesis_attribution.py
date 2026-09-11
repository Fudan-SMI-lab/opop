"""Do the hypothesis-attribution tests fail on the version that marked the analyst's whole list?

The defect is permanent and invisible: `failed_hypotheses` is journalled, replayed, and read by the
rewriter as "already tried, did NOT help", so an idea nobody implemented is retired for the rest of
the run on the strength of a sibling candidate's failure.
"""
import os, shutil, subprocess, sys, tempfile
from pathlib import Path

ROOT = Path(os.environ.get("KOPT_REPO_ROOT") or (
    Path(__file__).resolve().parent if (Path(__file__).resolve().parent / "src").is_dir()
    else Path(__file__).resolve().parents[1]))
ORCH = "src/kernel_optimizer/control/orchestrator.py"
TESTS = "tests/test_resume_restore.py"

VARIANTS = {
    # The original defect: every hypothesis in the report is marked failed.
    "mark_the_whole_report": ([(
        """    tried = [{"id": h.id, "change": h.change, "round": round_no}
             for h in hypotheses if not attempted or h.id in attempted]""",
        """    tried = [{"id": h.id, "change": h.change, "round": round_no}
             for h in hypotheses]"""),
    ], ["test_a_failed_round_marks_only_the_hypotheses_it_implemented",
        "test_both_candidates_implementing_the_same_hypothesis_retires_only_that_one"]),
    # The over-correction: an empty attempt set marks NOTHING, losing a real round's evidence.
    "empty_attempts_mark_nothing": ([(
        """             for h in hypotheses if not attempted or h.id in attempted]""",
        """             for h in hypotheses if h.id in attempted]"""),
    ], ["test_no_declared_hypothesis_id_leaves_the_list_untouched"]),
    # The LIVE path never records an attempt. NO CASE FLIPS, and the reason is worth recording rather
    # than papering over: this line is in `_do_rewrite`, which needs a live rewriter agent, while
    # `test_the_rounds_attempted_hypotheses_are_restored...` drives the RESUME path -- a separate
    # reader of the same event. The two populate `round_hypotheses` independently, so a test of one
    # cannot see a defect in the other, and claiming otherwise would be the "a test that copies the
    # loop does not test it" shape. The live path is covered instead by
    # `test_the_live_and_resume_paths_agree_on_the_attempt_set`, which asserts they produce the same
    # set from the same events -- the only check available without an agent.
    "never_record_the_attempt_live": ([(
        """            self.round_hypotheses.setdefault(family_id, set()).add(rw.hypothesis_id)""",
        """            pass"""),
    ], []),
    # Retirement keyed on HYPOTHESES_FAILED instead of end-of-round: an IMPROVING round emits none,
    # so its attempts leak into the next round.
    "retire_on_hypotheses_failed": ([(
        """                self.round_hypotheses.pop(ev.payload["family_id"], None)
            elif ev.type == "FAMILY_ROUND_NOT_EVALUATED":""",
        """                pass
            elif ev.type == "FAMILY_ROUND_NOT_EVALUATED":"""),
    ], ["test_a_closed_round_does_not_leak_its_attempts_into_the_next"]),
}


def main() -> int:
    bad = 0
    for name, (subs, claimed) in VARIANTS.items():
        tmp = Path(tempfile.mkdtemp(prefix="rh-"))
        try:
            tree = tmp / "t"
            shutil.copytree(ROOT, tree, ignore=shutil.ignore_patterns(
                "runs*", ".git", "__pycache__", "*.pyc", ".tmp-arms", "external*"))
            p = tree / ORCH
            src = p.read_text(encoding="utf-8")
            ok = True
            for old, new in subs:
                if src.count(old) != 1:
                    print("  %-32s SKIPPED -- anchor occurs %d times" % (name, src.count(old)))
                    bad += 1; ok = False; break
                src = src.replace(old, new)
            if not ok:
                continue
            p.write_text(src, encoding="utf-8")
            flipped = []
            for t in claimed:
                r = subprocess.run(
                    [sys.executable, "-B", "-m", "pytest", TESTS, "-q", "--no-header",
                     "-p", "no:cacheprovider", "-k", t[5:48]],
                    cwd=tree, capture_output=True,
                    env={**os.environ, "PYTHONPATH": "src", "PYTHONDONTWRITEBYTECODE": "1"})
                if r.returncode != 0:
                    flipped.append(t)
            if not claimed:
                # Names no case on purpose: the rationale in VARIANTS says why none can reach it.
                print("  %-32s %s" % (name, ("UNCLAIMED-CATCH by %s" % ",".join(flipped))
                                      if flipped else "NOEVID (names no case; see rationale)"))
            elif flipped:
                print("  %-32s CAUGHT by %s" % (name, ", ".join(x[5:58] for x in flipped)))
            else:
                print("  %-32s *** NOT CAUGHT *** (claimed %s)" % (name, claimed))
                bad += 1
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    return 1 if bad else 0


if __name__ == "__main__":
    print("VARIANTS of per-hypothesis failure attribution:")
    raise SystemExit(main())
