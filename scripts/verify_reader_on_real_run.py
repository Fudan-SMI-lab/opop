"""Verify the authoritative reader against the REAL run whose shape defeated the old readers.

A test with synthetic fixtures proves the reader is self-consistent. It does not prove the fixtures
match reality -- and the four G21 failures were all about reality differing from an assumption. So
this runs the new reader against run-l3-48-20260909-115701 on box 1, where the old reader reported
"no complete trial" on a run holding 680 TRIAL_DONE events.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kernel_optimizer.store.read import (  # noqa: E402
    EmptyResult,
    best_trial,
    candidate_ids,
    candidate_source,
    completed_trials,
    events_of_type,
    read_events,
    trials_with_latency,
)


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: verify_reader_on_real_run.py <run_dir> [<run_dir> ...]")
        return 2
    failures = 0
    for run in sys.argv[1:]:
        print("=" * 78)
        print(run)
        try:
            ev = read_events(run)
            done = events_of_type(run, "TRIAL_DONE")
            comp = completed_trials(run)
            withlat = trials_with_latency(run)
            best = best_trial(run)
            cids = candidate_ids(run)
            src = candidate_source(run, best["candidate_id"])
            print("  events            %d" % len(ev))
            print("  TRIAL_DONE        %d" % len(done))
            print("  complete          %d" % len(comp))
            print("  with latency      %d" % len(withlat))
            print("  best              %.4f ms  (%s)" % (best["latency_ms_value"],
                                                          best["candidate_id"]))
            print("  candidates        %d" % len(cids))
            print("  best source       %d bytes" % len(src))
            # The old reader's answer on this same run, for contrast.
            wrong = [e for e in done if (e.get("payload") or {}).get("status") == "complete"]
            print("  --- old reader (payload.status): %d complete  <-- the silent failure"
                  % len(wrong))
            if len(comp) == 0:
                print("  FAIL: read zero complete trials")
                failures += 1
        except EmptyResult as exc:
            # On a real finished run this is itself the finding: either the run genuinely has
            # nothing, or the reader is wrong. Either way it is now LOUD.
            print("  EmptyResult: %s" % str(exc)[:300])
            failures += 1
        except Exception as exc:  # noqa: BLE001
            print("  %s: %s" % (type(exc).__name__, str(exc)[:300]))
            failures += 1
    print()
    if failures:
        print("VERDICT: %d run(s) could not be read -- investigate before trusting the reader." % failures)
        return 1
    print("VERDICT: OK -- the reader handles every real run given, and each raises rather than")
    print("  returning an empty container when something is missing.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
