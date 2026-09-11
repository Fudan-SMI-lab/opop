"""When does a run reach its first FAMILY_ROUND_RECORDED -- the checkpoint where G27's `conversion`
and S2d's ledger first have content?

Loop C cannot start until `_pipeline_batch` finishes EVERY seed candidate: the call is
`self._pipeline_batch(list(self.runs))` and the Loop C `while True:` sits after it. So the projection
is per-candidate cost times candidates remaining, and the cost has to be measured on the candidates
THIS run already finished, not borrowed from another box.

THE BOUNDARY IS `STEP_DONE`, NOT `BOTTLENECK_CLASSIFIED`. Read from box 2's live log, one candidate
looks like:

    ...  TUNING_DONE / STATS_DONE / BOTTLENECK_CLASSIFIED / DIMENSION_STATE
    ...  BOTTLENECK_REPORTED          <- the analyst agent
    ...  SPACE_PUBLISHED / SPACE_EXPANDED   <- K expansion, a SECOND space
    ...  SPACE_PRESCREENED
    ...  TUNING_DONE / STATS_DONE / BOTTLENECK_CLASSIFIED / DIMENSION_STATE   <- classified AGAIN
    ...  BOTTLENECK_REPORTED
    ...  STEP_DONE                    <- only NOW is the candidate finished

So `BOTTLENECK_CLASSIFIED` fires TWICE per candidate when the space is expanded, and counting it as
"candidates done" overcounts: box 2's 4 classifications were 2 candidates, not 4 -- I reported "3 of
4 classified" from that count and it was wrong. It also makes each candidate look half as expensive
as it is, because the first classification lands before the re-tune.

`first_work` must likewise exclude registration: `_generate_seeds` registers all four candidates at
t~0, so keying on CANDIDATE_REGISTERED reports every unstarted candidate as "in flight since the
beginning" and subtracts that phantom time from the ETA, collapsing a 2-candidate backlog to zero.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Events that name a candidate without work having begun on it.
_REGISTRATION = {"CANDIDATE_REGISTERED", "AGENT_CALL_STARTED", "AGENT_CALL_FINISHED"}


def _candidate_of(p: dict) -> str | None:
    for path in (("candidate_id",), ("trial", "candidate_id"), ("space", "candidate_id"),
                 ("candidate", "candidate_id")):
        cur: object = p
        for k in path:
            cur = (cur or {}).get(k) if isinstance(cur, dict) else None
        if isinstance(cur, str) and cur:
            return cur
    return None


def read(rd: Path, n_seeds: int) -> dict:
    evs = []
    with (rd / "events.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                e = json.loads(line)
            except ValueError:
                continue
            if isinstance(e.get("ts"), (int, float)):
                evs.append(e)
    if not evs:
        raise SystemExit("no timestamped events in %s" % rd)
    t0, now = evs[0]["ts"], evs[-1]["ts"]

    first_work: dict[str, float] = {}
    registered: list[str] = []
    # A candidate is DONE at the STEP_DONE that follows its last BOTTLENECK_REPORTED. STEP_DONE
    # carries no candidate_id, so it is attributed to whichever candidate was most recently active.
    done_at: dict[str, float] = {}
    classifications: dict[str, int] = {}
    expansions: dict[str, int] = {}
    last_cand: str | None = None
    # The single slowest trial, so a one-off tail can be separated from the typical cost. Measured:
    # box 1's first candidate cost 2.08 h of which one 1800 s timeout was 30 min.
    worst_trial = 0.0
    worst_kind = ""
    prev_ts = t0
    for e in evs:
        t = e.get("type")
        p = e.get("payload") or {}
        cid = _candidate_of(p)
        if t == "TRIAL_DONE":
            gap = e["ts"] - prev_ts
            if gap > worst_trial:
                worst_trial = gap
                worst_kind = ((p.get("trial") or p).get("failure_kind")) or "ok"
        prev_ts = e["ts"]
        if cid:
            if t == "CANDIDATE_REGISTERED":
                registered.append(cid)
            if t not in _REGISTRATION:
                first_work.setdefault(cid, e["ts"])
                last_cand = cid
            if t == "BOTTLENECK_CLASSIFIED":
                classifications[cid] = classifications.get(cid, 0) + 1
            if t == "SPACE_EXPANDED":
                expansions[cid] = expansions.get(cid, 0) + 1
        elif t == "STEP_DONE" and last_cand and last_cand in first_work:
            # Only a STEP_DONE that follows real per-candidate work closes a candidate; the
            # baseline and seed-generation steps also emit one, and those have no last_cand yet.
            if last_cand not in done_at:
                done_at[last_cand] = e["ts"]

    durations = [done_at[c] - first_work[c] for c in done_at if c in first_work]
    return {"elapsed_h": (now - t0) / 3600.0,
            "registered": len(registered),
            "n_seeds": n_seeds,
            "done": len(done_at),
            "classifications": sum(classifications.values()),
            "expansions": sum(expansions.values()),
            "durations_h": sorted(d / 3600.0 for d in durations),
            "in_flight": {c: (now - first_work[c]) / 3600.0
                          for c in first_work if c not in done_at},
            "never_started": n_seeds - len(first_work),
            "worst_trial_h": worst_trial / 3600.0,
            "worst_trial_kind": worst_kind,
            "rounds": sum(1 for e in evs if e.get("type") == "FAMILY_ROUND_RECORDED")}


def report(lbl: str, r: dict) -> None:
    # POOLED counts, and the pair does NOT satisfy `classifications == expansions + 1`: that
    # invariant is PER CANDIDATE, and pooling adds the in-flight candidate's half-finished state
    # (classified once, not yet expanded, or the reverse). Box 1 read "5 classifications over 3
    # expansions" while all three of its per-candidate rows were exactly cls == exp + 1. Labelled
    # as pooled so the pair is not read as a violated invariant -- which is how I first read it.
    print("%-10s elapsed %.2f h   candidates DONE %d of %d   (%d classifications over %d "
          "expansions, POOLED -- the cls == exp + 1 invariant is per candidate, and pooling "
          "includes the in-flight one)" % (
              lbl, r["elapsed_h"], r["done"], r["n_seeds"], r["classifications"],
              r["expansions"]))
    if r["rounds"]:
        print("%-10s   FAMILY_ROUND_RECORDED already present: %d. Loop C has started." % (
            "", r["rounds"]))
        return
    for d in r["durations_h"]:
        print("%-10s   finished candidate: %.2f h" % ("", d))
    for c, h in sorted(r["in_flight"].items(), key=lambda kv: -kv[1]):
        print("%-10s   in flight: %s for %.2f h" % ("", c[-8:], h))
    if r["never_started"] > 0:
        print("%-10s   not started at all: %d" % ("", r["never_started"]))

    remaining = r["n_seeds"] - r["done"]
    if remaining <= 0:
        print("%-10s => seed pipeline DONE; Loop C starts at the next round check" % "")
        return
    ds = r["durations_h"]
    burned = sum(r["in_flight"].values())
    if len(ds) < 2:
        print("%-10s => NOT PROJECTABLE from n=%d finished candidate(s): one duration is a point, "
              "not a rate" % ("", len(ds)))
        if ds:
            # A single duration must not be extrapolated if a one-off tail dominates it. Box 1's
            # first candidate took 2.08 h, but its own per-trial MEDIAN is 25.7 s against box 2's
            # 30.6 s -- it is the FASTER box, and 30 of those 125 minutes are one 1800 s timeout.
            # Multiplying that duration by the remaining candidates charges every future candidate
            # for a timeout that happened once, which is the same tail-versus-body error
            # `check_arm_search_parity.py` exists to avoid.
            naive = ds[0] * remaining - burned
            print("%-10s    naive: if %.2f h were typical, >= %.2f h more" % ("", ds[0], naive))
            if r["worst_trial_h"] and r["worst_trial_h"] > 0.15 * ds[0]:
                adj = (ds[0] - r["worst_trial_h"]) * remaining - burned
                print("%-10s    BUT its slowest single trial was %.2f h (%.0f%% of that "
                      "candidate's whole cost, kind=%s). A one-off tail must not be charged to "
                      "every future candidate: net of it, ~%.2f h more" % (
                          "", r["worst_trial_h"], 100 * r["worst_trial_h"] / ds[0],
                          r["worst_trial_kind"], max(0.0, adj)))
                print("%-10s    Report the RANGE %.2f-%.2f h, not either endpoint alone." % (
                    "", max(0.0, adj), naive))
        return
    med = ds[len(ds) // 2]
    eta = max(0.0, med * remaining - burned)
    print("%-10s => median %.2f h/candidate (n=%d), %.2f h burned on in-flight => first "
          "FAMILY_ROUND_RECORDED in ~%.2f h (at %.2f h elapsed)" % (
              "", med, len(ds), burned, eta, r["elapsed_h"] + eta))


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__)
        return 2
    n_seeds = 4
    args = list(argv)
    if args[0] == "--seeds":
        n_seeds = int(args[1])
        args = args[2:]
    for lbl, rd in zip(args[0::2], args[1::2]):
        report(lbl, read(Path(rd), n_seeds))
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
