"""S7 feasibility, replayed on real trial ORDERS before any code is written.

THE QUESTION S7 DEPENDS ON, and it is not "is a wall findable" -- 2e already answers that AFTER tuning
ends. S7 moves the wall search INSIDE the tuning loop, so what has to be true is:

  (a) a probe-worthy wall exists EARLY enough that acting on it still has budget left, and
  (b) the wall's IDENTITY is stable enough that acting on it early is not acting on noise.

Both are properties of the trial ORDER, which the event log preserves, so both are answerable offline
at zero GPU cost. Replayed here at 25/50/75/100% of each candidate's own trial sequence, running the
SHIPPING `find_walls` + `select_for_probing` on the prefix -- not a reimplementation.

WHAT WOULD SINK S7. Three findings, any one of which is a reason not to spend a 12h pair:

  * walls appear only in the last quarter          => no budget left to act on them
  * the WALLED KNOB changes between prefixes       => early guidance points at a different knob than
                                                      the one that turns out to matter, and enqueueing
                                                      on it spends trials on noise
  * the walls vanish as sampling widens            => structurally expected (a wall is a refused value
                                                      OUTSIDE the measured range, and the range grows),
                                                      but if it happens on most candidates then the
                                                      early signal is an artifact of not having looked
                                                      yet rather than a property of the space

The third is the one I expect and the one whose SIZE matters: `cand-948343ba` was already observed
going 2 walls at 25% -> 0 at 75% -> 1 at 100% with a DIFFERENT knob. That is one candidate; this
replays every candidate in the local corpus so the rate is measured rather than anecdotal.

WHICH SIGNAL THE REPLAY CAN ACTUALLY USE HERE, and this had to be found by running it. The HARD wall
produces ZERO walls at every prefix on all 151 local candidates -- not because it is early or late, but
because every locally backed-up run predates the shared-memory screen, so `infeasible_shared_memory`
never appears and `find_walls`' only input is absent. So the hard-wall version of this question is
UNANSWERABLE on this host, and the "a wall is available by the halfway point" figure recorded earlier
came from the three-arm run's logs, which are not on this machine either.

The SOFT wall (item 2) needs no refusals -- it reads `n_spills` from each trial's profile -- and it does
fire here (50 walls over 22 candidates). So the replay is run for BOTH criteria and reports them apart:
the soft arm answers (a) and (b) on real data, and the hard arm reports its own absence rather than a
zero that could be mistaken for "walls are never early".
"""
from __future__ import annotations

import argparse
import statistics
import sys
from collections import defaultdict
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src"
sys.path.insert(0, str(SRC))

from kernel_optimizer.evaluation import soft_wall as sw  # noqa: E402
from kernel_optimizer.evaluation import wall_attribution as wa  # noqa: E402
from kernel_optimizer.models.reports import ParamStat, TuningStats  # noqa: E402
from kernel_optimizer.store import read as store_read  # noqa: E402

PREFIXES = (0.25, 0.50, 0.75, 1.00)


def _stats_from(trials: list[dict], cid: str) -> TuningStats:
    """`param_stats` with the per-choice MEDIAN table, rebuilt from a PREFIX of the trials.

    Rebuilt rather than taken from `STATS_DONE`, and that is the whole point: the recorded stats are
    computed once at the END of tuning, so using them would tell the replay what the sampler could not
    have known yet -- the wall would look available at 25% because its evidence came from trial 80.
    """
    by_knob: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for t in trials:
        if t.get("status") != "complete":
            continue
        ms = store_read.latency_ms_of(t)
        if ms is None:
            continue
        for knob, raw in (((t.get("params") or {}).get("values")) or {}).items():
            by_knob[str(knob)][str(raw)].append(ms)
    out = []
    for knob, table in sorted(by_knob.items()):
        med = {v: statistics.median(xs) for v, xs in table.items()}
        if not med:
            continue
        best = min(med, key=lambda v: med[v])
        out.append(ParamStat(name=knob, best_value=best, latency_by_value=med))
    return TuningStats(candidate_id=cid, space_id="sp",
                       n_complete=sum(1 for t in trials if t.get("status") == "complete"),
                       n_fail=sum(1 for t in trials if t.get("status") != "complete"),
                       param_stats=out)


def _walls_at(trials: list[dict], cid: str) -> tuple[int, list[str]]:
    """`(probe-worthy count, knob names)` from the SHIPPING HARD-wall functions on this prefix."""
    refused = [(t.get("params") or {}).get("values") or {} for t in trials
               if t.get("failure_kind") == "infeasible_shared_memory" and t.get("params")]
    if not refused:
        return 0, []
    walls = wa.find_walls(_stats_from(trials, cid), refused)
    worthy, _ = wa.select_for_probing(walls, 8)
    return len(worthy), [w.param for w in worthy]


def _soft_walls_at(trials: list[dict], cid: str) -> tuple[int, list[str]]:
    """`(count, knob names)` from the SHIPPING soft-wall function on this prefix.

    The soft criterion is the one this corpus can actually exercise: it reads `n_spills` off each
    trial's profile and needs no refusal record. Same prefix discipline -- the stats handed in are
    rebuilt from the prefix, so the scan sees only what the sampler would have seen.
    """
    scan = sw.find_soft_walls(_stats_from(trials, cid), trials)
    return len(scan.walls), [w.param for w in scan.walls]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    out: list[str] = []

    def say(s: str = "") -> None:
        out.append(s)

    rows: list[tuple[str, str, int, list[tuple[int, list[str]]]]] = []
    tot: dict[str, dict[str, int]] = {"hard": defaultdict(int), "soft": defaultdict(int)}
    first_seen: dict[str, list[float]] = {"hard": [], "soft": []}
    n_cands = 0
    n_refusing = 0

    for rd in args.runs:
        try:
            events = store_read.read_events(rd)
        except Exception as exc:  # noqa: BLE001
            say(f"!! {rd}: {type(exc).__name__}: {exc}")
            continue
        # Trials IN ORDER: the event log's order is the order the tuner asked, which is the only thing
        # that makes this replay meaningful.
        seq: dict[str, list[dict]] = defaultdict(list)
        for ev in events:
            if ev.get("type") != "TRIAL_DONE":
                continue
            t = store_read.trial_of(ev)
            cid = str(t.get("candidate_id") or "")
            if cid:
                seq[cid].append(t)

        for cid, trials in sorted(seq.items()):
            if len(trials) < 12:
                continue
            n_cands += 1
            if any(t.get("failure_kind") == "infeasible_shared_memory" for t in trials):
                n_refusing += 1
            for kind, fn in (("hard", _walls_at), ("soft", _soft_walls_at)):
                c = tot[kind]
                per_prefix: list[tuple[int, list[str]]] = []
                for frac in PREFIXES:
                    k = max(1, int(round(len(trials) * frac)))
                    per_prefix.append(fn(trials[:k], cid))
                if not any(n for n, _ in per_prefix):
                    c["never_any_wall"] += 1
                    continue
                c["ever_a_wall"] += 1
                if kind == "soft":
                    rows.append((Path(rd).name, cid, len(trials), per_prefix))
                idx = next(i for i, (n, _) in enumerate(per_prefix) if n)
                first_seen[kind].append(PREFIXES[idx])
                if idx <= 1:
                    c["wall_by_halfway"] += 1
                early = set(per_prefix[idx][1])
                final = set(per_prefix[-1][1])
                if early & final:
                    c["knob_survives"] += 1
                elif final:
                    c["knob_changed"] += 1
                else:
                    c["wall_vanished"] += 1

    say("=" * 100)
    say("S7 FEASIBILITY -- replaying the SHIPPING wall finders on PREFIXES of each candidate's own")
    say("trial order (25 / 50 / 75 / 100%). Zero GPU.")
    say("=" * 100)
    say("The per-choice median table is rebuilt FROM THE PREFIX, not taken from STATS_DONE: the")
    say("recorded stats are computed at the end of tuning, so using them would let the replay know")
    say("what the sampler could not have known yet.")
    say()
    say(f"candidates with >= 12 trials: {n_cands}")
    say(f"  ...of which any trial was refused for shared memory: {n_refusing}")
    say()
    say("HARD WALL (shared memory):")
    if not n_refusing:
        say(f"  UNANSWERABLE ON THIS CORPUS -- 0 of {n_cands} candidates have a single refusal")
        say("  recorded, because every locally backed-up run predates the shared-memory screen. The")
        say("  hard wall's ONLY input is the refused parameter sets, so its 0 here is the absence of")
        say("  input and must NOT be read as 'walls are never early'. The earlier figure (a")
        say("  probe-worthy wall by the halfway point, 40 trials left) came from the three-arm run's")
        say("  logs, which are not on this host either -- so it stands unverified here and has to be")
        say("  re-measured on a run that carries refusals.")
    else:
        h = tot["hard"]
        say(f"  ever had a probe-worthy wall: {h['ever_a_wall']}")
        if h["ever_a_wall"]:
            say(f"  first by the halfway point:   {h['wall_by_halfway']}/{h['ever_a_wall']}")
            say(f"  early knob still walled:      {h['knob_survives']}/{h['ever_a_wall']}, "
                f"changed {h['knob_changed']}, vanished {h['wall_vanished']}")
    say()
    say("SOFT WALL (n_spills) -- the criterion this corpus CAN exercise, since it needs no refusal:")
    s = tot["soft"]
    e = s["ever_a_wall"]
    say(f"  ever had a wall at any prefix: {e}   (never: {s['never_any_wall']})")
    if e:
        say()
        say("  (a) IS IT EARLY ENOUGH TO ACT ON?")
        say(f"      first wall by the halfway point: {s['wall_by_halfway']}/{e} = "
            f"{100.0 * s['wall_by_halfway'] / e:.0f}%")
        if first_seen["soft"]:
            say(f"      earliest prefix holding a wall: median "
                f"{statistics.median(first_seen['soft']):.2f} of the trial sequence")
        say()
        say("  (b) IS THE WALLED KNOB STABLE?")
        say(f"      the early knob is still walled at 100%: {s['knob_survives']}/{e} = "
            f"{100.0 * s['knob_survives'] / e:.0f}%")
        say(f"      a DIFFERENT knob is walled at 100%:     {s['knob_changed']}/{e}")
        say(f"      no wall left at 100% (vanished):        {s['wall_vanished']}/{e}")
        say()
        say("      For the SOFT wall, vanishing has a different cause than for the hard one: the soft")
        say("      criterion needs a monotone spill curve over >= 3 measured values, and an early")
        say("      prefix has fewer values and so a curve that is more easily monotone by accident.")
        say("      So a high vanish rate here reads as 'the early curve was underdetermined', which is")
        say("      exactly the failure mode S7's guidance would inherit.")
    if rows:
        say()
        say("PER CANDIDATE (soft wall; count:knobs at each prefix, first 25 rows):")
        say("-" * 100)
        say(f"{'run':24s} {'candidate':15s} {'trials':>7s}   {'25%':>12s} {'50%':>12s} "
            f"{'75%':>12s} {'100%':>12s}")
        say("-" * 100)
        for run, cid, n, per in rows[:25]:
            cells = []
            for cnt, knobs in per:
                cells.append(f"{cnt}:{','.join(k[:10] for k in knobs[:1])}" if cnt else "-")
            say(f"{run[:24]:24s} {cid[:15]:15s} {n:7d}   " + " ".join(f"{c:>12s}" for c in cells))
        say("-" * 100)
    say()
    say("HOW TO READ THIS FOR THE S7 DECISION:")
    say("  (a) low  => no budget left to act on the signal        => S7 has nothing to work with")
    say("  (b) low  => the early knob is not the one that matters => S7's guidance is noise")
    say("  both high => the mechanism has room; the OPEN question stays the one this cannot answer,")
    say("  namely whether slope is a good allocation prior at all (measured rho <= 0.24 against")
    say("  remaining gain, below the 0.43/0.52 incumbents), which is what the 12h pair is for.")

    text = "\n".join(out) + "\n"
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        sys.stdout.buffer.write(text.encode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
