"""Report generation from the event log (proves the trace is complete)."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from kernel_optimizer.evaluation.conversion_report import conversion_lines
from kernel_optimizer.models.core import latency_cell
from kernel_optimizer.reporting.accuracy_report import accuracy_lines
from kernel_optimizer.reporting.completeness import completeness_lines
from kernel_optimizer.reporting.wall_report import wall_lines
from kernel_optimizer.store.run_store import RunStore


def _robust_ms(lat: dict | None) -> float | None:
    """The statistic that DECIDED the run, read from a raw event payload.

    The dict mirror of `LatencyStats.robust_ms` (models/core.py): median when present, else the
    mean. Reporting must rank by the same statistic the orchestrator ranked by, or a report
    names a different winner than the run chose. That divergence was harmless only while no
    timing path produced a median at all; once `capture_timing_samples` gives every path one,
    a mean-based reconstruction would start disagreeing with the live summary.
    """
    if not lat:
        return None
    med = lat.get("median")
    if med is not None and med > 0:
        return float(med)
    mean = lat.get("mean")
    return float(mean) if mean is not None else None


def _reconstruct_summary(events, candidates: dict, trials: list) -> dict:
    """Rebuild the RUN_FINISHED summary shape from events, for a run still in flight.

    Only fields the events actually support. In particular NO final_reeval_ms and NO
    honest_verdict: both require the fresh-process re-eval that runs at finalize, and
    tuned_ms is systematically optimistic against it by 1.5-6.7%, so synthesising them
    from trial data would manufacture the very number the reeval-gap rule says not to
    trust.
    """
    best_per_cand: dict[str, dict] = {}
    for t in trials:
        if t.get("status") != "complete" or not t.get("latency_ms"):
            continue
        cid = t["candidate_id"]
        cur = best_per_cand.get(cid)
        cur_ms = _robust_ms(cur.get("latency_ms")) if cur else None
        t_ms = _robust_ms(t.get("latency_ms"))
        if t_ms is None:
            continue
        if cur_ms is None or t_ms < cur_ms:
            best_per_cand[cid] = t

    rounds: dict[str, int] = {}
    history: dict[str, list] = {}
    # Rounds that spent budget without evaluating anything. Kept separate from `history` on
    # purpose: a reader who sees a flat history concludes the structure has no headroom left,
    # which is exactly the wrong reading when the rewrite never ran. Surfaced per family so a
    # `frozen_converged` verdict can be checked against it -- in
    # run-l1-42-20260907-193510 one of the two converged families was this case.
    not_evaluated: dict[str, int] = {}
    for e in events:
        if e.type == "FAMILY_ROUND_RECORDED":
            fid = e.payload["family_id"]
            rounds[fid] = rounds.get(fid, 0) + 1
            history.setdefault(fid, []).append(e.payload.get("best_ms"))
        elif e.type == "FAMILY_ROUND_NOT_EVALUATED":
            fid = e.payload["family_id"]
            rounds[fid] = rounds.get(fid, 0) + 1
            not_evaluated[fid] = not_evaluated.get(fid, 0) + 1

    families: dict[str, dict] = {}
    for cid, cand in candidates.items():
        fid = cand["family_id"]
        fam = families.setdefault(fid, {
            "status": "active (run in progress)", "best_ms": None,
            "rewrite_rounds_used": rounds.get(fid, 0),
            "rounds_not_evaluated": not_evaluated.get(fid, 0),
            "history": history.get(fid, []), "members": [],
        })
        fam["members"].append({
            "id": cid, "origin": cand["origin"], "parents": cand.get("parent_ids") or [],
            "approach": cand.get("approach_summary") or "",
        })
        t = best_per_cand.get(cid)
        t_ms = _robust_ms(t.get("latency_ms")) if t else None
        if t_ms is not None and (fam["best_ms"] is None or t_ms < fam["best_ms"]):
            fam["best_ms"] = t_ms

    out: dict = {"families": families, "best": None}
    if best_per_cand:
        winner = min(best_per_cand.values(), key=lambda t: _robust_ms(t["latency_ms"]))
        cand = candidates.get(winner["candidate_id"], {})
        out["best"] = {
            "candidate_id": winner["candidate_id"],
            "family_id": cand.get("family_id", "?"),
            "tuned_ms": _robust_ms(winner["latency_ms"]),
            "params": winner["params"],
        }
    return out


def _plausibility_bound(events: list) -> dict | None:
    """The derived speedup ceiling from the log, or None if none was derived.

    Reads the event rather than recomputing, so the report states the bound the RUN actually
    flagged against. Recomputing here from the task cost and calibration would silently diverge
    the moment `plausibility.py` changed -- the report would then justify a flag with a threshold
    that was never applied, which is worse than no explanation. Recorded as
    `a-correct-reader-does-not-prevent-the-guess`.
    """
    evs = [e for e in events if e.type == "PLAUSIBILITY_BOUND"]
    if not evs:
        return None
    return (evs[-1].payload or {}).get("bound") or None


def _search_budget_lines(summary: dict | None, trials: list,
                         dead_events: list, provisional: bool) -> list[str]:
    """How much of the intended search actually ran, stated next to the headline number.

    A latency result means something different when the search that produced it used two
    thirds of its rewrite budget than when it used one sixth, and a reader quoting the
    speedup has no way to tell from the number alone. Two real cases motivated this:

    - `run-l3-21-20260905-071312` stopped at 2.05h of 12h with 2 of 6 rewrite rounds used,
      because two families with no correct candidate filled both active slots and ended
      the loop (docs/finding-run-stops-with-budget-unused.md). Its winning family was
      still improving 24% per round when frozen.
    - The same run spent 80 of 374 trials on a kernel that never launched
      (docs/finding-optimization-behind-a-dead-mode-branch.md).

    Both facts belong beside the verdict, not buried in a per-family section further down.
    Reports only what the events say; no interpretation of whether the result is good.
    """
    if not summary:
        return []
    families = summary.get("families") or {}
    if not isinstance(families, dict):
        return []
    used = sum(f.get("rewrite_rounds_used") or 0 for f in families.values())
    # rewrite_rounds_used is absent from older runs' summaries: 0 there means unrecorded,
    # so do not present a budget fraction we cannot substantiate.
    recorded = any(f.get("rewrite_rounds_used") is not None for f in families.values())
    empty = [fid for fid, f in families.items() if f.get("best_ms") is None]

    lines = ["### Search budget actually used\n"]
    substantive = False
    if recorded and families:
        lines.append(f"- rewrite rounds used: **{used}** across {len(families)} families "
                     f"({[f.get('rewrite_rounds_used') for f in families.values()]})")
        substantive = True
    if empty:
        lines.append(f"- families with **no correct candidate**: {len(empty)} of "
                     f"{len(families)} — {', '.join(f'`{f}`' for f in empty)}")
        substantive = True
    eh = summary.get("elapsed_hours")
    if eh is not None:
        # Never carries the section alone: elapsed is already in the report footer, and a
        # heading with nothing but a duration under it says less than no heading.
        lines.append(f"- elapsed: **{eh} h**")
    if dead_events:
        wasted = 0
        for ev in dead_events:
            wasted += ev.get("n_trials_measured") or 0
        names = sorted({k for ev in dead_events
                        for k in (ev.get("never_launched") or [])})
        lines.append(
            f"- ⚠ **{wasted} of {len(trials)} trials measured a candidate carrying a "
            f"kernel that never launched** ({', '.join(f'`{n}`' for n in names)}): those "
            f"budgets timed a fallback path, not the advertised optimization")
        substantive = True
    if not provisional and recorded and empty and used < len(families):
        lines.append(
            "- ⚠ this run may have stopped before its rewrite budget was spent: a family "
            "with no correct candidate is frozen without counting as progress, so enough "
            "of them ends the outer loop early "
            "(`docs/finding-run-stops-with-budget-unused.md`). Read the speedup above as "
            "the product of THIS much search, not of a converged one.")
    lines.append("")
    return lines if substantive else []


def _attribution_lines(best: dict, trials: list) -> list[str]:
    """Which GPU kernels the winning configuration actually launched.

    A candidate is free to hand part of the computation back to PyTorch and still win on
    latency -- and on L3:43 the run's fastest candidate did exactly that: `cand-60fdcae9`
    (8.06 ms) launches only `_fused_qkv_projection` and `_head_layout_projection`, so the
    attention core is torch's `scaled_dot_product_attention`. Fusing c_attn + QKV packing
    into one Triton GEMM is a real result, but it is not an attention kernel, and the best
    FULLY hand-written candidate in that run was `cand-9f6af7bd` at 9.43 ms. Delegation is
    also not a family-level property: that family's five members flip between hand-written
    and delegating, and the delegating one won.

    So this prints the launched kernel names as a FACT and does not classify them. A
    keyword rule ("does one of these look like attention?") is task-specific by
    construction -- it would need a different word list per operator -- and a wrong
    attribution label is worse than none. Deciding whether the winner covers the
    reference's dominant operator needs the reference's own operator profile, which the
    harness does not record yet; until it does, the reader gets the raw evidence.
    """
    cand_id = best.get("candidate_id")
    params = (best.get("params") or {}).get("values")
    if not cand_id:
        return []
    # The winning trial is the one whose params match the reported best, falling back to
    # this candidate's fastest completed trial (params round-trip through JSON, so compare
    # decoded dicts rather than strings).
    mine = [t for t in trials
            if t.get("candidate_id") == cand_id and t.get("status") == "complete"]
    if not mine:
        return []
    exact = [t for t in mine if (t.get("params") or {}).get("values") == params]
    pool = exact or mine
    winner = min(pool, key=lambda t: (t.get("latency_ms") or {}).get("mean") or float("inf"))
    names = ((winner.get("profile") or {}).get("kernel_names")) or []
    if not names:
        return ["- kernels launched by the winning configuration: **none recorded** "
                "(CUDA backend, or profiling unavailable) — attribution cannot be read "
                "from this run"]
    return [f"- kernels launched by the winning configuration: "
            f"{', '.join(f'`{n}`' for n in names)}",
            "  - a kernel the reference computes but that does not appear here was "
            "delegated to PyTorch, not written by the search; check this list before "
            "attributing the speedup"]


def _fp64_rescue_line(best: dict, trials: list[dict], eval_cfg: dict) -> list[str]:
    """F7: how much of the winner's correctness came from the fp64 relative arm.

    WHY. `fp64_rescued_trials` reached the event log and stopped there -- no reader, no
    report line, no effect on ranking. That hid one half of a comparison we actually made:
    our L3:48 winner at 1.411 ms passes correctness on 5 of 5 trials ONLY through the fp64
    relative arm, while an external CUDA kernel at 1.477 ms passes the primary gate outright
    with zero rescues. At 4.5% apart, theirs is the more accurate kernel, and quoting our
    number against theirs without this is quoting half the result.

    Absence is reported as carefully as presence. A run with the gate DISABLED has no
    rescues by construction, and printing "0 rescues" there would read as a clean bill of
    health for a check that never ran -- the same mistake as an unmeasurable signal left
    blank. So the gate's configured state is stated first, and it is read from the config
    rather than inferred from the counts.

    This is reporting only. Correctness still decides acceptance, deliberately: the rescues
    are not a defect to tune away. Measured on L3:48, switching that candidate to bf16 to
    avoid them fails outright (0 of 5, correctness_mismatch) -- fp16 is necessary there, so
    the rescues are a real cost of the task and exactly the kind of thing a reader must see.
    """
    if not eval_cfg.get("fp64_relative_gate"):
        # Only worth a line when a reader might otherwise assume the arm was available.
        return ["- fp64 relative gate: **disabled** for this run, so no candidate could be "
                "accepted by it (the absence of rescues below is not evidence of accuracy)"]

    cid = best.get("candidate_id")
    counted = [t for t in trials
               if t.get("candidate_id") == cid
               and t.get("fp64_rescued_trials") is not None]
    if not counted:
        return ["- fp64 relative gate: enabled, but no trial of the winning candidate "
                "recorded a rescue count (older run, or the field was not journalled)"]
    worst = max(t["fp64_rescued_trials"] for t in counted)
    # The denominator is `quick_correctness_trials`, NOT `correctness_trials`: these counts
    # come from TUNING trials, which run the quick path (3 trials by default), while the
    # final re-eval runs the full one (5). Dividing a quick-path count by the full-path
    # total produced "3 of 5 rescued" for a candidate whose quick trials were 3 of 3 --
    # understating it, and mixing two different measurements in one ratio.
    total = eval_cfg.get("quick_correctness_trials")
    if worst <= 0:
        return ["- fp64 relative gate: enabled, **0 rescues** — the winner passed the "
                "primary relaxed gate on its own"]
    of = f" of {total}" if total else ""
    line = (f"- fp64 relative gate: **{worst}{of} correctness trials rescued** by the "
            f"relative arm during tuning (the primary relaxed gate had already failed on "
            f"them)")
    out = [line]
    if total and worst >= total:
        out.append("  - ⚠ **this candidate's correctness rests entirely on the fp64 "
                   "relative arm.** It is accepted -- the arm is a legitimate pass, not a "
                   "loophole -- but a competing kernel of similar speed that clears the "
                   "primary gate outright is numerically the better result, and a "
                   "like-for-like comparison must say so.")
    return out


def _precision_of(params: dict) -> str | None:
    """The arithmetic-precision value a trial's params carry, or None if it declares none.

    Name-agnostic on purpose: candidates have used COMPUTE_DTYPE, PREC, DOT_PRECISION and
    BC_CACHE_DTYPE for this, so keying on one spelling would silently report "no precision
    knob" for most of them. A knob counts when its VALUE is one of the precision tokens --
    that set is small, closed, and does not collide with tile sizes or warp counts.
    """
    tokens = {"fp16", "bf16", "tf32", "ieee", "fp32", "float16", "bfloat16", "float32",
              "tf32x3", "half"}
    for name, value in (params.get("values") or {}).items():
        if isinstance(value, str) and value.lower() in tokens:
            # A cache/IO dtype is secondary to the compute precision; prefer a knob that
            # names the arithmetic when both are present.
            upper = name.upper()
            if "COMPUTE" in upper or "PREC" in upper or "DOT" in upper:
                return value.lower()
    for value in (params.get("values") or {}).values():
        if isinstance(value, str) and value.lower() in tokens:
            return value.lower()
    return None


def _trials_by_precision(trials: list[dict]) -> list[str]:
    """F6: trials grouped by the precision they ran at.

    WHY. The run-level line above is `N total, C complete, F failed`, and precision appears
    nowhere except on the single best candidate. That made a specific failure invisible:
    L3:43's theta_best declared four precisions in its own space and could only LAUNCH at
    one -- its tile was chosen at fp16 (2 bytes/element) and needs 131072-164352 bytes of
    shared memory at 4 bytes/element against a 101376 limit. Every tf32 and ieee trial
    failed, and in the report that was indistinguishable from ordinary noise among the
    other failures.

    A precision whose `complete` count is 0 is the signal to look for here: it means the
    candidate cannot run at a precision it claims to support, so the tuner never compared
    it against the alternatives.
    """
    groups: dict[str, dict] = {}
    for t in trials:
        prec = _precision_of(t.get("params") or {}) or "(no precision knob)"
        g = groups.setdefault(prec, {"n": 0, "complete": 0, "best": None, "kinds": {}})
        g["n"] += 1
        if t.get("status") == "complete":
            g["complete"] += 1
            lat = t.get("latency_ms") or {}
            ms = lat.get("median") or lat.get("mean")
            if ms is not None and (g["best"] is None or ms < g["best"]):
                g["best"] = ms
        else:
            kind = t.get("failure_kind") or "unknown"
            g["kinds"][kind] = g["kinds"].get(kind, 0) + 1
    if len(groups) <= 1 and "(no precision knob)" in groups:
        return []  # nothing to say: no candidate in this run exposed a precision knob
    out = ["### Trials by precision\n",
           "| precision | trials | complete | best ms | dominant failure |",
           "|---|---|---|---|---|"]
    for prec, g in sorted(groups.items(), key=lambda kv: -kv[1]["n"]):
        top = max(g["kinds"].items(), key=lambda kv: kv[1]) if g["kinds"] else None
        best = f"{g['best']:.3f}" if g["best"] is not None else "—"
        out.append(f"| `{prec}` | {g['n']} | {g['complete']} | {best} | "
                   f"{f'{top[0]} ({top[1]})' if top else '—'} |")
    dead = [(p, g) for p, g in groups.items()
            if g["complete"] == 0 and p != "(no precision knob)"]
    if dead:
        out.append("")
        names = ", ".join(f"`{p}`" for p, _ in dead)
        out.append(f"- **{names}: zero completed trials.** A precision the space offers but "
                   f"that never produced one measurement is not evidence that it is slower "
                   f"-- it is a precision the tuner could not evaluate, so the winning "
                   f"configuration was never compared against it.")
        # The cause is in the dominant failure, and it is NOT always the same one. Measured
        # across the L3:43 runs on disk: bf16 died 190-208 times per run with
        # `correctness_mismatch` (a numerical problem -- and on L3:48 bf16 genuinely cannot
        # hold that task's exponent range), while the shared-memory story belongs to trials
        # that fail `infeasible_shared_memory` because the tile was sized for a narrower
        # dtype. Naming one cause for both would send the reader to the wrong fix, so the
        # hint is derived per precision from what actually failed.
        for prec, g in dead:
            top = max(g["kinds"].items(), key=lambda kv: kv[1]) if g["kinds"] else None
            if not top:
                continue
            kind = top[0]
            if kind == "infeasible_shared_memory":
                why = ("the tile does not fit at this dtype's byte width -- it was sized "
                       "for a narrower one and cannot launch here. A tile/stage domain "
                       "that is feasible at 2 bytes/element is often infeasible at 4.")
            elif kind == "correctness_mismatch":
                why = ("the arithmetic, not the configuration: this precision does not "
                       "hold the task's numerics. Check the task's own two-precision "
                       "noise floor before reading it as a candidate defect.")
            elif kind == "runtime_error":
                why = "it failed at launch or during execution; read a trial's log tail."
            else:
                why = "see the trial records for this precision."
            out.append(f"  - `{prec}` ({kind}): {why}")
    out.append("")
    return out


def _vendor_library_usage(events) -> list[str]:
    """F8: which candidates handed computation to cuBLAS/cuDNN, from the static warnings.

    KernelBench reports `torch_computation_ops` and `pytorch_wrap` as WARNINGS, never
    errors, which is what makes delegating a large regular GEMM to the vendor library a
    legal choice -- and the contract now says so explicitly, because at strict IEEE fp32
    cuBLAS beat a hand-written Triton GEMM by 1.33x on a real task.

    The warnings were already being recorded on the worker result and read by nobody, so
    that choice was invisible. This surfaces it. Visibility, NOT enforcement: promoting the
    check would re-forbid the route the contract just opened, and the hard floor (a real
    kernel must exist, and must do the work the candidate claims) is enforced elsewhere.
    """
    seen: dict[str, set[str]] = {}
    for e in events:
        if e.type != "QUICKTEST_DONE":
            continue
        warns = (e.payload or {}).get("static_warnings") or []
        relevant = {w for w in warns
                    if "computation op" in w.lower() or "compute layer" in w.lower()}
        if relevant:
            seen.setdefault(e.payload.get("candidate_id", "?"), set()).update(relevant)
    if not seen:
        return []
    out = ["### Vendor-library delegation\n",
           "These candidates call a torch computation op (cuBLAS/cuDNN through "
           "`F.linear`/`F.conv2d`/`torch.matmul`, or an `nn` compute layer). This is "
           "PERMITTED and often correct — the library is usually near the hardware roof "
           "for a large regular GEMM — and is listed so the speedup can be attributed "
           "honestly, not as a defect.\n"]
    for cid, warns in sorted(seen.items()):
        out.append(f"- `{cid}`: {'; '.join(sorted(warns))}")
    out.append("")
    return out


def _why_the_run_ended(events, convergence: list[dict], budgets: dict) -> list[str]:
    """Name the reason the outer loop stopped, and whether budget was left on the table.

    The report used to show only the last ten convergence decisions, which never says *why*
    the run ended. That is why the D2 defect survived 19 runs: a run frozen by the outer
    loop's blanket sweep and a run that genuinely exhausted its families produced identical
    reports. Measured afterwards with `scripts/audit_run_termination_reasons.py`: only 1 of
    19 runs was ended by the wall clock, and four ended with 0-2 of 12 rewrite rounds used
    and no family freeze verdict at all. Every one of those looked normal here.

    So this states the ending, the clock spent, and the rewrite rounds spent -- the three
    numbers that make a premature ending visible without a separate audit script.
    """
    out: list[str] = []
    stuck = [e for e in events if e.type == "OUTER_LOOP_STUCK"]
    unrewritable = [e.payload.get("family_id")
                    for e in events if e.type == "FAMILY_FROZEN_UNREWRITABLE"]
    finished = [e for e in events if e.type == "RUN_FINISHED"]
    last_global = next((c["decision"] for c in reversed(convergence)
                        if (c.get("decision") or {}).get("scope") == "global"), None)

    rounds_used = sum(1 for e in events
                      if e.type in ("FAMILY_ROUND_RECORDED", "FAMILY_ROUND_NOT_EVALUATED"))
    # Both event types count as budget SPENT. A round whose rewrite never reached evaluation
    # still consumed one of the family's rounds; it just contributed no point to best_history
    # (appending one would read as convergence). Counting only the evaluated ones here would
    # report a run as having used less budget than it did.
    # The denominator is per-FAMILY, not per-seed. Seeds are only the families the run
    # STARTS with: Loop D adds more, and each new family carries its own
    # `rewrite_rounds_per_family` allowance. Deriving the total from `max_seed_candidates`
    # therefore understates it by exactly the novel families' share -- which was invisible
    # while Loop D never ran, and became wrong the moment it did. Observed on
    # run-l1-19-20260906-220044: 2 seeds x 2 rounds printed "6 of 4", an impossible
    # fraction, because Loop D had added 2 more families for a real total of 8.
    #
    # So count the families the log actually shows, falling back to the seed count only for
    # a run that died before seeding. Do NOT clamp to `max_families_total`: that budget
    # gates whether a NEW family may be created, and it can legitimately sit below the
    # number that exist -- run-l3-21-20260905-195615 seeded 4 families under
    # `max_families_total: 3` (the D1 defect: the gate counted differently than the seeder).
    # Clamping there reintroduced the same impossible fraction in the other direction,
    # printing "10 of 9". The log is the authority on how many families existed.
    per_family = budgets.get("rewrite_rounds_per_family")
    seeds = budgets.get("max_seed_candidates")
    families_seen = {e.payload.get("family_id") for e in events
                     if e.payload.get("family_id")} - {None}
    n_families = len(families_seen) or seeds
    rounds_avail = (n_families * per_family) if (n_families and per_family is not None) \
        else None

    elapsed = None
    if finished:
        elapsed = (finished[-1].payload.get("summary") or {}).get("elapsed_hours")
    wc = budgets.get("wall_clock_hours")

    if stuck:
        pl = stuck[-1].payload
        out.append(f"- **ended: OUTER_LOOP_STUCK** after {pl.get('idle_rounds')} idle "
                   f"rounds — this is the liveness guard, so it indicates a DEFECT, not a "
                   f"finished search. Family statuses at that point: "
                   f"{pl.get('families')}")
    elif not finished:
        out.append("- **ended: no RUN_FINISHED event** — the run was killed or crashed; "
                   "these numbers are partial")
    elif last_global and last_global.get("stop_kind") == "budget_exhausted" and wc \
            and elapsed is not None and elapsed >= wc * 0.98:
        out.append(f"- ended: **wall clock** ({elapsed} h of {wc} h) — the budget was "
                   f"actually spent")
    elif last_global and last_global.get("verdict") == "freeze":
        pct = f"{elapsed / wc * 100:.0f}%" if (wc and elapsed is not None) else "?"
        out.append(f"- ended: **every family frozen** "
                   f"(`{last_global.get('stop_kind')}`) at {elapsed} h of {wc} h ({pct} of "
                   f"the clock)")

    if rounds_avail:
        left = rounds_avail - rounds_used
        flag = ("  <- a freeze rule, not the budget, decided this ending"
                if left > 0 and elapsed is not None and wc and elapsed < wc * 0.9 else "")
        novel = max(0, len(families_seen) - (seeds or 0))
        how = (f"{n_families} families x {per_family}"
               + (f", incl. {novel} from Loop D" if novel else ""))
        out.append(f"- rewrite rounds spent: **{rounds_used} of {rounds_avail}** "
                   f"({how}){flag}")
    if unrewritable:
        out.append(f"- families frozen as unrewritable (no correct candidate, so no rewrite "
                   f"parent): {', '.join(f'`{f}`' for f in unrewritable)}")
    return out + [""] if out else []


class ReportGenerator:
    def generate(self, store: RunStore) -> Path:
        events = store.iter_events()
        report_dir = store.run_dir / "report"
        report_dir.mkdir(exist_ok=True)
        try:
            manifest = json.loads(
                (store.run_dir / "manifest.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            manifest = {}
        budgets = ((manifest.get("config") or {}).get("budgets") or {})
        eval_cfg = ((manifest.get("config") or {}).get("evaluation") or {})

        baselines = [e.payload["baseline"] for e in events if e.type == "BASELINE_DONE"]
        # Whether ANY baseline carries a real median. A median-labelled speedup needs one on
        # both sides; the baseline timing path returns summary statistics only, so usually
        # none does, and a stored `speedups_median` from before the orchestrator's gate is
        # then a baseline_MEAN / candidate_MEDIAN mixed ratio. See where this is used.
        baseline_medians = any(
            (b.get("latency_ms") or {}).get("median") for b in baselines
        )
        candidates = {e.payload["candidate"]["candidate_id"]: e.payload["candidate"]
                      for e in events if e.type == "CANDIDATE_REGISTERED"}
        trials = [e.payload["trial"] for e in events if e.type == "TRIAL_DONE"]
        tuning_done = [e.payload for e in events if e.type == "TUNING_DONE"]
        bottlenecks = [e.payload for e in events if e.type == "BOTTLENECK_REPORTED"]
        convergence = [e.payload for e in events if e.type == "CONVERGENCE_DECIDED"]
        agent_calls = [e.payload for e in events if e.type == "AGENT_CALL_FINISHED"]
        agent_failures = [e.payload for e in events
                          if e.type == "AGENT_CALL_FAILED" and e.payload.get("final")]
        # REWRITE_REJECTED is listed alongside the older two because a rewrite refused as a
        # structural duplicate is a rejection the reader wants to see. It used to be recorded as
        # NOVELTY_REJECTED, so both names are read: an older run's log still has the old type,
        # and dropping it here would make those rejections disappear from a replayed report.
        rejected = [e.payload for e in events
                    if e.type in ("SPACE_REJECTED", "NOVELTY_REJECTED", "REWRITE_REJECTED")]
        dead_kernels = [e.payload for e in events if e.type == "KERNELS_NEVER_LAUNCHED"]
        # SPACE_EXPANDED names the candidate, not the space; the expanded space is the one
        # published immediately before it.
        expanded_spaces: set[str] = set()
        last_published: str | None = None
        for e in events:
            if e.type == "SPACE_PUBLISHED":
                last_published = e.payload["space"]["space_id"]
            elif e.type == "SPACE_EXPANDED" and last_published:
                expanded_spaces.add(last_published)
        summary = next((e.payload["summary"] for e in reversed(events)
                        if e.type == "RUN_FINISHED"), None)
        # An unfinished run has no RUN_FINISHED, and reading ONLY that event made the
        # report claim "no correct candidate survived" and render an empty families
        # section on a run with 338 trials and nine successful tunings on disk. Since
        # `kernel-opt report` is the documented way to inspect a run -- including one
        # that was interrupted -- reconstruct the same shape from the events instead,
        # clearly marked provisional. The reconstruction deliberately omits
        # final_reeval_ms and honest_verdict: those come from a fresh-process re-eval
        # that has not happened, and inventing them would be the exact overclaim the
        # reeval-gap rule exists to prevent.
        provisional = summary is None
        if provisional:
            summary = _reconstruct_summary(events, candidates, trials)

        # trials.csv
        with (report_dir / "trials.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            # latency_median_ms is here because it is the statistic that actually DECIDED
            # every selection in the run (the tuning objective, the incumbent comparison
            # and the convergence test all read `robust_ms`). Emitting only the mean left
            # the deliverable trials.csv unable to explain any ranking it contains: on
            # level2:37 the best trial's mean was 32.20 us and its median 14.11, so a
            # reader sorting this file by mean gets a different winner than the run chose.
            writer.writerow(["trial_id", "candidate_id", "space_id", "status",
                             "failure_kind", "latency_mean_ms", "latency_median_ms",
                             "latency_std_ms", "n_regs", "n_spills", "shared_bytes",
                             "params"])
            for t in trials:
                lat = t.get("latency_ms") or {}
                prof = t.get("profile") or {}
                writer.writerow([
                    t["trial_id"], t["candidate_id"], t["space_id"], t["status"],
                    t.get("failure_kind") or "", lat.get("mean", ""),
                    latency_cell(lat.get("median")), lat.get("std", ""),
                    prof.get("n_regs", ""), prof.get("n_spills", ""),
                    prof.get("shared_bytes", ""),
                    json.dumps(t["params"]["values"]),
                ])

        total_cost = sum(c.get("cost", 0.0) for c in agent_calls)
        n_complete = sum(1 for t in trials if t["status"] == "complete")
        n_fail = len(trials) - n_complete

        lines: list[str] = []
        lines.append(f"# Run report — {store.run_dir.name}\n")
        if provisional:
            lines.append(
                "> **PROVISIONAL — this run has not finished.** Everything below is "
                "reconstructed from the event log so far. The final independent re-eval "
                "has not run, so there is no `final_reeval_ms` and no honest "
                "same-precision verdict: the latencies here are `tuned_ms` from "
                "quick_test, which is systematically optimistic against a full re-eval "
                "by 1.5–6.7%. Treat every number as provisional and do not quote a "
                "speedup from this report.\n")

        lines.append("## Baselines\n")
        for b in baselines:
            lat = b["latency_ms"]
            note = f" ({b['note']})" if b.get("note") else ""
            lines.append(f"- **{b['kind']}**: {lat['mean']} ms "
                         f"(std {lat['std']}, n={lat['n_samples']}){note}")
        lines.append("")

        # This box's measured ceilings. Reported because every bottleneck verdict in this run is
        # a fraction of them: a reader who cannot see the denominators cannot check the
        # classification, and a SUSPECT calibration silently inflates every %-of-peak figure.
        cal_ev = [e for e in events
                  if e.type in ("CALIBRATION_MEASURED", "CALIBRATION_LOADED")]
        if cal_ev:
            c = cal_ev[-1].payload
            lines.append("## Device calibration\n")
            lines.append(f"- source: **{'measured this run' if cal_ev[-1].type == 'CALIBRATION_MEASURED' else 'cached (same box)'}**"
                         f" — {c.get('device', 'unknown')}")
            if c.get("dram_tbs"):
                lines.append(f"- measured DRAM ceiling: {c['dram_tbs']:.4f} TB/s")
            if c.get("fp32_tflops"):
                lines.append(f"- measured fp32 ceiling: {c['fp32_tflops']:.2f} TFLOP/s"
                             + (f" · tf32 {c['tf32_tflops']:.2f} TFLOP/s"
                                if c.get("tf32_tflops") else ""))
            if c.get("empty_launch_floor_ms"):
                lines.append(f"- empty-launch floor: "
                             f"{c['empty_launch_floor_ms'] * 1e3:.1f} us "
                             f"(a kernel at this is at the floor; no tiling change can help)")
            th = c.get("thresholds") or {}
            if th:
                lines.append(f"- derived thresholds: dram≥{th.get('dram_saturated_frac')}, "
                             f"compute≥{th.get('compute_saturated_frac')}, "
                             f"idle<{th.get('idle_frac')}, "
                             f"cpu/gpu≥{th.get('launch_bound_cpu_ratio')} "
                             f"(measured on this box, not constants)")
            for s in (c.get("suspect") or []):
                lines.append(f"- ⚠ **SUSPECT calibration**: {s}")
            # Which signals this box could produce at all. Without it, a report showing no
            # instruction mix is ambiguous between "the box had no disassembler" and "the kernels
            # genuinely used no tensor cores" -- opposite conclusions.
            tiers = c.get("tiers") or {}
            if tiers and "error" not in tiers:
                t1 = tiers.get("tier1_sass_and_occupancy")
                lines.append(
                    f"- profiling tiers: Tier 0 (timing, FLOP/byte) available; "
                    f"Tier 1 (SASS instruction mix, occupancy) "
                    f"**{'available' if t1 else 'UNAVAILABLE'}**"
                    + ("" if t1 else " — tensor-core use, spills and access widths are reported "
                                    "as unknown in this run, not as absent")
                    + "; Tier 3 (hardware counters) unavailable "
                      "(ERR_NVGPUCTRPERM: needs a host-side permission a container cannot set)")
            lines.append("")
        elif any(e.type == "CALIBRATION_FAILED" for e in events):
            lines.append("## Device calibration\n")
            lines.append("- ⚠ **calibration FAILED** — bottleneck verdicts in this run report "
                         "`unknown` rather than comparing against a guessed ceiling.\n")

        # What the TASK requires (step 3). Reported next to the ceilings because the two together
        # are what any "% of peak" claim in this run means, and because `fusion headroom` is the
        # one number here that points at an action.
        tc_ev = [e for e in events if e.type == "TASK_COST_MEASURED"]
        if tc_ev:
            tc = tc_ev[-1].payload.get("task_cost") or {}
            flop = tc.get("flop_count", 0) or 0
            comp = tc.get("compulsory_bytes", 0) or 0
            ref_b = tc.get("reference_bytes", 0) or 0
            lines.append("## Task cost (measured on the reference)\n")
            if flop:
                lines.append(f"- required arithmetic: {flop/1e9:.3f} GFLOP")
            else:
                lines.append("- required arithmetic: **0 FLOP** — no multiply-accumulate, so "
                             "this task's ceiling is bandwidth, not FLOP/s")
            if comp:
                lines.append(f"- unavoidable traffic: {comp/1e6:.2f} MB "
                             f"(inputs + parameters + outputs, each counted once)")
            if comp and ref_b:
                lines.append(f"- reference materializes: {ref_b/1e6:.2f} MB over "
                             f"{tc.get('op_count', 0)} ops = **{ref_b/comp:.1f}x** the "
                             f"unavoidable traffic (fusion headroom)")
            if flop and comp:
                ai = flop / comp
                ridge = None
                if cal_ev:
                    ridge = (cal_ev[-1].payload or {}).get("ridge_flop_per_byte")
                side = ""
                if ridge:
                    side = (f" — above this card's ridge of {ridge:.1f}, so this task CAN be "
                            f"compute-bound" if ai > ridge else
                            f" — below this card's ridge of {ridge:.1f}, so no correct "
                            f"implementation of this task can be compute-bound here")
                lines.append(f"- highest intensity any correct implementation can reach: "
                             f"{ai:.2f} FLOP/byte{side}")
            for n in (tc.get("notes") or []):
                lines.append(f"- note: {n}")
            lines.append("")

        # The physical speedup ceiling this run flagged against. Reported next to the task cost
        # because it is derived from it, and reported even when NOTHING flagged: "the bound was
        # 17.48x and the winner reached 14.29x" is the sentence that makes a legal result
        # defensible, and "no bound could be derived" is a finding about this box that would
        # otherwise be indistinguishable from a check that silently never ran.
        pb = _plausibility_bound(events)
        pb_ev = [e for e in events if e.type == "PLAUSIBILITY_BOUND"]
        if pb:
            lines.append("## Implausible-speedup bound (derived, not configured)\n")
            lines.append("- a correct implementation of this task cannot run faster than "
                         f"**{pb.get('floor_ms', 0.0):.4f} ms** on this box "
                         f"({pb.get('binding_term', '?')}-bound)")
            lines.append(f"- so the largest possible speedup vs the {pb.get('reference_ms', 0.0):.3f} "
                         f"ms reference is **{pb.get('ceiling_x', 0.0):.2f}x**; the run flags above "
                         f"**{pb.get('threshold_x', 0.0):.2f}x** "
                         f"(x{pb.get('margin', 0.0):.2f} margin)")
            if not pb.get("dram_applicable", True):
                lines.append("- ⚠ the DRAM term was DROPPED: this task's compulsory traffic fits "
                             "in this card's measured L2, so the logical byte count does not "
                             "describe bus traffic. The bound rests on arithmetic alone.")
            lines.append(f"- derivation: {pb.get('derivation', '')}")
            lines.append("")
        elif pb_ev:
            lines.append("## Implausible-speedup bound (derived, not configured)\n")
            lines.append("- ⚠ **no bound could be derived** — %s. No speedup in this run was "
                         "checked for physical plausibility; correctness is unaffected (it is "
                         "the only thing that ever decided acceptance)."
                         % ((pb_ev[-1].payload or {}).get("reason") or "reason not recorded"))
            lines.append("")

        # The harness's own bottleneck verdicts (steps 6+7). Reported because they are the
        # deterministic half of the feedback loop: unlike the analyst's report they are
        # reproducible from the event log, so a reader can check them.
        bn = [e.payload for e in events if e.type == "BOTTLENECK_CLASSIFIED"]
        if bn:
            lines.append("## Measured bottleneck verdicts\n")
            kinds: dict[str, int] = {}
            for b in bn:
                kinds[b.get("kind", "?")] = kinds.get(b.get("kind", "?"), 0) + 1
            lines.append("- distribution: "
                         + ", ".join(f"**{k}** x{v}" for k, v in sorted(kinds.items())) + "\n")
            for b in bn:
                ev = b.get("evidence") or {}
                bits = []
                if ev.get("pct_of_dram_peak") is not None:
                    bits.append(f"{ev['pct_of_dram_peak']}% DRAM")
                if ev.get("pct_of_compute_peak") is not None:
                    bits.append(f"{ev['pct_of_compute_peak']}% "
                                f"{ev.get('compute_ceiling_used', 'compute')}")
                if ev.get("occupancy") is not None:
                    bits.append(f"occupancy {ev['occupancy']*100:.0f}% "
                                f"({ev.get('occupancy_limiter')})")
                if ev.get("uses_tensor_cores") is not None:
                    bits.append("tensor cores"
                                + ("" if ev["uses_tensor_cores"] else " NOT used"))
                lines.append(f"- `{b.get('candidate_id')}`: **{b.get('kind')}**"
                             + (f" — {', '.join(bits)}" if bits else ""))
                if b.get("disagreement"):
                    # A caveat that is not surfaced is a caveat that misleads.
                    lines.append(f"  - ⚠ low confidence: {b['disagreement']}")
            lines.append("")

        if summary and summary.get("best"):
            best = summary["best"]
            lines.append("## Best result\n")
            lines.append(f"- candidate: `{best['candidate_id']}` "
                         f"(family `{best['family_id']}`)")
            lines.append(f"- tuned latency: {best['tuned_ms']} ms")
            if "final_reeval_ok" in best:
                lines.append(f"- final independent re-eval: "
                             f"{'PASS' if best['final_reeval_ok'] else 'FAIL'} "
                             f"at {best.get('final_reeval_ms')} ms")
            else:
                lines.append("- final independent re-eval: **not run yet** (run in "
                             "progress); tuned_ms above is NOT a verified latency")
            if best.get("precision"):
                lines.append(f"- candidate arithmetic precision: **{best['precision']}**")
            lines.extend(_fp64_rescue_line(best, trials, eval_cfg))
            # The honest same-precision verdict comes FIRST, before the raw per-baseline
            # speedups. All three task references are plain fp32 while the winning
            # candidates compute in a lower precision, so most of the raw ratios compare
            # across precisions and read high: on L3:43 the baseline choice alone is worth
            # 1.91x (4.23x vs torch_compile, 2.21x vs torch_compile_tf32), nearly the whole
            # honest speedup. Three historical runs are recorded as FAILS on this verdict
            # while showing 1.08-1.86x against the fp32 baselines, so the ordering decides
            # which number a reader (or a paper draft) takes away first.
            hv = best.get("honest_verdict")
            if hv:
                verdict = hv.get("same_precision_speedup")
                against = hv.get("compared_against", "?")
                if verdict is not None:
                    beats = hv.get("beats_same_precision_baseline")
                    mark = "✅ beats" if beats else "❌ does not beat"
                    lines.append(
                        f"- **honest same-precision verdict**: candidate is "
                        f"{best.get('precision', '?')}; vs same-precision baseline "
                        f"`{against}` = **{verdict}x** — {mark} the same-precision "
                        f"baseline")
            speedups = best.get("speedups")
            if speedups:
                lines.append("- raw speedup vs each baseline "
                             "(baseline_ms / candidate_ms, >1 = faster) — **cross-precision "
                             "where the baseline's precision differs from the candidate's, "
                             "so NOT directly comparable; use the honest verdict above**:")
                for kind in sorted(speedups):
                    lines.append(f"  - vs `{kind}`: **{speedups[kind]}x**")
                # Both conventions, side by side, because a cross-framework comparison
                # across conventions is meaningless. The mean figures above are the
                # headline and the conservative claim (they include the scheduling stalls a
                # user actually observes); the medians below are what the tuner optimized
                # and what many kernel benchmarks publish. On a task whose stalls are large
                # relative to the kernel the two diverge a lot -- on level2:37 a trial's
                # mean/min ratio reached 2.04x -- and quoting one against the other's number
                # invents a difference no kernel produced.
                med = best.get("speedups_median")
                # VALIDATE, do not trust. `report` regenerates from the event log, including
                # logs written before the orchestrator gained its both-sides gate -- and a
                # summary from such a run stores `speedups_median` already computed as
                # baseline_MEAN / candidate_MEDIAN. run-l2-37-20260907-020707 is exactly that
                # case: the run started 02:07, the gate landed 03:40, the module was imported
                # at launch, so its stored medians are the mixed ratio and this report
                # published them under a MEDIANS heading with a "+33.1%" uplift.
                #
                # So the check belongs on both sides of the boundary: the orchestrator must
                # not compute a mixed ratio, and the report must not print one it is handed.
                # A median-labelled ratio requires a median in the BASELINE record too, which
                # is what `baseline_medians` carries.
                if med and not baseline_medians:
                    med = None
                    mixed_note = (
                        "not shown: this run's stored `speedups_median` was computed before "
                        "the both-sides gate existed, and its baselines carry no median (the "
                        "baseline timing path returns summary statistics only). The stored "
                        "values are baseline_MEAN / candidate_MEDIAN, which inflates the "
                        "ratio because the numerator includes scheduling stalls and the "
                        "denominator does not. Quote the mean-based figures above."
                    )
                else:
                    mixed_note = ""
                if med:
                    lines.append("- the same ratios computed from MEDIANS (robust to "
                                 "scheduling stalls; this is the statistic the tuner "
                                 "optimized, and the convention many published kernel "
                                 "numbers use — compare like with like):")
                    for kind in sorted(med):
                        delta = ""
                        if kind in speedups and speedups[kind]:
                            pct = (med[kind] / speedups[kind] - 1) * 100
                            delta = f" ({pct:+.1f}% vs the mean-based figure)"
                        lines.append(f"  - vs `{kind}`: **{med[kind]}x**{delta}")
                    if best.get("final_reeval_median_ms"):
                        lines.append(
                            f"  - re-eval latency: mean "
                            f"{best.get('final_reeval_ms')} ms, median "
                            f"{latency_cell(best.get('final_reeval_median_ms'))} ms")
                elif mixed_note or best.get("speedups_median_note"):
                    # Say why the median column is absent rather than leaving the reader to
                    # assume the two conventions agree. They do not: on level2:37 the
                    # candidate's mean was 32.20 us against a median of 14.11, so a reader
                    # comparing our mean-based figure to an externally published median or
                    # min is comparing different quantities.
                    lines.append(f"- median-based speedups: "
                                 f"{mixed_note or best['speedups_median_note']}")
                    if best.get("final_reeval_median_ms"):
                        lines.append(
                            f"  - the candidate's own re-eval latency, both ways: mean "
                            f"{best.get('final_reeval_ms')} ms, median "
                            f"{latency_cell(best.get('final_reeval_median_ms'))} ms — quote "
                            f"the mean against a mean, the median against a median")
            else:
                if "speedup_vs_eager" in best:
                    lines.append(f"- speedup vs eager: **{best['speedup_vs_eager']}x**")
                if "speedup_vs_compile" in best:
                    lines.append(f"- speedup vs torch.compile: "
                                 f"**{best['speedup_vs_compile']}x**")
            if best.get("excessive_speedup_flag"):
                # WITH the derivation, always. A flag whose number cannot be checked by the person
                # reading it is what the old 10x constant was: three L3:48 runs each flagged a
                # verified-correct winner and each cost a manual re-verification, because the
                # report said only "excessive". The bound is now a physical one, so the report can
                # say what it was and how close the result came to it.
                bound = _plausibility_bound(events)
                if bound:
                    lines.append(
                        "- ⚠ flagged: measured speedup exceeds this task's physical ceiling on "
                        "this box (%.2fx floor-derived ceiling, flag at %.2fx). Inspect before "
                        "trusting. Derivation: %s"
                        % (bound.get("ceiling_x", 0.0), bound.get("threshold_x", 0.0),
                           bound.get("derivation", "")))
                else:
                    lines.append("- ⚠ flagged: excessive speedup — inspect before trusting")
            lines.append(f"- best params: `{json.dumps(best['params']['values'])}`")
            lines.extend(_attribution_lines(best, trials))
            lines.append("")
            lines.extend(_search_budget_lines(summary, trials, dead_kernels,
                                              provisional))
        else:
            lines.append("## Best result\n\n- no correct candidate survived\n")

        lines.append("## Families / lineage\n")
        families = (summary or {}).get("families", {})
        for fid, fam in families.items():
            best_ms = fam.get("best_ms")
            rounds = fam.get("rewrite_rounds_used")
            headline = (f"best {best_ms} ms" if best_ms is not None
                        else "**no measured candidate**")
            lines.append(f"### `{fid}` — {fam['status']}, {headline}")
            # Three genuinely different states end up looking alike in the status field,
            # and conflating them misreads the search. A family with no best never got a
            # working candidate at all, so "0 rewrite rounds" says nothing about its
            # structure -- claiming its headroom is unknown-but-promising would be wrong.
            # On L3:48, fam-dc0697c9 is exactly this: its only seed (cand-eb910a18)
            # exhausted every repair attempt on non-finite output and was dropped.
            if best_ms is None:
                lines.append(
                    "- **no candidate in this family ever passed correctness**, so it was "
                    "never tuned and never rewritten. This is a FAILED branch, not an "
                    "unexplored one: the structure could not be made correct within the "
                    "repair budget."
                    if not provisional else
                    "- no candidate in this family has passed correctness YET; the run is "
                    "still going, so this is not (yet) a failed branch."
                )
            elif rounds == 0:
                # "frozen without the rewriter being invoked" is only true of a FINISHED
                # run. Mid-run, 0 rounds usually means the round is in flight right now --
                # on L3:48 fam-b1ee96ac had two rewrites under evaluation while this
                # branch called it frozen, which is simply false.
                lines.append(
                    "- **never entered structural search** (0 rewrite rounds): this "
                    "branch was frozen without the rewriter ever being invoked on it, "
                    "so its structural headroom is UNKNOWN, not exhausted."
                    if not provisional else
                    "- no completed rewrite round yet (a round may be in flight); nothing "
                    "can be concluded about this branch's structural headroom."
                )
            elif rounds is not None:
                lines.append(f"- rewrite rounds used: {rounds}")
            # A `converged` verdict rests on best_history being flat. If some of those rounds
            # never evaluated a rewrite, the flatness is not evidence about this structure --
            # it is the absence of evidence, and reads as the opposite of what it means. Stated
            # next to the status because that is where a reader forms the conclusion.
            # Measured: run-l1-42-20260907-193510 reported two frozen_converged families, and
            # fam-50ba7c87 was this case (its rewriter call failed all three attempts).
            unevaluated = fam.get("rounds_not_evaluated") or 0
            if unevaluated:
                lines.append(
                    f"- ⚠ **{unevaluated} of these rounds evaluated no rewrite at all** "
                    "(agent failure, or every candidate was a structural duplicate). Those "
                    "rounds contribute no point to the history below, so a flat history here "
                    "must NOT be read as exhausted headroom"
                    + (" — and this family's `converged` status is therefore not supported "
                       "by a measured rewrite." if "converged" in str(fam.get("status", ""))
                       else ".")
                )
            lines.append(f"- best history: {fam['history']}")
            for member in fam["members"]:
                lines.append(f"  - `{member['id']}` ({member['origin']}"
                             f"{', parents ' + str(member['parents']) if member['parents'] else ''}): "
                             f"{member['approach'][:150]}")
            lines.append("")

        lines.append("## Tuning\n")
        lines.append(f"- trials: {len(trials)} total, {n_complete} complete, {n_fail} failed")
        for t in tuning_done:
            # Include the space_id: a candidate that got a K expansion is tuned twice and
            # rendered as two identical-looking lines otherwise, which reads like a
            # duplicated entry rather than a re-tune over a widened space. The expansion
            # marker makes the improvement (or lack of it) attributable.
            expanded = " (expanded space)" if t.get("space_id") in expanded_spaces else ""
            lines.append(f"- `{t['candidate_id']}` [`{t.get('space_id', '?')}`]"
                         f"{expanded}: best {t.get('best_ms')} ms "
                         f"({(t.get('snapshot') or {}).get('asked', '?')} asked)")
        lines.append("")
        lines.extend(_trials_by_precision(trials))
        lines.extend(_vendor_library_usage(events))
        # S4': the conversion verdicts, and the complementary-slackness check on our OWN verdicts.
        # `conversion_verdict` was computed and journalled and read zero times here -- a verdict with
        # no consumer is not implemented, because nothing acts on it and nothing can notice it being
        # wrong. Reads only the event log, so `report` still regenerates purely from events.jsonl.
        lines.extend(conversion_lines(
            events,
            min_improvement_pct=float((budgets or {}).get("min_improvement_pct", 2.0))))

        if bottlenecks:
            lines.append("## Bottleneck reports\n")
            for b in bottlenecks:
                rep = b["report"]
                lines.append(f"- `{b['candidate_id']}`: {rep['summary'][:300]} "
                             f"(suggested: {rep['suggested_action']})")
                for lim in rep.get("parameter_limits", []):
                    lines.append(f"  - {lim['param']} wants {lim['headroom_direction']}, "
                                 f"blocked by {lim['blocked_by']}")
            lines.append("")

        # 2e, immediately after the analyst's own `parameter_limits`: the two sections answer the
        # same question, one by asking the agent and one by asking the compiler, and putting them
        # apart would hide that they disagree. Measured on box 2, they agree on 8 of 53 claims.
        lines.extend(wall_lines(events))

        # 2b(2)/2d, after 2e because it is about the same ledger read one level down: 2e asks the
        # compiler which knob is walled, this asks which DIMENSIONS the agent's own declarations are
        # reliable about. Both exist to stop a single pooled number standing in for eight unequal ones.
        lines.extend(accuracy_lines(events))

        lines.append("## Convergence decisions\n")
        lines.extend(_why_the_run_ended(events, convergence, budgets))
        for c in convergence[-10:]:
            d = c["decision"]
            scope_id = c.get("family_id", "global")
            lines.append(f"- {d['scope']} `{scope_id}`: {d['verdict']}"
                         f"{' (' + str(d.get('stop_kind')) + ')' if d.get('stop_kind') else ''}")
        lines.append("")

        # 1c. Which of the four loops actually ran, and where the agent-free time went.
        # Immediately after the ending, because "why it ended" and "how far it got" are the same
        # question asked twice, and the pair is what makes a silently truncated run visible.
        # `_why_the_run_ended` reports the ending; this reports whether the process was complete.
        lines.extend(completeness_lines(list(events), budgets))

        if rejected:
            lines.append("## Rejections\n")
            for r in rejected[:20]:
                lines.append(f"- {r.get('reason')}: {str(r.get('detail', ''))[:150]}")
            lines.append("")

        lines.append("## Agent usage\n")
        lines.append(f"- successful calls: {len(agent_calls)}; "
                     f"failed (final): {len(agent_failures)}")
        lines.append(f"- total cost: ${total_cost:.4f}")
        by_module: dict[str, int] = {}
        for c in agent_calls:
            by_module[c.get("module", "?")] = by_module.get(c.get("module", "?"), 0) + 1
        for module, count in sorted(by_module.items()):
            lines.append(f"  - {module}: {count} calls")
        lines.append("")

        if summary:
            lines.append(f"\n_Elapsed: {summary.get('elapsed_hours')} h; "
                         f"candidates: {len(candidates)}_\n")

        report_path = report_dir / "report.md"
        report_path.write_text("\n".join(lines), encoding="utf-8")
        return report_path
