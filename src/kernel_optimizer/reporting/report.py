"""Report generation from the event log (proves the trace is complete)."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from kernel_optimizer.store.run_store import RunStore


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
        if cur is None or t["latency_ms"]["mean"] < cur["latency_ms"]["mean"]:
            best_per_cand[cid] = t

    rounds: dict[str, int] = {}
    history: dict[str, list] = {}
    for e in events:
        if e.type == "FAMILY_ROUND_RECORDED":
            fid = e.payload["family_id"]
            rounds[fid] = rounds.get(fid, 0) + 1
            history.setdefault(fid, []).append(e.payload.get("best_ms"))

    families: dict[str, dict] = {}
    for cid, cand in candidates.items():
        fid = cand["family_id"]
        fam = families.setdefault(fid, {
            "status": "active (run in progress)", "best_ms": None,
            "rewrite_rounds_used": rounds.get(fid, 0),
            "history": history.get(fid, []), "members": [],
        })
        fam["members"].append({
            "id": cid, "origin": cand["origin"], "parents": cand.get("parent_ids") or [],
            "approach": cand.get("approach_summary") or "",
        })
        t = best_per_cand.get(cid)
        if t and (fam["best_ms"] is None or t["latency_ms"]["mean"] < fam["best_ms"]):
            fam["best_ms"] = t["latency_ms"]["mean"]

    out: dict = {"families": families, "best": None}
    if best_per_cand:
        winner = min(best_per_cand.values(), key=lambda t: t["latency_ms"]["mean"])
        cand = candidates.get(winner["candidate_id"], {})
        out["best"] = {
            "candidate_id": winner["candidate_id"],
            "family_id": cand.get("family_id", "?"),
            "tuned_ms": winner["latency_ms"]["mean"],
            "params": winner["params"],
        }
    return out


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

    rounds_used = sum(1 for e in events if e.type == "FAMILY_ROUND_RECORDED")
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

        baselines = [e.payload["baseline"] for e in events if e.type == "BASELINE_DONE"]
        candidates = {e.payload["candidate"]["candidate_id"]: e.payload["candidate"]
                      for e in events if e.type == "CANDIDATE_REGISTERED"}
        trials = [e.payload["trial"] for e in events if e.type == "TRIAL_DONE"]
        tuning_done = [e.payload for e in events if e.type == "TUNING_DONE"]
        bottlenecks = [e.payload for e in events if e.type == "BOTTLENECK_REPORTED"]
        convergence = [e.payload for e in events if e.type == "CONVERGENCE_DECIDED"]
        agent_calls = [e.payload for e in events if e.type == "AGENT_CALL_FINISHED"]
        agent_failures = [e.payload for e in events
                          if e.type == "AGENT_CALL_FAILED" and e.payload.get("final")]
        rejected = [e.payload for e in events
                    if e.type in ("SPACE_REJECTED", "NOVELTY_REJECTED")]
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
            writer.writerow(["trial_id", "candidate_id", "space_id", "status",
                             "failure_kind", "latency_mean_ms", "latency_std_ms",
                             "n_regs", "n_spills", "shared_bytes", "params"])
            for t in trials:
                lat = t.get("latency_ms") or {}
                prof = t.get("profile") or {}
                writer.writerow([
                    t["trial_id"], t["candidate_id"], t["space_id"], t["status"],
                    t.get("failure_kind") or "", lat.get("mean", ""), lat.get("std", ""),
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
                            f"{best.get('final_reeval_median_ms')} ms")
            else:
                if "speedup_vs_eager" in best:
                    lines.append(f"- speedup vs eager: **{best['speedup_vs_eager']}x**")
                if "speedup_vs_compile" in best:
                    lines.append(f"- speedup vs torch.compile: "
                                 f"**{best['speedup_vs_compile']}x**")
            if best.get("excessive_speedup_flag"):
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

        lines.append("## Convergence decisions\n")
        lines.extend(_why_the_run_ended(events, convergence, budgets))
        for c in convergence[-10:]:
            d = c["decision"]
            scope_id = c.get("family_id", "global")
            lines.append(f"- {d['scope']} `{scope_id}`: {d['verdict']}"
                         f"{' (' + str(d.get('stop_kind')) + ')' if d.get('stop_kind') else ''}")
        lines.append("")

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
