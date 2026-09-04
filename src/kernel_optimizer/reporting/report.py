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


class ReportGenerator:
    def generate(self, store: RunStore) -> Path:
        events = store.iter_events()
        report_dir = store.run_dir / "report"
        report_dir.mkdir(exist_ok=True)

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
            speedups = best.get("speedups")
            if speedups:
                lines.append("- speedup vs each baseline "
                             "(baseline_ms / candidate_ms, >1 = faster):")
                for kind in sorted(speedups):
                    lines.append(f"  - vs `{kind}`: **{speedups[kind]}x**")
            else:
                if "speedup_vs_eager" in best:
                    lines.append(f"- speedup vs eager: **{best['speedup_vs_eager']}x**")
                if "speedup_vs_compile" in best:
                    lines.append(f"- speedup vs torch.compile: "
                                 f"**{best['speedup_vs_compile']}x**")
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
            if best.get("excessive_speedup_flag"):
                lines.append("- ⚠ flagged: excessive speedup — inspect before trusting")
            lines.append(f"- best params: `{json.dumps(best['params']['values'])}`")
            lines.append("")
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
