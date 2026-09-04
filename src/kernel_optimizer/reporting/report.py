"""Report generation from the event log (proves the trace is complete)."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from kernel_optimizer.store.run_store import RunStore


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
        summary = next((e.payload["summary"] for e in reversed(events)
                        if e.type == "RUN_FINISHED"), None)

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
            lines.append(f"- final independent re-eval: "
                         f"{'PASS' if best['final_reeval_ok'] else 'FAIL'} "
                         f"at {best.get('final_reeval_ms')} ms")
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
                )
            elif rounds == 0:
                lines.append(
                    "- **never entered structural search** (0 rewrite rounds): this "
                    "branch was frozen without the rewriter ever being invoked on it, "
                    "so its structural headroom is UNKNOWN, not exhausted."
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
            lines.append(f"- `{t['candidate_id']}`: best {t.get('best_ms')} ms "
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
