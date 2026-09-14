"""v4.1 runtime bridge: wires the conditional layer into the orchestrator's tuning loop.

Separated from orchestrator.py so the whole mechanism has ONE seam: the orchestrator calls
`maybe_make(...)` once per tuning pass and `step(...)` once per told trial. Everything else
(probe dispatch, wall derivation, scan admission, fresh enqueue, event journalling) lives
here. Never raises into the tuning loop — a diagnostic that ended a candidate would present
a bookkeeping defect as a candidate defect (the same rule every v3 collaborator follows).

Isolation (v4.1 §3): probe sources are materialized under `runs/<id>/uw_probes/<cand>/`
and screened through `prescreen_batch` — which ONLY caches compiler answers; it never emits
production refusals, never touches guard/PRUNED, and the scanner reads the verdicts through
`cached_shared_verdict`/`screen_cache_entry`, the same read-only path 2e's wall probes use.
The production sampler ALSO reads that cache (`_shared_memory_ok`), which is read-only
sharing of compiler facts — the spec forbids the probe layer to REFUSE production configs
or fabricate production events, not to share a cache of compiler answers.

Modes: observe = probes + walls + events, scanner never admits (zero latency trials);
active = full C/E scan with fresh enqueue.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from kernel_optimizer.conditional.probe import (
    ConditionedWall,
    ProbePlanner,
    retry_prefix,
)
from kernel_optimizer.conditional.scanner import ConditionalScanner, ScanBlock, ScanPoint
from kernel_optimizer.conditional.tokens import TokenStore
from kernel_optimizer.models.core import ParamSet, TrialRecord
from kernel_optimizer.paramspace import materializer


@dataclass
class ScanRuntime:
    """Per-space live state: planner + scanner + the pending scan-point ledger."""

    mode: str
    planner: ProbePlanner
    scanner: ConditionalScanner
    cadence: int
    probe_dir: Path
    # params-key -> (block, point): scan points enqueued and awaiting their TRIAL_DONE.
    pending_points: dict[str, list[tuple[ScanBlock, ScanPoint]]] = field(
        default_factory=dict)
    last_cadence_told: int = 0

    def take_pending(self, params_key: str) -> tuple[ScanBlock, ScanPoint] | None:
        row = self.pending_points.get(params_key)
        if not row:
            return None
        item = row.pop(0)
        if not row:
            del self.pending_points[params_key]
        return item


class ConditionalScanBridge:
    """Owns the v4.1 layer's runtime for ONE candidate across its spaces."""

    def __init__(self, cfg: Any, store: Any, evaluator: Any, task: Any,
                 device_limit: int) -> None:
        self.cfg = cfg
        self.store = store
        self.evaluator = evaluator
        self.task = task
        self.limit = device_limit
        self.tokens = TokenStore()          # candidate-scoped: survives across spaces
        self.last_phase = "C"
        self.last_runtime: ScanRuntime | None = None  # for the §6 brief after tuning

    # ------------------------------------------------------------------ construction

    def make_runtime(self, crun: Any, run_dir: Path, seed: int,
                     budget_b: int, legal: Callable[[dict], bool]) -> ScanRuntime | None:
        v4cfg = self.cfg.v4.conditional_scan
        if v4cfg.mode == "off" or crun.space is None:
            return None
        planner = ProbePlanner(crun.space, self.limit, legal, pcap=v4cfg.pcap)
        scanner = ConditionalScanner(
            crun.space, crun.candidate.candidate_id, crun.candidate.backend,
            budget_b, self.tokens, seed=seed, initial_phase=self.last_phase,
            f_frac=v4cfg.f_frac)
        probe_dir = run_dir / "uw_probes" / crun.candidate.candidate_id
        probe_dir.mkdir(parents=True, exist_ok=True)
        rt = ScanRuntime(mode=v4cfg.mode, planner=planner, scanner=scanner,
                         cadence=v4cfg.cadence_told, probe_dir=probe_dir)
        self.last_runtime = rt
        return rt

    # ------------------------------------------------------------------ per-trial step

    def step(self, rt: ScanRuntime, crun: Any, tuner: Any,
             trial_id: str, record: TrialRecord) -> None:
        """Called after EVERY told trial. Folds scan results in, fires the completion
        hook, runs the cadence (probe batch + scan admission)."""
        # 1. scan-point result?
        pending = rt.take_pending(record.params.key()) if record.params else None
        if pending is not None:
            block, point = pending
            ok = record.status == "complete" and record.latency_ms is not None
            lat = record.latency_ms.robust_ms if ok else None
            token = rt.scanner.tell(block, point, lat, ok)
            self.store.append("SCAN_POINT_DONE", {
                "candidate_id": crun.candidate.candidate_id,
                "space_id": crun.space.space_id, "trial_id": trial_id,
                **point.payload(), "kind": block.kind, "ok": ok, "latency_ms": lat})
            if block.done:
                self._journal_block(crun, block, token)
                if token is not None and rt.mode == "active":
                    # §4.3 completion hook: immediate EVALUATION (admission attempt now,
                    # not at the next checkpoint); start remains whenever the tuner draws it.
                    self._admit_and_enqueue(rt, crun, tuner)
        else:
            rt.scanner.note_external_measurement(dict(record.params.values)
                                                 if record.params else {})
            if record.failure_kind == "infeasible_shared_memory" and record.params:
                incumbent = self._incumbent(crun)
                if incumbent:
                    rt.planner.note_refusal(dict(record.params.values), incumbent)

        # 2. cadence: ready work -> probe batch; then scan admission.
        told = tuner.n_told
        if told - rt.last_cadence_told >= rt.cadence:
            rt.last_cadence_told = told
            self._probe_batch(rt, crun)
            if rt.mode == "active":
                self._admit_and_enqueue(rt, crun, tuner)

    # ------------------------------------------------------------------ probe batch

    def _probe_batch(self, rt: ScanRuntime, crun: Any) -> None:
        try:
            incumbent = self._incumbent(crun)
            if incumbent is None or rt.planner.in_flight:
                return
            planned = rt.planner.plan_batch(incumbent)
            if not planned:
                return
            projected = 30.0 + 3.0 * len(planned)     # the shipping timeout model
            if not rt.planner.budget.admit(projected):
                return
            self._dispatch(rt, crun, planned, is_retry=False)
        except Exception as exc:  # noqa: BLE001 — diagnostics never end a candidate
            self.store.append("UW_PROBE_FAILED", {
                "candidate_id": crun.candidate.candidate_id,
                "error": f"{type(exc).__name__}: {exc}"[:300]})

    def _dispatch(self, rt: ScanRuntime, crun: Any, planned: list[dict],
                  is_retry: bool) -> None:
        paths: list[Path] = []
        kept: list[dict] = []
        for i, values in enumerate(planned):
            try:
                src = materializer.materialize(crun.source, ParamSet(values=values))
            except materializer.MaterializeError:
                continue
            p = rt.probe_dir / f"b{rt.planner.budget.attempts_started}-{i:03d}.py"
            p.write_text(src, encoding="utf-8")
            paths.append(p)
            kept.append(values)
        if not paths:
            return
        rt.planner.in_flight = True
        t0 = time.time()
        failed = False
        try:
            self.evaluator.prescreen_batch(
                self.task, paths, tag=f"{crun.candidate.candidate_id}-uw",
                backend=crun.candidate.backend)
        except Exception:  # noqa: BLE001
            failed = True
        elapsed = time.time() - t0
        results: dict[str, dict] = {}
        answered = 0
        for p in paths:
            src = p.read_text(encoding="utf-8")
            entry = self.evaluator.screen_cache_entry(src, crun.candidate.backend)
            if entry is None or not entry.get("ok"):
                results[str(p)] = {"ok": False, "reason": "not answered"}
            else:
                results[str(p)] = entry
                answered += 1
        timed_out = answered == 0 and not failed
        rt.planner.record(results, kept)
        if is_retry:
            rt.planner.budget.after_retry(elapsed, timed_out, failed)
        else:
            rt.planner.budget.record(elapsed, timed_out, failed, len(kept))
            if timed_out and rt.planner.budget.retried and not rt.planner.budget.stopped:
                prefix = retry_prefix(kept)
                projected = 30.0 + 3.0 * len(prefix)
                if rt.planner.budget.admit(projected):
                    self._dispatch(rt, crun, prefix, is_retry=True)
                else:
                    rt.planner.budget.stopped = True
                    rt.planner.budget.stop_reason = "retry does not fit remaining envelope"
        walls = rt.planner.walls(self._incumbent(crun) or {})
        self.store.append("UW_PROBE_BATCH", {
            "candidate_id": crun.candidate.candidate_id,
            "space_id": crun.space.space_id,
            "n_planned": len(planned), "n_dispatched": len(kept),
            "n_answered": answered, "elapsed_s": round(elapsed, 1),
            "is_retry": is_retry,
            "walls": [w.payload() for w in walls],
            "corner_witnesses": rt.planner.corner_witness(self._incumbent(crun) or {}),
            **rt.planner.snapshot()})

    # ------------------------------------------------------------------ scan admission

    def _admit_and_enqueue(self, rt: ScanRuntime, crun: Any, tuner: Any) -> None:
        try:
            incumbent = self._incumbent(crun)
            walls = rt.planner.walls(incumbent) if incumbent else []
            fits = self._fits_screen_map(rt, crun)
            b_remaining = tuner.budget - tuner._asked
            block = rt.scanner.next_block(walls, fits, b_remaining)
            if block is None:
                return
            accepted, refused = [], []
            for point in block.points:
                reason = tuner.enqueue_fresh(ParamSet(values=dict(point.values)))
                key = ParamSet(values=dict(point.values)).key()
                if reason is None:
                    rt.pending_points.setdefault(key, []).append((block, point))
                    accepted.append(point.payload())
                else:
                    refused.append({**point.payload(), "refused": reason})
                    block.failed = True
                    block.told += 1              # a refused slot is a consumed opportunity
            self.store.append("SCAN_BLOCK_ADMITTED", {
                "candidate_id": crun.candidate.candidate_id,
                "space_id": crun.space.space_id,
                "scan_id": block.scan_id, "kind": block.kind, "axis": block.axis,
                "partner_key": block.partner_key,
                "enqueued": accepted, "refused": refused,
                "token": block.token.payload() if block.token else None,
                "q_used": rt.scanner.q_used, "q_total": rt.scanner.q_total,
                "phase_after": rt.scanner.phase})
            self.last_phase = rt.scanner.phase
        except Exception as exc:  # noqa: BLE001 — scan admission never ends a candidate
            self.store.append("SCAN_ADMIT_FAILED", {
                "candidate_id": crun.candidate.candidate_id,
                "error": f"{type(exc).__name__}: {exc}"[:300]})

    def _fits_screen_map(self, rt: ScanRuntime, crun: Any) -> dict[str, bool]:
        out: dict[str, bool] = {}
        for key, ans in rt.planner.answers.items():
            if ans.fit == "fit":
                out[key] = True
            elif ans.fit == "refused":
                out[key] = False
        return out

    # ------------------------------------------------------------------ shared helpers

    @staticmethod
    def _incumbent(crun: Any) -> dict | None:
        best: tuple[float, dict] | None = None
        for t in crun.trials:
            if t.status != "complete" or t.latency_ms is None or t.params is None:
                continue
            ms = t.latency_ms.robust_ms
            if best is None or ms < best[0]:
                best = (ms, dict(t.params.values))
        return best[1] if best else None

    def _journal_block(self, crun: Any, block: ScanBlock, token: Any) -> None:
        contrast = block.contrast
        payload: dict[str, Any] = {
            "candidate_id": crun.candidate.candidate_id,
            "scan_id": block.scan_id, "kind": block.kind, "axis": block.axis,
            "failed": block.failed,
        }
        if contrast is not None:
            payload.update({
                "contrast_id": contrast.contrast_id,
                "full": contrast.full, "g_d": contrast.g_d, "y": contrast.y,
                "f_lat": contrast.f_lat, "n_lat": contrast.n_lat,
            })
            if contrast.mint is not None:
                m = contrast.mint
                payload.update({
                    "direction": m.direction, "reason": m.reason,
                    "rho_f": m.rho_f, "rho_n": m.rho_n,
                    "j_out": list(m.outward.interval), "j_in": list(m.inward.interval),
                    "tau": m.outward.tau,
                })
        payload["token_minted"] = token.payload() if token is not None else None
        self.store.append("SCAN_BLOCK_DONE", payload)

    def final_snapshot(self, rt: ScanRuntime) -> dict[str, Any]:
        return {"mode": rt.mode, "scanner": rt.scanner.snapshot(),
                "probe": rt.planner.snapshot()}

    # ------------------------------------------------------------------ §6 brief

    def brief_text(self, rt: ScanRuntime, crun: Any) -> str | None:
        """v4.1 §6: the conditioned wall/slope brief for the rewriter/analyst prompt.

        Conditioned wording throughout ("under the partners of this candidate's best
        configuration…"); gate-passing slopes carry numbers + disclosure; the original
        wallward g is reported WITH ITS SIGN even when the action direction is inward;
        unresolved contrasts say so; E1 results are cited at their actual evidence level
        (one new measured point, never a certified slope).
        """
        incumbent = self._incumbent(crun)
        if incumbent is None:
            return None
        walls = rt.planner.walls(incumbent)
        contrasts = [c for c in self.tokens.contrasts
                     if c.space_id == crun.space.space_id]
        corners = rt.planner.corner_witness(incumbent)
        if not walls and not contrasts and not corners:
            return None
        lines: list[str] = [
            "`analysis/resource_walls.md` — CONDITIONED measurements (v4.1): every fact "
            "below was measured with all other knobs frozen at the incumbent's partners. "
            "A wall here is a compiler refusal beside a fitting neighbour under those exact "
            "partners; it says nothing about other partner settings."]
        for w in walls:
            seg = (f"- [{w.kind} wall] axis `{w.axis}` under partners "
                   f"{w.partner_values}: point map {w.point_map}")
            if w.refused_value is not None:
                seg += (f"; refused at {w.refused_value!r}"
                        + (f" (requires {w.over_ratio:.2f}x the limit)"
                           if w.over_ratio else ""))
            if w.witness_kernel:
                seg += f"; witness kernel `{w.witness_kernel}`"
            lines.append(seg)
        for c in contrasts:
            if not c.full or c.mint is None:
                lines.append(f"- [contrast {c.contrast_id}] axis `{c.axis}`: incomplete "
                             "(provisional; no certified slope)")
                continue
            m = c.mint
            wallward = f"{m.outward.center:+.1%}"
            seg = (f"- [measured slope] axis `{c.axis}` F={c.f_value!r} vs N={c.n_value!r} "
                   f"under partners {c.partner_values}: toward-wall gain {wallward} "
                   f"(interval [{m.outward.interval[0]:+.1%}, {m.outward.interval[1]:+.1%}], "
                   f"rho_F={m.rho_f:.1%}, rho_N={m.rho_n:.1%}, tau={m.outward.tau:.1%}; "
                   f"empirical re-measurement envelope, NOT a 95% CI; "
                   f"2 fresh trials/end, ~20 timings/trial)")
            if m.direction == "inward":
                seg += (f"; action direction INWARD on the reversed reading "
                        f"[{m.inward.interval[0]:+.1%}, {m.inward.interval[1]:+.1%}] — "
                        "the original wallward sign above still stands")
            elif m.direction is None:
                seg += "; slope UNRESOLVED at this noise level"
            lines.append(seg)
        for cw in corners:
            lines.append(
                f"- [interaction witness] axes {cw['axes']} at {cw['values']}: each "
                "single change fits on its own, the combined corner is refused — a joint "
                "budget fact, more directive than either single-axis wall.")
        e_points = [b for b in rt.scanner.completed if b.kind in ("E1", "E2")
                    and not b.failed]
        for b in e_points:
            lat = b.lat.by_role.get("E_new")
            if lat is not None:
                lines.append(
                    f"- [E action result] axis `{b.axis}` {b.points[0].direction}: one new "
                    f"measured point at {b.points[0].values.get(b.axis)!r} -> {lat:.4f} ms "
                    "(single measurement; not a certified slope)")
        return "\n".join(lines) if len(lines) > 1 else None
