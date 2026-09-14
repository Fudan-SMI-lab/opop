"""v4.1 §4: the C/E four-slot phase machine — sources (C4) and actions (E1/E2/E4).

One instance per candidate SPACE (phase carries over into same-candidate continuation
spaces via `initial_phase` / inherited TokenStore). The scanner owns NO GPU work: it plans
scan trials, the orchestrator runs them through the ordinary production pipeline with
fresh-measurement intent, and tells the results back in.

Budget (§4): Q = floor(0.1 * B). C4 atomically reserves 4 B-slots + 4 scan opportunities;
E1/E2/E4 reserve their action length e = 1/2/4. Nothing starts unless the reservation fits;
un-asked reserved slots return to TPE and never become retry credit.

Phase (§4.4, the author-selected table): ordinary admission flips by REQUESTED phase;
token-priority admission is its own path that executes the token's REGISTERED action kind
and explicitly sets phase C on success — no flip-restore.

C4 protocol (§4.1, inherited v3): discovery pair + validation pair, each internally
randomized, forced mirror ordering FORBIDDEN (ABBA under linear drift manufactures
−(bΔ)² covariance); geometry frozen before the first slot (F = inner, N = wall-side;
θ* on the boundary → N = A, F = inward control); old winners leave the baseline: all four
endpoints freshly measured.

E geometry (§4.2, direction-mirrored):
  outward — nearest unmeasured screen-fit value beyond N toward the wall; then in-interval
            values nearest N; never across unknown/refused/component gaps.
  inward  — nearest unmeasured screen-fit value beyond F away from the wall; then
            in-interval values nearest F.

Ordering (§4.2): stratify by declared choice span; rotate strata; within a stratum sort by
the DIRECTION-AWARE gain score (positive for the minted direction), then fewest E
admissions, longest wait, stable ID.
"""

from __future__ import annotations

import random
import uuid
from dataclasses import dataclass, field
from typing import Any, Literal

from kernel_optimizer.conditional import gates
from kernel_optimizer.conditional.identity import (
    config_key,
    execution_identity,
    partner_projection,
)
from kernel_optimizer.conditional.probe import ConditionedWall
from kernel_optimizer.conditional.tokens import SourceContrast, Token, TokenStore
from kernel_optimizer.models.core import ParameterSpace

Phase = Literal["C", "E"]
SlotRole = Literal["F_d", "N_d", "F_v", "N_v", "E_new", "E_anchor"]


@dataclass
class ScanPoint:
    """One planned scan trial. `values` is the FULL committed configuration — partners are
    frozen at admission and never replaced by live partners (§4.3)."""

    scan_id: str
    role: SlotRole
    axis: str
    values: dict[str, Any]
    order: int
    token_id: str | None = None
    direction: str | None = None

    def payload(self) -> dict[str, Any]:
        return {"scan_id": self.scan_id, "role": self.role, "axis": self.axis,
                "order": self.order, "token_id": self.token_id,
                "direction": self.direction}


@dataclass
class SourceLatencies:
    """Role-keyed C4 latencies. Discovery/validation membership comes from the ROLE the
    geometry froze, never from arrival order — the two pairs execute in randomized order,
    so `lat[0]` is not 'the discovery value'."""

    by_role: dict[str, float] = field(default_factory=dict)


@dataclass
class ScanBlock:
    """An admitted C4 or E action: its points, its reservation, its life-cycle."""

    scan_id: str
    kind: Literal["C4", "E1", "E2", "E4"]
    axis: str
    partner_key: str
    points: list[ScanPoint]
    contrast: SourceContrast | None = None      # C4 only, frozen at admission
    token: Token | None = None                  # E only
    told: int = 0
    failed: bool = False
    lat: SourceLatencies = field(default_factory=SourceLatencies)

    @property
    def done(self) -> bool:
        return self.told >= len(self.points)


@dataclass
class EligibilityRecord:
    """§8 decision record: what the E decision saw, for the RR-vs-gain identifiability
    reporting. RR target is computed from the same snapshot, pure, no TPE involvement."""

    n_pending: int
    n_eligible: int
    competition_by_span: dict[int, int]
    rr_target: str | None
    gain_target: str | None
    chosen: str | None
    differed: bool


class ConditionalScanner:
    """The §4 phase machine for one candidate space."""

    def __init__(self, space: ParameterSpace, candidate_id: str, backend: str,
                 budget_b: int, tokens: TokenStore, seed: int = 0,
                 initial_phase: Phase = "C", f_frac: float = 0.1) -> None:
        self.space = space
        self.candidate_id = candidate_id
        self.backend = backend
        self.identity = execution_identity(candidate_id, space.space_id, backend)
        self.tokens = tokens
        self.q_total = int(f_frac * budget_b)
        self.q_used = 0
        self.b_reserved = 0
        self.phase: Phase = initial_phase
        self.rng = random.Random(seed)
        self.active: ScanBlock | None = None
        self.completed: list[ScanBlock] = []
        self.priority_token_id: str | None = None   # completion hook's one-shot priority
        # §4.1 coverage: axis+partner combinations with an admitted C4 (attempt-keyed).
        self._c4_admitted: dict[str, int] = {}
        self._e_admitted: dict[str, int] = {}
        self._wait_seq = 0
        self._queue_entry: dict[str, int] = {}
        self.eligibility_log: list[EligibilityRecord] = []
        self.funnel: list[dict[str, str]] = []       # eight-gate tri-state records
        # Axis values measured by ORDINARY trials, keyed (axis, partner_projection).
        # Instance state: a class-level mutable would silently share evidence across
        # candidates, which is exactly what ruling ⑥ forbids.
        self._external: dict[tuple[str, str], set[str]] = {}

    # ------------------------------------------------------------------ admission

    def q_remaining(self) -> int:
        return self.q_total - self.q_used

    def next_block(self, walls: list[ConditionedWall],
                   fits_screen: "dict[str, bool] | None" = None,
                   b_remaining: int = 1 << 30) -> ScanBlock | None:
        """Decide and ADMIT the next scan block, or None. Called at a cadence checkpoint or
        from the completion hook. Consumes reservation on admission. `b_remaining` is the
        tuner's unasked B budget — a block must fit BOTH Q and B (atomic reservation)."""
        if self.active is not None:
            return None                        # one in-flight block per space
        # 1. token-priority path (§4.4): executes the REGISTERED action kind, sets phase C.
        if self.priority_token_id is not None:
            tok = next((t for t in self.tokens.pending(self.identity)
                        if t.token_id == self.priority_token_id), None)
            self.priority_token_id = None
            if tok is not None:
                block = self._admit_e(tok, fits_screen, b_remaining)
                if block is not None:
                    self.phase = "C"
                    return block
        # 2. ordinary admission by requested phase, fallback to the other queue.
        requested = self.phase
        for served in ([requested, "E" if requested == "C" else "C"]):
            block = (self._admit_c4(walls, b_remaining) if served == "C"
                     else self._admit_best_e(fits_screen, b_remaining))
            if block is not None:
                # flip by REQUESTED phase (v3 rule, kept by §4.4).
                self.phase = "E" if requested == "C" else "C"
                return block
        return None

    # ---------------------------------------------------------------- C4 admission

    def _admit_c4(self, walls: list[ConditionedWall],
                  b_remaining: int = 1 << 30) -> ScanBlock | None:
        if self.q_remaining() < 4 or b_remaining < 4:
            return None
        candidates: list[tuple[int, int, str, ConditionedWall]] = []
        for w in walls:
            if w.f_value is None or w.n_value is None or repr(w.f_value) == repr(w.n_value):
                continue   # needs two distinct FIT endpoints in one component
            cov = f"{w.axis}|{w.partner_key}"
            if cov not in self._queue_entry:
                self._wait_seq += 1
                self._queue_entry[cov] = self._wait_seq
            candidates.append((self._c4_admitted.get(cov, 0),
                               self._queue_entry[cov], cov, w))
        if not candidates:
            return None
        # fewest admitted attempts → longest waiting (earliest entry) → stable ID.
        candidates.sort(key=lambda t: (t[0], t[1], t[2]))
        _, _, cov, wall = candidates[0]
        self._c4_admitted[cov] = self._c4_admitted.get(cov, 0) + 1

        scan_id = f"c4-{uuid.uuid4().hex[:8]}"
        base = dict(wall.partner_values)
        f_vals = {**base, wall.axis: wall.f_value}
        n_vals = {**base, wall.axis: wall.n_value}
        # Discovery pair and validation pair, EACH independently randomized; never forced
        # mirror (§4.1). Roles are frozen by GEOMETRY here, before any slot runs.
        d_pair = [("F_d", f_vals), ("N_d", n_vals)]
        v_pair = [("F_v", f_vals), ("N_v", n_vals)]
        self.rng.shuffle(d_pair)
        self.rng.shuffle(v_pair)
        points = [ScanPoint(scan_id, role, wall.axis, dict(vals), i)
                  for i, (role, vals) in enumerate([*d_pair, *v_pair])]
        contrast = SourceContrast(
            contrast_id=f"sc-{uuid.uuid4().hex[:8]}",
            comparable_key=(f"{self.identity}|{wall.partner_key}|{wall.axis}"
                            f"|F={wall.f_value!r}|N={wall.n_value!r}|C4"),
            execution_identity=self.identity,
            candidate_id=self.candidate_id, space_id=self.space.space_id,
            axis=wall.axis, partner_projection=wall.partner_key,
            partner_values=base, f_value=wall.f_value, n_value=wall.n_value)
        self.q_used += 4
        self.active = ScanBlock(scan_id, "C4", wall.axis, wall.partner_key,
                                points, contrast=contrast)
        return self.active

    # ---------------------------------------------------------------- E admission

    def _eligible_tokens(self, fits_screen: "dict[str, bool] | None") -> list[Token]:
        out = []
        for tok in self.tokens.pending(self.identity):
            if self._new_value_for(tok, fits_screen) is not None:
                out.append(tok)
        return out

    def _admit_best_e(self, fits_screen: "dict[str, bool] | None",
                      b_remaining: int = 1 << 30) -> ScanBlock | None:
        pending = self.tokens.pending(self.identity)
        eligible = self._eligible_tokens(fits_screen)
        record = self._record_decision(pending, eligible)
        if not eligible:
            return None
        chosen = self._rank(eligible)[0]
        record.chosen = chosen.token_id
        return self._admit_e(chosen, fits_screen, b_remaining)

    def _rank(self, eligible: list[Token]) -> list[Token]:
        """§4.2: stratify by declared choice span, rotate strata, sort within a stratum by
        direction-aware gain (desc) → fewest E admissions → longest wait → stable ID."""
        def sort_key(tok: Token):
            contrast = self.tokens.contrast_of(tok)
            domain = next((d for d in self.space.domains if d.name == tok.axis), None)
            span = contrast.choice_span(list(domain.choices)) if domain else 0
            gain = contrast.gain_score() or 0.0
            adm = self._e_admitted.get(tok.axis, 0)
            if tok.token_id not in self._queue_entry:
                self._wait_seq += 1
                self._queue_entry[tok.token_id] = self._wait_seq
            return (span, -gain, adm, self._queue_entry[tok.token_id], tok.token_id)
        return sorted(eligible, key=sort_key)

    def _rr_target(self, eligible: list[Token]) -> str | None:
        """RR arm's counterfactual pick from the SAME snapshot: stable rotation over the
        same eligible set (by axis's stable declared index then token id) — pure
        computation, no TPE ask, no RNG consumption (§8)."""
        if not eligible:
            return None
        names = [d.name for d in self.space.domains]
        ordered = sorted(eligible,
                         key=lambda t: (names.index(t.axis) if t.axis in names else 99,
                                        t.token_id))
        idx = sum(self._e_admitted.values()) % len(ordered)
        return ordered[idx].token_id

    def _record_decision(self, pending: list[Token], eligible: list[Token]) -> EligibilityRecord:
        spans: dict[int, int] = {}
        for tok in eligible:
            contrast = self.tokens.contrast_of(tok)
            domain = next((d for d in self.space.domains if d.name == tok.axis), None)
            span = contrast.choice_span(list(domain.choices)) if domain else 0
            spans[span] = spans.get(span, 0) + 1
        gain_target = self._rank(eligible)[0].token_id if eligible else None
        rr_target = self._rr_target(eligible)
        rec = EligibilityRecord(
            n_pending=len(pending), n_eligible=len(eligible),
            competition_by_span=spans, rr_target=rr_target, gain_target=gain_target,
            chosen=None, differed=(rr_target is not None and rr_target != gain_target))
        self.eligibility_log.append(rec)
        return rec

    def _admit_e(self, tok: Token, fits_screen: "dict[str, bool] | None",
                 b_remaining: int = 1 << 30) -> ScanBlock | None:
        e_len = {"E1": 1, "E2": 2, "E4": 4}[tok.action_kind]
        if self.q_remaining() < e_len or b_remaining < e_len:
            return None                          # §5: preregistered kinds never downgrade
        new_value = self._new_value_for(tok, fits_screen)
        if new_value is None:
            return None
        contrast = self.tokens.contrast_of(tok)
        # ADMISSION consumes (§4.1) — applicability was checked against current partners
        # here; the committed points keep these exact partners even if the incumbent moves
        # later (§4.3: current-at-admission).
        self.tokens.consume(tok.token_id, f"admitted {tok.action_kind}")
        self._e_admitted[tok.axis] = self._e_admitted.get(tok.axis, 0) + 1
        scan_id = f"e-{uuid.uuid4().hex[:8]}"
        base = dict(contrast.partner_values)
        points = [ScanPoint(scan_id, "E_new", tok.axis,
                            {**base, tok.axis: new_value}, 0,
                            token_id=tok.token_id, direction=tok.direction)]
        if tok.action_kind == "E2":
            anchor_val = contrast.f_value if tok.direction == "inward" else contrast.n_value
            points.append(ScanPoint(scan_id, "E_anchor", tok.axis,
                                    {**base, tok.axis: anchor_val}, 1,
                                    token_id=tok.token_id, direction=tok.direction))
        self.q_used += e_len
        self.active = ScanBlock(scan_id, tok.action_kind, tok.axis,
                                contrast.partner_projection, points, token=tok)
        return self.active

    def _new_value_for(self, tok: Token,
                       fits_screen: "dict[str, bool] | None") -> Any | None:
        """§4.2 direction-mirrored geometry. `fits_screen` maps full-config point keys to
        screen-fit booleans (from the probe layer); None means unknown — never crossed."""
        contrast = self.tokens.contrast_of(tok)
        domain = next((d for d in self.space.domains if d.name == tok.axis), None)
        if domain is None:
            return None
        choices = list(domain.choices)
        try:
            fi, ni = choices.index(contrast.f_value), choices.index(contrast.n_value)
        except ValueError:
            return None
        if fi == ni:
            return None
        measured = set()
        for p in self._measured_axis_values(tok.axis, contrast.partner_projection):
            measured.add(p)
        step = 1 if ni > fi else -1

        def screen_ok(value: Any) -> bool:
            if fits_screen is None:
                return True
            key = ConditionalScannerKeys.point_key(
                {**contrast.partner_values, tok.axis: value})
            verdict = fits_screen.get(key)
            return verdict is True                # unknown/refused: never crossed

        def first_unmeasured(indices: list[int]) -> Any | None:
            for i in indices:
                if not (0 <= i < len(choices)):
                    return None                   # ran off the domain
                v = choices[i]
                if not screen_ok(v):
                    return None                   # gap: stop, do not step across
                if repr(v) not in measured:
                    return v
            return None

        if tok.direction == "outward":
            beyond = list(range(ni + step, ni + step * len(choices), step))
            inside = list(range(ni - step, fi, -step))
            fallback = [fi - step] if 0 <= fi - step < len(choices) else []
        else:
            beyond = list(range(fi - step, fi - step * len(choices), -step))
            inside = list(range(fi + step, ni, step))
            fallback = []
        for indices in (beyond, inside, fallback):
            v = first_unmeasured(indices)
            if v is not None:
                return v
        return None

    def _measured_axis_values(self, axis: str, partner_key: str) -> set[str]:
        out: set[str] = set(self._external.get((axis, partner_key), set()))
        for block in [*self.completed, *( [self.active] if self.active else [] )]:
            if block.partner_key != partner_key or block.axis != axis:
                continue
            for p in block.points[: block.told]:
                out.add(repr(p.values.get(axis)))
        return out

    def note_external_measurement(self, values: dict[str, Any]) -> None:
        """Ordinary TPE trials also measure axis values; fold them in so E never re-buys
        a measured point. Keyed per axis under ITS partner projection."""
        for d in self.space.domains:
            if d.name not in values:
                continue
            pk = partner_projection(values, d.name)
            self._external.setdefault((d.name, pk), set()).add(repr(values[d.name]))

    # ---------------------------------------------------------------- results

    def tell(self, block: ScanBlock, point: ScanPoint, latency_ms: float | None,
             ok: bool) -> "Token | None":
        """Fold one scan trial result in. Returns a freshly minted token when the block is
        a C4 that just completed and passed its gates (the completion hook then takes it)."""
        block.told += 1
        if not ok or latency_ms is None or latency_ms <= 0:
            block.failed = True
        elif block.contrast is not None:
            block.lat.by_role[point.role] = latency_ms
        if not block.done:
            return None
        self.completed.append(block)
        self.active = None
        if block.kind != "C4" or block.contrast is None:
            return None
        c = block.contrast
        roles = block.lat.by_role
        if block.failed or not all(r in roles for r in ("F_d", "N_d", "F_v", "N_v")):
            c.full = False                       # incomplete: provisional, never mints
            self.tokens.register(c)
            self._funnel_record(c, "F", "fail" if block.failed else "unknown")
            return None
        import math as _m
        # Role-keyed, never arrival-order: pairs execute in randomized order (§4.1).
        c.f_lat = [roles["F_d"], roles["F_v"]]
        c.n_lat = [roles["N_d"], roles["N_v"]]
        c.g_d = 1.0 - _m.exp(_m.log(roles["N_d"]) - _m.log(roles["F_d"]))
        c.y = 1.0 - _m.exp(_m.log(roles["N_v"]) - _m.log(roles["F_v"]))
        c.mint = gates.evaluate_gates(c.f_lat, c.n_lat)
        token = self.tokens.register(c)
        self._funnel_record(c, "G", "pass" if token is not None else "fail")
        if token is not None:
            self.priority_token_id = token.token_id     # completion hook: one-shot priority
        return token

    def _funnel_record(self, c: SourceContrast, first_fail_gate: str, state: str) -> None:
        self.funnel.append({"contrast_id": c.contrast_id, "gate": first_fail_gate,
                            "state": state})

    # ---------------------------------------------------------------- reporting

    def snapshot(self) -> dict[str, Any]:
        decisions = self.eligibility_log
        return {
            "q_total": self.q_total, "q_used": self.q_used, "phase": self.phase,
            "blocks_completed": [
                {"scan_id": b.scan_id, "kind": b.kind, "axis": b.axis,
                 "failed": b.failed} for b in self.completed],
            "tokens": self.tokens.snapshot(),
            "e_decisions": len(decisions),
            "e_decisions_with_competition": sum(1 for r in decisions if r.n_eligible >= 2),
            "rr_gain_differed": sum(1 for r in decisions if r.differed),
            "funnel": self.funnel,
        }


class ConditionalScannerKeys:
    """Key helper shared with the probe layer (same typed point identity)."""

    @staticmethod
    def point_key(values: dict[str, Any]) -> str:
        from kernel_optimizer.conditional.identity import typed_repr
        return ";".join(f"{k}={typed_repr(values[k])}" for k in sorted(values))


__all__ = ["ConditionalScanner", "ScanBlock", "ScanPoint", "EligibilityRecord",
           "config_key"]
