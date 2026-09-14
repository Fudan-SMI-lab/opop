"""v4.1 §3: the U/W probe layer — conditioned, compile-only, batched.

Answers ONE question per point: "under these exact frozen partners, does this axis value
fit or is it refused, and what is its resource vector?" — no latency, no launch. The batch
runs through the SAME `run_compile_probe` worker path the production screen uses (7 ms
marginal per point, ~11 s per batch), but into a SEPARATE diagnostic cache: this layer never
writes the production `_screen_cache`, never emits a production refusal, never touches
guard/PRUNED (v4.1 §3 isolation).

Point categories (v4.1 §3):
  U — axis points not yet answered under the CURRENT frozen partners (fewest-committed axis
      first; within an axis, nearest declared neighbours outward, lower before upper).
  W — bounded category rotation: missing inward controls → observed-transition neighbours →
      at most ONE 2x2 corner template per batch (TARGETED/UNFILTERED alternating) → remote
      refusal walk-back (≤2 dependency layers, lower layer waits for the next batch).
      Cursors advance on ATTEMPT, not success; completions are recorded separately.

Wall criteria (v4.1 §2, the conditioned replacements for the marginal gatekeepers):
  hard — a probed point whose compiler-reported max_shared EXCEEDS the device limit under
         the frozen partners, beside at least one FIT point on the inner side. Non-monotone
         point maps are recorded as-is ([fit, refused, fit, unknown] keeps BOTH local edges);
         no bisecting, no extrapolating, no "first infeasible" claim.
  soft — a locally RISING n_spills segment toward the high side under frozen partners,
         witness kernel tracked per point, missing fields stay unknown (never zero).

Budget (v4.1 §3 five-branch bootstrap): D = min(2*m_ref, 600s) once a reference batch cost
exists. The retry rule is the frozen-order prefix with n2 = ceil(n1/2) — members are chosen
by the frozen declaration order, never by which points succeeded.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Literal

from kernel_optimizer.conditional.identity import partner_projection, typed_repr
from kernel_optimizer.models.core import ParameterSpace

PCAP_DEFAULT = 48
MAX_ATTEMPTS = 4
BOOTSTRAP_TOTAL_S = 600.0

Fit = Literal["fit", "refused", "unknown"]


@dataclass
class ProbeAnswer:
    fit: Fit
    max_shared: int | None = None
    limit: int | None = None
    kernels: list[dict[str, Any]] = field(default_factory=list)
    coverage: str = "reached_kernels"     # forward exceptions are swallowed upstream, so
    reason: str | None = None             # only reached kernels are ever observed

    def max_spills(self) -> tuple[int | None, str | None]:
        """(max n_spills, witness kernel name); (None, None) when no kernel reports the
        field — missing is unknown, never zero (the getattr-false-zero lesson)."""
        best: tuple[int, str] | None = None
        for k in self.kernels:
            v = k.get("n_spills")
            if isinstance(v, int) and (best is None or v > best[0]):
                best = (v, str(k.get("name")))
        return best if best else (None, None)


@dataclass
class ConditionedWall:
    """One conditioned wall: axis + frozen partners + local edge(s). Both hard and soft
    walls carry the SAME shape so the brief and the scanner treat them uniformly."""

    kind: Literal["hard", "soft"]
    axis: str
    partner_key: str
    partner_values: dict[str, Any]
    # Declared-order point map over this axis: choice -> fit/refused/unknown ("hard"), or
    # choice -> n_spills/None ("soft").
    point_map: dict[str, str]
    # The two C4 endpoints this wall proposes: F (inner) and N (wall-side), both FIT.
    f_value: Any
    n_value: Any
    refused_value: Any | None            # hard only: the refused point beside N
    witness_kernel: str | None = None
    over_ratio: float | None = None      # hard: required / limit at the refused point

    def payload(self) -> dict[str, Any]:
        return {"kind": self.kind, "axis": self.axis, "partner_key": self.partner_key,
                "point_map": self.point_map, "f_value": repr(self.f_value),
                "n_value": repr(self.n_value),
                "refused_value": repr(self.refused_value) if self.refused_value is not None
                else None,
                "witness_kernel": self.witness_kernel, "over_ratio": self.over_ratio}


@dataclass
class DBudget:
    """The §3 five-branch bootstrap ledger. All costs in seconds, all spends recorded."""

    m_ref: float | None = None           # reference full-batch cost, once known
    frozen_d: float | None = None        # min(2*m_ref, 600)
    spent_s: float = 0.0
    bootstrap_spent_s: float = 0.0
    attempts_started: int = 0
    retried: bool = False
    stopped: bool = False                # active diagnostics stopped for this space
    stop_reason: str | None = None

    def admit(self, projected_s: float) -> bool:
        if self.stopped or self.attempts_started >= MAX_ATTEMPTS:
            return False
        if self.frozen_d is not None:
            return self.spent_s + projected_s <= self.frozen_d
        # Bootstrap: pre-declared 600s total envelope.
        return self.bootstrap_spent_s + projected_s <= BOOTSTRAP_TOTAL_S

    def record(self, cost_s: float, timed_out: bool, failed: bool, n_points: int) -> None:
        self.spent_s += cost_s
        self.attempts_started += 1
        if self.frozen_d is not None:
            return
        self.bootstrap_spent_s += cost_s
        if not timed_out and not failed:
            # First complete success freezes D from ITS cost; a later retry success never
            # re-freezes (v4.1 §3: no re-opening D from retry cost).
            self.m_ref = cost_s
            self.frozen_d = min(2.0 * cost_s, BOOTSTRAP_TOTAL_S)
            return
        if timed_out and not self.retried and n_points >= 2:
            self.retried = True          # one legal smaller retry may follow
            return
        self.stopped = True
        self.stop_reason = ("non-timeout failure in bootstrap" if failed
                            else "timeout with no legal retry")

    def after_retry(self, cost_s: float, timed_out: bool, failed: bool) -> None:
        self.spent_s += cost_s
        self.bootstrap_spent_s += cost_s
        self.attempts_started += 1
        # Whatever the retry outcome: active diagnostics stop; the answer (if any) is kept;
        # D is NOT re-frozen from the retry cost (170s+30s=200s, never D=60).
        self.stopped = True
        self.stop_reason = "post-retry stop (retry " + ("succeeded" if not (timed_out or failed)
                                                        else "failed") + ")"

    def snapshot(self) -> dict[str, Any]:
        return {"m_ref": self.m_ref, "frozen_d": self.frozen_d,
                "spent_s": round(self.spent_s, 1),
                "bootstrap_spent_s": round(self.bootstrap_spent_s, 1),
                "attempts_started": self.attempts_started, "retried": self.retried,
                "stopped": self.stopped, "stop_reason": self.stop_reason}


class ProbePlanner:
    """Plans U/W batches and derives conditioned walls from the answers.

    One instance per candidate SPACE. Answers are keyed by the full typed configuration, so
    a partner change never aliases: the same axis value under different partners is a
    different point.
    """

    def __init__(self, space: ParameterSpace, shared_limit: int,
                 legal: Callable[[dict[str, Any]], bool],
                 pcap: int = PCAP_DEFAULT) -> None:
        self.space = space
        self.limit = shared_limit
        self.legal = legal
        self.pcap = pcap
        self.answers: dict[str, ProbeAnswer] = {}
        self.budget = DBudget()
        self.in_flight = False
        # W rotation state (cursors advance on attempt; v4.1 §3).
        self._w_category = 0
        self._corner_cursor = 0
        self._corner_mode: Literal["targeted", "unfiltered"] = "targeted"
        self._walkback: list[tuple[dict[str, Any], dict[str, Any], int]] = []  # (r, a, layer)

    # ------------------------------------------------------------------ keys

    @staticmethod
    def point_key(values: dict[str, Any]) -> str:
        return ";".join(f"{k}={typed_repr(values[k])}" for k in sorted(values))

    def _answer_at(self, values: dict[str, Any]) -> ProbeAnswer | None:
        return self.answers.get(self.point_key(values))

    # ------------------------------------------------------------------ planning

    def ready_work(self, incumbent: dict[str, Any]) -> bool:
        return bool(self.plan_batch(incumbent))

    def plan_batch(self, incumbent: dict[str, Any]) -> list[dict[str, Any]]:
        """U,W,U,W… rounds against evidence FROZEN at call time (batch-start freeze)."""
        picked: list[dict[str, Any]] = []
        seen: set[str] = set()
        u_queue = self._u_candidates(incumbent)
        w_queue = self._w_candidates(incumbent)
        take_u = True
        while len(picked) < self.pcap and (u_queue or w_queue):
            queue = u_queue if (take_u and u_queue) or not w_queue else w_queue
            values = queue.pop(0)
            take_u = not take_u
            key = self.point_key(values)
            if key in seen or key in self.answers or not self.legal(values):
                continue
            seen.add(key)
            picked.append(values)
        return picked

    def _u_candidates(self, incumbent: dict[str, Any]) -> list[dict[str, Any]]:
        """Unanswered axis points under the frozen partners: fewest-answered axis first,
        within an axis nearest declared neighbours outward (lower before upper)."""
        per_axis: list[tuple[int, str, list[dict[str, Any]]]] = []
        for d in self.space.domains:
            if d.name not in incumbent:
                continue
            answered = 0
            ordered: list[dict[str, Any]] = []
            choices = list(d.choices)
            try:
                center = choices.index(incumbent[d.name])
            except ValueError:
                center = 0
            # nearest-first ring around the incumbent value, lower side before upper.
            ring: list[int] = []
            for step in range(1, len(choices)):
                for idx in (center - step, center + step):
                    if 0 <= idx < len(choices):
                        ring.append(idx)
            for idx in [center, *ring]:
                values = dict(incumbent)
                values[d.name] = choices[idx]
                if self._answer_at(values) is not None:
                    answered += 1
                else:
                    ordered.append(values)
            per_axis.append((answered, d.name, ordered))
        per_axis.sort(key=lambda t: (t[0], t[1]))
        out: list[dict[str, Any]] = []
        for _, _, pts in per_axis:
            out.extend(pts)
        return out

    def _w_candidates(self, incumbent: dict[str, Any]) -> list[dict[str, Any]]:
        """Bounded category rotation: one request per non-empty category, then repeat."""
        cats = [self._missing_inward_controls(incumbent),
                self._transition_neighbours(incumbent),
                self._corner_template(incumbent),
                self._walkback_points()]
        out: list[dict[str, Any]] = []
        idx = self._w_category
        rounds = 0
        while rounds < 4 * max(len(c) for c in cats) + 4 and any(cats):
            cat = cats[idx % 4]
            idx += 1
            rounds += 1
            if cat:
                out.append(cat.pop(0))
        self._w_category = idx % 4
        return out

    def _missing_inward_controls(self, incumbent: dict[str, Any]) -> list[dict[str, Any]]:
        """For every axis with a refused answer, the inner neighbour of that refusal must be
        answered — the wall criterion is two-sided by construction."""
        out: list[dict[str, Any]] = []
        for d in self.space.domains:
            choices = list(d.choices)
            for i, c in enumerate(choices):
                values = dict(incumbent)
                values[d.name] = c
                ans = self._answer_at(values)
                if ans is None or ans.fit != "refused":
                    continue
                for j in range(i - 1, -1, -1):
                    inner = dict(incumbent)
                    inner[d.name] = choices[j]
                    if self._answer_at(inner) is None:
                        out.append(inner)
                        break
        return out

    def _transition_neighbours(self, incumbent: dict[str, Any]) -> list[dict[str, Any]]:
        """Unanswered points adjacent to an observed fit→refused transition."""
        out: list[dict[str, Any]] = []
        for d in self.space.domains:
            choices = list(d.choices)
            states = []
            for c in choices:
                values = dict(incumbent)
                values[d.name] = c
                a = self._answer_at(values)
                states.append(a.fit if a else None)
            for i in range(len(choices) - 1):
                pair = {states[i], states[i + 1]}
                if "fit" in pair and "refused" in pair:
                    for j in (i - 1, i + 2):
                        if 0 <= j < len(choices) and states[j] is None:
                            values = dict(incumbent)
                            values[d.name] = choices[j]
                            out.append(values)
        return out

    def _corner_template(self, incumbent: dict[str, Any]) -> list[dict[str, Any]]:
        """≤1 template per batch: the missing corners of one 2x2 over an axis pair.

        TARGETED rounds draw the pair from axes on which refusals differ from the incumbent;
        UNFILTERED rounds rotate over ALL choose(d_eligible, 2) pairs by stable declared
        index (v4.1 §3). Falls back to unfiltered when the targeted queue is empty, keeping
        the same pair-identity dedup.
        """
        eligible = [d for d in self.space.domains
                    if len(d.choices) >= 2 and d.name in incumbent]
        if len(eligible) < 2:
            return []
        pairs = [(a.name, b.name) for i, a in enumerate(eligible)
                 for b in eligible[i + 1:]]
        if self._corner_mode == "targeted":
            targeted = self._targeted_pairs(incumbent, pairs)
            chosen = targeted or pairs
        else:
            chosen = pairs
        pair = chosen[self._corner_cursor % len(chosen)]
        self._corner_cursor += 1                       # advance on ATTEMPT
        self._corner_mode = ("unfiltered" if self._corner_mode == "targeted" else "targeted")
        k1, k2 = pair
        d1 = next(d for d in eligible if d.name == k1)
        d2 = next(d for d in eligible if d.name == k2)
        v1 = self._one_step_up(d1, incumbent[k1])
        v2 = self._one_step_up(d2, incumbent[k2])
        if v1 is None or v2 is None:
            return []
        corners = [dict(incumbent),
                   {**incumbent, k1: v1},
                   {**incumbent, k2: v2},
                   {**incumbent, k1: v1, k2: v2}]
        return [c for c in corners if self._answer_at(c) is None]

    def _targeted_pairs(self, incumbent: dict[str, Any],
                        pairs: list[tuple[str, str]]) -> list[tuple[str, str]]:
        diff_axes: set[str] = set()
        for key, ans in self.answers.items():
            if ans.fit != "refused":
                continue
            parts = dict(p.split("=", 1) for p in key.split(";"))
            for d in self.space.domains:
                if d.name in incumbent and d.name in parts:
                    if parts[d.name] != typed_repr(incumbent[d.name]):
                        diff_axes.add(d.name)
        return [p for p in pairs if p[0] in diff_axes and p[1] in diff_axes]

    @staticmethod
    def _one_step_up(domain: Any, value: Any) -> Any | None:
        choices = list(domain.choices)
        try:
            i = choices.index(value)
        except ValueError:
            return None
        return choices[i + 1] if i + 1 < len(choices) else None

    def note_refusal(self, refused_params: dict[str, Any], incumbent: dict[str, Any]) -> None:
        """Feed a PRODUCTION refusal in as walk-back seed (read-only reference)."""
        if len(self._walkback) < 8:
            self._walkback.append((dict(refused_params), dict(incumbent), 0))

    def _walkback_points(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for r, a, layer in list(self._walkback):
            if layer >= 2:
                continue
            for axis in sorted(r):
                if axis not in a or r[axis] == a[axis]:
                    continue
                step = dict(r)
                step[axis] = a[axis]
                if self._answer_at(step) is None:
                    out.append(step)
        return out

    # ------------------------------------------------------------------ recording

    def record(self, results: dict[str, dict[str, Any]],
               planned: list[dict[str, Any]]) -> None:
        """Fold one batch's worker results in. `results` is keyed by materialized path;
        `planned` is the batch in dispatch order (path i ↔ planned i)."""
        self.in_flight = False
        for values, entry in zip(planned, results.values()):
            key = self.point_key(values)
            if not entry.get("ok"):
                self.answers[key] = ProbeAnswer(fit="unknown",
                                                reason=str(entry.get("reason"))[:200])
                continue
            max_shared = entry.get("max_shared")
            fit: Fit = "unknown"
            if isinstance(max_shared, int):
                fit = "refused" if max_shared > self.limit else "fit"
            self.answers[key] = ProbeAnswer(
                fit=fit, max_shared=max_shared, limit=self.limit,
                kernels=list(entry.get("kernels") or []))
        # Walk-back layers advance only for points that ACTUALLY answered refused.
        nxt: list[tuple[dict[str, Any], dict[str, Any], int]] = []
        for r, a, layer in self._walkback:
            ans = self._answer_at(r)
            if ans is not None and ans.fit == "refused" and layer < 2:
                nxt.append((r, a, layer + 1))
        self._walkback = nxt

    # ------------------------------------------------------------------ walls

    def walls(self, incumbent: dict[str, Any]) -> list[ConditionedWall]:
        out: list[ConditionedWall] = []
        pkey_cache: dict[str, str] = {}
        for d in self.space.domains:
            if d.name not in incumbent:
                continue
            choices = list(d.choices)
            states: list[tuple[Any, ProbeAnswer | None]] = []
            for c in choices:
                values = dict(incumbent)
                values[d.name] = c
                states.append((c, self._answer_at(values)))
            pkey = pkey_cache.setdefault(
                d.name, partner_projection({**incumbent}, d.name))
            partner_values = {k: v for k, v in incumbent.items() if k != d.name}
            out.extend(self._hard_walls(d.name, pkey, partner_values, states))
            soft = self._soft_wall(d.name, pkey, partner_values, states)
            if soft is not None:
                out.append(soft)
        return out

    def _hard_walls(self, axis: str, pkey: str, partners: dict[str, Any],
                    states: list[tuple[Any, ProbeAnswer | None]]) -> list[ConditionedWall]:
        """Every fit→refused local edge is a wall — non-monotone maps keep ALL edges."""
        point_map = {repr(c): (a.fit if a else "unknown") for c, a in states}
        walls: list[ConditionedWall] = []
        for i in range(len(states) - 1):
            (c_lo, a_lo), (c_hi, a_hi) = states[i], states[i + 1]
            if a_lo is None or a_hi is None:
                continue
            if a_lo.fit == "fit" and a_hi.fit == "refused":
                f_val = states[i - 1][0] if i >= 1 and states[i - 1][1] is not None \
                    and states[i - 1][1].fit == "fit" else None
                over = (a_hi.max_shared / a_hi.limit
                        if a_hi.max_shared and a_hi.limit else None)
                witness = None
                if a_hi.kernels:
                    top = max((k for k in a_hi.kernels if k.get("shared") is not None),
                              key=lambda k: k["shared"], default=None)
                    witness = str(top.get("name")) if top else None
                if f_val is None:
                    # Single fit point beside the refusal: N=that point, F unavailable →
                    # the scanner needs TWO fit endpoints, so this wall is recorded but
                    # not scannable until an inward control answers.
                    f_val = c_lo
                    n_val = c_lo
                else:
                    n_val = c_lo
                walls.append(ConditionedWall(
                    kind="hard", axis=axis, partner_key=pkey, partner_values=partners,
                    point_map=point_map, f_value=f_val, n_value=n_val,
                    refused_value=c_hi, witness_kernel=witness, over_ratio=over))
        return walls

    def _soft_wall(self, axis: str, pkey: str, partners: dict[str, Any],
                   states: list[tuple[Any, ProbeAnswer | None]]) -> ConditionedWall | None:
        """A locally rising spill segment toward the high side (≥2 consecutive measured
        points, strictly rising, ending at the highest measured choice)."""
        spills: list[tuple[Any, int, str]] = []
        for c, a in states:
            if a is None or a.fit != "fit":
                continue
            v, witness = a.max_spills()
            if v is not None:
                spills.append((c, v, witness or "?"))
        if len(spills) < 2:
            return None
        tail = spills[-2:]
        if not (tail[0][1] < tail[1][1] and tail[1][1] > 0):
            return None
        point_map = {repr(c): str(v) for c, v, _ in spills}
        return ConditionedWall(
            kind="soft", axis=axis, partner_key=pkey, partner_values=partners,
            point_map=point_map, f_value=tail[0][0], n_value=tail[1][0],
            refused_value=None, witness_kernel=tail[1][2])

    def corner_witness(self, incumbent: dict[str, Any]) -> list[dict[str, Any]]:
        """Group witnesses: complete 2x2s where each single change fits but the combined
        corner is refused. Requires ALL FOUR corners actually answered (v4.1 §3 —
        template attempt ≠ template completion; unknown stays unknown)."""
        out: list[dict[str, Any]] = []
        eligible = [d for d in self.space.domains
                    if len(d.choices) >= 2 and d.name in incumbent]
        for i, d1 in enumerate(eligible):
            v1 = self._one_step_up(d1, incumbent[d1.name])
            if v1 is None:
                continue
            for d2 in eligible[i + 1:]:
                v2 = self._one_step_up(d2, incumbent[d2.name])
                if v2 is None:
                    continue
                a00 = self._answer_at(dict(incumbent))
                a10 = self._answer_at({**incumbent, d1.name: v1})
                a01 = self._answer_at({**incumbent, d2.name: v2})
                a11 = self._answer_at({**incumbent, d1.name: v1, d2.name: v2})
                if not all(a is not None and a.fit != "unknown"
                           for a in (a00, a10, a01, a11)):
                    continue
                if (a00.fit == "fit" and a10.fit == "fit" and a01.fit == "fit"
                        and a11.fit == "refused"):
                    out.append({"axes": [d1.name, d2.name],
                                "values": [repr(v1), repr(v2)],
                                "verdict": "individually_fit_combined_refused"})
        return out

    def snapshot(self) -> dict[str, Any]:
        fits = sum(1 for a in self.answers.values() if a.fit == "fit")
        refused = sum(1 for a in self.answers.values() if a.fit == "refused")
        unknown = sum(1 for a in self.answers.values() if a.fit == "unknown")
        return {"n_answers": len(self.answers), "fit": fits, "refused": refused,
                "unknown": unknown, "budget": self.budget.snapshot()}


def retry_prefix(planned: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The §3 shrink rule, chosen and pre-declared: the frozen-order PREFIX of size
    ceil(n1/2). Members come from the frozen dispatch order — never from which points
    happened to answer in the timed-out batch."""
    return planned[: math.ceil(len(planned) / 2)]
