"""Item 2: the register-spill SOFT wall -- a knob whose range is clipped without being refused.

WHY A SECOND CRITERION FUNCTION AND NOT A PARAMETER ON `find_walls`. The shared-memory wall is a HARD
REFUSAL: the compiler's `metadata.shared` exceeds the device limit, the configuration cannot launch,
and the screen records it as `failure_kind == "infeasible_shared_memory"`. `find_walls`' ENTIRE input
is that list of refused parameter sets. Register spilling refuses nothing -- the configuration
compiles, launches and returns a latency, it is merely CLIPPED: the compiler runs out of registers and
puts live values in local memory, roughly two orders of magnitude slower. There is no refused
parameter set to read, so the existing function cannot see it however it is parameterised.

WHY `n_spills` AND NOT `occupancy`, WHICH WAS MEASURED AND REJECTED. Both passed the independence
threshold (|rho| >= 0.9 disqualifies; the largest was 0.497). What separated them is a second, stricter
test -- does the advice point where the measured optimum actually IS -- over 2788 completed trials and
56 candidates across two card types:

                             n_spills                     occupancy
  can it fire               43.8% of trials non-zero     87.6% of trials < 0.30 (nearly always true)
  shape of the relation     27 peaked / 18 monotone      37 PEAKED / 13 monotone
  winner's position in
    its own range           median 0.00, 56/56 at the    median 0.33, only 2/56 at the top
                            BOTTOM
  criterion writable?       yes, direction agrees        NO -- "raise occupancy" points AWAY from the
                            with the optimum             measured optimum

So "reduce spilling" never conflicts with where the optimum sits, because the optimum is already at
the bottom of its spill range. "Raise occupancy" would send the rewriter after a direction the
measurement contradicts. `occupancy_limiter` is still used here -- but as the EXPLANATION of a spill
wall (at the winners it reads 38 registers / 18 shared_memory), never as a wall of its own.

TWO LIMITS THAT MUST BE REPORTED RATHER THAN GLOSSED.

  1. APPLICABILITY IS 43%. 32 of 56 winners already spill ZERO, and a knob cannot be walled by
     spilling at a point that does not spill. So "no wall found" and "not applicable" are different
     states and this module returns the denominator for both. Reporting only "we found N" would make a
     structural non-event look like a negative result.
  2. THERE IS NO INDEPENDENT CONFIRMATION. The hard wall's attribution is checked by a second,
     independent question to the compiler (the ablation probe, 6 of 6 from the optimum). A soft wall
     has ONE observation of an already-measured field, and no probe can be added cheaply because
     "would this spill" is not a yes/no the compiler answers separately from compiling. Every rendered
     sentence says so.

WHAT THIS DOES NOT TOUCH. Nothing about the tuning loop: no configuration is refused, no sampling
changes, no trial budget moves, and Optuna receives exactly the states it received before. That is the
structural difference from S7 -- a soft wall cannot break trial-budget parity between runs, so it
needs no paired control arm. Like the hard wall, it records and never enforces.
"""

from __future__ import annotations

import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from kernel_optimizer.evaluation.wall_attribution import _as_num, _literal_for
from kernel_optimizer.models.reports import TuningStats

# Same floor and tail length as the hard wall, and for the same reasons: two points are monotone
# through any two points, and three matches the `at_boundary` convention already used by
# `TuningStatsAnalyzer`.
_MIN_VALUES = 3
_TAIL_LEN = 3


@dataclass
class SoftWall:
    """One knob along which the compiler's register spilling rises monotonically into a bound state.

    Deliberately NOT a `Wall`. Three fields of a hard wall have no meaning here and inventing values
    for them would let a reader take an unconfirmed observation for a probed one:
      * `refused_value`  -- nothing was refused; there is a value where spilling BEGINS.
      * `verdict`        -- there is no probe to return a verdict.
      * `second_origin`  -- no origin can be re-probed, so the hard wall's conditionality check
                            (measured 1 of 6 agreement, the single most load-bearing fact in the
                            2e design) has no counterpart. That absence is a weakness, and it is
                            written into the rendered text rather than hidden by a lookalike field.
    """

    param: str
    # The values actually measured, ascending, and the per-value medians beside them.
    ran_values: list[float]
    spills_by_value: list[float]
    latency_by_value: list[float]
    # Where spilling first becomes non-zero -- the analogue of "the smallest step that hits the wall".
    onset_value: float
    # Latency across the last `_TAIL_LEN` values, far-to-near, and its relative change. Positive means
    # latency is still IMPROVING toward the spilling end, i.e. the knob still has slope to give.
    tail_values: list[float] = field(default_factory=list)
    tail_latencies: list[float] = field(default_factory=list)
    tail_gain_pct: float = 0.0
    # The compiler's own account of what limits residency at the candidate's best trial -- "registers"
    # or "shared_memory". The EXPLANATION of a spill wall, never a wall itself.
    limiter_at_best: str | None = None
    spills_at_best: float | None = None

    def payload(self) -> dict[str, Any]:
        return {
            "kind": "n_spills",
            "param": self.param,
            "ran_values": list(self.ran_values),
            "spills_by_value": list(self.spills_by_value),
            "latency_by_value": [round(v, 6) for v in self.latency_by_value],
            "onset_value": self.onset_value,
            "tail_values": list(self.tail_values),
            "tail_latencies": [round(v, 6) for v in self.tail_latencies],
            "tail_gain_pct": round(self.tail_gain_pct, 2),
            "limiter_at_best": self.limiter_at_best,
            "spills_at_best": self.spills_at_best,
            # Recorded on every row so no reader can mistake this for a probed verdict, including a
            # reader of the raw log who never sees `for_prompt`'s wording.
            "probe_confirmed": False,
        }


@dataclass
class SoftWallScan:
    """The result of one candidate's scan, WITH its denominators.

    `walls` alone cannot be read: 0 walls because the winner never spilled and 0 walls because every
    knob's spill curve was flat are different facts, and 43% applicability means the first is the
    common case. `probe-needs-a-positive-control` in the other direction -- a count with no
    denominator is not a measurement.
    """

    walls: list[SoftWall] = field(default_factory=list)
    applicable: bool = False
    reason: str = ""
    n_knobs_examined: int = 0
    n_knobs_with_spill_variation: int = 0
    spills_at_best: float | None = None
    limiter_at_best: str | None = None

    def payload(self) -> dict[str, Any]:
        return {
            "applicable": self.applicable,
            "reason": self.reason,
            "n_knobs_examined": self.n_knobs_examined,
            "n_knobs_with_spill_variation": self.n_knobs_with_spill_variation,
            "spills_at_best": self.spills_at_best,
            "limiter_at_best": self.limiter_at_best,
            "n_walls": len(self.walls),
            "walls": [w.payload() for w in self.walls],
        }


def _profile_field(trial: Any, name: str) -> Any:
    """One profile field, whether the trial is a model or a replayed dict.

    `occupancy` is NESTED (`profile.occupancy["occupancy"]`) and a flat read returns None, which reads
    as "not measured" for a field that WAS measured -- already recorded as
    `occupancy-is-nested-and-a-flat-read-fakes-unmeasured`. That is why the limiter is fetched through
    this one place rather than inline at each use.
    """
    prof = getattr(trial, "profile", None)
    if prof is None and isinstance(trial, dict):
        prof = trial.get("profile")
    if prof is None:
        return None
    if isinstance(prof, dict):
        return prof.get(name)
    return getattr(prof, name, None)


def _occupancy_limiter(trial: Any) -> str | None:
    """The compiler's residency limiter, from the nested occupancy record.

    The key is `limiter`, NOT `occupancy_limiter`: `ProfileRecord.occupancy_limiter` is a PROPERTY that
    reads `self.occupancy["limiter"]`, so the dict a replayed trial carries has the short name. A first
    version of this function looked for `occupancy_limiter` in the dict and would have returned None on
    every real trial -- an unmeasured reading for a field that was measured, which is
    `occupancy-is-nested-and-a-flat-read-fakes-unmeasured` one level deeper.
    """
    prof = getattr(trial, "profile", None)
    if prof is None and isinstance(trial, dict):
        prof = trial.get("profile")
    # The model exposes the property; prefer it so this function cannot drift from the model's own
    # spelling of the key.
    if prof is not None and not isinstance(prof, dict):
        v = getattr(prof, "occupancy_limiter", None)
        if v is not None:
            return str(v)
    occ = _profile_field(trial, "occupancy")
    if isinstance(occ, dict):
        v = occ.get("limiter")
        return str(v) if v is not None else None
    return None


def _robust_ms(trial: Any) -> float | None:
    """Median else mean, reproduced here because `robust_ms` is a @PROPERTY and never serialized.

    A reader that looked for a `robust_ms` KEY in a replayed trial would find nothing and silently
    treat every trial as untimed. The fallback order is the project's measured objective: median ranks
    20-sample pairs correctly 93.2% of the time against the mean's 64.8%, and `min` is never used
    because it reports the luckiest sample.
    """
    lat = getattr(trial, "latency_ms", None)
    if lat is None and isinstance(trial, dict):
        lat = trial.get("latency_ms")
    if lat is None:
        return None
    if isinstance(lat, dict):
        for key in ("median", "mean"):
            v = lat.get(key)
            if isinstance(v, (int, float)) and not isinstance(v, bool) and v > 0:
                return float(v)
        return None
    v = getattr(lat, "robust_ms", None)
    return float(v) if isinstance(v, (int, float)) and v > 0 else None


def _params(trial: Any) -> dict[str, Any]:
    p = getattr(trial, "params", None)
    if p is None and isinstance(trial, dict):
        p = trial.get("params")
    if p is None:
        return {}
    if isinstance(p, dict):
        return dict(p.get("values") or {})
    return dict(getattr(p, "values", {}) or {})


def _status(trial: Any) -> str | None:
    s = getattr(trial, "status", None)
    if s is None and isinstance(trial, dict):
        s = trial.get("status")
    return str(s) if s is not None else None


def find_soft_walls(stats: TuningStats, trials: list[Any]) -> SoftWallScan:
    """Every knob whose spilling rises monotonically into a bound state with latency still improving.

    Pure: no GPU, no I/O, no probe. The input is the trials the candidate already ran, because
    `n_spills` is in every trial's profile -- which is why this costs nothing and also why it can never
    be independently confirmed.

    THE APPLICABILITY GATE COMES FIRST. If the candidate's best trial does not spill, there is no wall
    to find AT THE POINT THE AGENT REWRITES FROM, and the scan says "not applicable" rather than "no
    wall". Measured: 32 of 56 winners are in that state. The gate is on the BEST trial specifically,
    for the same reason the hard wall's ablation starts at the optimum -- starting anywhere else
    reports a property of a configuration nobody is going to ship.
    """
    scan = SoftWallScan()
    done = [t for t in trials if _status(t) == "complete" and _robust_ms(t) is not None]
    if not done:
        scan.reason = "no completed, timed trial"
        return scan

    best = min(done, key=lambda t: _robust_ms(t) or float("inf"))
    spills_at_best = _profile_field(best, "n_spills")
    scan.limiter_at_best = _occupancy_limiter(best)
    if not isinstance(spills_at_best, (int, float)) or isinstance(spills_at_best, bool):
        scan.reason = "the best trial has no n_spills measurement"
        return scan
    scan.spills_at_best = float(spills_at_best)
    if float(spills_at_best) <= 0.0:
        # NOT a negative result. 32 of 56 measured winners land here.
        scan.reason = "the best trial does not spill (no soft wall exists at the optimum)"
        return scan
    scan.applicable = True

    # Per-knob, per-value medians of spills and latency. Medians rather than means for the same reason
    # the framework's own objective is a median, and computed over the SAME completed trials so the two
    # curves cannot be drawn from different populations.
    buckets: dict[str, dict[float, list[tuple[float, float]]]] = defaultdict(
        lambda: defaultdict(list))
    for t in done:
        sp = _profile_field(t, "n_spills")
        ms = _robust_ms(t)
        if not isinstance(sp, (int, float)) or isinstance(sp, bool) or ms is None:
            continue
        for knob, raw in _params(t).items():
            v = _as_num(raw)
            if v is not None:
                buckets[str(knob)][v].append((float(sp), ms))

    # Restricted to the knobs the harness itself reported on, so a knob that exists in the trials but
    # not in `param_stats` cannot enter through this door -- the two views of "this candidate's knobs"
    # must not diverge.
    known = {ps.name for ps in stats.param_stats} or set(buckets)
    for knob in sorted(k for k in buckets if k in known):
        by_value = buckets[knob]
        xs = sorted(by_value)
        if len(xs) < _MIN_VALUES:
            continue
        scan.n_knobs_examined += 1
        spills = [statistics.median(s for s, _ in by_value[x]) for x in xs]
        lats = [statistics.median(m for _, m in by_value[x]) for x in xs]
        if max(spills) <= min(spills):
            continue
        scan.n_knobs_with_spill_variation += 1
        # (1) monotone non-decreasing and strictly higher at the end. PER KNOB, never a global rule:
        # the relation is peaked on 27 of 56 candidates, so a candidate whose curve is peaked simply
        # does not fire rather than being forced into a monotone reading.
        if not all(spills[i + 1] >= spills[i] for i in range(len(spills) - 1)):
            continue
        if spills[-1] <= spills[0]:
            continue
        # (2) the top end is actually in the bound state.
        if spills[-1] <= 0.0:
            continue
        # (3) latency still improving toward that end -- the same slope filter as the hard wall. A
        # knob whose latency WORSENS toward the spilling end is capped and worthless: telling the
        # rewriter to free it would spend a rewrite on a non-problem (measured on the hard wall: this
        # removes 3 of 6 walls, worst at -54.8%).
        tail = xs[-_TAIL_LEN:]
        tail_lats = [lats[xs.index(v)] for v in tail]
        gain = ((tail_lats[0] - tail_lats[-1]) / tail_lats[0] * 100.0
                if len(tail_lats) >= 2 and tail_lats[0] else 0.0)
        if gain <= 0.0:
            continue
        onset = next((x for x, s in zip(xs, spills) if s > 0.0), xs[-1])
        scan.walls.append(SoftWall(
            param=knob, ran_values=xs, spills_by_value=spills, latency_by_value=lats,
            onset_value=onset, tail_values=tail, tail_latencies=tail_lats, tail_gain_pct=gain,
            limiter_at_best=scan.limiter_at_best, spills_at_best=scan.spills_at_best))
    scan.walls.sort(key=lambda w: w.tail_gain_pct, reverse=True)
    if scan.applicable and not scan.walls:
        scan.reason = ("the best trial spills, but no knob's spill curve rises monotonically with "
                       "latency still improving")
    return scan


def for_prompt(scan: SoftWallScan, max_rows: int = 3) -> str | None:
    """Agent-facing text, or None when there is nothing measured.

    THREE THINGS EVERY RENDERING MUST SAY, because each of them has a way of being over-read:
      * the numbers come from the per-trial profile, NOT from a probe -- there is no second,
        independent confirmation the way the hard wall has one;
      * the conclusion is about THIS candidate's measured points, not an intrinsic limit of the knob;
      * it is a hint. The agent decides whether a restructure is possible; a hand-written resource
        rule presented as truth measured a median 32% of the real limit in this project.

    Carries no resource vector and no candidate latency, only how latency MOVED along one knob -- the
    same boundary `wall_attribution.for_prompt` keeps, so `assert_no_raw_vector`'s territory is not
    re-entered through this door.
    """
    if not scan.applicable or not scan.walls:
        return None
    lines = ["寄存器溢出墙(实测,来自逐 trial profile,**未经独立探针确认** —— "
             "以下结论仅在该候选已测的参数点上成立):"]
    for w in scan.walls[:max_rows]:
        lo, hi = w.ran_values[0], w.ran_values[-1]
        sp_lo, sp_hi = w.spills_by_value[0], w.spills_by_value[-1]
        trend = " -> ".join(f"{v:.4f}" for v in w.tail_latencies)
        lines.append(
            f"  {w.param}:该 knob 从 {_literal_for(lo)} 增到 {_literal_for(hi)} 时,"
            f"编译器报告的寄存器溢出从 {sp_lo:.0f} 升到 {sp_hi:.0f} 个槽位"
            f"(从 {_literal_for(w.onset_value)} 开始出现),"
            f"而这一段上延迟仍在改善:{trend} ms(尾部 {w.tail_gain_pct:+.1f}%)。")
        if w.limiter_at_best:
            lines.append(
                f"    在该候选的最优点处,编译器报告限制驻留的是 {w.limiter_at_best}"
                f"(该点溢出 {w.spills_at_best:.0f} 个槽位)。")
    lines.append(
        "  ⇒ 这一维仍有斜率,但继续增大会把中间结果挤到 local memory(比寄存器慢约两个数量级)。"
        "可考虑的方向:减少每线程的活跃变量 —— 拆分循环、缩小累加器、"
        "或把部分中间量改放共享内存。这是一个提示,不是指令 —— 是否可行由你判断。")
    return "\n".join(lines)
