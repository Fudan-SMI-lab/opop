"""2e: attribute a shared-memory wall to a single knob, by ablation at the optimum.

WHAT THIS ANSWERS. A tuning space can hold configurations the hardware refuses -- the compiler's
own `metadata.shared` exceeds the device's per-block opt-in limit, so a launch could only raise
`out of resource: shared memory`. The refusal is already recorded per configuration
(`CONFIG_SCREENED_INFEASIBLE`), but it names a WHOLE parameter set of 6-12 knobs, so it cannot say
which knob is responsible. This module asks that question by ablation: take the candidate's own
best configuration, change ONE knob to the refused value, and see whether the compiler still
refuses.

WHY IT IS WORTH ASKING. The framework already has a field for this conclusion --
`BottleneckReport.parameter_limits`, with `param` / `blocked_by` / `predicted_gain_pct` -- and it
is filled entirely by the analyst agent, unchecked. Measured on box 2's two L3:43 runs: of 53
claims across the six candidates that have a wall, the harness confirms 8. The failure shapes are
not subtle -- the wrong resource named (`registers` for what is a shared-memory wall), walls
claimed for knobs that have none (5 of one candidate's 10 claims), and `predicted_gain_pct` with
the sign inverted (+1.0% claimed where the measured tail is -41.1%).

WHY ABLATION AND NOT A FORMULA. Shared memory is roughly a product of tile dimensions and stages,
so the obvious move is a closed form. That has been audited and refuted twice over in this project:
`BLOCK_M * BLOCK_N * stages` has failing-min BELOW passing-max in 15 of 15 candidates, so the
feasible and infeasible sets OVERLAP in any such product; and agent-written shared-memory
constraints came in at a median 32% of the true limit. A probe does not need a threshold -- it asks
whether ONE concrete point compiles, which is a question the compiler answers exactly.

THE CONCLUSION IS CONDITIONAL, AND THAT IS MEASURED, NOT ASSUMED. Repeating the same ablation from
the space's DEFAULT configuration instead of the optimum attributes only 1 of 6 walls, against 6 of
6 from the optimum. The mechanism is plain: at the default every dimension sits at its smallest, so
enlarging one knob still fits; at the optimum the others are already large, so one more step
overflows. The wall's position therefore DEPENDS on the other knobs, and every rendering of this
result has to say "at this candidate's optimum" -- an unconditional "BLOCK_M is capped by shared
memory" would send the rewriter after a wall that is not there at the point it is rewriting from.
That is why `second_origin` is a recorded field and not an optional diagnostic.

WHAT IT DELIBERATELY DOES NOT DO. It does not remove, shrink or reorder any domain: a wall is
recorded, never enforced. The tuner keeps sampling exactly what it sampled before, and the existing
screen keeps refusing exactly what it refused before.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from kernel_optimizer.models.reports import TuningStats

# A tail needs at least this many measured values to have a direction at all. Two points make a
# line through any two points; three is the smallest number that can be non-monotone, which is the
# thing being tested.
_MIN_RAN_VALUES = 3
# How many measured values at the wall end define the slope. Three, to match the `at_boundary`
# convention already used by `TuningStatsAnalyzer` ("last 3 choices monotone toward the edge").
_TAIL_LEN = 3

Verdict = Literal["attributed", "not_attributed", "undecidable"]


@dataclass
class Wall:
    """One knob whose refused value lies outside the range that actually ran."""

    param: str
    refused_value: float
    ran_values: list[float]
    side: Literal["high", "low"]
    # Latency at the last `_TAIL_LEN` measured values, ordered from far to near the wall.
    tail_values: list[float] = field(default_factory=list)
    tail_latencies: list[float] = field(default_factory=list)
    monotone: bool = False
    # Relative latency change across the tail, positive when latency is still IMPROVING toward the
    # wall. The sign is what separates "capped and it costs us" from "capped and it costs nothing":
    # on box 2, 3 of 6 walls had a positive tail and 3 were negative, the worst at -54.8%.
    tail_gain_pct: float = 0.0
    # Filled by the probe. `None` until then; "undecidable" when the probe could not answer, which
    # is NOT the same as "not attributed" and must never be collapsed into it.
    verdict: Verdict | None = None
    max_shared: int | None = None
    limit: int | None = None
    over_ratio: float | None = None
    # The same ablation from the space's default configuration. See the module docstring: 1 of 6
    # agreement is why this is not optional.
    second_origin: Verdict | None = None
    second_origin_max_shared: int | None = None

    def payload(self) -> dict[str, Any]:
        return {
            "param": self.param,
            "refused_value": self.refused_value,
            "ran_values": list(self.ran_values),
            "side": self.side,
            "tail_values": list(self.tail_values),
            "tail_latencies": list(self.tail_latencies),
            "monotone": self.monotone,
            "tail_gain_pct": round(self.tail_gain_pct, 2),
            "verdict": self.verdict,
            "max_shared": self.max_shared,
            "limit": self.limit,
            "over_ratio": (round(self.over_ratio, 3)
                           if isinstance(self.over_ratio, float) else self.over_ratio),
            "second_origin": self.second_origin,
            "second_origin_max_shared": self.second_origin_max_shared,
        }


def _as_num(x: Any) -> float | None:
    """Numeric value of a knob setting, or None when the domain is not ordered.

    Categorical domains are skipped on purpose: their "first" and "last" choices are an artifact of
    list order, so a slope over them measures the order the agent happened to write, not the
    hardware. `bool` is excluded because `float(True)` is 1.0 and would make an on/off switch look
    like an ordered axis.
    """
    if isinstance(x, bool):
        return None
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, str):
        try:
            return float(x)
        except ValueError:
            return None
    return None


def _literal_for(value: float) -> int | float:
    """The value as it must be written back into `PARAMS`.

    An int knob rendered as `512.0` changes the source text and can change what the kernel does
    (a float where a `tl.constexpr` int is expected), so an integral value goes back as an int.
    """
    return int(value) if float(value).is_integer() else value


def find_walls(stats: TuningStats, refused_params: list[dict[str, Any]]) -> list[Wall]:
    """Every knob with a refused value outside its measured range. Pure; no GPU, no I/O.

    Reads `ParamStat.latency_by_value`, which is the harness's own per-choice MEDIAN table, rather
    than recomputing a per-value latency here. Two reasons: median is this project's measured choice
    (93.2% rank-correctness against 64.8% for mean), and a second implementation of the same table
    would be free to disagree with the one the report and the analyst see.
    """
    walls: list[Wall] = []
    if not refused_params:
        return walls
    for ps in stats.param_stats:
        by_value = ps.latency_by_value or {}
        ran: dict[float, float] = {}
        for key, ms in by_value.items():
            v = _as_num(key)
            if v is not None and isinstance(ms, (int, float)):
                ran[v] = float(ms)
        if len(ran) < _MIN_RAN_VALUES:
            continue
        refused = sorted({v for p in refused_params
                          if (v := _as_num(p.get(ps.name))) is not None})
        if not refused:
            continue
        lo, hi = min(ran), max(ran)
        above = [v for v in refused if v > hi]
        below = [v for v in refused if v < lo]
        if not (above or below):
            # A refused value INSIDE the measured range is not a truncation: the tuner reached both
            # sides of it, so nothing was cut off. Common (158 of 164 on box 2) and correctly a
            # non-event here.
            continue
        # Nearest refused value on the truncated side -- the smallest step that would hit the wall.
        # `above` first: an upper wall is the case the tail slope can be read toward.
        if above:
            side: Literal["high", "low"] = "high"
            target = min(above)
        else:
            side = "low"
            target = max(below)
        xs = sorted(ran)
        tail = xs[-_TAIL_LEN:] if side == "high" else xs[:_TAIL_LEN][::-1]
        lats = [ran[v] for v in tail]
        monotone = len(lats) >= 2 and all(lats[i + 1] < lats[i] for i in range(len(lats) - 1))
        gain = ((lats[0] - lats[-1]) / lats[0] * 100.0) if len(lats) >= 2 and lats[0] else 0.0
        walls.append(Wall(param=ps.name, refused_value=target, ran_values=xs, side=side,
                          tail_values=tail, tail_latencies=lats,
                          monotone=monotone, tail_gain_pct=gain))
    return walls


def select_for_probing(walls: list[Wall], max_probes: int) -> tuple[list[Wall], list[Wall]]:
    """Split into (worth probing, skipped), highest tail gain first.

    A wall whose latency is FLAT or WORSENING toward it is real but worthless: lifting it buys
    nothing, and telling the rewriter to spend another resource dimension freeing it would spend a
    rewrite on a non-problem. Measured on box 2, this filter removes 3 of 6 walls, the worst of them
    at -54.8%.

    `max_probes` caps the DIAGNOSTIC depth, not the search space -- no configuration is excluded
    from sampling by anything in this module, exactly as an independent screen timeout caps cost
    without restricting the space. Skipped walls are returned rather than dropped so the event can
    say how many there were.
    """
    worth = [w for w in walls if w.monotone and w.tail_gain_pct > 0.0]
    worth.sort(key=lambda w: w.tail_gain_pct, reverse=True)
    if max_probes >= 0:
        return worth[:max_probes], worth[max_probes:]
    return worth, []


def ablation_params(theta: dict[str, Any], wall: Wall) -> dict[str, Any]:
    """`theta` with exactly one knob moved to the refused value."""
    out = dict(theta)
    out[wall.param] = _literal_for(wall.refused_value)
    return out


def verdict_from(fits: bool | None) -> Verdict:
    """Map `cached_shared_verdict`'s three-valued answer onto this module's names.

    `None` becomes "undecidable", never "not_attributed": an unanswered probe and a probe that
    said the configuration fits are different facts, and collapsing them would let a worker
    timeout read as evidence that the knob is innocent.
    """
    if fits is None:
        return "undecidable"
    return "not_attributed" if fits else "attributed"


def summarize(walls: list[Wall]) -> dict[str, int]:
    counts = {"attributed": 0, "not_attributed": 0, "undecidable": 0}
    for w in walls:
        if w.verdict in counts:
            counts[w.verdict] += 1
    return counts


def for_prompt(walls: list[Wall]) -> str | None:
    """Agent-facing text for the attributed walls, or None when there is nothing measured.

    EVERY sentence is conditional on the optimum. See the module docstring for the 1-of-6
    second-origin result that makes this mandatory rather than careful.

    Carries no resource vector and no candidate latency -- only how latency MOVED along this one
    knob, which is a property of the axis rather than a ranking of the candidate. That distinction
    is what keeps this out of `assert_no_raw_vector`'s territory: `pct_of_dram_peak` and
    `pct_of_compute_peak` are 1/latency rescaled and would tell the agent how fast this candidate
    is; a per-knob trend does not.
    """
    rows = [w for w in walls if w.verdict == "attributed"]
    if not rows:
        return None
    lines = ["共享内存墙(实测,非估计 —— 以下结论仅在该候选的最优参数点处成立):"]
    for w in rows:
        over = f"{w.over_ratio:.2f}x" if isinstance(w.over_ratio, float) else "?"
        trend = " -> ".join(f"{v:.4f}" for v in w.tail_latencies)
        lines.append(
            f"  {w.param}:在最优点处单独改成 {_literal_for(w.refused_value)} 时,"
            f"编译器要求 {w.max_shared} 字节共享内存,本卡上限 {w.limit}({over})。"
            f"已测的 {', '.join(str(_literal_for(v)) for v in w.tail_values)} 上延迟为 {trend} ms"
            f"(尾部 {w.tail_gain_pct:+.1f}%),所以这一维仍有斜率但被共享内存截断。")
        if w.second_origin == "not_attributed":
            lines.append(
                "    注意:从空间默认配置出发时同一改动并不触墙,"
                "所以这道墙由该点上其他 knob 的取值共同决定,不是这个 knob 的固有上限。")
    lines.append(
        "  可考虑的方向:让每个 block 需要更少共享内存(例如沿 K 分块、两段式归约、"
        "或把中间结果放寄存器/全局内存),从而把这一维的取值域重新打开。"
        "这是一个提示,不是指令 —— 是否可行由你判断。")
    return "\n".join(lines)
