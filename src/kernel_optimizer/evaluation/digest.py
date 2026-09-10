"""S2 (b) and (c): the digestion layer, and the gate that keeps raw vectors out of prompts.

WHY A LAYER AT ALL. Handing an agent raw counters measurably makes things worse: in KernelPro's
three-arm comparison the raw-counter arm reached 1.77x against 3.35x for no feedback at all, one-
sided Wilcoxon p=0.0007. Our own G9 three-arm run agrees on the other side -- structured analysis
separated cleanly from nothing (7/7 per candidate), while `verdict` and `rich` did NOT separate
(4.19% apart against a 4.72% within-arm spread) at 2.1x the cost. So: structure helps, volume
does not.

WHAT THIS LAYER IS NOT. It is not a compressor. The plan's first version called it one, and that
description would have led to dropping dimensions at implementation time -- which is v2's original
defect (19 of 20 reports carrying one label). The rule is D-9: **change of form, never reduction of
content**. Every applicable dimension appears; each appears as a judgement instead of a number.

THE FOUR-TUPLE, per dimension:

    {dimension, severity, root_cause, ranked_recommendation, expected_delta}

  severity              three bands, from DimensionRecord.verdict. Two would mis-route the middle:
                        L3:48 at 90.4% of its DRAM roof and L3:43 with 69x fusion headroom only
                        separate under three.
  root_cause            points at a knob or a structure. Never a restatement of the symptom.
  ranked_recommendation the ORDER MUST CARRY ITS BASIS. There is no measured rule for "which
                        dimension to move first" -- the best candidate rule scored 7.1% against a
                        21.4% control -- so the order here is by severity and confidence only, and
                        it says so. It is advice, and the harness never reads it.
  expected_delta        may be `unknown`; may never be invented. Enters the prompt as a direction
                        hint and enters NO selection logic.

G28, THE GATE. `agents/modules.py` currently renders `verdict.evidence` key by key straight into the
analyst's document, and writes `## Verdict: **{kind}**`. That is the opposite of J2-3. `for_prompt()`
here returns only digested text, and `assert_no_raw_vector()` refuses anything that looks like a raw
evidence dict or a DimensionRecord -- so the prompt-building path cannot be handed one by accident.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict

from kernel_optimizer.evaluation.bounds import describe
from kernel_optimizer.evaluation.dimensions import DimensionRecord, DimensionState

# Severity ordering for the recommendation list. Binding first, then near-binding; slack and
# not-applicable are reported but never recommended -- there is nothing to act on.
_SEVERITY_RANK = {"binding": 0, "near-binding": 1, "unknown": 2, "slack": 3, "not-applicable": 4}

# What a binding dimension implies, in terms of a knob or a structure. Advice, and deliberately
# short: G9 measured that more volume buys nothing. Keyed by dimension so it cannot be mistaken for
# a general cost model -- there is no cost table, by decision, because change rates are
# task-specific and not derivable in advance (no closed form for shared memory, 0 of 96 exact; the
# map is not separable, 0 of 10; even the SIGN is unreliable, 13 non-monotone slices).
_ROOT_CAUSE: dict[str, str] = {
    "occupancy": ("too few resident warps to hide latency; the limiter field says which budget "
                  "caps residency, and that budget is the knob to change"),
    "n_regs": ("register demand per thread caps how many warps can be resident; fewer live values "
               "per thread (smaller tile, shorter software pipeline) is the lever"),
    "n_spills": ("the compiler ran out of registers and spilled to local memory; every spill is a "
                 "memory access added to the inner loop"),
    "shared_bytes": ("the block's shared-memory request is against this card's per-block limit, so "
                     "the tile cannot grow further in its current shape"),
    "peak_alloc_bytes": ("peak device allocation; relevant when a larger tile or an extra buffer "
                         "would not fit, not as a speed lever"),
    "candidate_aten_bytes": ("traffic visible at the aten level. A LOWER BOUND -- better fusion "
                             "makes this read lower, so it cannot be used to prove a ceiling"),
    "candidate_aten_ops": ("aten operation count; a proxy for how much is left unfused, and a "
                           "lower bound for the same reason as the byte count"),
    "threads_launched": "launch width, reported as a quantity; it has no better/worse direction",
}


class DimensionFinding(BaseModel):
    """One dimension, digested. The only shape allowed to reach a prompt."""

    model_config = ConfigDict(frozen=True)

    dimension: str
    severity: str
    root_cause: str
    ranked_recommendation: str = ""
    # A DIRECTION or `unknown`, never a rate. Measured three ways that a rate is not derivable in
    # advance, so asking for one would be asking to be lied to.
    expected_delta: str = "unknown"
    # Carried through so a reader can tell "measured and fine" from "never measured", and
    # "does not apply here" from either.
    applicable: bool = True
    # Whether a reading EXISTS -- a boolean, never the reading itself, so this does not smuggle a raw
    # number past the G28 gate. It is here because `severity == "unknown"` has two causes that must
    # not be rendered with the same sentence: nothing was measured, or something WAS measured and has
    # no ceiling to be banded against (aten traffic is a lower bound with no roof). Calling the
    # second one "NOT MEASURED" in a prompt would be a false statement about the box, and it is the
    # exact inversion this layer exists to prevent -- an absent reading and an unbandable one lead to
    # different actions.
    measured_present: bool = False
    confidence: float = 1.0
    note: str = ""
    # S3: "how much room is left", or an explicit statement that it is not derivable. Rendered as
    # TEXT, not as the bound object -- the prompt boundary takes strings (see `for_prompt`), and an
    # invented floor is worse than no floor, so `unknown` has to survive into the sentence.
    room_left: str = ""


class Digest(BaseModel):
    """Every applicable dimension as a finding, plus the basis of the ordering."""

    model_config = ConfigDict(frozen=True)

    candidate_id: str = ""
    findings: tuple[DimensionFinding, ...] = ()
    ordering_basis: str = ""
    # Dimensions that exist in the vocabulary but could not be read here. Named rather than dropped:
    # an absent dimension must not read as a clean bill of health.
    unmeasured: tuple[str, ...] = ()
    # J2-4: roofs that exist on this box but are NOT what this candidate is measured against. Their
    # own section rather than a finding, because they are statements about a DENOMINATOR, not about a
    # dimension's state -- and because J2-4's failing condition is "it still appears in the
    # recommendation", which only means something if they are structurally outside the ranked list.
    unreachable_ceilings: tuple[str, ...] = ()
    # S3: where the compute denominator came from, and a WARNING when it was measured at a different
    # precision than the kernel computes in. Its own field for the same reason as above: it is about
    # the denominator, and the 107.8% incident was a wrong denominator that made a kernel with 40-46%
    # headroom read as saturated.
    ceiling_notes: tuple[str, ...] = ()

    def binding(self) -> tuple[DimensionFinding, ...]:
        return tuple(f for f in self.findings if f.severity == "binding")


class RawVectorInPromptError(RuntimeError):
    """Raised when a raw evidence dict or record reaches the prompt-building path (G28 / J2-3)."""


def assert_no_raw_vector(value: Any) -> None:
    """Refuse a raw vector at the prompt boundary.

    Checks the TYPES that carry raw numbers, not a list of key names: a name list would pass the
    moment a key was renamed, and would also have to be kept in sync with `classify()` forever.

    The dict case looks for the evidence dict's own signature keys. It is deliberately narrow --
    a digest's own `model_dump()` must still be allowed through, or the gate would block the thing
    it exists to permit.
    """
    if isinstance(value, (DimensionRecord, DimensionState)):
        raise RawVectorInPromptError(
            f"a {type(value).__name__} reached the prompt path; prompts may only receive a Digest. "
            "Raw per-dimension records go to events.jsonl and report.md, never into a prompt "
            "(J2-3): handing an agent raw counters measured WORSE than handing it nothing "
            "(1.77x against 3.35x, p=0.0007).")
    if isinstance(value, dict):
        raw_markers = {"thresholds", "gpu_ms", "pct_of_dram_peak", "pct_of_compute_peak",
                       "arithmetic_intensity", "backend_reachable_frac", "shared_used_frac"}
        hit = raw_markers & set(value)
        if hit:
            raise RawVectorInPromptError(
                f"a raw evidence dict reached the prompt path (keys {sorted(hit)}); prompts may "
                "only receive a Digest. Note two of those keys are 1/latency in disguise -- "
                "`pct_of_dram_peak` and `pct_of_compute_peak` have task-level-constant numerators, "
                "verified constant to 0.07-0.36% while gpu_ms varied 4.6x.")


def digest(state: DimensionState, *, ceiling_notes: dict[str, str] | None = None,
           unreachable: tuple[str, ...] = (),
           denominator_notes: tuple[str, ...] = ()) -> Digest:
    """Turn the per-dimension state into findings. One finding per record, no merging, no dropping.

    D-9 is enforced structurally rather than by intent: the loop is over `state.records`, so the
    output length equals the input length and a dimension cannot be dropped without changing this
    function's shape. J2-9 asserts that equality from the outside as well.

    `unreachable` comes from `dimensions.unreachable_ceilings()` and is carried through untouched.
    `denominator_notes` (S3) carries the compute ceiling's provenance and any precision mismatch.
    """
    notes = ceiling_notes or {}
    findings: list[DimensionFinding] = []
    unmeasured: list[str] = []

    for rec in state.records:
        if rec.measured is None and rec.applicable:
            unmeasured.append(rec.dimension_id)

        root = _ROOT_CAUSE.get(rec.dimension_id, "no root-cause statement for this dimension")
        if not rec.applicable:
            root = rec.not_applicable_reason or root

        note = notes.get(rec.dimension_id, "")
        if rec.applicable and rec.verdict == "not-applicable" and rec.not_applicable_reason:
            # Present and measured, but unbandable. The reason must travel: without it the reader
            # cannot tell this apart from "the box does not have this dimension", and those two lead
            # to different actions.
            note = (note + " " if note else "") + rec.not_applicable_reason
        if rec.confidence < 1.0 and rec.applicable:
            # Say WHY the confidence is reduced, in the finding itself. A bare 0.5 is not
            # actionable and invites being ignored.
            note = (note + " " if note else "") + (
                "reduced confidence: this reading is a bound, not a value")

        findings.append(DimensionFinding(
            dimension=rec.dimension_id,
            severity=rec.verdict,
            root_cause=root,
            expected_delta="unknown",
            applicable=rec.applicable,
            measured_present=rec.measured is not None,
            confidence=rec.confidence,
            note=note.strip(),
            # S3. Only for a dimension with a measurement -- "room left" on an unmeasured dimension
            # would be a statement about a number nobody has.
            room_left=(describe(rec.dimension_id, rec.bound)
                       if rec.bound is not None and rec.measured is not None else ""),
        ))

    # Order by severity then by confidence. STATED, because there is no measured rule for which
    # dimension to move first -- the best rule tried scored 7.1% against a 21.4% random control, so
    # presenting this order as a priority would be presenting a refuted rule as fact.
    ordered = sorted(
        findings,
        key=lambda f: (_SEVERITY_RANK.get(f.severity, 9), -f.confidence, f.dimension))

    ranked = [
        f.model_copy(update={"ranked_recommendation": (
            f"#{i + 1} by severity then confidence" if f.severity in ("binding", "near-binding")
            else "not recommended: nothing to act on at this severity")})
        for i, f in enumerate(ordered)
    ]

    n_binding = sum(1 for f in ranked if f.severity == "binding")
    basis = (
        "ordered by severity (binding > near-binding > unknown > slack) then by confidence. "
        "This order is NOT a measured priority: no rule for 'which dimension to move first' beat "
        "its controls (7.1% against 21.4%), so treat it as a reading order, not a plan. "
        + (f"{n_binding} dimensions are binding at once; that is the finding, not a tie to break."
           if n_binding > 1 else
           ("one dimension is binding." if n_binding == 1 else
            "no dimension is binding: nothing measured here is the limit."))
    )

    return Digest(candidate_id=state.candidate_id, findings=tuple(ranked),
                  ordering_basis=basis, unmeasured=tuple(unmeasured),
                  unreachable_ceilings=tuple(unreachable),
                  ceiling_notes=tuple(denominator_notes))


def for_prompt(d: Digest) -> str:
    """The digest as prompt text. The ONLY function allowed to produce agent-facing resource text.

    Short by design. G9 measured that adding per-candidate measurements on top of a verdict bought
    nothing detectable (4.19% apart against a 4.72% within-arm spread) at 2.1x the token cost, so
    the four-tuple is kept small on purpose rather than for tidiness.
    """
    assert_no_raw_vector(d)
    lines: list[str] = ["## Resource state, one line per dimension", ""]
    for f in d.findings:
        if not f.applicable:
            lines.append(f"- **{f.dimension}** — does not apply here: {f.root_cause}")
            continue
        if f.severity == "not-applicable":
            # Present and measured, but no band applies -- `threads_launched` has no better/worse
            # direction. Distinct from the line above, which says the box does not have it at all.
            lines.append(f"- **{f.dimension}** — measured; no band applies: {f.note or f.root_cause}")
            continue
        if f.severity == "unknown":
            if f.measured_present:
                # Measured, but there is no ceiling to band it against. Saying "NOT MEASURED" here
                # would be false, and saying "slack" would be worse -- it would claim headroom.
                lines.append(f"- **{f.dimension}** — measured, but NOT RANKED: this dimension has "
                             f"no ceiling on this box to compare against, so no band is claimed. "
                             f"{f.root_cause}")
            else:
                lines.append(f"- **{f.dimension}** — NOT MEASURED on this candidate. "
                             f"Do not read this as headroom.")
            continue
        head = f"- **{f.dimension}** — {f.severity}"
        if f.severity in ("binding", "near-binding"):
            head += f": {f.root_cause}"
        lines.append(head + (f" ({f.note})" if f.note else ""))
    if any(f.room_left for f in d.findings):
        # S3, its own section rather than appended to each severity line: "how far from the roof" and
        # "how much room is left" are different questions, and for most dimensions the second answer
        # is UNKNOWN. Mixing an unknown into a line that also states a band invites reading the band
        # as the room figure.
        lines += ["", "**How much room is left** (distance to a FLOOR, not to the roof above). "
                      "Most dimensions have no derivable floor and say so — an invented floor would "
                      "be worse than none, because it would make a dimension look exhausted:"]
        lines += [f"- {f.room_left}" for f in d.findings if f.room_left]
    lines += ["", f"**Reading order.** {d.ordering_basis}"]
    if d.unmeasured:
        lines += ["", "**Unmeasured dimensions** (absence is not a clean bill of health): "
                      + ", ".join(d.unmeasured)]
    if d.unreachable_ceilings:
        # Its own section, AFTER the reading order, so it cannot be read as an item to act on. J2-4
        # fails if an unreachable roof appears in the recommendation; here it appears as a caveat on
        # the denominators instead.
        lines += ["", "**Roofs that are NOT what you are measured against.** Aiming at one of these "
                      "is aiming at a number nothing here was compared to:"]
        lines += [f"- {c}" for c in d.unreachable_ceilings]
    if d.ceiling_notes:
        # S3. A wrong denominator is the one defect that makes a kernel with headroom read as
        # saturated, so where the denominator came from belongs beside the percentages it qualifies.
        lines += ["", "**Where the compute denominator came from.**"]
        lines += [f"- {c}" for c in d.ceiling_notes]
    return "\n".join(lines)
