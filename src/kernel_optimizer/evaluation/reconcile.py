"""S2d(b): reconcile what the agent SAID a rewrite would do against what was MEASURED.

WHY THIS IS THE RIGHT KIND OF CHECK. The second half of the ledger already exists and is trustworthy
for a stated reason: `conversion_verdict` compares two REAL MEASUREMENTS of the same family, before
and after, rather than comparing one measurement against an assumed ceiling. Its `no_conversion`
outcome -- a resource improved materially and latency did not move -- is direct evidence that THAT
resource was not the limit, and no utilisation label can produce that. This module adds the first
half: whether the agent's own stated direction was right, held to the same measured standard.

FOUR OUTCOMES per dimension, and the distinction between the last two is the point:

  hit          declared direction matches the measured direction
  miss         declared direction is the opposite, or declared unchanged and it moved
  vacuous      declared `unknown` -- counted, never silently skipped, because an agent that may
               answer `unknown` freely can otherwise escape the check permanently
  unpredicted  the dimension MOVED and the agent did not mention it

`miss` and `unpredicted` are not the same and must not be pooled: one is "thought wrong", the other
is "did not think of it", and they imply different things to say in the next round.

POLARITY IS NOT INVOLVED HERE, and that is deliberate. `up`/`down` are about the NUMBER, not about
better or worse. Occupancy going up is `up` whether or not that is an improvement. Mixing the two
notions is precisely the error that reported ZERO multi-dimension binding while the counter-example
sat in the input -- so the improvement question stays where it already lives (`conversion.py`'s
`direction`, which does consult polarity), and this module answers only "did the number move the way
you said".

THE MATERIALITY THRESHOLD IS REUSED, NOT RE-INVENTED. `conversion._MATERIAL_RESOURCE_DELTA = 0.05`:
a resource must move more than 5% of its own previous value to count as having moved. Not a physical
quantity -- it exists so that a 1-register or 128-byte difference is not called a resource change with
a latency verdict attached. Importing it rather than restating it means the two halves of the ledger
cannot drift into disagreeing about whether something moved.

EVERY LEDGER ENTRY CARRIES A CAVEAT, and it is not boilerplate. `conversion_verdict` compares the
parent's theta_best against the child's theta_best, and A REWRITE IS RE-TUNED -- so a `miss` may mean
the tuner went somewhere else, not that the agent judged wrong. That is a code fact, not a worry.
Omitting the caveat would let the ledger blame the agent for the tuner's choice, and an agent that
adjusts to wrong feedback is worse off than one with no feedback at all -- which is the measured shape
of KernelPro's raw-counter arm (1.77x against 3.35x for no feedback, p=0.0007).
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from kernel_optimizer.evaluation.conversion import _DIMENSIONS, _MATERIAL_RESOURCE_DELTA

Expect = Literal["up", "down", "unchanged", "unknown"]
Actual = Literal["up", "down", "flat", "unknown"]
Match = Literal["hit", "miss", "vacuous", "unpredicted", "unmeasured"]

# The closed vocabulary an expectation's `dimension` must name. Exposed so the agent module can list
# it back to the agent on a rejection: an error that does not say what the legal values are turns a
# schema check into a guessing game, and this project has burned three repair calls on a message that
# described an encoding problem as a content problem.
DIMENSION_VOCABULARY: tuple[str, ...] = tuple(sorted(_DIMENSIONS))


def unknown_dimensions(names: list[str]) -> list[str]:
    """Which of `names` are not in the vocabulary. Empty list means all are legal."""
    return [n for n in names if n not in _DIMENSIONS]


class DimensionReconciliation(BaseModel):
    """One dimension, declared against measured."""

    model_config = ConfigDict(frozen=True)

    dimension: str
    expected: Expect = "unknown"
    actual: Actual = "unknown"
    match: Match = "vacuous"
    before: float | None = None
    after: float | None = None
    delta: float | None = None
    rel: float | None = None
    unit: str = ""
    why: str = ""


class Reconciliation(BaseModel):
    """A round's ledger entry: the per-dimension table plus its counts and the caveat."""

    model_config = ConfigDict(frozen=True)

    hypothesis_id: str = ""
    per_dimension: tuple[DimensionReconciliation, ...] = ()
    hits: int = 0
    misses: int = 0
    vacuous: int = 0
    # Dimensions that MOVED and were not mentioned. Named, not counted only: "you did not think of
    # shared memory" is a different sentence from "you were wrong about it".
    dimensions_unpredicted: tuple[str, ...] = ()
    # Dimensions that were declared but could not be checked -- no reading on one side. Distinct from
    # `vacuous`, which is the agent declining, and from `miss`, which is a judgement.
    dimensions_unmeasured: tuple[str, ...] = ()
    caveat: str = ""

    def n_declared(self) -> int:
        return len(self.per_dimension) - len(self.dimensions_unpredicted)


# The caveat, in one place so every entry carries the same words and a reader learns to recognise it.
CAVEAT = (
    "ATTRIBUTION CAVEAT: this compares the parent's best configuration against the child's best "
    "configuration, and a rewrite is RE-TUNED. A `miss` may therefore mean the tuner settled "
    "somewhere else, not that your reasoning was wrong. Treat a miss as a question, not a verdict, "
    "and do not abandon a structural idea on one miss alone."
)


def _actual_direction(delta_info: dict[str, Any]) -> Actual:
    """Which way the NUMBER moved, from `conversion_verdict`'s own delta entry.

    Reads `rel` and the sign of `delta`, NOT `direction`: `direction` is already polarity-aware
    ("improved"/"worsened"), and an expectation is about the number. Deriving `up` from "improved"
    would silently invert every dimension where lower is better -- which is most of them.
    """
    delta = delta_info.get("delta")
    rel = delta_info.get("rel")
    if not isinstance(delta, (int, float)) or not isinstance(rel, (int, float)):
        return "unknown"
    if rel <= _MATERIAL_RESOURCE_DELTA:
        return "flat"
    return "up" if delta > 0 else "down"


def _classify(expected: Expect, actual: Actual) -> Match:
    if expected == "unknown":
        return "vacuous"
    if actual == "unknown":
        return "unmeasured"
    if expected == "unchanged":
        return "hit" if actual == "flat" else "miss"
    # `up` or `down`. A flat reading is a miss: the agent said it would move and it did not.
    return "hit" if actual == expected else "miss"


def reconcile(expectations: list[Any], resource_deltas: dict[str, dict] | None,
              *, hypothesis_id: str = "") -> Reconciliation:
    """Build the ledger entry. Pure arithmetic over two inputs, no I/O, no GPU.

    `expectations` are `ResourceExpectation`-shaped (read by attribute, so a replayed dict shim works
    too); `resource_deltas` is `conversion_verdict()`'s own `resource_deltas` sub-dict, reused rather
    than recomputed so the two halves of the ledger cannot disagree about what moved.

    A declared dimension with no delta entry becomes `unmeasured`, never `flat`: reading a missing
    measurement as "it did not move" would score a hit for an agent that said `unchanged` about a
    dimension nobody looked at.
    """
    deltas = resource_deltas or {}
    rows: list[DimensionReconciliation] = []
    hits = misses = vacuous = 0
    unmeasured: list[str] = []
    declared: set[str] = set()

    for exp in expectations:
        dim = str(getattr(exp, "dimension", "") or "")
        if not dim:
            continue
        declared.add(dim)
        expected: Expect = getattr(exp, "expect", "unknown") or "unknown"
        info = deltas.get(dim) or {}
        actual = _actual_direction(info) if info else "unknown"
        match = _classify(expected, actual)
        if match == "hit":
            hits += 1
        elif match == "miss":
            misses += 1
        elif match == "vacuous":
            vacuous += 1
        else:
            unmeasured.append(dim)
        rows.append(DimensionReconciliation(
            dimension=dim, expected=expected, actual=actual, match=match,
            before=info.get("before"), after=info.get("after"),
            delta=info.get("delta"), rel=info.get("rel"),
            unit=str(info.get("unit", "")), why=str(getattr(exp, "why", "") or "")))

    # Dimensions that moved materially and were never mentioned.
    unpredicted: list[str] = []
    for dim, info in deltas.items():
        if dim in declared:
            continue
        if _actual_direction(info) in ("up", "down"):
            unpredicted.append(dim)
            rows.append(DimensionReconciliation(
                dimension=dim, expected="unknown", actual=_actual_direction(info),
                match="unpredicted", before=info.get("before"), after=info.get("after"),
                delta=info.get("delta"), rel=info.get("rel"),
                unit=str(info.get("unit", ""))))

    return Reconciliation(
        hypothesis_id=hypothesis_id,
        per_dimension=tuple(rows), hits=hits, misses=misses, vacuous=vacuous,
        dimensions_unpredicted=tuple(sorted(unpredicted)),
        dimensions_unmeasured=tuple(sorted(unmeasured)),
        caveat=CAVEAT)


def _fmt(v: float | None) -> str:
    if v is None:
        return "?"
    if abs(v) >= 1000 or (v and abs(v) < 0.01):
        return "%.3g" % v
    return "%g" % round(v, 3)


def render_ledger(entries: list[dict[str, Any]]) -> str:
    """The ledger as prompt text: ONE PARAGRAPH PER CANDIDATE, ONE LINE PER DIMENSION.

    Per CANDIDATE, not per round, because a round has two of them (9 of 9 measured rounds) asked for
    different hypotheses. A heading that named only the round produced two rows per dimension under
    one title with nothing to separate them -- so `shared_bytes` appeared twice, once `up` as
    predicted and once `unchanged` and wrong, describing two different pieces of code. The heading
    carries the candidate id when the entry has one, and older entries without it still render.

    Not a JSON dump, and the length is the design constraint rather than an afterthought. G9 measured
    that adding per-candidate measurements on top of a structured verdict bought nothing detectable
    (4.19% apart against a 4.72% within-arm spread) at 2.1x the token cost, so a ledger that grew into
    a second raw vector would reproduce KernelPro's 1.77x arm -- the one that lost to no feedback at
    all. J2d-5 asserts the growth is linear in rounds with one line per dimension.

    `entries` are ledger dicts as journalled (`{id, change, round, candidate_id, reconciliation,
    conversion, latency_gain_pct}`), read with `.get` so a replayed older entry without the S2d fields
    renders as the pre-S2d line rather than raising.
    """
    if not entries:
        return ""
    out: list[str] = [
        "# What earlier rounds PREDICTED, and what was measured", "",
        "Your own stated resource directions from previous rounds, checked against the "
        "measurements. This is here so a structural idea is judged on what happened, not on how it "
        "sounded. One section per candidate: a round produces several, and each is judged only on "
        "the code it actually changed.", ""]
    for e in entries:
        rnd = e.get("round")
        hid = e.get("id") or "?"
        cid = e.get("candidate_id")
        out.append("## Round %s — %s%s" % (rnd if rnd is not None else "?", hid,
                                          " (%s)" % cid if cid else ""))
        change = (e.get("change") or "").strip()
        if change:
            out.append("*Change:* %s" % change)
        rec = e.get("reconciliation") or {}
        rows = rec.get("per_dimension") or []
        if not rows:
            out.append("- No resource directions were declared for this round, so nothing could be "
                       "checked. Declaring them is how a structural claim becomes falsifiable.")
        for r in rows:
            verdict = {
                "hit": "as you predicted",
                "miss": "**NOT as you predicted**",
                "vacuous": "you declared `unknown`",
                "unpredicted": "**you did not mention this dimension**",
                "unmeasured": "could not be checked (no reading on one side)",
            }.get(r.get("match", ""), r.get("match", ""))
            out.append("- **%s**: you said `%s`, measured `%s` (%s → %s %s) — %s" % (
                r.get("dimension"), r.get("expected"), r.get("actual"),
                _fmt(r.get("before")), _fmt(r.get("after")), r.get("unit", ""), verdict))
        gain = e.get("latency_gain_pct")
        conv = e.get("conversion")
        if conv:
            out.append("- *Latency:* %s%s" % (
                conv, "" if not isinstance(gain, (int, float)) else " (%.2f%%)" % gain))
            if conv == "no_conversion":
                out.append("  A resource improved and latency did not move, so THAT RESOURCE WAS NOT "
                           "THE LIMIT for this structure. That is a fact about where the limit is "
                           "not, and it is worth more than a utilisation reading.")
        out.append("")
    out.append(CAVEAT)
    return "\n".join(out)
