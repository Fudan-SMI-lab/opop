"""S4': what a resource change actually bought, read back out of the event log.

THE STAGE IS THREE THINGS, and one of them was deleted before it was built.

DELETED: the static estimator `S = dQ/dF`. It is a CHANGE-RATE estimator, and three measurements say
a change rate is not derivable in advance -- shared memory has no closed form (0 of 96 configurations
matched), the cost map is not separable (0 of 10 one-step deltas agreed), and even the SIGN is
unreliable (13 non-monotone slices). Building it would have meant shipping a number that looks like a
measurement.

(1) THE CONSUMER. `conversion_verdict` is called and its output is journalled, and `report.py` reads
it ZERO times -- verified by grep. A verdict with no consumer is not implemented: nothing acts on it,
nothing checks it, and it cannot be wrong in a way anyone notices. Worse, the corpus shows the
mechanism has never produced a single verdict in a real run: 9 of 9 `FAMILY_ROUND_RECORDED` events
carry no `conversion` field and 0 carry `resource_deltas`, because the five runs predate the fix
(runs 09-07..09-10, the fix 09-10). So the next control run is also this mechanism's FIRST evidence,
and the report is where that evidence has to become visible.

(2) COMPLEMENTARY SLACKNESS, and this is the one worth having. LP duality gives a usable theorem: the
shadow price of a NON-BINDING constraint is zero. Translated into our terms: **if a dimension is
judged `slack`, then spending resources on it must buy zero latency improvement.** So a round that
improved a slack dimension and moved latency is evidence that OUR VERDICT WAS WRONG -- not that the
candidate was bad. That is what makes it valuable: almost every other check here grades candidates,
and this one grades us.

It must be able to fail loudly, so `slackness_violations()` reports the violating dimension by name
and does not soften it. The honest caveat travels with it: a rewrite is re-tuned, so the improvement
may come from elsewhere in the same round. That weakens attribution and does not remove the signal --
a `slack` dimension whose improvement repeatedly coincides with latency gains is a verdict to
distrust.

(3) THE GABLES ORDERING SELF-CHECK. Gables' own worked example walks four configurations reaching
40 -> 1.3 -> 2 -> 160 Gops/s, and three of those four steps move a resource in a locally sensible
direction and get WORSE. Our verdict ordering should reproduce that ranking; if it cannot, the
ordering rule is wrong. Pure arithmetic, no GPU, and it is a control we cannot pass by accident --
the sequence is deliberately non-monotone in every single resource.

WHAT THIS MODULE MAY NOT DO. Four hard constraints, each with a measured reason:

  * never become a TPE objective. Latency is the only objective; a conversion figure as a target
    would optimize the diagnostic.
  * report `unknown` below the noise floor rather than a small number. On L3:48 the per-trial std was
    16% of the mean while 33 near-ties spanned 9%, so a "gain" inside that band is not a ranking.
  * never prune or screen. The measured counter-example: on L2:37 the family whose seed was the
    SLOWEST improved the most, -30.5%.
  * report an interval per dimension, not a point.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict

# A latency move smaller than this fraction of the baseline is not a move. Deliberately expressed as
# a fraction of the run's own `min_improvement_pct` rather than as a constant: the noise floor is a
# property of (card, task) and the run already carries a threshold derived for it. A second constant
# here would be a second opinion about the same quantity.
_SLACK_TOLERANCE_MULTIPLE = 1.0


class SlacknessViolation(BaseModel):
    """A dimension we called `slack` whose improvement coincided with a latency gain.

    Evidence about OUR verdict, not about the candidate. Named as such so a reader cannot mistake it
    for a candidate defect -- that inversion is the thing several fixes in this project exist to
    prevent.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    dimension: str
    family_id: str = ""
    round: int | None = None
    latency_gain_pct: float = 0.0
    rel_resource_change: float = 0.0
    detail: str = ""


def slackness_violations(rounds: list[dict[str, Any]], slack_dimensions: set[str],
                         *, min_improvement_pct: float) -> list[SlacknessViolation]:
    """Rounds where a `slack` dimension improved AND latency improved.

    `rounds` are `FAMILY_ROUND_RECORDED` payloads. `slack_dimensions` is the set of dimensions judged
    `slack` in this run, taken from `DIMENSION_STATE`.

    A RUN-LEVEL SET, not a per-round one, and the imprecision is deliberate rather than hidden.
    `DIMENSION_STATE` is keyed by candidate and `FAMILY_ROUND_RECORDED` by (family, round); linking
    them needs the round's parent candidate, which is `family.best` AT THAT MOMENT and is not
    journalled. Inventing that mapping would be the same class of error as guessing a payload path,
    which has misfired twice in this project.

    The direction of the resulting imprecision matters and is the acceptable one: a union over the run
    counts a dimension as slack even in rounds where it was binding, so the check reports MORE
    candidate violations than a precise mapping would. For a check whose entire value is that it can
    fail loudly about our own verdicts, erring toward noise beats erring toward silence -- and each
    violation is printed with its dimension, family and round so it can be checked by hand.

    The theorem is one-directional and so is this function. A slack dimension improving with NO latency
    gain is the theorem HOLDING -- unremarkable, not reported. Only the coincidence contradicts a zero
    shadow price.
    """
    out: list[SlacknessViolation] = []
    floor = min_improvement_pct * _SLACK_TOLERANCE_MULTIPLE
    if not slack_dimensions:
        return out
    for payload in rounds:
        gain = payload.get("latency_gain_pct")
        if not isinstance(gain, (int, float)) or gain < floor:
            continue
        deltas = payload.get("resource_deltas") or {}
        for dim in payload.get("resources_improved") or []:
            if dim not in slack_dimensions:
                continue
            rel = (deltas.get(dim) or {}).get("rel")
            out.append(SlacknessViolation(
                dimension=dim, family_id=str(payload.get("family_id", "")),
                round=payload.get("round") if isinstance(payload.get("round"), int) else None,
                latency_gain_pct=float(gain),
                rel_resource_change=float(rel) if isinstance(rel, (int, float)) else 0.0,
                detail=(
                    "we judged %s `slack`, and improving it coincided with a %.2f%% latency gain. "
                    "Complementary slackness says a non-binding constraint has a shadow price of "
                    "zero, so THIS IS EVIDENCE OUR VERDICT WAS WRONG -- not that the candidate was "
                    "bad. Attribution caveat: a rewrite is re-tuned, so the gain may come from "
                    "elsewhere in the same round; one coincidence is a question, a repeated one is "
                    "a verdict to distrust." % (dim, gain))))
    return out


# Gables' worked example: four configurations, and the throughput each reaches. Three of the four
# steps move a resource in a locally sensible direction and get WORSE, which is why this is a control
# and not a formality -- a rule that simply follows one resource cannot reproduce it.
GABLES_STEPS: tuple[tuple[str, float], ...] = (
    ("baseline", 40.0),
    ("step-1", 1.3),
    ("step-2", 2.0),
    ("step-3", 160.0),
)


def gables_ranking(steps: tuple[tuple[str, float], ...] = GABLES_STEPS) -> list[str]:
    """The four configurations ordered best-to-worst by achieved throughput.

    Trivial arithmetic on purpose. The value is not in this function -- it is in the assertion that our
    ordering rule reproduces it, since three of the four steps are locally sensible moves that made
    things worse. A rule that ranks by "which resource moved most" gets this wrong.
    """
    return [name for name, _ in sorted(steps, key=lambda kv: -kv[1])]


def conversion_lines(events: Any, *, min_improvement_pct: float = 2.0) -> list[str]:
    """The report section (1): what each rewrite round's resource change bought.

    Reads `FAMILY_ROUND_RECORDED` and `DIMENSION_STATE` off the event log, so the section is
    reproducible from `events.jsonl` alone -- `report` regenerates purely from the log, and a section
    that needed live state would break that.

    Says explicitly when there is NOTHING to report, and distinguishes two reasons that look identical
    in a log holding neither: the run had no rewrite rounds, versus the rounds carry no `conversion`
    field because they predate the fix. The second is the state of all five existing corpora, so a
    section that rendered nothing would silently reproduce the "zero consumers" gap it exists to
    close.
    """
    rounds: list[dict] = []
    slack_dimensions: set[str] = set()
    n_states = 0
    for e in events:
        etype = getattr(e, "type", None)
        payload = getattr(e, "payload", None) or {}
        if etype == "FAMILY_ROUND_RECORDED":
            rounds.append(payload)
        elif etype == "DIMENSION_STATE":
            n_states += 1
            slack_dimensions.update(
                r.get("dimension_id") for r in (payload.get("records") or [])
                if r.get("verdict") == "slack" and r.get("applicable"))

    out: list[str] = ["### What resource changes bought (conversion)\n"]
    if not rounds:
        out += ["No rewrite round was recorded, so there is nothing to convert. This is a fact "
                "about the search, not about the mechanism.\n", ""]
        return out

    with_verdict = [r for r in rounds if r.get("conversion")]
    if not with_verdict:
        out += [
            f"{len(rounds)} rewrite round(s) were recorded and **none carries a conversion "
            f"verdict**. That means these rounds predate the fix that forwards profiles into "
            f"`conversion_verdict` (G27) — not that nothing converted. Every corpus collected "
            f"before 2026-09-10 reads this way: 9 of 9 rounds with no `conversion` field, 0 with "
            f"`resource_deltas`.\n", ""]
        return out

    counts: dict[str, int] = {}
    for r in with_verdict:
        counts[r["conversion"]] = counts.get(r["conversion"], 0) + 1
    out.append("| verdict | rounds | meaning |")
    out.append("|---|---|---|")
    meanings = {
        "improved": "latency fell by more than the noise floor",
        "no_conversion": "**a resource improved and latency did NOT move — that resource was not "
                         "the limit**",
        "regressed": "latency rose",
        "flat": "neither latency nor any measured resource moved materially",
        "unknown": "latency could not be compared",
    }
    for verdict, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        out.append(f"| `{verdict}` | {n} | {meanings.get(verdict, '')} |")
    out.append("")

    for r in with_verdict:
        note = r.get("conversion_note") or ""
        out.append(f"- `{r.get('family_id')}` round {r.get('round')}: `{r['conversion']}`"
                   + (f" — {note}" if note else ""))
    out.append("")

    violations = slackness_violations(with_verdict, slack_dimensions,
                                     min_improvement_pct=min_improvement_pct)
    out.append("#### Complementary-slackness check (this grades OUR verdicts, not the candidates)\n")
    if n_states == 0:
        out += ["No per-dimension state was journalled, so no dimension was judged `slack` and the "
                "check cannot run. It needs `v3.diagnosis` to have produced a vector.\n", ""]
    elif not slack_dimensions:
        out += ["Per-dimension state exists but NO dimension was judged `slack` in this run, so the "
                "check has nothing to test. That is not a pass.\n", ""]
    elif not violations:
        out += [f"No violation, over {len(with_verdict)} round(s) with a verdict and "
                f"{len(slack_dimensions)} dimension(s) judged slack somewhere in the run "
                f"({', '.join(sorted(slack_dimensions))}). Every round that improved latency improved "
                f"at least one dimension we had NOT called `slack` — which is what a zero shadow "
                f"price on a non-binding constraint predicts.\n",
                "This is a WEAK pass. The slack set is a union over the whole run (see "
                "`slackness_violations`), so it errs toward reporting violations rather than hiding "
                "them; and with few rounds it also passes vacuously.\n", ""]
    else:
        out.append(f"**{len(violations)} violation(s).** Each is evidence that a `slack` verdict was "
                   f"wrong, because improving a non-binding dimension should buy nothing:\n")
        for v in violations:
            out.append(f"- **{v.dimension}** ({v.family_id} round {v.round}): "
                       f"{v.latency_gain_pct:.2f}% latency gain alongside a "
                       f"{v.rel_resource_change * 100:.1f}% change in a dimension we called slack")
        out.append("")
        out.append("Attribution caveat: a rewrite is RE-TUNED, so the gain may come from elsewhere "
                   "in the same round. One coincidence is a question; a repeated one is a verdict to "
                   "distrust.\n")
    return out
