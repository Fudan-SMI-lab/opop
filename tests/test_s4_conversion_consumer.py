"""S4': the conversion consumer, complementary slackness, and the Gables ordering control.

The stage's four acceptance criteria and the reverse control for each:

  a consumer exists                 a verdict nothing reads is not implemented
  complementary slackness           MUST be able to fail loudly -- it grades OUR verdicts
  Gables ordering                    reproduce 40 -> 1.3 -> 2 -> 160, where three of four steps are
                                     locally sensible moves that made things WORSE
  four hard constraints              not a TPE objective / `unknown` below the noise floor / never
                                     prunes / an interval per dimension
"""

from __future__ import annotations

from kernel_optimizer.evaluation.conversion import conversion_verdict
from kernel_optimizer.evaluation.conversion_report import (
    GABLES_STEPS,
    conversion_lines,
    gables_ranking,
    slackness_violations,
)


class _Ev:
    """An event as `report.py` sees it (attribute access), so the tests drive the real reader."""

    def __init__(self, type_: str, payload: dict):
        self.type = type_
        self.payload = payload


def _round(family="fam-1", rnd=1, conversion="improved", gain=8.0,
           improved=("shared_bytes",), deltas=None) -> _Ev:
    return _Ev("FAMILY_ROUND_RECORDED", {
        "family_id": family, "round": rnd, "best_ms": 1.5,
        "conversion": conversion, "latency_gain_pct": gain,
        "conversion_note": "a note",
        "resources_improved": list(improved),
        "resource_deltas": deltas or {
            d: {"before": 100.0, "after": 50.0, "delta": -50.0, "rel": 0.5, "unit": "u",
                "direction": "improved"} for d in improved},
    })


def _state(slack=("shared_bytes",), binding=("occupancy",)) -> _Ev:
    records = [{"dimension_id": d, "verdict": "slack", "applicable": True} for d in slack]
    records += [{"dimension_id": d, "verdict": "binding", "applicable": True} for d in binding]
    return _Ev("DIMENSION_STATE", {"candidate_id": "c1", "records": records})


# --------------------------------------------------------------------------------------------------
# The consumer
# --------------------------------------------------------------------------------------------------

def test_the_report_has_a_conversion_section_that_reads_the_event_log():
    """The gap this closes, verbatim from the plan: "`report.py` 里 `conversion` 0 次命中" and
    "没有消费者的判决等于没实现".

    Revert-checked against not calling `conversion_lines`: the section is absent, and the verdict
    remains computed, journalled, and read by nothing.
    """
    lines = conversion_lines([_round(), _state()], min_improvement_pct=2.0)
    text = "\n".join(lines)
    assert "What resource changes bought (conversion)" in text
    assert "`improved`" in text
    assert "fam-1" in text


def test_no_conversion_is_the_row_that_gets_the_emphasis():
    """`no_conversion` is the informative outcome -- a resource improved and latency did not move, so
    THAT resource was not the limit. It is worth more than any utilisation reading because it is a
    controlled comparison between two real measurements of the same family."""
    lines = conversion_lines([_round(conversion="no_conversion", gain=0.3), _state()],
                             min_improvement_pct=2.0)
    text = "\n".join(lines)
    assert "was not the limit" in text


def test_a_run_with_no_rewrite_rounds_says_so_rather_than_rendering_nothing():
    """An empty section and a section saying "no rounds" are different documents. 4 of 19 runs used
    0-2 of their rewrite rounds, so this is the common case, not an edge one."""
    text = "\n".join(conversion_lines([], min_improvement_pct=2.0))
    assert "nothing to convert" in text
    assert "a fact about the search, not about the mechanism" in text


def test_rounds_without_a_conversion_field_are_reported_as_predating_the_fix():
    """This is the state of EVERY existing corpus: 9 of 9 `FAMILY_ROUND_RECORDED` events carry no
    `conversion` field and 0 carry `resource_deltas`, because the five runs predate G27's fix.

    A section that rendered nothing here would silently reproduce the "zero consumers" gap -- the
    reader could not tell "nothing converted" from "the mechanism never ran".

    Revert-checked against skipping the branch: the section shows an empty table and a reader
    concludes the mechanism produced no verdicts, which is true but for the wrong reason.
    """
    bare = _Ev("FAMILY_ROUND_RECORDED", {"family_id": "fam-1", "round": 1, "best_ms": 1.5})
    text = "\n".join(conversion_lines([bare, bare], min_improvement_pct=2.0))
    assert "none carries a conversion verdict" in text
    assert "G27" in text
    assert "not that nothing converted" in text


def test_the_section_is_built_from_the_event_log_alone():
    """`report` regenerates purely from `events.jsonl` -- that is how the trace is proven complete --
    so a section needing live state would break it. Asserted by passing ONLY events."""
    text = "\n".join(conversion_lines([_round(), _round(rnd=2, conversion="flat", gain=0.1),
                                       _state()], min_improvement_pct=2.0))
    assert "round 1" in text and "round 2" in text


# --------------------------------------------------------------------------------------------------
# Complementary slackness -- the check that grades us
# --------------------------------------------------------------------------------------------------

def test_a_slack_dimension_that_bought_speed_is_reported_as_our_error():
    """The criterion, verbatim: "若买到改善 ⇒ 我们的判决是错的". LP duality: the shadow price of a
    non-binding constraint is zero, so improving a `slack` dimension must buy nothing.

    Revert-checked against returning `[]` unconditionally: the check silently passes forever, which is
    the shape of a probe with no positive control -- and this project has already read five negative
    "results" off a broken probe.
    """
    v = slackness_violations([_round(gain=8.0, improved=("shared_bytes",)).payload],
                             {"shared_bytes"}, min_improvement_pct=2.0)
    assert len(v) == 1
    assert v[0].dimension == "shared_bytes"
    assert v[0].latency_gain_pct == 8.0
    assert "OUR VERDICT WAS WRONG" in v[0].detail
    assert "not that the candidate was" in v[0].detail


def test_a_slack_dimension_that_bought_nothing_is_the_theorem_holding_and_is_not_reported():
    """One-directional on purpose. Reporting the holding case would bury the violating one in noise --
    and `no_conversion` on a slack dimension is exactly what the theorem predicts."""
    v = slackness_violations([_round(gain=0.3, conversion="no_conversion").payload],
                             {"shared_bytes"}, min_improvement_pct=2.0)
    assert v == []


def test_a_binding_dimension_that_bought_speed_is_not_a_violation():
    """The control that keeps the check from firing on every improvement: a binding dimension buying
    speed is the system working."""
    v = slackness_violations([_round(gain=8.0, improved=("occupancy",)).payload],
                             {"shared_bytes"}, min_improvement_pct=2.0)
    assert v == []


def test_a_gain_below_the_runs_own_threshold_is_not_a_violation():
    """Reuses the run's `min_improvement_pct` rather than a local constant. The noise floor is a
    property of (card, task) -- on L3:48 the per-trial std was 16% of the mean while 33 near-ties
    spanned 9% -- so a second constant here would be a second opinion about the same quantity."""
    v = slackness_violations([_round(gain=1.5).payload], {"shared_bytes"},
                             min_improvement_pct=2.0)
    assert v == []
    v2 = slackness_violations([_round(gain=1.5).payload], {"shared_bytes"},
                              min_improvement_pct=1.0)
    assert len(v2) == 1


def test_no_slack_dimensions_means_the_check_cannot_run_and_says_so_rather_than_passing():
    """"Nothing was slack" is not a pass. A check that reports success when it had nothing to test is
    how five negative results were read off a broken probe."""
    text = "\n".join(conversion_lines([_round(), _state(slack=(), binding=("occupancy",))],
                                      min_improvement_pct=2.0))
    assert "NO dimension was judged `slack`" in text
    assert "That is not a pass" in text


def test_no_dimension_state_at_all_says_the_check_could_not_run():
    text = "\n".join(conversion_lines([_round()], min_improvement_pct=2.0))
    assert "no dimension was judged `slack` and the check cannot run" in text


def test_a_clean_pass_is_labelled_weak_and_says_why():
    """The slack set is a union over the whole run, and with few rounds the check also passes
    vacuously. Both caveats have to be in the output, or a weak pass gets cited as a strong one."""
    text = "\n".join(conversion_lines([_round(gain=8.0, improved=("occupancy",)), _state()],
                                      min_improvement_pct=2.0))
    assert "No violation" in text
    assert "WEAK pass" in text
    assert "union over the whole run" in text
    assert "vacuously" in text


def test_the_violation_carries_its_attribution_caveat():
    """A rewrite is RE-TUNED, so the gain may come from elsewhere in the same round. Without the
    caveat a single coincidence would read as a refutation."""
    text = "\n".join(conversion_lines([_round(gain=8.0, improved=("shared_bytes",)), _state()],
                                      min_improvement_pct=2.0))
    assert "violation(s)" in text
    assert "RE-TUNED" in text
    assert "One coincidence is a question" in text


def test_the_slack_set_is_a_union_and_the_direction_of_that_error_is_toward_noise():
    """Documented in the function and asserted here: linking `DIMENSION_STATE` (per candidate) to
    `FAMILY_ROUND_RECORDED` (per family, round) needs the round's parent AT THAT MOMENT, which is not
    journalled. Inventing the mapping would be the payload-path guess that has misfired twice here.

    The union over-reports rather than under-reports, which for a check whose value is that it can
    fail loudly is the acceptable direction.
    """
    events = [_round(gain=8.0, improved=("shared_bytes",)),
              _state(slack=("shared_bytes",)),
              _state(slack=("n_regs",))]
    text = "\n".join(conversion_lines(events, min_improvement_pct=2.0))
    assert "violation(s)" in text
    assert "shared_bytes" in text


# --------------------------------------------------------------------------------------------------
# The Gables ordering control
# --------------------------------------------------------------------------------------------------

def test_the_gables_four_step_ordering_is_reproduced():
    """Gables' worked example: 40 -> 1.3 -> 2 -> 160 Gops/s. Three of the four steps move a resource in
    a locally sensible direction and get WORSE, so a rule that follows one resource cannot reproduce
    this ranking. That is why it is a control rather than a formality.
    """
    assert gables_ranking() == ["step-3", "baseline", "step-2", "step-1"]


def test_the_gables_sequence_is_non_monotone_which_is_what_makes_it_a_control():
    """Asserted so a future edit cannot replace the figures with a monotone sequence and keep the test
    green. A monotone example would be passable by any rule at all.
    """
    values = [v for _, v in GABLES_STEPS]
    ups = sum(1 for a, b in zip(values, values[1:]) if b > a)
    downs = sum(1 for a, b in zip(values, values[1:]) if b < a)
    assert ups >= 1 and downs >= 1, "a monotone sequence would be passable by any rule"
    assert values[0] > values[1], "the first step must make things worse"


def test_the_ordering_rule_ranks_by_achieved_throughput_not_by_resource_movement():
    """The rule under test. "Which resource moved most" is BANNED: shared memory's median swing is
    13.5x registers', purely from the normalisation, so an argmax-|delta| rule degenerates into
    "always pick shared" (measured: 61.5% of the time) and discovers nothing.
    """
    scrambled = (("step-1", 1.3), ("step-3", 160.0), ("baseline", 40.0), ("step-2", 2.0))
    assert gables_ranking(scrambled) == ["step-3", "baseline", "step-2", "step-1"]


# --------------------------------------------------------------------------------------------------
# The four hard constraints
# --------------------------------------------------------------------------------------------------

def test_conversion_is_never_a_tuning_objective():
    """Latency is the only objective. A conversion figure as a target would optimize the diagnostic.

    A grep-shaped assertion run as a test so it cannot rot -- checked against the modules that define
    what the tuner and the selection chain optimize.
    """
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "src" / "kernel_optimizer"
    for rel in ("tuning/tpe.py", "tuning/stats.py", "control/convergence.py"):
        body = (root / rel).read_text(encoding="utf-8").lower()
        assert "conversion" not in body, (
            f"{rel} mentions conversion; it must never enter the objective or the freeze decision")


def test_conversion_never_prunes_or_screens_a_candidate():
    """The measured counter-example: on L2:37 the family whose seed was the SLOWEST improved the most,
    -30.5%. Pruning on an early conversion figure would have killed it.

    `conversion_report` is a REPORTING module -- it returns strings and a list of violations, and has
    no path that removes anything. Asserted on its public surface.
    """
    import kernel_optimizer.evaluation.conversion_report as cr

    exported = [n for n in dir(cr) if not n.startswith("_")]
    for name in exported:
        assert not any(bad in name.lower() for bad in ("prune", "screen", "reject", "kill")), (
            f"{name} suggests a pruning path; conversion may never remove a candidate")


def test_a_latency_move_inside_the_noise_floor_reads_flat_not_as_a_small_gain():
    """`unknown`/`flat` below the noise floor, never a small number. Drives the real
    `conversion_verdict`, since that is where the threshold lives."""
    v = conversion_verdict(1.500, 1.495, None, None, min_improvement_pct=2.0)
    assert v["conversion"] in ("flat", "no_conversion")
    assert v["conversion"] != "improved"


def test_latency_that_cannot_be_compared_reads_unknown():
    v = conversion_verdict(None, 1.5, None, None, min_improvement_pct=2.0)
    assert v["conversion"] == "unknown"
    assert "not both available" in v["conversion_note"]


def test_each_dimension_is_reported_with_its_own_interval_not_a_single_number():
    """"每维报区间" -- before, after, delta and rel per dimension, so a reader can see the size of the
    move rather than only its direction."""
    prof_a = type("P", (), {"n_regs": 96, "shared_bytes": 65536, "n_spills": 0,
                            "peak_alloc_bytes": None, "candidate_aten_bytes": None,
                            "candidate_aten_ops": None, "threads_launched": None,
                            "occupancy": {"occupancy": 0.5}})()
    prof_b = type("P", (), {"n_regs": 218, "shared_bytes": 16384, "n_spills": 0,
                            "peak_alloc_bytes": None, "candidate_aten_bytes": None,
                            "candidate_aten_ops": None, "threads_launched": None,
                            "occupancy": {"occupancy": 0.17}})()
    v = conversion_verdict(2.0, 1.5, prof_a, prof_b, min_improvement_pct=2.0)
    deltas = v["resource_deltas"]
    for dim in ("n_regs", "shared_bytes"):
        assert set(deltas[dim]) >= {"before", "after", "delta", "rel", "unit", "direction"}


def test_the_section_reaches_the_GENERATED_report_not_only_the_helper(tmp_path):
    """The one test that actually proves S4' landed.

    Everything above drives `conversion_lines` directly, which is the same mistake the gap itself was:
    `conversion_verdict` was correct, journalled, and READ BY NOTHING. A helper that renders perfectly
    and is never called is indistinguishable from no helper at all -- so this builds a real RunStore,
    generates the real report, and reads the markdown off disk.

    Revert-checked against removing the `conversion_lines(...)` call from `report.py`: every other test
    in this file still passes and only this one fails, which is exactly the blind spot being covered.
    """
    from kernel_optimizer.reporting.report import ReportGenerator
    from kernel_optimizer.store.run_store import RunStore

    store = RunStore.create(tmp_path, run_id="s4-consumer", manifest={
        "config": {"budgets": {"min_improvement_pct": 2.0}}})
    store.append("CANDIDATE_REGISTERED", {"candidate": {
        "candidate_id": "cand-aaa", "family_id": "fam-1", "parent_ids": [],
        "origin": "seed", "backend": "triton", "source_sha": "x",
        "structural_signature": "s", "approach_summary": "a fused scan"}})
    store.append("TRIAL_DONE", {"trial": {
        "trial_id": "tr-1", "candidate_id": "cand-aaa", "space_id": "sp-1",
        "params": {"values": {"B": 64}}, "status": "complete",
        "latency_ms": {"mean": 2.5, "std": 0.1, "min": 2.4, "max": 2.9, "n_samples": 20}}})
    store.append("TUNING_DONE", {"candidate_id": "cand-aaa", "space_id": "sp-1",
                                 "best_ms": 2.5, "snapshot": {"asked": 40}})
    # A round that improved latency while improving a dimension we called `slack` -- so the generated
    # report must carry BOTH the conversion table and a slackness violation.
    store.append("DIMENSION_STATE", _state(slack=("shared_bytes",)).payload)
    store.append("FAMILY_ROUND_RECORDED",
                 _round(gain=8.0, improved=("shared_bytes",)).payload)

    text = ReportGenerator().generate(store).read_text(encoding="utf-8")
    assert "What resource changes bought (conversion)" in text, (
        "the conversion section did not reach the generated report; a verdict with no consumer is "
        "not implemented")
    assert "Complementary-slackness check" in text
    assert "violation(s)" in text
    assert "shared_bytes" in text
    # And the run's own threshold must be the one used -- not a constant inside the helper.
    assert "8.00% latency gain" in text


def test_a_violation_model_refuses_an_invented_field():
    """Same reason as `ResourceExpectation` and `LowerBound`: pydantic's default silently drops an
    unknown field, so a `severity` someone adds in passing would validate and vanish."""
    import pytest

    from kernel_optimizer.evaluation.conversion_report import SlacknessViolation

    with pytest.raises(Exception):
        SlacknessViolation(dimension="n_regs", severity="high")
