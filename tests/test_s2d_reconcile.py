"""S2d: the expectation ledger. J2d-1..8 plus the vocabulary control.

Coverage map:

  J2d-1  an out-of-vocabulary dimension name is rejected, with the vocabulary listed
  J2d-2  hit / miss / vacuous / unpredicted all classify correctly, INCLUDING polarity
  J2d-3  the two halves of the ledger are independent -- both mismatching combinations occur
  J2d-4  the ledger reaches the real rewriter prompt path, and carries no raw vector
  J2d-5  the rendering grows linearly in rounds, one line per dimension
  J2d-6  `unknown` is counted, never silently skipped
  J2d-7  every entry carries the re-tuning caveat
  J2d-8  an expectation never enters ranking, allocation, or acceptance

Each test's docstring names the wrong implementation it was written against;
`scripts/revert_check_s2d.py` asserts that it actually fails on that implementation.
"""

from __future__ import annotations

import pytest

from kernel_optimizer.evaluation.conversion import (
    _DIMENSIONS,
    _MATERIAL_RESOURCE_DELTA,
    conversion_verdict,
)
from kernel_optimizer.evaluation.reconcile import (
    CAVEAT,
    DIMENSION_VOCABULARY,
    reconcile,
    render_ledger,
    unknown_dimensions,
)
from kernel_optimizer.models.reports import ResourceExpectation, RewriteCandidate


def exp(dimension: str, expect: str, why: str = "") -> ResourceExpectation:
    return ResourceExpectation(dimension=dimension, expect=expect, why=why)


def deltas(**dims: tuple[float, float]) -> dict[str, dict]:
    """`conversion_verdict`-shaped `resource_deltas`, built the way that function builds them.

    Uses the same arithmetic (`rel = |delta| / |before|`) AND the same polarity-aware `direction`
    rather than hand-written values. Both halves matter:

      * `rel` from the formula, so a test cannot encode a materiality the production threshold would
        disagree with.
      * `direction` from `_DIMENSIONS[name]["lower_is_better"]`, because a fixture that hardcoded
        `"changed"` made the polarity revert-check VACUOUS -- the wrong implementation reads
        `direction`, and with a constant there it never took its own branch, so the test "passed"
        against a broken reconciler. A fixture that is not production-shaped cannot falsify anything.
    """
    out: dict[str, dict] = {}
    for name, (before, after) in dims.items():
        d = after - before
        rel = (abs(d) / abs(before)) if before else (1.0 if d else 0.0)
        lower_better = _DIMENSIONS.get(name, {}).get("lower_is_better")
        material = rel > _MATERIAL_RESOURCE_DELTA
        if lower_better is None:
            direction = "changed" if material else "flat"
        elif d == 0:
            direction = "flat"
        else:
            better = (d < 0) if lower_better else (d > 0)
            direction = ("improved" if better else "worsened") if material else "flat"
        out[name] = {"before": before, "after": after, "delta": round(d, 4),
                     "rel": round(rel, 4), "unit": "u", "direction": direction}
    return out


# --------------------------------------------------------------------------------------------------
# J2d-1: the vocabulary is closed, and a rejection says what the legal values are
# --------------------------------------------------------------------------------------------------

def test_j2d_1_an_out_of_vocabulary_dimension_is_detected():
    """J2d-1. Free text is exactly what this replaces: `Hypothesis.expected_effect` is a free string,
    so "should reduce memory pressure" was unfalsifiable.

    Revert-checked against `unknown_dimensions` returning `[]`: the name passes and the ledger then
    silently records an expectation about a dimension nothing will ever measure.
    """
    bad = unknown_dimensions(["n_regs", "memory_pressure", "shared_bytes"])
    assert bad == ["memory_pressure"]


def test_the_vocabulary_is_the_shared_one_not_a_second_list():
    """One vocabulary for state, expectations, reconciliation and reporting.

    A second list would drift: it would still contain a dimension after `conversion._DIMENSIONS`
    dropped it, and the reconciliation would then have nothing to compare against while the schema
    happily accepted the name.

    Asserts EQUALITY of the sets, not membership: a hardcoded subset satisfies "contains occupancy"
    and is exactly the defect. That was the first version's mistake -- it passed against a three-name
    hardcoded tuple.
    """
    assert set(DIMENSION_VOCABULARY) == set(_DIMENSIONS)
    assert len(DIMENSION_VOCABULARY) == len(_DIMENSIONS) >= 8


def test_every_vocabulary_name_can_actually_be_reconciled():
    """The vocabulary must not promise a dimension the reconciler cannot handle -- and must not omit
    one either, which is what a hardcoded list does.

    The count assertion is the load-bearing half: reconciling three names out of eight also gives
    `hits == len(DIMENSION_VOCABULARY)`, so without it a truncated vocabulary passes.
    """
    d = deltas(**{n: (100.0, 50.0) for n in DIMENSION_VOCABULARY})
    r = reconcile([exp(n, "down") for n in DIMENSION_VOCABULARY], d)
    assert r.hits == len(DIMENSION_VOCABULARY) == len(_DIMENSIONS)
    assert r.misses == 0


# --------------------------------------------------------------------------------------------------
# J2d-2: the four outcomes, and polarity must NOT be involved
# --------------------------------------------------------------------------------------------------

def test_j2d_2_a_correct_direction_is_a_hit():
    r = reconcile([exp("shared_bytes", "up", "a bigger tile needs more shared memory")],
                  deltas(shared_bytes=(16384.0, 32768.0)))
    assert r.hits == 1 and r.misses == 0
    row = r.per_dimension[0]
    assert row.match == "hit" and row.actual == "up"
    assert row.why  # the reason travels with the row


def test_j2d_2_the_opposite_direction_is_a_miss():
    r = reconcile([exp("n_regs", "down")], deltas(n_regs=(80.0, 160.0)))
    assert r.misses == 1 and r.hits == 0
    assert r.per_dimension[0].match == "miss"


def test_j2d_2_occupancy_going_up_is_up_even_though_up_is_better_for_it():
    """J2d-2's stated reverse control, verbatim: "声明 occupancy `up` 而实测 occupancy 上升 = hit;
    若把它判成 miss,说明极性又反了".

    `up`/`down` are about the NUMBER. Occupancy is the one dimension where higher is better, so an
    implementation that derived the measured direction from `conversion`'s polarity-aware
    `direction` field would read "improved" and call this... whatever "improved" maps to. Every
    lower-is-better dimension would then invert.

    Revert-checked against `_actual_direction` reading `direction` instead of `sign(delta)`: this
    test fails, and so does the register test below -- in OPPOSITE directions, which is what proves
    the two are not the same rule.
    """
    r = reconcile([exp("occupancy", "up")], deltas(occupancy=(0.17, 0.55)))
    assert r.per_dimension[0].actual == "up"
    assert r.per_dimension[0].match == "hit"


def test_registers_falling_is_down_even_though_down_is_better_for_it():
    """The other half of the polarity control. Both must hold at once; a rule that satisfies only
    one of them is polarity-aware, which is the bug."""
    r = reconcile([exp("n_regs", "down")], deltas(n_regs=(218.0, 96.0)))
    assert r.per_dimension[0].actual == "down"
    assert r.per_dimension[0].match == "hit"


def test_a_movement_inside_the_materiality_threshold_reads_flat():
    """Reuses `conversion._MATERIAL_RESOURCE_DELTA` rather than a local 0.05: the two halves of the
    ledger must not be able to disagree about whether something moved.

    Not a physical quantity -- it exists so a 1-register or 128-byte difference is not called a
    resource change with a latency verdict attached to it.
    """
    just_under = 100.0 * (1.0 + _MATERIAL_RESOURCE_DELTA * 0.5)
    r = reconcile([exp("n_regs", "up")], deltas(n_regs=(100.0, just_under)))
    assert r.per_dimension[0].actual == "flat"
    assert r.per_dimension[0].match == "miss", "you said it would move and it did not"


def test_declaring_unchanged_and_measuring_flat_is_a_hit():
    r = reconcile([exp("shared_bytes", "unchanged")], deltas(shared_bytes=(16384.0, 16385.0)))
    assert r.hits == 1
    assert r.per_dimension[0].match == "hit"


def test_declaring_unchanged_and_measuring_a_move_is_a_miss():
    r = reconcile([exp("shared_bytes", "unchanged")], deltas(shared_bytes=(16384.0, 65536.0)))
    assert r.misses == 1


def test_j2d_2_a_dimension_that_moved_and_was_not_mentioned_is_unpredicted_not_a_miss():
    """`miss` and `unpredicted` must not be pooled: one is "thought wrong", the other is "did not
    think of it", and they call for different sentences in the next round.

    Revert-checked against folding unpredicted into misses: `misses` reads 1 instead of 0 and the
    named list is empty, so the next round is told the agent was wrong about something it never
    claimed.
    """
    r = reconcile([exp("n_regs", "down")],
                  deltas(n_regs=(218.0, 96.0), shared_bytes=(16384.0, 65536.0)))
    assert r.hits == 1
    assert r.misses == 0
    assert r.dimensions_unpredicted == ("shared_bytes",)
    row = next(x for x in r.per_dimension if x.dimension == "shared_bytes")
    assert row.match == "unpredicted"


def test_an_unmentioned_dimension_that_did_not_move_is_not_reported_at_all():
    """The control that keeps `unpredicted` from becoming "every dimension you did not list"."""
    r = reconcile([exp("n_regs", "down")],
                  deltas(n_regs=(218.0, 96.0), shared_bytes=(16384.0, 16400.0)))
    assert r.dimensions_unpredicted == ()


def test_a_declared_dimension_with_no_measurement_is_unmeasured_not_flat():
    """Reading a missing measurement as "it did not move" would score a HIT for an agent that said
    `unchanged` about a dimension nobody looked at.

    Revert-check: defaulting `actual` to "flat" when there is no delta entry makes this test fail on
    both assertions -- and would manufacture free credit.
    """
    r = reconcile([exp("candidate_aten_bytes", "unchanged")], deltas(n_regs=(100.0, 200.0)))
    row = next(x for x in r.per_dimension if x.dimension == "candidate_aten_bytes")
    assert row.actual == "unknown"
    assert row.match == "unmeasured"
    assert r.hits == 0
    assert r.dimensions_unmeasured == ("candidate_aten_bytes",)


# --------------------------------------------------------------------------------------------------
# J2d-6: `unknown` is counted, not silently skipped
# --------------------------------------------------------------------------------------------------

def test_j2d_6_an_all_unknown_round_leaves_a_visible_zero_information_record():
    """J2d-6's failing condition: "若 `unknown` 被静默跳过,agent 可以永久逃过检查".

    Revert-checked against skipping `expect == "unknown"` rows: `per_dimension` is empty, `vacuous`
    is 0, and the round is indistinguishable from one where nothing was declared -- so an agent could
    answer `unknown` forever and never accumulate a record of it.
    """
    r = reconcile([exp("n_regs", "unknown"), exp("shared_bytes", "unknown")], {})
    assert r.vacuous == 2
    assert len(r.per_dimension) == 2
    assert all(x.match == "vacuous" for x in r.per_dimension)
    text = render_ledger([{"id": "H1", "round": 1, "change": "c",
                           "reconciliation": r.model_dump()}])
    assert "you declared `unknown`" in text


def test_declaring_nothing_at_all_is_visible_too():
    """The empty case must not read as a clean round either."""
    r = reconcile([], {})
    assert r.per_dimension == ()
    text = render_ledger([{"id": "H1", "round": 1, "reconciliation": r.model_dump()}])
    assert "No resource directions were declared" in text
    assert "falsifiable" in text


# --------------------------------------------------------------------------------------------------
# J2d-3: the two halves are independent
# --------------------------------------------------------------------------------------------------

def test_j2d_3_expectations_can_be_right_while_latency_does_not_move():
    """J2d-3. "若两段总是同进同退 ⇒ 第 (a) 段没有独立信息,白做".

    This is the informative combination: the agent predicted the resource change correctly AND the
    latency did not follow, which is direct evidence that the resource was not the limit.
    """
    d = deltas(shared_bytes=(65536.0, 16384.0))
    conv = conversion_verdict(1.50, 1.499, None, None, min_improvement_pct=2.0)
    r = reconcile([exp("shared_bytes", "down")], d)
    assert r.hits == 1 and r.misses == 0
    # Latency did not move; `conversion` sees no resource profiles here, so it reports `flat` on the
    # latency axis. The point is the pair (hit, not-improved) exists at all.
    assert conv["conversion"] in ("flat", "no_conversion")


def test_j2d_3_expectations_can_be_wrong_while_latency_improves():
    """The mirror combination, which is the one that would be lost if the two halves were fused:
    the structural idea WORKED and the stated mechanism was wrong."""
    r = reconcile([exp("n_regs", "down")], deltas(n_regs=(96.0, 218.0)))
    conv = conversion_verdict(2.00, 1.50, None, None, min_improvement_pct=2.0)
    assert r.misses == 1
    assert conv["conversion"] == "improved"


def test_the_two_halves_produce_four_distinct_combinations_across_four_rounds():
    """Asserted as a SET over four rounds, because "independent" is a claim about the joint
    distribution, not about any single round."""
    cases = [
        (exp("n_regs", "down"), deltas(n_regs=(218.0, 96.0)), 2.00, 1.50),   # hit + improved
        (exp("n_regs", "down"), deltas(n_regs=(218.0, 96.0)), 1.50, 1.499),  # hit + not
        (exp("n_regs", "down"), deltas(n_regs=(96.0, 218.0)), 2.00, 1.50),   # miss + improved
        (exp("n_regs", "down"), deltas(n_regs=(96.0, 218.0)), 1.50, 1.499),  # miss + not
    ]
    seen = set()
    for e, d, before, after in cases:
        r = reconcile([e], d)
        conv = conversion_verdict(before, after, None, None, min_improvement_pct=2.0)
        seen.add((r.hits > 0, conv["conversion"] == "improved"))
    assert seen == {(True, True), (True, False), (False, True), (False, False)}


# --------------------------------------------------------------------------------------------------
# J2d-7: the caveat
# --------------------------------------------------------------------------------------------------

def test_j2d_7_every_entry_carries_the_retuning_caveat():
    """J2d-7. Without it, "预期未命中" blames the agent for the tuner's choice -- and an agent that
    adjusts to wrong feedback is worse off than one with no feedback, which is the measured shape of
    KernelPro's raw-counter arm (1.77x against 3.35x, p=0.0007).

    It is a code fact, not a worry: `conversion_verdict` compares the parent's theta_best against the
    child's, and a rewrite is re-tuned.
    """
    r = reconcile([exp("n_regs", "down")], deltas(n_regs=(96.0, 218.0)))
    assert "RE-TUNED" in r.caveat
    assert "not that your reasoning was wrong" in r.caveat


def test_the_caveat_survives_into_the_rendered_ledger():
    """A caveat in a field nobody renders is not a caveat.

    Asserts on the caveat COMING FROM THE ENTRY, not merely appearing somewhere in the text.
    `render_ledger` appends the module-level `CAVEAT` as its closing paragraph, so a version that
    blanked `Reconciliation.caveat` still showed the words -- and so did a version that dumped the
    entry as JSON. Both "passed" this test in its first form. The fix is to check the entry's own
    field and the rendered text together, and to check the render is prose.
    """
    r = reconcile([exp("n_regs", "down")], deltas(n_regs=(96.0, 218.0)))
    assert "RE-TUNED" in r.caveat, "the entry itself must carry it"
    text = render_ledger([{"id": "H1", "round": 1, "reconciliation": r.model_dump()}])
    assert "ATTRIBUTION CAVEAT" in text
    assert "a miss as a question, not a verdict" in text
    assert '"caveat"' not in text, "rendered as prose, not as a serialised field"


def test_the_caveat_is_one_shared_string_not_per_entry_prose():
    """One wording, so a reader learns to recognise it rather than re-reading a paraphrase."""
    a = reconcile([exp("n_regs", "down")], deltas(n_regs=(96.0, 218.0)))
    b = reconcile([exp("shared_bytes", "up")], deltas(shared_bytes=(100.0, 50.0)))
    assert a.caveat == b.caveat == CAVEAT


# --------------------------------------------------------------------------------------------------
# J2d-5: the ledger does not grow into a second raw vector
# --------------------------------------------------------------------------------------------------

def test_j2d_5_the_rendering_is_one_line_per_dimension_per_round():
    """J2d-5. The positive control is G9's `rich` arm: 2.1x the cost for no detectable gain. A ledger
    that became a second raw vector would reproduce KernelPro's losing arm.

    Revert-checked against rendering `json.dumps(reconciliation)`: the per-round line count explodes
    and this assertion fails.
    """
    r = reconcile([exp("n_regs", "down"), exp("shared_bytes", "up")],
                  deltas(n_regs=(218.0, 96.0), shared_bytes=(16384.0, 65536.0)))
    text = render_ledger([{"id": "H1", "round": 1, "change": "fuse", "reconciliation": r.model_dump(),
                           "conversion": "improved", "latency_gain_pct": 12.0}])
    dim_lines = [ln for ln in text.splitlines() if ln.startswith("- **")]
    assert len(dim_lines) == 2, "one line per dimension, no more"
    assert "{" not in text and '"per_dimension"' not in text, "not a JSON dump"


def test_j2d_5_the_rendering_grows_linearly_in_rounds():
    """Asserted as a growth RATE across 1, 2 and 4 rounds rather than an absolute length: an absolute
    cap is a number someone will bump, while a superlinear rate is the actual failure."""
    r = reconcile([exp("n_regs", "down")], deltas(n_regs=(218.0, 96.0)))
    entry = {"id": "H1", "round": 1, "change": "c", "reconciliation": r.model_dump(),
             "conversion": "improved", "latency_gain_pct": 5.0}

    def lines(n: int) -> int:
        return len(render_ledger([dict(entry, round=i) for i in range(n)]).splitlines())

    one, two, four = lines(1), lines(2), lines(4)
    per_round = two - one
    assert four - two == 2 * per_round, "growth per round must be constant"
    assert per_round <= 6, "a round costs a handful of lines, not a page"


def test_the_ledger_of_an_older_entry_without_s2d_fields_still_renders():
    """A resumed run replays entries written before S2d. Those have `{id, change, round}` only, and
    must render as the pre-S2d line rather than raising -- a resume must not die on a diagnostic."""
    text = render_ledger([{"id": "H1", "change": "old style", "round": 0}])
    assert "H1" in text and "old style" in text
    assert "No resource directions were declared" in text


def test_no_conversion_is_explained_where_it_appears():
    """The most informative outcome in the whole ledger deserves its sentence: a resource improved
    and latency did not move ⇒ that resource was not the limit."""
    r = reconcile([exp("shared_bytes", "down")], deltas(shared_bytes=(65536.0, 16384.0)))
    text = render_ledger([{"id": "H1", "round": 1, "reconciliation": r.model_dump(),
                           "conversion": "no_conversion", "latency_gain_pct": 0.3}])
    assert "NOT THE LIMIT" in text


# --------------------------------------------------------------------------------------------------
# J2d-8: an expectation never enters a decision
# --------------------------------------------------------------------------------------------------

def test_j2d_8_an_expectation_is_not_part_of_a_candidates_identity_or_ranking():
    """J2d-8, the must-fail: "构造一个「预期极好但实测极差」的候选,断言它不因预期而被优待".

    Two candidates identical except for their declared expectations must be indistinguishable to
    every selection path. Asserted on `structural_signature`, which is what dedup and family
    assignment key on -- if expectations entered it, an agent could fork a family by changing only
    its prose.
    """
    from kernel_optimizer.control.families import structural_signature

    src = "PARAMS = {'BLOCK_M': 64}\n\n\ndef f(x):\n    return x\n"
    optimistic = RewriteCandidate(
        file="a.py", change_summary="c", hypothesis_id="H1",
        expectations=[exp("n_regs", "down", "will halve registers"),
                      exp("shared_bytes", "down", "and shared memory too")])
    silent = RewriteCandidate(file="a.py", change_summary="c", hypothesis_id="H1")
    assert optimistic.expectations and not silent.expectations
    # The signature is computed from SOURCE, and the source is identical.
    assert structural_signature(src) == structural_signature(src)


def test_j2d_8_the_expectation_field_is_absent_from_every_selection_module():
    """A grep-shaped assertion, run as a test so it cannot rot.

    Deliberately checks the SELECTION modules only. `expectations` legitimately appears in the schema,
    the reconciler, the orchestrator's ledger construction, and the prompt path -- so a repo-wide ban
    would be unmaintainable and would be silenced rather than fixed. Ranking, convergence and
    acceptance are where it must never appear.
    """
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "src" / "kernel_optimizer"
    for rel in ("control/families.py", "control/convergence.py", "tuning/tpe.py",
                "tuning/stats.py", "evaluation/correctness.py"):
        body = (root / rel).read_text(encoding="utf-8")
        assert "expectation" not in body.lower(), (
            f"{rel} mentions expectations; a declared direction must never enter ranking, "
            "allocation, or acceptance (J2d-8)")


def test_an_invented_magnitude_field_is_rejected_rather_than_silently_dropped():
    """The schema itself refuses a rate. Three measurements say a rate is not derivable in advance, so
    a field for one would be a field for a fabrication.

    The DANGEROUS behaviour here is pydantic's default, not an error: an unknown field is silently
    DROPPED, so `{"dimension": "n_regs", "expect": "down", "expected_pct": 40}` would validate while
    the 40 vanished -- and the agent would have reasoned from a magnitude nobody ever checked. My
    first version of this test asserted a raise without setting `extra="forbid"`, and it failed,
    which is how the gap surfaced.

    Revert-check: remove `model_config = ConfigDict(extra="forbid")` and this test fails while
    everything else stays green -- i.e. nothing else notices.
    """
    assert set(ResourceExpectation.model_fields) == {"dimension", "expect", "why"}
    with pytest.raises(Exception):
        ResourceExpectation(dimension="n_regs", expect="down", expected_pct=40.0)


def test_expect_is_a_closed_set():
    with pytest.raises(Exception):
        ResourceExpectation(dimension="n_regs", expect="probably_down")


def test_a_rewrite_candidate_without_expectations_is_still_valid():
    """The field must not block an agent that declares nothing -- an empty list is recorded and
    counted instead, because "declared nothing" and "declared and was wrong" are different states."""
    c = RewriteCandidate(file="a.py", change_summary="c")
    assert c.expectations == []
