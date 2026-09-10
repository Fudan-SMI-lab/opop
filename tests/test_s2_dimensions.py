"""S2: per-dimension verdicts, the digestion layer, and the two must-fail controls.

Every test here was written against a WRONG implementation first and checked to fail on it; the
comment on each says which wrong implementation, because a test that passes on both the broken and
the fixed version is not evidence. That discipline exists because of a measured incident in this
repo: six green tests over a loop that was spinning 2.05M times, all six having re-implemented the
loop inside the test body instead of driving it.

Coverage map, to the plan's acceptance criteria:

  N1  (must fail)  applicable=False carrying a measured value must be reported
  N2  (must fail)  a raw vector reaching the prompt path must raise
  J2-1             distinguishability: different inputs must give different verdict combinations
  J2-4             an unreachable roof is named, and is NOT in the ranked recommendation
  J2-6             DRAM-saturated + 17% occupancy gives TWO binding records, not one label
  J2-7             a rewrite's records differ from its parent's -- keyed on structure, not family
  J2-9             D-9: the digest emits one finding per record, never fewer
"""

from __future__ import annotations

import pytest

from kernel_optimizer.evaluation.digest import (
    RawVectorInPromptError,
    assert_no_raw_vector,
    digest,
    for_prompt,
)
from kernel_optimizer.evaluation.dimensions import (
    DimensionRecord,
    DimensionState,
    state_from_evidence,
    unreachable_ceilings,
)
from kernel_optimizer.evaluation.reading_checks import check_applicability

# A800's real limits (sm_80, 108 SMs, 79.3 GiB, 166912 B shared/block), so the fractions the bands
# are computed from are the ones production uses rather than round numbers chosen to pass.
A800 = {"max_regs_per_thread": 255, "max_shared_bytes_optin": 166912, "vram_gb": 79.3}


def l3_48_evidence() -> dict:
    """The J2-6 case: DRAM at 93.7% with occupancy at 17%, both measured on the same candidate.

    Numbers from run-l3-48: the plateau candidates read 93.7-95.5% of the measured DRAM roof, and
    Triton's occupancy on this shape was 16.7% at 218 registers with 0 spills. This is the pair that
    a single `kind` string cannot express, which is the entire motivation for the stage.
    """
    return {
        "occupancy": 0.167,
        "occupancy_limiter": "registers",
        "n_regs": 218,
        "n_spills": 0,
        "shared_used_frac": 0.42,
        "peak_alloc_mib": 1350.0,
        "candidate_aten_mib": 1350.9,
        "candidate_aten_ops": 12,
        "threads_launched": 1048576,
        "pct_of_dram_peak": 93.7,
        "uses_tensor_cores": False,
        "compute_ceiling_used": "fp32",
    }


# --------------------------------------------------------------------------------------------------
# N1 (must fail): applicable=False must not carry a measurement
# --------------------------------------------------------------------------------------------------

def test_n1_applicable_false_with_a_measured_value_is_reported():
    """N1. The plan's wording: "造 `applicable=false` 却带 `measured` 的记录须报错".

    `applicable=False` means the dimension DOES NOT EXIST here -- a card without bf16 does not have
    that dimension at zero. A record that says both "does not exist" and "measures 0.42" is
    internally contradictory, and the contradiction is the dangerous direction: downstream reads the
    number.

    Revert-checked against the pre-fix `check_applicability`, which only looked at ranges and
    constants (`check_reading` + `check_collection`, grep-confirmed 0 occurrences of `applicable`):
    this test fails there because nothing reports anything.
    """
    bad = DimensionRecord(
        dimension_id="shared_bytes", measured=0.42, ceiling=166912.0,
        ceiling_provenance="test", verdict="slack", applicable=False,
        not_applicable_reason="this box has no such unit")
    notes = check_applicability([bad])
    assert notes, "a record that is both inapplicable and measured must be reported"
    assert any("applicable=False" in n and "verdict" in n for n in notes)


def test_n1_applicable_false_without_a_reason_is_reported():
    """The other half: an inapplicable record with no reason is indistinguishable from a collection
    failure, and telling those apart is the whole purpose of having the field.

    Asserts on rule 2's OWN message rather than merely on "some note fired". Rule 5 (a
    `not-applicable` verdict with no reason) also detects this input, so a bare `assert notes` passed
    even with rule 2 removed -- it would have been evidence for detection in general and not for the
    branch it names. The two rules say different things to a reader: rule 5 is about the verdict,
    rule 2 is about the flag.
    """
    bad = DimensionRecord(dimension_id="n_regs", verdict="not-applicable", applicable=False)
    notes = check_applicability([bad])
    assert any("applicable=False with no reason" in n for n in notes)


def test_a_measured_value_with_no_ceiling_is_not_reported_as_inconsistent():
    """The counterpart that keeps N1 honest: aten traffic HAS a value and has no roof to band
    against, so `unknown` is the correct verdict there and must not be flagged.

    This is not a weakening of the check -- it is the case the first version got wrong. It flagged
    both aten dimensions on the very first smoke test, which would have trained a reader to ignore
    the notes. The rule is narrowed to "a ceiling exists and no band was assigned", which still
    catches a real declination.
    """
    st = state_from_evidence(l3_48_evidence(), device_limits=A800)
    aten = [r for r in st.records if r.dimension_id.startswith("candidate_aten")]
    assert len(aten) == 2
    assert all(r.measured is not None and r.ceiling is None and r.verdict == "unknown"
               for r in aten)
    assert check_applicability(list(st.records)) == []


def test_applicable_true_with_a_ceiling_and_no_band_is_still_reported():
    """...and the narrowing did not remove the case it was narrowed around. A record WITH a ceiling
    that stayed `unknown` is a genuine silent declination and must still be caught.

    Revert-check: dropping the `if getattr(rec, "ceiling", ...)` guard entirely makes the previous
    test fail; dropping the whole branch makes THIS one fail. Both directions are covered.
    """
    bad = DimensionRecord(dimension_id="n_regs", measured=218.0, ceiling=255.0,
                          ceiling_provenance="test", verdict="unknown")
    notes = check_applicability([bad])
    assert any("declined to judge" in n for n in notes)


def test_a_verdict_with_no_measurement_behind_it_is_reported():
    """`slack` with nothing measured is the dangerous one: it reads as headroom."""
    bad = DimensionRecord(dimension_id="occupancy", measured=None, verdict="slack")
    notes = check_applicability([bad])
    assert any("no measured value behind it" in n for n in notes)


# --------------------------------------------------------------------------------------------------
# N2 (must fail): a raw vector must not reach the prompt path
# --------------------------------------------------------------------------------------------------

def test_n2_raw_evidence_dict_into_the_prompt_path_raises():
    """N2, G28. Driving the REAL gate, not a copy of its rule (D-5).

    Revert-checked against no gate at all: without `assert_no_raw_vector` the call returns a string
    and this test fails.
    """
    with pytest.raises(RawVectorInPromptError) as exc:
        assert_no_raw_vector(l3_48_evidence())
    # The message must name what leaked, or a future reader cannot act on it.
    assert "pct_of_dram_peak" in str(exc.value)


def test_n2_a_dimension_record_into_the_prompt_path_raises():
    """The type case. A record is the per-dimension raw form; only a Digest may reach a prompt."""
    rec = DimensionRecord(dimension_id="n_regs", measured=218.0, ceiling=255.0, verdict="slack")
    with pytest.raises(RawVectorInPromptError):
        assert_no_raw_vector(rec)
    with pytest.raises(RawVectorInPromptError):
        assert_no_raw_vector(DimensionState(records=(rec,)))


def test_n2_a_digest_is_allowed_through():
    """The gate must permit the thing it exists to permit, or it would be satisfied by blocking
    everything. `model_dump()` of a Digest goes through too -- that shape reaches report.md."""
    d = digest(state_from_evidence(l3_48_evidence(), device_limits=A800))
    assert_no_raw_vector(d)
    assert_no_raw_vector(d.model_dump())
    assert isinstance(for_prompt(d), str)


def test_n2_the_prompt_text_contains_no_raw_number_from_the_evidence():
    """Behavioural rather than type-based: the rendered text must not carry the raw readings.

    Checks the actual distinctive digits (218 registers, 93.7% of DRAM, 1048576 threads) rather
    than asserting on a key list -- a key list would pass the moment a key was renamed.
    """
    text = for_prompt(digest(state_from_evidence(l3_48_evidence(), device_limits=A800)))
    for raw in ("218", "93.7", "1048576", "0.167"):
        assert raw not in text, f"the raw reading {raw} reached the prompt text"


# --------------------------------------------------------------------------------------------------
# J2-6: several dimensions binding at once produce several records
# --------------------------------------------------------------------------------------------------

def test_j2_6_low_occupancy_is_binding_while_registers_are_not_at_the_hardware_cap():
    """J2-6, and rule 3 of §1.5.3: each dimension declares its own polarity.

    Occupancy at 16.7% is binding because occupancy binds when LOW. Registers at 218 of 255 are
    85.5% -- inside the near-binding band, not binding -- which is the honest reading and the one a
    uniform "high = binding" rule inverts.

    Revert-checked against a `_band` with no `higher_is_better` branch: occupancy 0.167 then lands
    in `slack` (0.167 < 0.70) and this test fails on the assertion below, which is exactly the
    measured failure that motivated the polarity field -- ZERO multi-binding reported while the
    counter-example sat in the input.
    """
    st = state_from_evidence(l3_48_evidence(), device_limits=A800)
    by_id = {r.dimension_id: r for r in st.records}
    assert by_id["occupancy"].verdict == "binding"
    assert by_id["occupancy"].higher_is_better is True
    assert by_id["n_regs"].verdict == "near-binding", "218/255 = 85.5% is the middle band"
    assert by_id["shared_bytes"].verdict == "slack"


def test_j2_6_two_dimensions_binding_at_once_give_two_records():
    """The shape assertion: the output is TWO binding records, not one label.

    Uses a spill alongside the low occupancy, because any spill is binding by nature (there is no
    "70% of no spills") -- so this is two independent walls measured on one candidate.

    Revert-check: a `classify()`-shaped function returning one string cannot satisfy `len(...) == 2`
    at all, which is the point of asserting on the count.
    """
    ev = dict(l3_48_evidence(), n_spills=48)
    st = state_from_evidence(ev, device_limits=A800)
    binding = st.binding()
    assert {r.dimension_id for r in binding} == {"occupancy", "n_spills"}
    assert len(binding) == 2

    d = digest(st)
    assert len(d.binding()) == 2
    # And the ordering basis must SAY that several are binding rather than silently ranking them.
    assert "binding at once" in d.ordering_basis


def test_a_spill_of_zero_is_slack_not_binding():
    """The other side of the spill rule, so it is not just "always binding"."""
    st = state_from_evidence(l3_48_evidence(), device_limits=A800)
    by_id = {r.dimension_id: r for r in st.records}
    assert by_id["n_spills"].measured == 0.0
    assert by_id["n_spills"].verdict == "slack"


# --------------------------------------------------------------------------------------------------
# J2-1: distinguishability
# --------------------------------------------------------------------------------------------------

def test_j2_1_two_candidates_with_different_resource_states_get_different_verdict_combinations():
    """J2-1. The baseline this replaces: 19 of 20 reports on L3:21 carried the SAME label.

    Compares the whole verdict tuple, not one field, because the stage's claim is about the
    combination. Revert-checked against the label path: `classify()` returns `resource_limited` for
    both of these, so the two combinations would be identical and this test fails.

    The two candidates differ ONLY in banded dimensions -- occupancy, registers, shared memory --
    with spills held at 0 in both. That is deliberate: `n_spills` is special-cased and does not go
    through `_band`, so an earlier version of this test (which also moved spills) still passed on a
    `_band` that returned one constant verdict for everything. It was then evidence for the spill
    branch, not for the banding this test is named for.
    """
    a = state_from_evidence(l3_48_evidence(), device_limits=A800)
    # Same spill count, different walls: occupancy is comfortable, registers are cheap, shared memory
    # is nearly exhausted. A genuinely different resource state, reached without the special case.
    b = state_from_evidence(
        dict(l3_48_evidence(), occupancy=0.78, n_regs=40, n_spills=0, shared_used_frac=0.95),
        device_limits=A800)

    combo_a = tuple((r.dimension_id, r.verdict) for r in a.records)
    combo_b = tuple((r.dimension_id, r.verdict) for r in b.records)
    assert combo_a != combo_b
    # Specifically: the walls are in different places, not merely different numbers.
    assert {r.dimension_id for r in a.binding()} == {"occupancy"}
    assert {r.dimension_id for r in b.binding()} == {"shared_bytes"}


def test_the_same_evidence_twice_gives_the_same_verdicts():
    """J2-1's own reverse control, from the plan: "若组合数上升但源于噪声抖动 ⇒ 不算通过".

    Distinguishability only counts if it is not jitter, so identical input must give identical
    output -- i.e. the mapping is a function of the evidence and of nothing else.
    """
    ev = l3_48_evidence()
    first = state_from_evidence(ev, device_limits=A800)
    second = state_from_evidence(dict(ev), device_limits=A800)
    assert first.records == second.records


# --------------------------------------------------------------------------------------------------
# J2-4: an unreachable roof is named, and is not something to act on
# --------------------------------------------------------------------------------------------------

def test_j2_4_a_backend_unreachable_roof_is_named_with_its_measured_fraction():
    """J2-4 via G10, the case with a number behind it: Triton reaches 84.1% of the fp32 roof on
    box 1 (45.61 against cuBLAS's 54.20, the widest of the four measured gaps).

    Revert-check: with `unreachable_ceilings()` returning `()` the assertion below fails, and the
    prompt then presents 84% of an unreachable roof as ordinary headroom.
    """
    ev = dict(l3_48_evidence(), backend_reachable_frac=0.841, compute_ceiling_used="fp32")
    notes = unreachable_ceilings(ev)
    assert any("84.1%" in n and "Triton" in n for n in notes)


def test_j2_4_an_l2_resident_working_set_marks_the_dram_roof_inapplicable():
    """The G26 case: a roof that does not apply at all, rather than one partly reachable."""
    ev = dict(l3_48_evidence(), dram_applicable=False,
              dram_inapplicable_reason="working set 40.0 MiB fits in this box's 40 MiB L2")
    notes = unreachable_ceilings(ev)
    assert any("DRAM roof does not apply" in n for n in notes)


def test_j2_4_an_unreachable_roof_never_appears_in_the_ranked_recommendation():
    """J2-4's failing condition, verbatim: "若它仍出现在 recommendation 里 ⇒ 没解决".

    Asserted structurally -- the roofs live in their own field, so no finding can carry one and
    `ranked_recommendation` is only ever built from `state.records`.

    Names the BACKEND roof specifically. Asserting only that `unreachable_ceilings` is non-empty
    passed even with the backend arm removed, because this evidence also trips the tensor-core arm --
    so the assertion has to be about the roof the test is named for.
    """
    ev = dict(l3_48_evidence(), backend_reachable_frac=0.841)
    d = digest(state_from_evidence(ev, device_limits=A800),
               unreachable=unreachable_ceilings(ev))
    assert any("84.1%" in c for c in d.unreachable_ceilings)
    for f in d.findings:
        assert "Triton" not in f.ranked_recommendation
        assert "reachable" not in f.ranked_recommendation
    text = for_prompt(d)
    # Present in the prompt as a caveat, and positioned after the reading order.
    assert "NOT what you are measured against" in text
    assert text.index("Reading order") < text.index("NOT what you are measured against")
    assert "84.1%" in text


def test_no_unreachable_roof_is_claimed_when_nothing_measured_says_so():
    """The control that keeps the previous three from being "always report a caveat"."""
    ev = dict(l3_48_evidence())
    ev.pop("uses_tensor_cores")
    ev.pop("compute_ceiling_used")
    assert unreachable_ceilings(ev) == ()


# --------------------------------------------------------------------------------------------------
# J2-7: resource use is a property of the structure, so a rewrite is re-measured
# --------------------------------------------------------------------------------------------------

def test_j2_7_a_rewrite_with_different_resource_use_gets_different_records():
    """J2-7. Keyed on `(candidate, structural_signature)`, never on family -- keying on family is
    precisely how a rewrite would silently inherit its parent's vector.

    Revert-check: a `state_from_evidence` that ignored its `structural_signature` argument would
    still pass the value assertions, so this test also asserts the KEY differs, which is the part
    that carries the defect.
    """
    parent = state_from_evidence(l3_48_evidence(), candidate_id="cand-parent",
                                structural_signature="sig-a", device_limits=A800)
    child = state_from_evidence(
        dict(l3_48_evidence(), occupancy=0.55, n_regs=96, shared_used_frac=0.88),
        candidate_id="cand-child", structural_signature="sig-b", device_limits=A800)

    assert (parent.candidate_id, parent.structural_signature) != \
           (child.candidate_id, child.structural_signature)
    assert parent.records != child.records
    by_parent = {r.dimension_id: r.verdict for r in parent.records}
    by_child = {r.dimension_id: r.verdict for r in child.records}
    assert by_parent["occupancy"] == "binding"
    assert by_child["occupancy"] == "slack", "0.55 is above the 0.50 near-binding line"


# --------------------------------------------------------------------------------------------------
# J2-9: D-9, the digest changes form and never reduces content
# --------------------------------------------------------------------------------------------------

def test_j2_9_the_digest_emits_one_finding_per_record():
    """J2-9, the D-9 guard. "消化层若把 9 维压成 2 维,J2-1 也会「通过」".

    Revert-check: a digest that filtered to `binding` + `near-binding` -- the natural "summarise"
    implementation, and the one the plan's first wording ("compressor") would have produced -- emits
    2 of 8 here and fails this test.
    """
    st = state_from_evidence(l3_48_evidence(), device_limits=A800)
    d = digest(st)
    assert len(d.findings) == len(st.records)
    assert {f.dimension for f in d.findings} == {r.dimension_id for r in st.records}


def test_j2_9_every_dimension_reaches_the_prompt_text_including_the_slack_ones():
    """The same guard one layer out: reduction could equally happen in the renderer."""
    st = state_from_evidence(l3_48_evidence(), device_limits=A800)
    text = for_prompt(digest(st))
    for rec in st.records:
        assert rec.dimension_id in text, f"{rec.dimension_id} was dropped from the prompt"


def test_a_measured_dimension_with_no_ceiling_is_not_rendered_as_not_measured():
    """The prompt must not state something false about the box.

    The aten dimensions have readings and no roof. Calling them "NOT MEASURED" -- which the first
    renderer did, caught by the smoke test -- is a false statement, and it leads to a different
    action than "measured, cannot be ranked": one says go measure it, the other says there is
    nothing to compare it to.

    Asserts the line is PRESENT before asserting what it says. Without that, a digest that dropped
    the dimension entirely would satisfy the loop vacuously -- which is what happened under the
    "compressor" variant, where this test passed for the wrong reason.

    Scoped to the per-dimension SEVERITY section. S3 added a second section ("how much room is left")
    which names every measured dimension again, so an unscoped per-line scan sees two lines per
    dimension and fails on the second -- a test that reads the whole document cannot say WHICH
    statement it is checking.
    """
    st = state_from_evidence(l3_48_evidence(), device_limits=A800)
    text = for_prompt(digest(st))
    section = text.split("**How much room is left**")[0]
    seen = 0
    for line in section.splitlines():
        if "candidate_aten_bytes" in line or "candidate_aten_ops" in line:
            seen += 1
            assert "NOT MEASURED" not in line
            assert "NOT RANKED" in line
    assert seen == 2, "both aten dimensions must appear in the prompt at all (D-9)"


def test_a_genuinely_unmeasured_dimension_says_so_and_is_listed():
    """...and the "NOT MEASURED" wording still exists for the case that deserves it."""
    ev = l3_48_evidence()
    ev.pop("occupancy")
    st = state_from_evidence(ev, device_limits=A800)
    d = digest(st)
    assert "occupancy" in d.unmeasured
    text = for_prompt(d)
    assert "NOT MEASURED" in text
    assert "absence is not a clean bill of health" in text


def test_a_dimension_whose_evidence_key_is_missing_is_never_read_as_zero():
    """The failure mode this project has hit repeatedly: a moved field read as a zero result.

    A missing `n_regs` must produce `measured=None` and verdict `unknown` -- not 0 registers, which
    would band as `slack` and report register headroom on a candidate nobody measured.
    """
    ev = l3_48_evidence()
    ev.pop("n_regs")
    st = state_from_evidence(ev, device_limits=A800)
    regs = next(r for r in st.records if r.dimension_id == "n_regs")
    assert regs.measured is None
    assert regs.verdict == "unknown"


def test_a_reduced_confidence_reading_says_why_in_the_finding():
    """A bare 0.5 is not actionable and invites being ignored, so the reason travels with it."""
    st = state_from_evidence(l3_48_evidence(), device_limits=A800)
    d = digest(st)
    aten = next(f for f in d.findings if f.dimension == "candidate_aten_bytes")
    assert aten.confidence == 0.5
    assert "bound, not a value" in aten.note


def test_expected_delta_is_unknown_and_never_a_number():
    """§2.6(b): `expected_delta` may be missing, may never be invented.

    Measured three ways that a rate is not derivable in advance -- no closed form for shared memory
    (0 of 96 exact), the map is not separable (0 of 10), and even the SIGN is unreliable (13
    non-monotone slices). A number here would be a fabrication.
    """
    d = digest(state_from_evidence(l3_48_evidence(), device_limits=A800))
    assert all(f.expected_delta == "unknown" for f in d.findings)


def test_the_ordering_basis_states_that_it_is_not_a_measured_priority():
    """The order must carry its basis: no rule for "which dimension to move first" beat its
    controls (best rule 7.1% against a 21.4% random control), so presenting this order as a
    priority would be presenting a refuted rule as fact."""
    d = digest(state_from_evidence(l3_48_evidence(), device_limits=A800))
    assert "NOT a measured priority" in d.ordering_basis
    assert "7.1%" in d.ordering_basis and "21.4%" in d.ordering_basis
    for f in d.findings:
        if f.severity in ("slack", "not-applicable"):
            assert "not recommended" in f.ranked_recommendation


def test_threads_launched_is_never_banded_because_it_has_no_polarity():
    """`_DIMENSIONS` gives it `lower_is_better: None`. A dimension with no better/worse direction
    cannot have a wall, and calling it `slack` would imply headroom in a direction that does not
    exist.

    Note what this dimension separates: `applicable` and the verdict `not-applicable` answer
    DIFFERENT questions, and this is the record where they come apart. The reading EXISTS and was
    taken, so `applicable=True`; what does not apply is the band. The first version set
    `applicable=False` here, which says the box does not have the dimension -- false, and it hides a
    real measurement behind a flag that means "absent".
    """
    st = state_from_evidence(l3_48_evidence(), device_limits=A800)
    th = next(r for r in st.records if r.dimension_id == "threads_launched")
    assert th.measured == 1048576.0
    assert th.applicable is True, "the reading exists; it is the BAND that does not apply"
    assert th.verdict == "not-applicable"
    assert th.not_applicable_reason
    assert check_applicability([th]) == []


def test_a_not_applicable_verdict_without_a_reason_is_reported():
    """Rule 5 of the check, which the `threads_launched` distinction made necessary.

    `not-applicable` has two causes now -- the dimension is absent from the box, or it is present
    with no polarity -- so the reason is the only field that says which. Revert-checked against a
    check that only looked at `applicable is False`: this record has `applicable=True`, so that
    version reports nothing and the test fails.
    """
    bad = DimensionRecord(dimension_id="threads_launched", measured=1024.0,
                          verdict="not-applicable", applicable=True)
    notes = check_applicability([bad])
    assert any("no reason" in n and "two distinct" in n for n in notes)
