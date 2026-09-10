"""S3: per-dimension lower bounds, and where a ceiling came from.

Two acceptance criteria, and the reverse control for each is written into the criterion itself:

  every dimension has a floor anchor OR an explicit `unknown`
        reverse control: "`unknown` must be a REACHABLE state -- an invented lower bound is worse
        than no lower bound". So there are tests here asserting that specific dimensions report
        UNKNOWN, and asserting WHY, because a module that answered every dimension would be
        fabricating most of its answers.

  `ceiling_provenance` is non-empty, every ratio carries its denominator's origin
        positive control: an fp16 candidate scored against a tf32 ceiling read **107.8% of peak**
        when its real figure was 54-60%, which downstream reads as "saturated, stop optimizing".
        The instruction was completely inverted. So the test is not "provenance is non-empty" -- it
        is "provenance can EXPOSE that mismatch", which needs the precision in a field.
"""

from __future__ import annotations

from kernel_optimizer.evaluation.bounds import (
    CeilingProvenance,
    LowerBound,
    describe,
    lower_bound,
)
from kernel_optimizer.evaluation.conversion import _DIMENSIONS
from kernel_optimizer.evaluation.digest import digest, for_prompt
from kernel_optimizer.evaluation.dimensions import (
    compute_ceiling_provenance,
    state_from_evidence,
)
from kernel_optimizer.evaluation.task_cost import TaskCost
from tests.test_s2_dimensions import A800, l3_48_evidence

# L3:48's real figures: 1.351 GB of compulsory traffic across 12 dispatched ops.
L3_48_COST = TaskCost(flop_count=0, compulsory_bytes=1_351_000_000,
                      reference_bytes=1_351_000_000, op_count=12)


# --------------------------------------------------------------------------------------------------
# Criterion 1: a floor, or an explicit unknown
# --------------------------------------------------------------------------------------------------

def test_the_compulsory_traffic_floor_comes_from_task_cost_and_is_measured_on_the_reference():
    """The one real, task-specific floor. Fusion can remove every intermediate and cannot remove the
    distinct inputs, parameters and outputs.

    Measured on the REFERENCE by design -- a per-candidate count would measure what that candidate
    happens to do, which is the quantity being optimized and therefore useless as a yardstick. This
    module READS `task_cost.py` and does not modify it; replacing its task-level basis would destroy
    the L3:48 "only ~10% left" conclusion.
    """
    b = lower_bound("candidate_aten_bytes", 2_000_000_000.0, L3_48_COST)
    assert b.floor == 1_351_000_000.0
    assert b.source == "task_lower_bound"
    assert b.room == 649_000_000.0
    assert b.room_frac is not None and 0.32 < b.room_frac < 0.33
    assert "measured on the\nREFERENCE" in b.reason or "REFERENCE" in b.reason


def test_definitional_floors_are_marked_as_definitional_not_as_measured():
    """A spill floor of zero and an op floor of one are true by definition, not by measurement, and
    the record must say which -- a reader deciding whether to trust a floor needs to know if anyone
    measured it."""
    spills = lower_bound("n_spills", 48.0, None)
    assert spills.floor == 0.0 and spills.source == "definitional"
    assert spills.room == 48.0

    ops = lower_bound("candidate_aten_ops", 12.0, None)
    assert ops.floor == 1.0 and ops.source == "definitional"
    assert ops.room == 11.0
    assert "fully fused" in ops.reason


def test_registers_report_unknown_and_say_the_sign_is_unreliable():
    """The reverse control, and the reason is measured rather than asserted: 13 non-monotone slices,
    BK 16->32 dropping 58 registers while 32->64 added 87 and hit the 255 cap.

    Revert-checked against a plausible invented floor (`0` or "one register per live value"): this
    test fails, and the digest then reports register room a candidate does not have.
    """
    b = lower_bound("n_regs", 218.0, L3_48_COST)
    assert b.floor is None
    assert b.source == "none"
    assert "SIGN is unreliable" in b.reason
    assert "13 non-monotone" in b.reason


def test_shared_memory_reports_unknown_and_cites_the_refuted_formula():
    """A candidate closed-form formula matched 0 of 96 measured configurations, and the three points
    that appeared to confirm it were one family's coincidence."""
    b = lower_bound("shared_bytes", 70000.0, L3_48_COST)
    assert b.floor is None and b.source == "none"
    assert "0 of 96" in b.reason


def test_occupancy_reports_unknown_because_it_is_an_output_not_an_input():
    """Occupancy is derived from the register/shared/warp budgets, so its floor is whatever those
    allow -- and the register floor is itself not derivable. Deriving one would be deriving the
    other."""
    b = lower_bound("occupancy", 0.167, L3_48_COST)
    assert b.floor is None
    assert "OUTPUT" in b.reason


def test_threads_launched_reports_unknown_for_lack_of_polarity_not_lack_of_data():
    """Two different reasons produce `floor=None`, and they are not the same statement: "no floor is
    derivable" and "room left has no direction to be in". The reason field is what tells them apart.
    """
    b = lower_bound("threads_launched", 1048576.0, L3_48_COST)
    assert b.floor is None
    assert "no polarity" in b.reason


def test_a_dimension_with_no_entry_at_all_still_refuses_to_invent_a_floor():
    """The default path must be `unknown`, not zero. A new dimension added to the vocabulary without
    a floor must read as unknown rather than silently claiming it can be eliminated."""
    b = lower_bound("some_future_dimension", 5.0, L3_48_COST)
    assert b.floor is None and b.source == "none"
    assert "inventing it" in b.reason


def test_unknown_is_the_majority_answer_and_that_is_the_point():
    """Stated as a test because it is the acceptance criterion, not an accident of the current data.

    If a future edit made most dimensions report a floor, that would be the fabrication this
    criterion exists to forbid -- so the count is asserted, and it must be checked against the
    vocabulary rather than a hardcoded list.
    """
    with_floor = {d for d in _DIMENSIONS
                  if lower_bound(d, 1.0, L3_48_COST).floor is not None}
    without = set(_DIMENSIONS) - with_floor
    assert with_floor == {"n_spills", "candidate_aten_ops", "candidate_aten_bytes",
                          "peak_alloc_bytes"}
    assert without == {"n_regs", "shared_bytes", "occupancy", "threads_launched"}
    assert len(without) >= len(with_floor) // 2, (
        "most dimensions genuinely have no derivable floor; a module answering all of them would be "
        "fabricating"
    )


def test_a_floor_is_absent_rather_than_zero_when_the_task_cost_was_not_measured():
    """Reading an unmeasured task cost as a floor of 0 would say "you can eliminate all traffic",
    which is the strongest possible claim made from no data."""
    b = lower_bound("candidate_aten_bytes", 2e9, None)
    assert b.floor is None
    assert "not that the floor is zero" in b.reason


def test_the_allocation_floor_is_weak_and_says_so_with_reduced_confidence():
    """A bound on a NEIGHBOURING quantity is not a bound on this one: the caching allocator sits
    between compulsory traffic and peak allocation. A bare confidence of 1.0 here would invite
    reading the room figure as exact."""
    b = lower_bound("peak_alloc_bytes", 3e9, L3_48_COST)
    assert b.floor == 1_351_000_000.0
    assert b.confidence == 0.5
    assert "WEAK floor" in b.reason


# --------------------------------------------------------------------------------------------------
# A reading below its own floor is reported, never clamped
# --------------------------------------------------------------------------------------------------

def test_a_reading_below_its_floor_is_flagged_and_not_clamped():
    """Not hypothetical: `candidate_aten_bytes` is ITSELF a lower bound on the candidate's real
    traffic, because a fused kernel does its work inside one aten call and the intermediate bytes are
    never dispatched. So a well-fused candidate reads BELOW compulsory_bytes.

    Clamping to zero room would say "you are at the floor", a false statement about a candidate that
    simply is not measurable this way. Same discipline as `check_reading`'s out-of-range rule: a clamp
    preserves the wrong action.

    Revert-checked against `room = max(0.0, measured - floor)`: `below_floor` reads False, the room
    figure reads 0, and the prompt then tells the agent this dimension is exhausted.
    """
    b = lower_bound("candidate_aten_bytes", 900_000_000.0, L3_48_COST)
    assert b.below_floor is True
    assert b.room is not None and b.room < 0
    text = describe("candidate_aten_bytes", b)
    assert "BELOW its own floor" in text
    assert "NOT that the candidate is at the floor" in text


def test_a_reading_at_its_floor_is_not_flagged_as_below_it():
    """The control: exactly at the floor is not below it."""
    b = lower_bound("candidate_aten_ops", 1.0, None)
    assert b.below_floor is False
    assert b.room == 0.0


# --------------------------------------------------------------------------------------------------
# Criterion 2: provenance, and the 107.8% control
# --------------------------------------------------------------------------------------------------

def test_a_precision_mismatch_is_detected_and_names_the_inversion():
    """The positive control, which is an incident rather than a hypothesis: an fp16 kernel scored
    against a tf32 ceiling read 107.8% of peak while its real figure was about 54-60%.

    Revert-checked against provenance as a free-form STRING (which is what S2 shipped): a sentence
    saying "tensor-core ceiling" cannot be asked "of WHICH precision", so this test cannot even be
    written against it. That is why the fields are separate.
    """
    prov = CeilingProvenance(source="measured", precision="tf32", backend="cublas")
    msg = prov.precision_mismatch("fp16")
    assert msg is not None
    assert "PRECISION MISMATCH" in msg
    assert "107.8" in msg
    assert "fp16" in msg and "tf32" in msg


def test_a_matching_precision_produces_no_warning():
    """The control that keeps the check from being "always warn"."""
    prov = CeilingProvenance(source="measured", precision="fp16")
    assert prov.precision_mismatch("fp16") is None


def test_an_unknown_precision_on_either_side_produces_no_warning():
    """An absent precision is not evidence of a mismatch, and claiming one would be the same error in
    the other direction -- a false alarm that trains the reader to ignore the real one."""
    assert CeilingProvenance(source="measured", precision="").precision_mismatch("fp16") is None
    assert CeilingProvenance(source="measured", precision="tf32").precision_mismatch(None) is None
    assert CeilingProvenance(source="measured", precision="tf32").precision_mismatch("") is None


def test_the_compute_provenance_extracts_the_precision_from_the_classifiers_own_label():
    """`compute_ceiling_used` is a LABEL like "tensor-core (fp16)". The precision is read out of it
    rather than assumed -- assuming it would be the same class of guess as guessing an events payload
    path, which has misfired twice in this project.
    """
    prov = compute_ceiling_provenance({"compute_ceiling_used": "tensor-core (fp16)"})
    assert prov.precision == "fp16"
    assert prov.source == "measured"

    prov32 = compute_ceiling_provenance({"compute_ceiling_used": "fp32"})
    assert prov32.precision == "fp32"


def test_the_compute_provenance_reports_no_ceiling_rather_than_guessing_one():
    prov = compute_ceiling_provenance({})
    assert prov.source == "none"
    assert prov.precision == ""
    assert "no compute ceiling was in force" in prov.note


def test_the_backend_field_names_the_pair_the_ceiling_is_a_max_over():
    """G10: the ceiling is the max over measured paths, so naming ONE backend would be wrong. On box 1
    Triton reaches 84.1% of cuBLAS at fp32 but 109.6% at fp16 -- neither alone is the roof."""
    prov = compute_ceiling_provenance(
        {"compute_ceiling_used": "fp32", "backend_reachable_frac": 0.841})
    assert "max(cuBLAS, Triton)" in prov.backend
    assert "84.1%" in prov.backend


def test_a_calibration_identity_and_date_travel_with_a_measured_ceiling():
    """A ceiling measured under a different Triton generates different code, so a figure from another
    identity does not describe this box (G32). The identity is what lets a later reader tell."""

    class Cal:
        measured_at = "2026-09-10T04:00:00Z"

        def identity(self):
            return "NVIDIA A800 80GB PCIe|8.0|108|2.8.0|550.54|3.4.0"

    prov = compute_ceiling_provenance({"compute_ceiling_used": "tensor-core (bf16)"}, Cal())
    assert "3.4.0" in prov.calibration_identity
    assert prov.measured_at == "2026-09-10T04:00:00Z"


def test_a_device_query_ceiling_is_not_stamped_with_a_calibration_date():
    """A hardware limit is not invalidated by a recalibration, and stamping it with a calibration date
    would imply it was measured then. Registers-per-thread is a property of the card.

    Revert-check: stamping every provenance with the calibration date makes this fail -- and would
    make a reader think a recalibration could change the register cap.
    """

    class Cal:
        measured_at = "2026-09-10T04:00:00Z"

        def identity(self):
            return "ident"

    st = state_from_evidence(l3_48_evidence(), device_limits=A800, calibration=Cal(),
                             task_cost=L3_48_COST)
    regs = next(r for r in st.records if r.dimension_id == "n_regs")
    assert regs.provenance is not None
    assert regs.provenance.source == "device_query"
    assert regs.provenance.measured_at == ""
    assert regs.provenance.calibration_identity == ""


def test_every_record_carries_a_provenance_and_a_bound():
    """The criterion is per-RATIO, so a record without provenance is a ratio whose denominator has no
    stated origin -- which is the pre-S3 state for all of them."""
    st = state_from_evidence(l3_48_evidence(), device_limits=A800, task_cost=L3_48_COST)
    assert len(st.records) == 8
    for r in st.records:
        assert r.provenance is not None, f"{r.dimension_id} has no provenance"
        assert r.bound is not None, f"{r.dimension_id} has no bound record"
        # Every provenance must say something; an empty note with source "none" is the one case where
        # there is genuinely nothing to say beyond the source itself.
        assert r.provenance.source in ("measured", "device_query", "definitional",
                                       "task_lower_bound", "none")


def test_the_free_text_provenance_is_kept_alongside_the_structured_one():
    """The prose is what reaches a human reader, and dropping it to add fields would be a reduction.
    Both must be present."""
    st = state_from_evidence(l3_48_evidence(), device_limits=A800, task_cost=L3_48_COST)
    for r in st.records:
        assert r.ceiling_provenance, f"{r.dimension_id} lost its prose provenance"


# --------------------------------------------------------------------------------------------------
# The consumer: S3 must reach the prompt, or it is not implemented
# --------------------------------------------------------------------------------------------------

def test_the_room_left_figures_reach_the_prompt_with_their_unknowns_intact():
    """A bound nothing renders is not implemented -- the same argument that made `conversion`'s zero
    consumers a gap.

    Asserts both halves: the real floor appears WITH its number, and the undecidable ones appear as
    UNKNOWN. A renderer that dropped the unknowns would leave the reader to assume those dimensions
    were fine.
    """
    st = state_from_evidence(l3_48_evidence(), device_limits=A800, task_cost=L3_48_COST)
    text = for_prompt(digest(st))
    assert "How much room is left" in text
    assert "an invented floor would be worse than none" in text
    # The measured floor, with its source.
    assert "task_lower_bound" in text
    # And the undecidable ones, out loud.
    for dim in ("n_regs", "shared_bytes", "occupancy"):
        assert any(dim in ln and "UNKNOWN" in ln for ln in text.splitlines()), (
            f"{dim}'s unknown floor did not reach the prompt")


def test_the_denominator_notes_reach_the_prompt_as_their_own_section():
    """A wrong denominator is the one defect that makes a kernel with headroom read as saturated, so
    where the denominator came from belongs beside the percentages it qualifies."""
    st = state_from_evidence(l3_48_evidence(), device_limits=A800, task_cost=L3_48_COST)
    prov = CeilingProvenance(source="measured", precision="tf32")
    msg = prov.precision_mismatch("fp16")
    assert msg
    text = for_prompt(digest(st, denominator_notes=(msg,)))
    assert "Where the compute denominator came from" in text
    assert "PRECISION MISMATCH" in text


def test_room_left_is_absent_for_an_unmeasured_dimension_rather_than_unknown():
    """"Room left unknown" and "this dimension was never measured" are different statements, and the
    second is already said by the unmeasured section. Saying both would be two sentences about one
    absence, and the weaker one would dilute the stronger."""
    ev = l3_48_evidence()
    ev.pop("n_regs")
    st = state_from_evidence(ev, device_limits=A800, task_cost=L3_48_COST)
    d = digest(st)
    regs = next(f for f in d.findings if f.dimension == "n_regs")
    assert regs.room_left == ""
    assert "n_regs" in d.unmeasured


def test_the_bound_is_journalled_inside_the_record():
    """S3's output has to survive to the event log: a floor that exists only in a prompt cannot be
    re-checked offline, and every J2/J3 criterion is verified by replay."""
    st = state_from_evidence(l3_48_evidence(), device_limits=A800, task_cost=L3_48_COST)
    dumped = [r.model_dump() for r in st.records]
    aten = next(d for d in dumped if d["dimension_id"] == "candidate_aten_bytes")
    assert aten["bound"]["floor"] == 1_351_000_000.0
    assert aten["bound"]["source"] == "task_lower_bound"
    assert aten["provenance"]["source"] == "none"


def test_a_lower_bound_model_refuses_an_invented_field():
    """Same reason as `ResourceExpectation`: pydantic's default silently drops an unknown field, so a
    `derived_floor` someone adds in passing would validate and vanish."""
    import pytest

    with pytest.raises(Exception):
        LowerBound(floor=1.0, source="definitional", derived_by="a guess")
    with pytest.raises(Exception):
        CeilingProvenance(source="measured", guessed_precision="fp16")


def test_a_no_tensor_core_kernel_against_a_tensor_core_roof_is_reported_as_inconsistent():
    """Caught by a smoke test of my own output: with `uses_tensor_cores=False` and a tensor-core
    label, the caveat CONTRADICTED ITSELF -- it said the tensor-core roof is not the denominator and
    then named it as the denominator.

    `classify()` only names a tensor-core roof when `uses_tc` is truthy (`if peaks and uses_tc:`), so
    the pair should be unreachable. But a self-contradicting sentence in a prompt is worse than either
    half of it: a reader cannot act on it at all, and cannot tell which half to distrust. So the
    inconsistency is reported as one.

    Revert-checked against the single-sentence version: this test fails, and the prompt then carries
    the contradiction.
    """
    from kernel_optimizer.evaluation.dimensions import unreachable_ceilings

    ev = dict(l3_48_evidence(), uses_tensor_cores=False,
              compute_ceiling_used="tensor-core (tf32)")
    notes = unreachable_ceilings(ev)
    hit = [n for n in notes if "INCONSISTENT" in n]
    assert hit, "a no-tensor-core kernel against a tensor-core roof must be flagged"
    assert "unknown denominator" in hit[0]
    # And it must NOT also emit the ordinary caveat, which would restore the contradiction.
    assert not any("is NOT the denominator" in n for n in notes)


def test_a_no_tensor_core_kernel_against_an_fp32_roof_gets_the_ordinary_caveat():
    """The control: the consistent pair still produces the plain statement, so the branch above is not
    "always report an inconsistency"."""
    from kernel_optimizer.evaluation.dimensions import unreachable_ceilings

    ev = dict(l3_48_evidence(), uses_tensor_cores=False, compute_ceiling_used="fp32")
    notes = unreachable_ceilings(ev)
    assert any("is NOT the denominator" in n for n in notes)
    assert not any("INCONSISTENT" in n for n in notes)
