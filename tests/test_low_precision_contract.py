"""The precision knob must vary the ARITHMETIC only -- asserted on the prompt text agents read.

G37 established where the real lever is. The acceptance gate was audited over all 537
`correctness_mismatch` rejections and attributed ZERO of them to itself: the rejected candidates
missed the fp64 reference by 3.04x at the very best, 6.48x median, up to 3766x, so none was a
near-miss the gate could have been kinder about. The defect is on the CANDIDATE GENERATION side,
and it has two measured causes plus one that is commonly confused with them:

  (a) an uncompensated `dot` -- not enough mantissa in a single low-precision MMA. The three-MMA
      split passed 12/12 where the single MMA failed 11/11.
  (b) `PREC` gating the ALGORITHM rather than the dot, so a non-default precision runs code that
      was never stabilized. Measured across four candidates on one task: PREC confined to the dot
      helper passed 32/32, 17/17, 12/12; PREC also gating the algorithm failed bf16 0/13 and
      tf32 0/9 while passing fp16 43/43 (the branch that had been debugged).
  (c) fp16 RANGE overflow above 65504 -- an exponent problem, so neither of the above fixes it,
      and bf16 does not have it.

These tests assert the guidance reaches the four places an agent could bake the defect in
(contract, generator, parameterizer, repair). They are TEXT assertions, which this project has a
recorded failure mode for -- a source-text assertion can pass on broken code and fail after the
fix. That is accepted here for a specific reason: the artifact under test IS text. The prompt is
the deliverable, there is no behaviour to drive, and the failure being guarded against is prose
that says the wrong thing. What the tests must therefore avoid is being satisfied by a keyword
appearing anywhere, so each one checks a claim in its own context rather than a bare substring.
"""

from __future__ import annotations

import re
from importlib import resources

import pytest


def _contract() -> str:
    return (resources.files("kernel_optimizer.agents.prompts")
            .joinpath("candidate_contract.md").read_text(encoding="utf-8"))


def _modules_source() -> str:
    from pathlib import Path

    import kernel_optimizer.agents.modules as m

    return Path(m.__file__).read_text(encoding="utf-8")


# --- (b) the algorithm-gating failure, which is the expensive one ---------------------------


def test_the_contract_forbids_the_precision_knob_from_selecting_a_code_path():
    """The rule has to be stated as a PROHIBITION, not as a stylistic preference.

    Checked as a rule plus its evidence, because 'write the algorithm once' on its own reads as
    tidiness advice. The measured 0/13 and 0/9 are what make it a correctness rule.
    """
    text = _contract()
    assert re.search(r"ARITHMETIC only, never the algorithm", text), (
        "the contract does not state the rule that the precision knob may not change the "
        "algorithm -- the measured cause of bf16 0/13 and tf32 0/9")
    assert "0/13" in text and "0/9" in text, (
        "the rule is stated without the measurement behind it, so an agent weighing it against "
        "its own judgement has nothing to weigh")
    assert re.search(r"unconditionally at every precision", text, re.IGNORECASE), (
        "the contract says what not to do but not the REMEDY: a stabilization a precision needs "
        "must be applied at every precision, not branched on")


def test_the_contract_gives_the_diagnostic_that_separates_the_causes():
    """`if PREC ==` is the check that tells cause (b) from cause (a) by reading the source, and it
    is the reason the four-candidate table could be built at all. Without it an agent facing a
    low-precision failure has no way to tell which fix applies."""
    assert "if PREC ==" in _contract(), (
        "the contract omits the countable diagnostic (`if PREC ==` sites) that distinguishes an "
        "algorithm-gating knob from a genuine mantissa shortfall")


# --- (a) the compensated dot, offered as a knob rather than hard-wired ----------------------


def test_the_contract_offers_the_compensated_dot_as_a_tunable_knob():
    """Not as a recommendation. `split3` costs 3x the MMAs, so on a short reduction it is pure
    loss -- which makes it exactly the kind of question the tuner answers on measurements. A
    contract that told agents to always compensate would be trading one hard-coded choice for
    another."""
    text = _contract()
    assert "DOT_MODE" in text and "split3" in text, "the compensated dot is not offered as a knob"
    assert re.search(r'"plain".*"split3"|\["plain", "split3"\]', text), (
        "DOT_MODE is mentioned without its choices, so an agent cannot tell it is a two-value knob")
    assert "12/12" in text and "11/11" in text, (
        "the split form is proposed without the measurement that motivates it")
    # The knob must be described as arithmetic, or it becomes the very branch rule (b) forbids.
    assert re.search(r"arithmetic only, NOT a code path|same algorithm with the same shapes",
                     text, re.IGNORECASE), (
        "DOT_MODE is offered without saying both forms must compute the SAME algorithm -- as a "
        "code path it would reintroduce exactly the defect the section above forbids")


def test_the_split_form_in_the_contract_is_arithmetically_correct():
    """The snippet is copied by agents, so an error in it propagates into candidates.

    Asserts the three cross terms are the right three: (ah,bh) is the leading term and (ah,bl) +
    (al,bh) recover the next mantissa bits. The fourth term (al,bl) is legitimately dropped -- it
    is below the representable range of the result -- but dropping (ah,bl) or (al,bh) instead
    would silently halve the benefit while looking equally plausible.
    """
    text = _contract()
    for term in ("tl.dot(ah, bh)", "tl.dot(ah, bl)", "tl.dot(al, bh)"):
        assert term in text, "the compensated-dot snippet is missing %s" % term
    assert "(a - ah.to(tl.float32))" in text, (
        "the low part must be the residual a - ah computed in fp32; anything else is not a split")


# --- (c) the range failure, which is a different problem wearing the same clothes -----------


def test_the_contract_separates_fp16_range_from_mantissa():
    """The three causes all surface as `correctness_mismatch`, and this is the one that neither
    knob fixes. Conflating it produces an agent that adds `split3` to a kernel overflowing to Inf
    and cannot understand why nothing improved."""
    text = _contract()
    assert "65504" in text, "the fp16 range limit is not stated"
    assert re.search(r"RANGE, not mantissa|range.*not.*mantissa", text, re.IGNORECASE), (
        "the contract does not distinguish the range failure from the mantissa failure")
    assert re.search(r"bf16 keeps fp32's exponent range", text), (
        "bf16's exponent range -- the direct remedy for (c) -- is not stated next to it")


# --- the repair path, where a low-precision failure actually lands --------------------------


def test_the_repair_prompt_does_not_prescribe_abandoning_the_tensor_cores():
    """This is a FIX to the repair prompt, not only an addition.

    It previously told the agent that a small numerical error means using
    `input_precision="ieee"` -- which leaves the tensor cores idle. That is the advice whose
    outcome is on record: 8 of 8 tensor-core candidates rejected, 7 of 7 scalar accepted, and a
    best kernel that used no tensor cores at all.
    """
    src = _modules_source()
    numeric = src[src.index("    numeric = ("):src.index("    compile_ = (")]
    assert "input_precision" in numeric, "the numeric repair guidance no longer mentions precision"
    assert re.search(r"Do NOT reach for input_precision=\\?.ieee\\?. as the fix", numeric), (
        "the repair prompt still offers ieee as the remedy for a small numerical error, without "
        "the warning that it abandons the tensor cores")
    assert "8 of 8" in numeric, (
        "the warning is stated without its measured consequence, which is what makes it a rule "
        "rather than a preference")


def test_the_repair_prompt_names_all_three_causes_with_distinct_fixes():
    """A repair agent that cannot tell the three apart guesses, and two of the three guesses make
    the candidate worse. Asserted inside the numeric branch specifically -- the strings appearing
    elsewhere in the file would not help the agent handling a `correctness_mismatch`."""
    src = _modules_source()
    numeric = src[src.index("    numeric = ("):src.index("    compile_ = (")]
    assert "MANTISSA" in numeric, "cause (a) is not named in the numeric repair guidance"
    assert "ALGORITHM BRANCH" in numeric, "cause (b) is not named"
    assert "RANGE OVERFLOW" in numeric, "cause (c) is not named"
    assert "if PREC ==" in numeric, "the diagnostic that separates (a) from (b) is missing"
    assert "65504" in numeric, "the fp16 range limit is missing from the repair path"
    # The pre-existing semantic-mismatch guidance must survive: it is the FIRST thing to check and
    # has its own measured root cause (train-mode BatchNorm on L3:21).
    assert "BatchNorm" in numeric and "TRAIN mode" in numeric, (
        "the semantic-mismatch guidance was displaced by the precision guidance; a large constant "
        "offset is a different failure and must still be checked first")


# --- the two producers that could bake the defect in -----------------------------------------


@pytest.mark.parametrize("marker,who", [
    ("COMPUTE_DTYPE", "the unified precision knob"),
    ("DOT_MODE", "the compensated-dot knob"),
    ("ARITHMETIC ONLY", "the prohibition on gating the algorithm"),
])
def test_the_parameterizer_prompt_carries_the_rule(marker, who):
    """The parameterizer is where an existing kernel's `if PREC ==` becomes a knob, so it is the
    one module that can convert the defect into a tuned dimension. It must be told to FIX that
    structure rather than preserve it."""
    src = _modules_source()
    start = src.index("Your job: parameterize every tunable feature")
    body = src[start:start + 6000]
    assert marker in body, "the parameterizer prompt is missing %s" % who


def test_the_parameterizer_is_told_to_fix_a_gating_knob_not_preserve_it():
    """Parameterizing faithfully is the parameterizer's whole contract elsewhere ('rewrite the
    file so all knobs flow through PARAMS'), so leaving an algorithm-gating PREC in place is the
    DEFAULT reading of its job. The instruction to hoist it has to be explicit."""
    src = _modules_source()
    start = src.index("Your job: parameterize every tunable feature")
    body = src[start:start + 6000]
    assert re.search(r"a defect to FIX while\s+#?\s*parameterizing, not a structure to preserve",
                     body), (
        "the parameterizer is not told that an algorithm-gating precision variable must be fixed "
        "rather than faithfully parameterized")


def test_the_generator_prompt_no_longer_names_the_superseded_knob():
    """The generator used to ask for `"DOT_PRECISION": "tf32"` -- a single-value example that also
    only contrasted tf32 against ieee, leaving fp16/bf16 unmentioned. Two names for one concept
    across two prompts is how a candidate ends up with a dtype cast in the body and a mismatched
    knob in PARAMS."""
    src = _modules_source()
    start = src.index("PRECISION / TENSOR CORES:")
    body = src[start:start + 3000]
    assert "DOT_PRECISION" not in body, (
        "the generator prompt still names DOT_PRECISION while the contract and parameterizer use "
        "COMPUTE_DTYPE")
    assert "COMPUTE_DTYPE" in body, "the generator prompt does not name the unified knob"
    assert '["fp16", "bf16", "tf32", "ieee"]' in body, (
        "the generator is not shown the full choice list, so bf16 -- which passed where fp16 "
        "overflowed on one task -- is not offered")


def test_no_task_specific_special_casing_entered_the_prompts():
    """The fix must be general. A rule naming a task, a shape, or a specific candidate would be
    the case-specific change this project's standard forbids -- and it is the error S1b was
    withdrawn for (a correlation that held only in the corpus at hand).
    """
    text = _contract()
    start = text.index("### The precision knob must vary the ARITHMETIC only")
    section = text[start:start + 4200]
    for banned in ("level3", "L3:48", "L3:43", "L3:21", "Mamba", "MinGPT", "EfficientNet",
                   "cand-", "attention task"):
        assert banned not in section, (
            "the precision section names %r, which makes it guidance about one task rather than a "
            "general rule" % banned)

def test_the_contract_forbids_split3_falling_through_to_ieee_for_tf32():
    """FIRST PRODUCTION EVIDENCE drove this: 4 of 4 candidates on the first real run after the
    contract change implemented DOT_MODE correctly -- three-MMA split with the right cross terms,
    PREC confined to the dot helper, zero algorithm branching -- and ALL FOUR let
    ("split3", "tf32") fall through to `ieee`.

    Four out of four is not chance; it is the prompt being silent, so each agent filled the gap the
    same defensible way. The choice is arguable (split3 on tf32 gives ~21 effective mantissa bits
    against ieee's 23, at 3 MMAs against 1) but it silently DELETES a combination from the search
    space, and the tuner then reports on a pair it never measured. Whether splitting tf32 pays is a
    MEASUREMENT, not something to settle in the source -- the same argument that made DOT_MODE a
    knob rather than a recommendation.
    """
    text = _contract()
    assert "must be meaningful for every `COMPUTE_DTYPE` that reaches the tensor cores" in text, (
        "the contract does not require DOT_MODE to be expressible at tf32, so a candidate may "
        "silently drop the (split3, tf32) pair")
    assert '"tf32")` fall through to `ieee`' in text, (
        "the specific fall-through seen in 4/4 candidates is not named, so the guidance is not "
        "actionable against the thing that actually happened")
    assert "legitimately a no-op" in text, (
        "the contract now demands split3 everywhere, including ieee, where the inputs are already "
        "full fp32 -- that would be three MMAs for no mantissa gain")
    assert "say so in `approach_summary` rather than removing the option" in text, (
        "an agent that believes splitting tf32 cannot pay is given no way to record that belief "
        "except by deleting the option, which is what happened")

