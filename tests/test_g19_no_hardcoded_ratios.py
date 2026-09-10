"""G19: no hardware ratio may be hardcoded in the text an agent reads.

THE PRINCIPLE, and where it leaked. Every hardware NUMBER in this harness is measured on the box --
that part was already right. But the agent-facing prose additionally stated RATIOS between those
numbers as constants ("tensor cores are roughly 2x faster", "tf32 and fp32 differ by 1.6x"), and
nobody updated them when a new card arrived. Measured:

    ratio          4090     A800     prose said
    fp16 / tf32    1.80     2.05     "~2x"      -- fine on both, by luck
    tf32 / fp32    1.61     5.88     "1.6x"     -- wrong by 3.7x on the A800
    fp16 / fp32    2.88    12.05     "~2x"      -- wrong by 6x on the A800

The A800 is the card where leaving the IEEE path matters MOST (its fp32 is 19.0 TFLOP/s, 2.9x slower
than the 4090's) and it was the card being told that move is worth only 1.6x. Two sources for the
same quantity, one measured and one stale, with no way for the agent to know which to trust.

So the rule: prose describes the MECHANISM, the ceilings block supplies the NUMBERS. This test is the
guard, because the hardcoded ratios got there in the first place by someone writing a helpful
concrete example.
"""
from __future__ import annotations

import re
from importlib import resources
from pathlib import Path

# A throughput/speed claim with a number in it: "2x faster", "~2x", "1.6x in throughput", "3x the".
# Deliberately narrow: it must be a bare multiplier next to speed language, so legitimate uses (a
# COMPUTED "{ratio:.2f}x" in an f-string, a byte count, a version number) do not trip it.
_RATIO_CLAIM = re.compile(
    r"(?<![\w.{:])~?\d+(?:\.\d+)?\s*x\b(?![}\w])",
    re.IGNORECASE,
)

# Phrases that make a bare multiplier a claim about hardware speed rather than something else.
_SPEED_CONTEXT = re.compile(
    r"faster|slower|throughput|speedup|tflop|bandwidth|tensor core|peak", re.IGNORECASE)

# Numbers that are POLICY, not hardware priors, and are correctly hardcoded. The anti-cheat gate
# rejects a candidate measured >10x faster than the reference; that threshold is a rule this project
# chose, identical on every card, and is not a claim about what any GPU can do. Excluding it by name
# keeps the check honest rather than loosening the pattern until nothing trips it.
_POLICY_NUMBERS = re.compile(
    r"MORE THAN 10x FASTER|suspicious_speedup|excessive_speedup|10x harder|2x suspicious",
    re.IGNORECASE)

_PROMPT_FILES = ("candidate_contract.md", "triton_pitfalls.md")


def _prompt_text(name: str) -> str:
    return (resources.files("kernel_optimizer.agents.prompts")
            .joinpath(name).read_text(encoding="utf-8"))


def test_no_hardcoded_speed_ratio_in_the_agent_prompts():
    """A multiplier stated as fact in prompt prose is a hardcoded hardware prior."""
    offences: list[str] = []
    for name in _PROMPT_FILES:
        for i, line in enumerate(_prompt_text(name).splitlines(), 1):
            if not _SPEED_CONTEXT.search(line) or _POLICY_NUMBERS.search(line):
                continue
            for m in _RATIO_CLAIM.finditer(line):
                # An explicitly card-relative or negated mention is the CURE, not the disease:
                # "it is NOT a fixed 2x", "ranges from under 2x to nearly 6x depending on the card".
                if re.search(r"not\s+a\s+fixed|ranges from|depending on the card|do not assume",
                             line, re.IGNORECASE):
                    continue
                offences.append("%s:%d  %r  (matched %r)" % (name, i, line.strip()[:100], m.group()))
    assert not offences, (
        "hardcoded hardware speed ratios found in agent prompt prose. Every number an agent reads "
        "must come from this box's own calibration, because these ratios are card-specific: "
        "tf32/fp32 measures 1.61 on a 4090 and 5.88 on an A800, so a written-in '1.6x' understates "
        "the tensor-core lever by 3.7x on the card that needs it most. Describe the mechanism and "
        "let the ceilings block supply the figure.\n  " + "\n  ".join(offences))


def test_the_ceilings_block_computes_its_ratios_from_measurements():
    """The replacement must actually carry the numbers, or removing them just loses information."""
    from kernel_optimizer.agents.modules import _measured_ceilings_doc
    from kernel_optimizer.evaluation.calibration import Calibration

    # A800 figures: the case where every hardcoded ratio was wrong.
    a800 = Calibration(device_name="A800", capability=[8, 0], sm_count=108,
                       dram_tbs=1.686, fp32_tflops=19.0, tf32_tflops=111.8,
                       fp16_tflops=229.0, bf16_tflops=234.0)
    doc = _measured_ceilings_doc(a800)
    assert "5.88x" in doc, (
        "the tf32/fp32 ratio is not computed into the ceilings block, so removing the hardcoded "
        "'1.6x' from the prose loses the lever entirely instead of correcting it")
    assert "2.05x" in doc, "the fp16/tf32 ratio is not computed into the ceilings block"

    # And the same code on a 4090 must produce the DIFFERENT, correct ratio -- proof it is computed
    # rather than a constant that merely happens to match one card.
    b1 = Calibration(device_name="4090", capability=[8, 9], sm_count=128,
                     dram_tbs=0.911, fp32_tflops=54.8, tf32_tflops=88.0,
                     fp16_tflops=158.0, bf16_tflops=164.0)
    doc_b1 = _measured_ceilings_doc(b1)
    assert "1.61x" in doc_b1, "the 4090 tf32/fp32 ratio is wrong or not computed"
    assert "5.88x" not in doc_b1, (
        "the A800 ratio appeared on a 4090 calibration, so the figure is not derived from the "
        "measurements it is shown beside")


def test_no_hardcoded_speed_ratio_in_the_prompt_building_code():
    """The same rule applies to prose assembled in Python, which is equally agent-facing."""
    src = Path("src/kernel_optimizer/agents/modules.py")
    if not src.exists():          # running from an installed package rather than the repo
        return
    offences: list[str] = []
    for i, line in enumerate(src.read_text(encoding="utf-8").splitlines(), 1):
        if not _SPEED_CONTEXT.search(line) or _POLICY_NUMBERS.search(line):
            continue
        if "{" in line and "}" in line:      # an f-string interpolating a computed ratio
            continue
        for m in _RATIO_CLAIM.finditer(line):
            if re.search(r"not\s+a\s+fixed|ranges from|depending on the card|do not assume|"
                         r"is 1\.80 on a 4090|1\.61 vs 5\.88|2\.9x slower",
                         line, re.IGNORECASE):
                continue          # documenting WHY hardcoding is wrong
            offences.append("modules.py:%d  %r  (matched %r)"
                            % (i, line.strip()[:100], m.group()))
    assert not offences, (
        "hardcoded hardware speed ratios found in prompt-building code:\n  "
        + "\n  ".join(offences))
