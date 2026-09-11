"""A2: a rescued rewrite must keep its S2d expectations, and must not admit junk.

THE LOSS, measured. On box 3's `run-l3-48-20260911-052647` both rewrite candidates of the run's
single round arrived through `rescue_from_sandbox` after a transport failure. The rescue rebuilt
the FILES -- which is what it was written for, and it worked -- but `expectations` lived only in
the JSON response the dead connection took with it. So:

    change_summary = "[recovered from sandbox after a transport failure; ...]"
    hypothesis_id  = ""
    expectations   = []          <- both candidates
    ledger         = {}          <- the family's entire S2d record for that round

That round's S2d evidence was zero, and nothing said so: an empty ledger is exactly what an
agent that declared nothing produces. `rescue_from_sandbox`'s own docstring called
hypothesis_id/change_summary "narration the pipeline does not gate on" -- true before S2d, false
after, because S2d(a)/(b)/(c) are entirely built on the declarations.

THE FIX asks the rewriter to write `rewrites/expectations.json` beside its kernels, and reads it
back on rescue. The risk that comes with it, stated plainly: a sidecar file is agent-written
content entering the ledger on the path where the model's own validated answer is absent. So the
tests below are as much about what must be REFUSED as about what must be recovered -- a rescue
must never become a way to admit work the normal path would have rejected (the constraint
`rescue_from_sandbox`'s contract states, and which `_rescue` enforces by running `check_output`
over whatever this returns).
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from kernel_optimizer.agents.modules import StructureRewriterAgent
from kernel_optimizer.agents.sandbox import Sandbox

# A minimal Triton rewrite: `_detect_backend` must read "triton" from it, and the lint check that
# `check_output` runs must find a kernel.
_SRC = """
import torch, triton, triton.language as tl
PARAMS = {"BLOCK": 64}

@triton.jit
def k(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = i < n
    tl.store(y_ptr + i, tl.load(x_ptr + i, mask=m) * 2.0, mask=m)

class ModelNew(torch.nn.Module):
    def forward(self, x):
        y = torch.empty_like(x)
        n = x.numel()
        k[(triton.cdiv(n, PARAMS["BLOCK"]),)](x, y, n, BLOCK=PARAMS["BLOCK"])
        return y
"""


def _sandbox(files: dict[str, str]) -> tuple[Sandbox, tempfile.TemporaryDirectory]:
    """A real Sandbox over a temp dir, with `rewrites/` populated as an agent would leave it."""
    td = tempfile.TemporaryDirectory()
    root = Path(td.name)
    (root / "rewrites").mkdir(parents=True, exist_ok=True)
    for rel, content in files.items():
        (root / rel).write_text(content, encoding="utf-8")
    return Sandbox(root), td


def _agent() -> StructureRewriterAgent:
    """The real agent, built with `__new__`: `rescue_from_sandbox` needs no runtime or store."""
    return StructureRewriterAgent.__new__(StructureRewriterAgent)


def test_a_rescued_rewrite_recovers_its_expectations():
    """The defect. Box 3's round recorded zero declarations for two candidates that had them."""
    sb, td = _sandbox({
        "rewrites/rw_1.py": _SRC,
        "rewrites/expectations.json": json.dumps({
            "rw_1.py": [
                {"dimension": "shared_bytes", "expect": "up",
                 "why": "the tile grows from 64x64 to 128x64"},
                {"dimension": "candidate_aten_bytes", "expect": "down",
                 "why": "the fp32 round trip is gone"},
            ]}),
    })
    with td:
        out = _agent().rescue_from_sandbox(sb)
        assert out is not None and len(out.candidates) == 1
        exps = out.candidates[0].expectations
        assert len(exps) == 2, [e.dimension for e in exps]
        assert {e.dimension for e in exps} == {"shared_bytes", "candidate_aten_bytes"}
        assert [e.expect for e in exps] == ["up", "down"]
        assert "128x64" in exps[0].why, "the reasoning must survive too, not just the direction"


def test_the_files_are_still_recovered_when_there_is_no_sidecar():
    """The pre-existing behaviour must be untouched: a rescue that recovered nothing because the
    agent wrote no sidecar is still a successful rescue. Otherwise this fix would trade one lost
    round for another."""
    sb, td = _sandbox({"rewrites/rw_1.py": _SRC})
    with td:
        out = _agent().rescue_from_sandbox(sb)
        assert out is not None and len(out.candidates) == 1
        assert out.candidates[0].expectations == []
        assert "recovered from sandbox" in out.candidates[0].change_summary


def test_no_rewrites_at_all_still_returns_None():
    sb, td = _sandbox({"rewrites/expectations.json": json.dumps({"rw_1.py": []})})
    with td:
        assert _agent().rescue_from_sandbox(sb) is None, (
            "a sidecar without any kernel must not manufacture a candidate")


def test_the_sidecar_is_matched_by_basename_not_by_path():
    """`list_outputs` returns "rewrites/rw_1.py" while the agent is asked for "rw_1.py".

    Matching on the full path would find nothing and silently recover zero expectations -- which
    is indistinguishable from an agent that declared none, i.e. the exact ambiguity this fix
    exists to remove. Both spellings are accepted on the way in for the same reason.
    """
    sb, td = _sandbox({
        "rewrites/rw_1.py": _SRC,
        "rewrites/rw_2.py": _SRC.replace("* 2.0", "* 3.0"),
        "rewrites/expectations.json": json.dumps({
            "rw_1.py": [{"dimension": "n_regs", "expect": "up", "why": "more live values"}],
            # The agent wrote a path here instead of a name. Accepted rather than dropped: the
            # information is unambiguous and losing a round over a leading directory would be
            # absurd.
            "rewrites/rw_2.py": [{"dimension": "occupancy", "expect": "down",
                                  "why": "bigger tiles, fewer resident blocks"}],
        }),
    })
    with td:
        out = _agent().rescue_from_sandbox(sb)
        by_file = {c.file: c for c in out.candidates}
        assert [e.dimension for e in by_file["rewrites/rw_1.py"].expectations] == ["n_regs"]
        assert [e.dimension for e in by_file["rewrites/rw_2.py"].expectations] == ["occupancy"]


def test_a_malformed_sidecar_costs_the_expectations_and_not_the_round():
    """Every failure mode of the file must degrade to "no expectations", never to an exception.

    A rescue exists because a round is about to be lost; a rescue that RAISES loses it anyway,
    and for a reason unrelated to the kernel. Six shapes, all of them things a half-written or
    truncated file really produces.
    """
    for bad in ('{"rw_1.py": [', 'null', '[]', '"a string"', '{"rw_1.py": "not a list"}',
                '{"rw_1.py": [42]}'):
        sb, td = _sandbox({"rewrites/rw_1.py": _SRC, "rewrites/expectations.json": bad})
        with td:
            out = _agent().rescue_from_sandbox(sb)
            assert out is not None, "a malformed sidecar must not cost the rescue: %r" % bad
            assert out.candidates[0].expectations == [], bad


def test_an_unknown_dimension_is_dropped_without_taking_the_others_with_it():
    """The vocabulary is closed on the rescue path exactly as it is on the normal path.

    Dropped per ENTRY rather than per file: rejecting the whole result over one bad name would
    hand the round back to the failure the rescue is recovering from, while `check_output` -- which
    `_rescue` runs over whatever this returns -- would reject the entire rescued result if a bad
    name survived to it. So filtering here is what lets the good declarations through.
    """
    sb, td = _sandbox({
        "rewrites/rw_1.py": _SRC,
        "rewrites/expectations.json": json.dumps({"rw_1.py": [
            {"dimension": "shared_bytes", "expect": "up", "why": "bigger tile"},
            {"dimension": "l2_hit_rate", "expect": "up", "why": "not a dimension we measure"},
            {"dimension": "n_spills", "expect": "unchanged", "why": "no new live ranges"},
        ]}),
    })
    with td:
        out = _agent().rescue_from_sandbox(sb)
        kept = [e.dimension for e in out.candidates[0].expectations]
        assert kept == ["shared_bytes", "n_spills"], kept


def test_a_magnitude_in_the_sidecar_is_refused_just_as_it_is_in_the_response():
    """`extra="forbid"` on `ResourceExpectation` is load-bearing, and must apply here too.

    A number in this field would be invented -- the rate is not derivable in advance (no closed
    form for shared memory, 0 of 96 configurations matched; the cost map is not separable; even
    the sign reverses across a wide sweep). Pydantic's default would DROP the unknown key and
    accept the entry, and the agent would then have reasoned from a magnitude nobody checked.
    The sidecar must not become the loophole.
    """
    sb, td = _sandbox({
        "rewrites/rw_1.py": _SRC,
        "rewrites/expectations.json": json.dumps({"rw_1.py": [
            {"dimension": "n_regs", "expect": "down", "why": "fewer live values",
             "expected_pct": 40},
            {"dimension": "n_spills", "expect": "down", "why": "fewer live values"},
        ]}),
    })
    with td:
        out = _agent().rescue_from_sandbox(sb)
        kept = [e.dimension for e in out.candidates[0].expectations]
        assert kept == ["n_spills"], (
            "an entry carrying a magnitude must be refused, not accepted with the number "
            "silently dropped: %r" % kept)


def test_an_illegal_direction_is_refused():
    """`expect` is a closed set of four values. "increase" is not one of them."""
    sb, td = _sandbox({
        "rewrites/rw_1.py": _SRC,
        "rewrites/expectations.json": json.dumps({"rw_1.py": [
            {"dimension": "n_regs", "expect": "increase", "why": "more live values"},
            {"dimension": "occupancy", "expect": "unknown", "why": "cannot tell in advance"},
        ]}),
    })
    with td:
        out = _agent().rescue_from_sandbox(sb)
        assert [e.dimension for e in out.candidates[0].expectations] == ["occupancy"]


def test_a_rescued_result_still_passes_the_normal_check_output_gate():
    """The contract: whatever a rescue returns goes through the SAME gate a normal answer does.

    Asserted by running `check_output` over the rescued result, so a future rescue that recovered
    something `check_output` would reject cannot pass unnoticed.
    """
    sb, td = _sandbox({
        "rewrites/rw_1.py": _SRC,
        "rewrites/expectations.json": json.dumps({"rw_1.py": [
            {"dimension": "shared_bytes", "expect": "up", "why": "bigger tile"}]}),
    })
    with td:
        agent = _agent()
        out = agent.rescue_from_sandbox(sb)
        assert agent.check_output(out, sb) is None, agent.check_output(out, sb)


def test_the_prompt_asks_for_the_sidecar_or_nothing_will_ever_write_one():
    """The producing half. A reader that works on a file no agent is told to write reads nothing.

    Recorded as `probe-needs-a-positive-control`: this project once read a probe's own failure as
    five negative results. The reader above is useless without this.

    Read from the SOURCE of `render_prompt` rather than by calling it: the prompt is an f-string
    built from a fully-populated `RewriterInputs` (a task, a bottleneck report, a calibration), and
    constructing all of that here would test the fixture more than the instruction. The claim being
    checked is textual -- "the prompt contains this instruction" -- so reading the text is the
    direct assertion, not a proxy for a behavioural one.
    """
    import inspect

    prompt = inspect.getsource(StructureRewriterAgent.render_prompt)
    assert "rewrites/expectations.json" in prompt, (
        "the rewriter is never asked to write the sidecar, so the rescue path will recover "
        "expectations from a file that never exists")
    # And it must say WHY, because an instruction that looks like redundant bookkeeping is the
    # kind an agent skips under time pressure.
    assert "connection" in prompt.lower() or "transport" in prompt.lower()
