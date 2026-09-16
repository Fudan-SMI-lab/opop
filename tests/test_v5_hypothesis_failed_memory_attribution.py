"""CPU regressions for failure memory with incomplete rewrite provenance."""

from pathlib import Path

import pytest

from kernel_optimizer.agents.modules import StructureRewriterAgent
from kernel_optimizer.agents.sandbox import Sandbox
from kernel_optimizer.control.orchestrator import _split_attempted
from kernel_optimizer.models.reports import Hypothesis


@pytest.fixture
def hypotheses() -> list[Hypothesis]:
    return [Hypothesis(id=hid, change=f"change {hid}", expected_effect="faster")
            for hid in ("H1", "H2", "H3")]


def test_failure_records_nothing_when_attribution_is_empty(
    hypotheses: list[Hypothesis],
) -> None:
    # Given: a failed round with no declared hypothesis provenance.
    attempted: set[str] = set()
    # When: the report is reconciled with those declarations.
    failed, undeclared = _split_attempted(hypotheses, attempted, 4)
    # Then: no idea is condemned or asserted to be definitely untried.
    assert failed == []
    assert undeclared == []


@pytest.mark.parametrize("attempted", [{"H9"}, {""}, {"H9", ""}])
def test_failure_records_nothing_when_no_declared_id_matches(
    hypotheses: list[Hypothesis], attempted: set[str],
) -> None:
    # Given: declarations that identify none of this report's ideas.
    # When: the failed round is attributed.
    failed, _ = _split_attempted(hypotheses, attempted, 4)
    # Then: unknown IDs neither fabricate entries nor condemn report hypotheses.
    assert failed == []


@pytest.mark.parametrize("attempted", [{"H2"}, {"H2", "H9"}, {"H2", ""}])
def test_failure_keeps_targeted_memory_when_a_known_id_is_declared(
    hypotheses: list[Hypothesis], attempted: set[str],
) -> None:
    # Given: only H2 is explicitly attributed to this report.
    # When: the failed round is attributed.
    failed, undeclared = _split_attempted(hypotheses, attempted, 7)
    # Then: payload and legacy nonempty-set behavior are preserved.
    assert failed == [{"id": "H2", "change": "change H2", "round": 7}]
    assert undeclared == ["H1", "H3"]  # Not declared, not proof of nonexecution.


def test_blank_id_rescue_remains_accepted_without_condemning_report(
    tmp_path: Path, hypotheses: list[Hypothesis],
) -> None:
    # Given: a completed artifact whose response (including attribution) was lost.
    rewrites = tmp_path / "rewrites"
    rewrites.mkdir()
    (rewrites / "rw_1.py").write_text(
        "import torch, triton, triton.language as tl\n"
        "@triton.jit\n"
        "def kernel(x, y, BLOCK: tl.constexpr):\n"
        "    i = tl.arange(0, BLOCK)\n"
        "    tl.store(y + i, tl.load(x + i))\n"
        "class ModelNew(torch.nn.Module):\n"
        "    def forward(self, x):\n"
        "        y = torch.empty_like(x)\n"
        "        kernel[(1,)](x, y, BLOCK=64)\n"
        "        return y\n",
        encoding="utf-8",
    )
    sandbox = Sandbox(tmp_path)
    agent = StructureRewriterAgent.__new__(StructureRewriterAgent)
    rescued = agent.rescue_from_sandbox(sandbox)
    assert rescued is not None
    assert len(rescued.candidates) == 1
    assert rescued.candidates[0].hypothesis_id == ""
    assert agent.check_output(rescued, sandbox) is None
    attempted = {c.hypothesis_id for c in rescued.candidates if c.hypothesis_id}
    # When: an unimproved rescued round reaches failed-memory attribution.
    failed, undeclared = _split_attempted(hypotheses, attempted, 0)
    # Then: the eligible artifact supplies no evidence against specific report ideas.
    assert failed == []
    assert undeclared == []
