"""Agent schema tests: canned structured outputs validate; fenced-JSON fallback."""

import httpx
import pytest

from kernel_optimizer.agents.runtime import AgentCallError, OpencodeClient, extract_fenced_json
from kernel_optimizer.models.reports import (
    BottleneckReport,
    GenerationResult,
    NoveltyResult,
    ParameterizationResult,
    RepairResult,
    RewriteResult,
)


def test_generation_result_validates():
    data = {"candidates": [
        {"file": "candidates/cand_1.py", "backend": "triton",
         "approach_summary": "tiled", "structural_axes": ["partitioning"]},
        {"file": "candidates/cand_2.py", "approach_summary": "fused"},
    ]}
    out = GenerationResult.model_validate(data)
    assert len(out.candidates) == 2
    assert out.candidates[1].backend == "triton"  # default


def test_parameterization_result_validates():
    data = {"file": "candidate/parameterized.py",
            "space": {"params": [
                {"name": "BLOCK_M", "kind": "int", "choices": [32, 64]},
                {"name": "MODE", "kind": "str", "choices": ["a", "b"],
                 "description": "impl variant"}],
                "constraints": [{"expr": "BLOCK_M <= 64", "rationale": "smem"}]}}
    out = ParameterizationResult.model_validate(data)
    assert out.space.params[0].choices == [32, 64]


def test_bottleneck_report_validates():
    data = {"summary": "shared-memory bound",
            "parameter_limits": [
                {"param": "TILE", "headroom_direction": "increase",
                 "blocked_by": "shared_memory", "predicted_gain_pct": 15.0,
                 "evidence": "latency monotone down to 256; 512 OOMs"}],
            "hypotheses": [{"id": "H1", "change": "split-K", "expected_effect": "less smem"}],
            "suggested_action": "rewrite"}
    out = BottleneckReport.model_validate(data)
    assert out.parameter_limits[0].blocked_by == "shared_memory"


def test_rewrite_and_novelty_and_repair_validate():
    RewriteResult.model_validate(
        {"candidates": [{"file": "rewrites/rw_1.py", "hypothesis_id": "H1",
                         "change_summary": "split-K"}]})
    NoveltyResult.model_validate(
        {"candidates": [{"file": "novel/nv_1.py", "approach_summary": "warp-level",
                         "difference_claim": "no shared memory at all"}]})
    RepairResult.model_validate(
        {"file": "candidate/fixed.py", "diagnosis": "bad mask", "change_summary": "fix mask"})


def test_fenced_json_extraction():
    text = """Here is my analysis.

```json
{"summary": "x", "suggested_action": "rewrite"}
```
"""
    data = extract_fenced_json(text)
    assert data == {"summary": "x", "suggested_action": "rewrite"}


def test_fenced_json_last_block_wins():
    text = """```json
{"a": 1}
```
some words
```json
{"b": 2}
```"""
    assert extract_fenced_json(text) == {"b": 2}


def test_bare_json_fallback():
    assert extract_fenced_json('prefix {"a": 1} suffix') == {"a": 1}


def test_no_json_returns_none():
    assert extract_fenced_json("no json here at all") is None


def test_prompt_timeout_becomes_agent_error_and_aborts():
    """A hung agent call (httpx.ReadTimeout) must surface as AgentCallError so the
    AgentModule retry loop handles it, and must abort the stuck session — never
    let the transport exception propagate and crash a multi-hour run.
    Regression for the level3:43 run that died on an uncaught ReadTimeout."""
    client = OpencodeClient.__new__(OpencodeClient)  # skip real HTTP client construction
    client.base_url = "http://127.0.0.1:0"
    client.timeout_s = 1.0

    aborted: list[str] = []

    class FakeHttp:
        def post(self, *a, **k):
            raise httpx.ReadTimeout("timed out")

    client._http = FakeHttp()
    client.abort = lambda session_id: aborted.append(session_id)

    with pytest.raises(AgentCallError) as ei:
        client.prompt("ses_stuck", "hi", model="openai/gpt-5.6-sol",
                      schema={"type": "object"})
    assert "ReadTimeout" in str(ei.value)
    assert aborted == ["ses_stuck"]
