"""The proof gate must judge against the limit the quality gate ENFORCED, not a literal.

WHAT THIS PINS, and why it is not a cosmetic refactor. `prove_candidate` re-verifies a candidate that
`evaluate_quality` already accepted. It used to compare `logits_relative_l2_max` against a hardcoded
`.01` -- the contract's DECLARED limit. That is only the enforced limit when no equivalence floor was
measured. With a floor, `runner_quality.criteria` raises the limit, so the two gates in one pipeline
disagreed about the same candidate:

    dev epoch F5-dev-0921-185739, on disk:
      floor measured  0.016163086   (73-module site group, automatic 1-ULP)
      candidate       l2 0.011807271831267152, local_checks_passed 1.0
      evaluate_quality  -> valid: true, score 270.725 ms vs baseline 283.324 (+4.45%)
      prove_candidate   -> FALSE, because 0.011807 > 0.01

The epoch kept drawing and burned three further agent calls. Every lane that measures a floor was
capped by the declared limit regardless, which made the whole floor mechanism inert at the decision
that matters.

The limit is now read from the quality receipt's own `criteria` block. These tests pin both halves:
the floor case is admitted, and -- the part that must not regress -- the no-floor case is still judged
at exactly the declared limit, and a receipt that cannot be read is a refusal rather than a pass.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from kernel_optimizer.evaluation.task_eval import TaskEvaluation

from examples.c3_qwen3.fast_proof import logits_limit_enforced, prove_candidate
from examples.c3_qwen3.runner_quality import CRITERIA, FLOOR_METRIC, criteria

DECLARED = 0.01
MEASURED_FLOOR = 0.016163086
ADMITTED_L2 = 0.011807271831267152   # the real candidate, between the two


def write_quality(path: Path, limit: float, *, metric: str = FLOOR_METRIC,
                  wrapped: bool = True) -> Path:
    """A quality receipt shaped exactly like `ResidentRunner._raw("quality", ...)` writes one.

    `wrapped` reproduces the real {kind, binding, data} envelope -- the shape verified against
    F5-dev-0921-185739's own quality-*.json. With `wrapped=False` the criteria sit at the top level,
    which the reader also accepts so it does not depend on the envelope.
    """
    body = {
        "prompts": [], "metrics": {}, "valid": True, "detail": None, "forward_calls": 1,
        "criteria": [{"metric": metric, "limit": limit, "measures": "worst-prompt relative L2"},
                     {"metric": "paired_mean_nll_delta_nat_per_token", "limit": 0.02, "measures": "nll"}],
        "criteria_source": "contract",
        "equivalence_floor": limit if limit > DECLARED else None,
    }
    document = {"kind": "quality", "binding": {"site_ids": ["pure-x"]}, "data": body} if wrapped else body
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def make_candidate(tmp_path: Path, l2: float, quality_raw: Path, *, score: float = 270.725,
                   omit_quality_raw: bool = False):
    """A candidate whose stored evidence is internally consistent, varying only the logits error."""
    from examples.c3_qwen3.device_records import StageIdentity

    stage = StageIdentity(profile="c3_fast_device", stage="development", framework_id="F")
    receipt = tmp_path / "fast-model.json"
    artifact = SimpleNamespace(author="agent", version=0, session_id="ses",
                               bundle=tmp_path / "bundle.json", bundle_sha256="b",
                               params_sha256="p", source_hashes={"operators.py": "h"})
    metrics = {"local_checks_passed": 1.0, FLOOR_METRIC: l2,
               "paired_mean_nll_delta_nat_per_token": 0.0022, "completed_requests": 4.0,
               "output_tokens": 4.0}
    data = {"stage": stage.model_dump(mode="json"), "forward_calls": 69, "wall_ms": 6544.6,
            "evaluation": {"valid": True, "score": score, "metrics": metrics, "detail": None},
            "measurement_raw": str(tmp_path / "measurement.json")}
    if not omit_quality_raw:
        data["quality_raw"] = str(quality_raw)
    receipt.write_text(json.dumps({
        "kind": "fast-model",
        "binding": {"bundle_sha256": "b", "params_sha256": "p",
                    "source_hashes": {"operators.py": "h"}},
        "data": data,
    }), encoding="utf-8")
    # The OUTER detail is {receipt, detail} only -- exactly what `evaluate_fast_model` rebuilds. It
    # deliberately does NOT carry quality_raw: a reader that expected it there rejected every real
    # candidate, so the test fixture must reproduce the real shape rather than a convenient one.
    detail = json.dumps({"receipt": str(receipt), "detail": None})
    # `stage` must be the REAL StageIdentity: `prove_candidate` compares `r.stage == stage`, and a
    # namespace never equals a pydantic record, so a stub here would fail the local-match clause and
    # make every admission test look like a limit failure.
    local = SimpleNamespace(bundle_sha256="b", params_sha256="p", stage=stage,
                            fixtures=(SimpleNamespace(description=SimpleNamespace(goal="ttft")),))
    candidate = SimpleNamespace(artifact=artifact, locals=(local,),
                                model=TaskEvaluation(valid=True, score=score, metrics=metrics, detail=detail))
    return candidate


@pytest.fixture
def stage():
    from examples.c3_qwen3.device_records import StageIdentity
    return StageIdentity(profile="c3_fast_device", stage="development", framework_id="F")


@pytest.fixture(autouse=True)
def _accept_the_local_half(monkeypatch):
    """Stub `proven_local` only.

    This suite is about ONE clause of `prove_candidate` -- which logits limit it compares against.
    The local-evidence half needs a real `LocalReport` with device proof, 20 reference and 20
    candidate timings per fixture, which is the subject of its own suites. Stubbing it keeps a
    failure here attributable to the limit, and the tests below still exercise every other clause
    (`local_checks_passed`, the NLL limit, the receipt identity chain) through the real code.
    """
    import examples.c3_qwen3.fast_proof as proof
    monkeypatch.setattr(proof, "proven_local", lambda report: True)


@pytest.fixture
def goal():
    return SimpleNamespace(id="ttft", requests=4, output_tokens=1)


# --------------------------------------------------------------- the limit reader

def test_reads_the_limit_the_receipt_declares(tmp_path):
    assert logits_limit_enforced(write_quality(tmp_path / "q.json", MEASURED_FLOOR)) == MEASURED_FLOOR


def test_reads_the_declared_limit_when_no_floor_was_measured(tmp_path):
    assert logits_limit_enforced(write_quality(tmp_path / "q.json", DECLARED)) == DECLARED


def test_a_receipt_without_a_criteria_block_raises(tmp_path):
    p = tmp_path / "q.json"
    p.write_text(json.dumps({"kind": "quality", "data": {"metrics": {}, "valid": True}}), encoding="utf-8")
    with pytest.raises(ValueError, match="no criteria block"):
        logits_limit_enforced(p)


def test_the_reader_accepts_an_unwrapped_receipt_too(tmp_path):
    """The envelope is an artefact of `_raw`, not of the criteria contract."""
    p = write_quality(tmp_path / "q.json", MEASURED_FLOOR, wrapped=False)
    assert logits_limit_enforced(p) == MEASURED_FLOOR


def test_quality_raw_is_read_from_inside_the_model_receipt(tmp_path, stage, goal):
    """The bug this nearly reintroduced: the OUTER detail has no `quality_raw`.

    `evaluate_fast_model` returns `detail = {"receipt": ..., "detail": ...}`, discarding the
    `quality_raw` it wrote into the receipt body. A reader looking in the outer detail finds nothing
    and refuses every candidate -- verified against F5-dev-0921-185739, where all three candidates'
    outer details carry exactly the keys ['receipt', 'detail'].
    """
    q = write_quality(tmp_path / "q.json", MEASURED_FLOOR)
    cand = make_candidate(tmp_path, ADMITTED_L2, q)
    assert set(json.loads(cand.model.detail)) == {"receipt", "detail"}
    assert prove_candidate(cand, stage, goal) is True


def test_a_model_receipt_without_quality_raw_is_refused(tmp_path, stage, goal):
    """`quality_raw` is a required field, so an old-shaped receipt fails closed, not open."""
    q = write_quality(tmp_path / "q.json", MEASURED_FLOOR)
    cand = make_candidate(tmp_path, ADMITTED_L2, q, omit_quality_raw=True)
    assert prove_candidate(cand, stage, goal) is False


def test_a_receipt_that_does_not_enforce_the_logits_metric_raises(tmp_path):
    p = write_quality(tmp_path / "q.json", DECLARED, metric="some_other_metric")
    with pytest.raises(ValueError, match="does not enforce"):
        logits_limit_enforced(p)


@pytest.mark.parametrize("bad", [None, "0.01", float("inf"), float("nan"), -1.0, True])
def test_a_non_finite_or_non_numeric_limit_raises(tmp_path, bad):
    p = tmp_path / "q.json"
    p.write_text(json.dumps({"kind": "quality",
                             "data": {"criteria": [{"metric": FLOOR_METRIC, "limit": bad}]}}), encoding="utf-8")
    with pytest.raises(ValueError):
        logits_limit_enforced(p)


def test_a_candidate_with_no_quality_raw_in_its_detail_is_refused(tmp_path, stage, goal):
    """An outer detail with no receipt pointer at all is a refusal."""
    q = write_quality(tmp_path / "q.json", MEASURED_FLOOR)
    cand = make_candidate(tmp_path, ADMITTED_L2, q)
    cand.model = cand.model.model_copy(update={"detail": json.dumps({"detail": None})})
    assert prove_candidate(cand, stage, goal) is False


# --------------------------------------------------------------- the gate itself

def test_the_real_candidate_is_admitted_under_its_measured_floor(tmp_path, stage, goal):
    """The exact numbers from F5-dev-0921-185739: this is the regression that mattered."""
    q = write_quality(tmp_path / "q.json", MEASURED_FLOOR)
    assert prove_candidate(make_candidate(tmp_path, ADMITTED_L2, q), stage, goal) is True


def test_the_same_candidate_is_refused_under_the_declared_limit(tmp_path, stage, goal):
    """Nothing became permissive: with no floor in the receipt, 0.011807 still fails 0.01."""
    q = write_quality(tmp_path / "q.json", DECLARED)
    assert prove_candidate(make_candidate(tmp_path, ADMITTED_L2, q), stage, goal) is False


def test_a_candidate_above_even_the_floor_is_refused(tmp_path, stage, goal):
    """The floor RAISES a limit; it does not remove one."""
    q = write_quality(tmp_path / "q.json", MEASURED_FLOOR)
    assert prove_candidate(make_candidate(tmp_path, MEASURED_FLOOR * 1.01, q), stage, goal) is False


def test_exactly_at_the_limit_is_admitted(tmp_path, stage, goal):
    q = write_quality(tmp_path / "q.json", MEASURED_FLOOR)
    assert prove_candidate(make_candidate(tmp_path, MEASURED_FLOOR, q), stage, goal) is True


def test_a_missing_quality_receipt_is_a_refusal_not_a_pass(tmp_path, stage, goal):
    """The failure mode to avoid: an unreadable receipt must never mean "no limit applies"."""
    assert prove_candidate(make_candidate(tmp_path, ADMITTED_L2, tmp_path / "absent.json"), stage, goal) is False


def test_an_outer_detail_carrying_only_the_receipt_is_still_valid(tmp_path, stage, goal):
    """`{"receipt": ...}` alone is enough, because the limit is found INSIDE the receipt.

    This is the positive counterpart of the removed assumption: nothing in the outer detail besides
    the receipt pointer is required, so a lane that stops writing the `detail` key does not break.
    """
    q = write_quality(tmp_path / "q.json", MEASURED_FLOOR)
    cand = make_candidate(tmp_path, ADMITTED_L2, q)
    receipt = json.loads(cand.model.detail)["receipt"]
    cand.model = cand.model.model_copy(update={"detail": json.dumps({"receipt": receipt})})
    assert prove_candidate(cand, stage, goal) is True


def test_the_nll_limit_is_still_enforced_under_a_floor(tmp_path, stage, goal):
    """A raised LOGITS limit must not carry the NLL criterion with it."""
    q = write_quality(tmp_path / "q.json", MEASURED_FLOOR)
    cand = make_candidate(tmp_path, ADMITTED_L2, q)
    metrics = {**cand.model.metrics, "paired_mean_nll_delta_nat_per_token": 0.021}
    cand.model = cand.model.model_copy(update={"metrics": metrics})
    assert prove_candidate(cand, stage, goal) is False


def test_local_checks_are_still_required(tmp_path, stage, goal):
    q = write_quality(tmp_path / "q.json", MEASURED_FLOOR)
    cand = make_candidate(tmp_path, ADMITTED_L2, q)
    cand.model = cand.model.model_copy(update={"metrics": {**cand.model.metrics, "local_checks_passed": 0.0}})
    assert prove_candidate(cand, stage, goal) is False


# ------------------------------------------------- the two gates cannot disagree

def test_the_gate_limit_equals_what_criteria_would_enforce(tmp_path):
    """The invariant behind the fix, stated directly against `runner_quality.criteria`.

    Whatever `criteria()` decides to enforce is what the receipt records and therefore what the proof
    gate reads. Pinning it this way means a future change to how the floor is applied cannot
    reintroduce the disagreement without failing here.
    """
    policy = SimpleNamespace(logits_relative_l2_max_per_prompt=DECLARED,
                             paired_mean_nll_delta_max_nat_per_token=0.02,
                             logits_relative_l2_floor_multiple=1.0)
    for floor in (None, 0.0, 0.005, MEASURED_FLOOR, 0.05):
        active = criteria(policy, floor)
        enforced = next(limit for key, limit, _ in active if key == FLOOR_METRIC)
        path = write_quality(tmp_path / f"q-{floor}.json", enforced)
        assert logits_limit_enforced(path) == enforced
        # And a floor below the declared limit never lowers it.
        if floor is not None and floor < DECLARED:
            assert enforced == DECLARED


def test_the_proof_gate_does_not_hardcode_the_logits_limit():
    """`fast_proof` must read the enforced limit, because its lane can have a measured floor."""
    import examples.c3_qwen3.fast_proof as proof

    body = Path(proof.__file__).read_text(encoding="utf-8")
    code = "\n".join(line.split("#")[0] for line in body.splitlines())
    for token in ("<= .01", "<= 0.01", "> .01", "> 0.01"):
        assert token not in code, f"fast_proof still hardcodes the logits limit: {token}"
    assert "logits_limit_enforced" in code


def test_the_manual_lane_states_its_limits_once_and_they_match_the_contract():
    """`manual_task` keeps literals, deliberately -- so the numbers are pinned here instead.

    That file is loaded STANDALONE by `TaskEvaluator`, copied out with only `manual_data` and
    `manual_types` beside it, so importing `runner_quality` to share the constants raises
    ModuleNotFoundError and kills the evaluator (measured: 10 tests in test_c3_manual_task failed
    exactly this way). It also passes no floor evidence, so its enforced limits ARE the declared ones.

    This test is what stops the two from drifting: if `runner_quality`'s declared limits change, the
    manual lane's constants must be updated or this fails.
    """
    import examples.c3_qwen3.manual_task as manual

    declared_logits = next(limit for key, limit, _ in CRITERIA if key == FLOOR_METRIC)
    declared_nll = next(limit for key, limit, _ in CRITERIA
                        if key == "paired_mean_nll_delta_nat_per_token")
    assert manual.DECLARED_LOGITS_LIMIT == declared_logits
    assert manual.DECLARED_NLL_LIMIT == declared_nll
    assert manual.LOGITS_METRIC == FLOOR_METRIC
    # And no inline literal survives in the comparison itself.
    code = "\n".join(line.split("#")[0] for line in Path(manual.__file__).read_text(encoding="utf-8").splitlines())
    for token in ("<= .01", "> .02", "<= 0.01", "> 0.02"):
        assert token not in code, f"manual_task still compares against an inline literal: {token}"


def test_the_manual_lane_imports_no_sibling_c3_modules():
    """The constraint that made the first attempt at this fix wrong, pinned so it is not relearned."""
    import ast

    import examples.c3_qwen3.manual_task as manual

    tree = ast.parse(Path(manual.__file__).read_text(encoding="utf-8"))
    relative = {node.module for node in ast.walk(tree)
                if isinstance(node, ast.ImportFrom) and node.level == 1 and node.module}
    assert relative <= {"manual_data", "manual_types"}, (
        f"manual_task is loaded standalone; it may only import its copied siblings, not {relative}")


def test_declared_module_default_still_matches_the_contract_constant():
    """If CRITERIA's own default drifts from 0.01, the no-floor assertions above would silently pass."""
    assert next(limit for key, limit, _ in CRITERIA if key == FLOOR_METRIC) == DECLARED
