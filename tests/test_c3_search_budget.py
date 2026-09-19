"""Shared admission, repair and failure tests through the real operator workflow."""

from contextlib import closing
import json
from pathlib import Path
from time import time

import pytest

from examples.c3_qwen3.operator_search import optimize_goal
from examples.c3_qwen3.search_records import EvaluationRequest, SlotBudget
from kernel_optimizer.models.core import ParamSet
from tests.c3_search_fakes import search_case


def test_selftests_consume_eight_slots_while_agent_waits(tmp_path):
    # Given: the provider synchronously waits for two real loopback self-tests per call.
    session, provider, agent, inputs = search_case(tmp_path, selftests=2)
    # When
    with session, closing(provider):
        result = optimize_goal(session, agent, inputs)
    # Then: the serving thread is active during prompt wait; native gets only the remainder.
    assert len(provider.helper_results) == 4 and all(r.evaluation.valid for r in provider.helper_results)
    assert [o.slots_used for o in result.opportunities] == [8, 8]
    admitted = [r for r in session.attempts if r["admitted"] and r["purpose"] != "baseline"]
    assert len(admitted) == 16
    assert len([r for r in admitted if r["purpose"] == "self_test"]) == 4
    assert sum(len(o.trials) for o in result.opportunities) == 11
    assert result.slots_reconciled and result.slot_counts == {"baseline": 1, "native": 11, "self_test": 4, "parent_alignment": 1}
    session.runner.close()


def test_exhausted_selftests_never_become_native_cached_winner(tmp_path):
    # Given: eight valid self-tests exhaust each opportunity before any native trial.
    session, provider, agent, inputs = search_case(tmp_path, selftests=8)
    # When
    with session, closing(provider):
        result = optimize_goal(session, agent, inputs)
    # Then
    assert all(o.slots_used == 8 and not o.trials for o in result.opportunities)
    assert result.selected.baseline_fallback and result.selected.evaluation == result.baseline.evaluation
    assert len(provider.requests) == 2
    session.runner.close()


def test_wrong_operator_gets_at_most_one_repair_without_refund(tmp_path):
    # Given: each initial source is wrong, while the one repair computes the correct operation.
    session, provider, agent, inputs = search_case(tmp_path, wrong=True)
    # When
    with session, closing(provider):
        result = optimize_goal(session, agent, inputs)
    # Then
    assert len(provider.requests) == 4
    assert all(o.repair_calls == 1 and o.slots_used == 8 for o in result.opportunities)
    assert any(not t.task_evaluation.valid for o in result.opportunities for t in o.trials)
    assert result.selected.evaluation.valid
    session.runner.close()


def test_bad_params_and_changed_frozen_inputs_consume_slots_without_measurement(tmp_path):
    # Given: a valid candidate emitted by the real fake-provider pipeline.
    session, provider, agent, inputs = search_case(tmp_path)
    with session, closing(provider):
        result = optimize_goal(session, agent, inputs)
        session.begin_opportunity(session.baseline, {}, deadline_unix_s=time() + 60)
        before = session.runner.backend.forward_calls
        first = Path(next(r["bundle"] for r in session.attempts if r["purpose"] == "native"))
        # When
        invalid = session.self_test(EvaluationRequest(bundle=first,
            params=ParamSet(values={"foreign": 100}), site_groups=result.selected.site_groups))
        # Then
        assert not invalid.valid and invalid.score is None and session.budget.used == 1
        assert session.runner.backend.forward_calls == before
        assert "invalid parameters" in invalid.detail
        inputs.profile.write_text(inputs.profile.read_text() + " ")
        changed = session.self_test(EvaluationRequest(bundle=first, params=ParamSet(values={"speed": 2}),
                                                     site_groups=result.selected.site_groups))
        assert not changed.valid and "frozen" in changed.detail and session.budget.used == 2
        assert session.runner.backend.forward_calls == before
    session.runner.close()


def test_startup_default_ten_and_pilot_two():
    # Given / When: the real installed Optuna constructor, not a substitute sampler.
    from kernel_optimizer.tuning.tpe import OptunaTPETuner
    from kernel_optimizer.models.core import ParameterSpace
    space = ParameterSpace(space_id="s", candidate_id="c", source_sha="", domains=[])
    legacy = OptunaTPETuner(space, lambda p: True, 8)
    pilot = OptunaTPETuner(space, lambda p: True, 8, n_startup_trials=2)
    # Then
    assert legacy.study.sampler._n_startup_trials == 10
    assert pilot.study.sampler._n_startup_trials == 2


def test_deadline_censors_both_rounds_without_extra_baseline_search(tmp_path):
    # Given: an expired scientific clock; the borrowed CPU runner already exists for the test.
    session, provider, agent, inputs = search_case(tmp_path)
    from examples.c3_qwen3.search_records import SearchClock
    inputs = inputs.model_copy(update={"clock": SearchClock(campaign_started_unix_s=1,
        search_cutoff_unix_s=9001, final_deadline_unix_s=10801)})
    before = session.runner.backend.forward_calls
    # When
    with session, closing(provider):
        result = optimize_goal(session, agent, inputs)
    # Then
    assert all(o.status == "censored" and o.slots_used == 0 for o in result.opportunities)
    assert not provider.requests and session.runner.backend.forward_calls == before
    assert result.baseline.evaluation.score is None
    session.runner.close()


def test_helper_cli_uses_same_resident_without_importing_a_model(tmp_path):
    # Given: a live helper service and an invalid request, from an independent helper process.
    import subprocess
    import sys
    from examples.c3_qwen3.search_helper import HelperService
    from kernel_optimizer.agents.sandbox import Sandbox
    session, provider, _, _ = search_case(tmp_path)
    session.begin_opportunity(session.baseline, {}, deadline_unix_s=time() + 60)
    with session, closing(provider), HelperService(session) as helper:
        sandbox = Sandbox(tmp_path / "client")
        helper.seed_sandbox(sandbox)
        payload = tmp_path / "request.json"
        payload.write_text(json.dumps({"bundle": str(tmp_path / "absent.json"), "params": {"values": {}}}))
        output = tmp_path / "reply.json"
        # When
        process = subprocess.run([sys.executable, "-B", "-m", "examples.c3_qwen3.search_helper",
            "--endpoint", str(sandbox.root / "task/operator-helper.json"), "--request", str(payload),
            "--output", str(output)], capture_output=True, text=True, timeout=30)
        # Then
        assert process.returncode == 1 and json.loads(output.read_text())["slots_used"] == 1
        assert session.runner.backend.loads == 1
    session.runner.close()


def test_agent_drain_censors_later_admission_without_clock_reset(tmp_path, monkeypatch):
    # Given: an admitted agent turn finishes just after the same goal deadline.
    from examples.c3_qwen3.search_records import SearchClock
    session, provider, agent, inputs = search_case(tmp_path)
    clock = [1000.0]
    session.now = session.clock = lambda: clock[0]
    inputs = inputs.model_copy(update={"clock": SearchClock(campaign_started_unix_s=1000,
        search_cutoff_unix_s=10000, final_deadline_unix_s=11800)})
    invoke = agent.invoke
    def delayed(request):
        result = invoke(request)
        clock[0] = 6401
        return result
    monkeypatch.setattr(agent, "invoke", delayed)
    # When
    with session, closing(provider):
        result = optimize_goal(session, agent, inputs)
    # Then: ninety minutes includes the turn; neither native work nor round two is newly admitted.
    assert result.deadline_unix_s == 6400 and result.drain_s == 1
    assert len(provider.requests) == 1 and all(o.status == "censored" for o in result.opportunities)
    assert result.slot_counts["native"] == 0
    session.runner.close()
