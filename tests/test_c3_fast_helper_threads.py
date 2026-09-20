import json
from contextlib import closing
from concurrent.futures import Future
from contextvars import ContextVar
from threading import get_ident

import pytest

from examples.c3_qwen3.device_records import LocalRequest, SourceEligibility
from examples.c3_qwen3.fast_generation import _LocalCall, _OwnerCallbacks
from examples.c3_qwen3.runner_records import RunnerError
from examples.c3_qwen3.search_helper import request_fixture
from kernel_optimizer.agents.runtime import AgentCallError
from tests.c3_device_fakes import case
from tests.c3_framework_fakes import workflow_case


def test_fast_helper_runs_on_owner_while_agent_context_stays_scoped(tmp_path, monkeypatch):
    # Given: actual agent lifecycle/RPC, with only the existing numerical/provider boundaries fake.
    generator, provider, _, _ = workflow_case(tmp_path)
    generator.evaluator.capture()
    owner = get_ident()
    marker = ContextVar("helper_thread_test", default="unset")
    token = marker.set("owner")
    seen = {}
    original_prompt = provider.prompt
    original_local = generator.evaluator.local
    old_hook = generator.agent.self_test_context
    deadline = generator.evaluator.budget.spec.deadline_unix_s
    def local(request, *, from_agent=False):
        seen["callback"] = (get_ident(), marker.get())
        return original_local(request, from_agent=from_agent)
    def prompt(*args, **kwargs):
        seen["agent"] = (get_ident(), marker.get())
        marker.set("agent-only")
        result = original_prompt(*args, **kwargs)
        folder = kwargs["directory"]
        request = LocalRequest(mode="fixture_only", stage=generator.evaluator.stage,
            bundle_path=folder / "bundle.json", params={}, site_id=generator.evaluator.target.brief.site_id,
            kernels=result.structured["device_kernels"],
            source_gate=SourceEligibility(eligible=False, bundle_sha256="", params_sha256=""))
        endpoint = json.loads((folder / "task/operator-helper.json").read_text())
        seen["reply"] = request_fixture(endpoint, request.model_dump(mode="json"))
        return result
    monkeypatch.setattr(generator.evaluator, "local", local)
    monkeypatch.setattr(provider, "prompt", prompt)
    # When: real DraftGenerator invokes the agent and services its blocking helper RPC.
    try:
        with closing(provider):
            result = generator.invoke(0)
        # Then: GPU boundary is on owner context; agent validation retains its scoped parent context.
        assert seen["callback"] == (owner, "owner")
        assert seen["agent"][0] != owner
        assert seen["agent"][1] == "owner"
        assert marker.get() == "owner"
        assert seen["reply"].valid
        assert result.status == "draft_ready", result.error
        assert generator.agent.self_test_context is old_hook
        assert generator.evaluator.budget.spec.deadline_unix_s == deadline
    finally:
        marker.reset(token)
        generator.evaluator.runtime.runner.close()


def test_callback_exception_reaches_waiter_without_stopping_next_request(tmp_path):
    # Given: one callback fails; the transport observes the failure and can finish normally.
    runtime, request, fixture = case(tmp_path)
    from examples.c3_qwen3.local_eval import evaluate_local
    calls = []
    def callback(value):
        calls.append(get_ident())
        if len(calls) == 1:
            raise RunnerError("local callback failed")
        return evaluate_local(value, (fixture,), runtime)
    bridge = _OwnerCallbacks(callback)
    owner = get_ident()
    def operation():
        with pytest.raises(RunnerError, match="local callback failed"):
            bridge.request(request)
        return bridge.request(request)
    # When / Then: exception/result both cross the bridge without changing execution owner.
    result = bridge.run(operation)
    assert result.valid, result.detail
    assert calls == [owner, owner]
    with pytest.raises(RunnerError, match="closed"):
        bridge.request(request)
    runtime.runner.close()


def test_transport_exception_closes_bridge_without_new_callback(tmp_path):
    # Given: the agent transport terminates without an admitted helper operation.
    runtime, request, _ = case(tmp_path)
    calls = []
    bridge = _OwnerCallbacks(lambda value: calls.append(value))
    def failure():
        raise AgentCallError("fixed transport timeout")
    # When / Then: original failure propagates and late calls cannot execute GPU callbacks.
    with pytest.raises(AgentCallError, match="fixed transport timeout"):
        bridge.run(failure)
    with pytest.raises(RunnerError, match="closed"):
        bridge.request(request)
    assert calls == []
    runtime.runner.close()


def test_close_releases_pending_waiter_without_executing_callback(tmp_path):
    # Given: a queued callback has not reached GPU admission when transport ends.
    runtime, request, _ = case(tmp_path)
    calls = []
    bridge = _OwnerCallbacks(lambda value: calls.append(value))
    pending = Future()
    bridge.queue.put(_LocalCall(request, pending))
    # When / Then: the waiting RPC gets a terminal error, not an indefinite wait or fabricated score.
    bridge._close()
    with pytest.raises(RunnerError, match="agent ended"):
        pending.result(timeout=1)
    assert calls == []
    runtime.runner.close()
