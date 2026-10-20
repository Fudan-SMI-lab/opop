import importlib.util
import sys

import pytest

from kernel_optimizer.project.candidate_diagnostics import CandidateTrace, should_trace


def test_caught_exception_and_guard_reported_without_changing_behavior(tmp_path):
    path = tmp_path/"candidate.py"
    path.write_text(
        "def guarded(x):\n"
        "    try:\n"
        "        if x: raise ValueError('capture failed')\n"
        "        return True\n"
        "    except ValueError:\n"
        "        return False\n"
    )
    spec = importlib.util.spec_from_file_location("diagnostic_candidate", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    previous = sys.gettrace()
    trace = CandidateTrace([{"file": str(path), "id": "bundle-a"}])
    with trace:
        assert module.guarded(True) is False
        assert module.guarded(True) is False
        trace.phase = "run"
        assert module.guarded(False) is True
        try:
            raise RuntimeError("unrelated caller")
        except RuntimeError:
            pass
    assert sys.gettrace() is previous
    rows = trace.report()["events"]
    assert len(rows) == 2
    assert {r["event"] for r in rows} == {"exception", "return"}
    assert all(r["bundle"] == "bundle-a" and r["phase"] == "setup" and r["count"] == 2 for r in rows)
    assert next(r for r in rows if r["event"] == "return")["source"] == "return False"
    with pytest.raises(ValueError):
        with CandidateTrace([]):
            raise ValueError("outer failure")
    assert sys.gettrace() is previous


def test_false_return_keeps_guard_location_across_context_manager_cleanup(tmp_path):
    path = tmp_path/"guard.py"
    path.write_text(
        "from contextlib import nullcontext\n"
        "def guard():\n"
        "    with nullcontext():\n"
        "        return False\n"
    )
    namespace = {}
    exec(compile(path.read_text(), str(path), "exec"), namespace)
    with CandidateTrace([{"file": str(path), "id": "guard"}]) as trace:
        assert namespace["guard"]() is False
    event = trace.report()["events"][0]
    assert {"line": 4, "source": "return False"} in event["recent_lines"]
    assert not trace.recent


def test_reference_trigger_is_optional_and_not_an_acceptance_rule():
    run = {"ranks": [{"result": {"coverage": {"b": {
        "counts": {"x": 2, "y": 2}, "reference_invocations": {"x": 2, "y": 2}}}}}]}
    assert should_trace(run, "b", {})
    assert not should_trace(run, "b", {"candidate_diagnostics": "off"})
    assert not should_trace(run, "missing", {})
    run["ranks"][0]["result"]["coverage"]["b"]["reference_invocations"]["y"] = 1
    assert not should_trace(run, "b", {})
    assert should_trace(run, "b", {"candidate_diagnostics": "always"})
    with pytest.raises(ValueError):
        should_trace(run, "b", {"candidate_diagnostics": "invalid"})
