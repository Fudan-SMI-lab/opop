"""Full promotion through real native studies and adapters with external CPU fakes."""

from collections import Counter
from pathlib import Path

import pytest

from kernel_optimizer.agents.runtime import OpencodeClient, OpencodeServer, PromptResult
from kernel_optimizer.gpu.worker_client import WslGpuWorker, to_wsl_path
from kernel_optimizer.models.core import ParamSet
from kernel_optimizer.paramspace.materializer import extract_defaults, materialize
from kernel_optimizer.store.run_store import RunStore
from scripts.experiments import c2_information
from scripts.experiments.c2_information_inputs import InformationResult, InformationRun
from scripts.experiments.c2_method_protocol import Deadline
from tests.test_c2_information_inputs import prepared


@pytest.mark.parametrize("group", ["G0", "H"])
@pytest.mark.parametrize("mode", ["overlap", "negative", "conflict", "deadline", "deadline_pair2", "bad_confirmation",
    "bad_samples", "invalid", "small", "artifact_error", "final_failed", "partial", "no_best", "parent_failure", "upstream_error"])
def test_full_promotion_after_all_spaces_is_bounded_and_source_faithful(tmp_path, monkeypatch, prepared, group, mode):
    # Given: real cap1 studies; confirmation requests depend on fresh full evidence.
    shared, acquisition, cfg = prepared
    cfg.gpu.compile_screen_enabled = False
    for name in ("analyst", "rewriter", "parameterizer"):
        getattr(cfg.agents, name).max_retries = 0
        getattr(cfg.agents, name).max_transport_retries = 0
    root = tmp_path / "run"
    clock = [0.0]
    roles, calls = [], []
    full_counts = Counter()
    default = 2 if mode == "partial" else 7
    initial = f"import triton\nPARAMS={{'x': {default}}}\n@triton.jit\ndef work(): return PARAMS['x']\n"
    small = mode in {"small", "artifact_error", "final_failed", "partial", "no_best", "upstream_error"}
    parent_screen = [10.0] * 3 if small else [10.0, 11.0, 12.0]
    child_screen = [9.99] * 3 if small else [11.0, 11.0, 13.0] if mode == "negative" else [10.5] * 3
    helper = tmp_path / "owner.py"
    helper.write_text("OWNER = 'parent'\n")
    run = InformationRun(cfg, group, root, deadline=Deadline(0, lambda: clock[0], 100),
        helpers=(helper,), helper_root=tmp_path, space_expansions_per_candidate=1, promotion_policy="full",
        require_b40=mode == "parent_failure")

    def prompt(self, session_id, text, **kwargs):
        directory, title = kwargs["directory"], kwargs["schema"]["title"]
        expansion = "retune" in directory.parts
        roles.append((title, expansion))
        source = initial.replace(f"'x': {default}", "'x': 59").replace("return PARAMS['x']", "return PARAMS['x'] + 100") if expansion else initial
        if title == "ParameterizationResult":
            output = directory / "candidate/parameterized.py"
            (output.parent / "owner.py").write_text("OWNER = 'child'\n")
            file = "candidate/parameterized.py"
        else:
            output, file = directory / "rewrite.py", "rewrite.py"
        output.write_text(source)
        choices = [-1, *range(60)] if expansion else list(range(4 if mode == "partial" else 60))
        answers = {
            "BottleneckReport": {"summary": "fixture", "hypotheses": []},
            "RewriteResult": {"candidates": [{"file": file, "backend": "triton", "hypothesis_id": "H1", "change_summary": "fixture"}]},
            "ParameterizationResult": {"file": file, "space": {"params": [{"name": "x", "kind": "int", "choices": choices}]}},
        }
        return PromptResult(text="", structured=answers[title], session_id=session_id)

    def worker(self, job, timeout_s, tag, **kwargs):
        if job["job_type"] == "static_check":
            return {"ok": True}
        assert job["seed"] == 0
        source_path = Path(job["kernel_src_path"])
        source = source_path.read_text(encoding="utf-8")
        params = extract_defaults(source)
        confirmation = "confirmation" in self.jobs_dir.parts
        parent = source_path.parent == root / "common"
        arm = "parent" if parent else "child"
        n = job["num_perf_trials"]
        if n == 100:
            assert job["num_correct_trials"] == 5
        if (mode == "invalid" and not parent) or (mode == "parent_failure" and parent):
            return {"ok": False, "failure_kind": "correctness_mismatch", "log_tail": "fixture"}
        if confirmation:
            assert n == 100
            snapshots = [e.payload["snapshot"]["asked"] for e in RunStore.open(root / "retune").iter_events()
                         if e.type == "TUNING_DONE"]
            assert snapshots == [40, 40]
            own = root / "common/imports" if parent else root / "retune/inputs"
            other = root / "retune/inputs" if parent else root / "common/imports"
            assert to_wsl_path(own) in self.cfg.extra_pythonpath
            assert to_wsl_path(other) not in self.cfg.extra_pythonpath
            assert (own / "owner.py").read_text() == f"OWNER = '{arm}'\n"
            expected = materialize(shared.source, shared.parent.params) if parent else materialize(initial, ParamSet(values={"x": 0}))
            assert source == expected
            calls.append(arm)
            if mode == "deadline" and parent:
                clock[0] = 101
            if mode == "deadline_pair2" and len(calls) == 3:
                clock[0] = 101
            if mode == "bad_confirmation" and not parent:
                return {"ok": False, "failure_kind": "correctness_mismatch", "log_tail": "fixture"}
            value = 10.0 if parent and mode == "negative" else 11.0 if parent else 12.0 if mode == "negative" or (mode == "conflict" and len(calls) == 3) else 10.0
        elif n == 100:
            index = full_counts[arm]
            full_counts[arm] += 1
            value = (parent_screen if parent else child_screen)[index]
        else:
            value = (1.0 if small else 10.0) + int(params["x"]) + (100 if "+ 100" in source else 0)
        samples = 20 if mode == "bad_samples" and n == 100 and not parent else n
        return {"ok": True, "latency_ms": {"mean": value, "median": value, "min": value, "max": value,
            "std": 0.0, "n": samples, "samples": [value] * samples}}

    monkeypatch.setattr(OpencodeServer, "start", lambda self: "http://unused.invalid")
    monkeypatch.setattr(OpencodeServer, "stop", lambda self: None)
    if mode == "upstream_error":
        def stop_failure(self):
            raise OSError("fixture runtime cleanup failure")
        monkeypatch.setattr(OpencodeServer, "stop", stop_failure)
    monkeypatch.setattr(OpencodeClient, "create_session", lambda self, directory, title: title)
    monkeypatch.setattr(OpencodeClient, "prompt", prompt)
    monkeypatch.setattr(WslGpuWorker, "run_job", worker)
    if mode in {"artifact_error", "final_failed", "no_best"}:
        original_retune = c2_information.retune
        def native_status(*args, **kwargs):
            return original_retune(*args, **kwargs).model_copy(update={"status": mode})
        monkeypatch.setattr(c2_information, "retune", native_status)
    # When: the full policy consumes existing screens and only its requested pairs.
    result = c2_information.run_information(shared, acquisition, run)
    # Then: ordering, data retention, native qualification and decision stay separate from heldout.
    expected_calls = ["parent", "child", "child", "parent"] if mode in {"overlap", "conflict"} else ["parent", "child"] if mode in {"negative", "bad_confirmation"} else ["parent"] if mode == "deadline" else ["parent", "child", "child"] if mode == "deadline_pair2" else []
    assert calls == expected_calls
    assert result.selected == ("child" if mode in {"overlap", "small"} else "parent")
    expected_status = "failed" if mode in {"invalid", "no_best", "bad_confirmation", "final_failed", "parent_failure"} else "censored" if mode in {"deadline", "deadline_pair2", "bad_samples", "artifact_error", "partial", "upstream_error"} else "valid"
    assert result.status == expected_status
    assert result.promotion_policy == "full" and result.promotion_decision is not None
    assert result.promotion_decision.outcome == (result.selected if expected_status == "valid" else "failed")
    assert len(result.confirmation_pairs) == (len(calls) + 1) // 2
    saved = InformationResult.model_validate_json((root / "result.json").read_text(encoding="utf-8"))
    assert saved.confirmation_pairs == result.confirmation_pairs and saved.promotion_decision == result.promotion_decision
    if mode == "parent_failure":
        assert all(t.status == "fail" for t in result.parent_finals) and roles == []
        return
    assert [t.latency_ms.median for t in result.parent_finals] == parent_screen
    if mode != "invalid":
        assert [t.latency_ms.median for t in result.child.finals] == child_screen
        assert full_counts == {"parent": 3, "child": 3}
        assert roles == [("BottleneckReport", False), ("RewriteResult", False), ("ParameterizationResult", False), ("ParameterizationResult", True)]
        assert result.child.selected_space.space_id == result.child.spaces[0].space_id
    if mode == "deadline":
        assert result.confirmation_pairs[0].parent is not None and result.confirmation_pairs[0].child is None
    if mode == "deadline_pair2":
        assert result.confirmation_pairs[1].child is not None and result.confirmation_pairs[1].parent is None
    if mode == "partial":
        assert [s.asked for s in result.studies] == [4, 40]
    assert (root / result.selected_artifact).is_file()
