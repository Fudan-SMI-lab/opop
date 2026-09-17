"""Real CPU protocol with fakes only at external model/worker boundaries."""

from pathlib import Path
from threading import Barrier, Event, get_ident

import pytest

from kernel_optimizer.agents.runtime import AgentCallError, OpencodeClient, OpencodeServer, PromptResult
from kernel_optimizer.gpu.worker_client import WslGpuWorker, to_wsl_path
from kernel_optimizer.models.core import ParamSet, ParamValue
from kernel_optimizer.paramspace.materializer import extract_defaults
from tests.test_c2_information_inputs import prepared as existing_prepared
from tests.test_c2_method_program import method_module

prepared = existing_prepared


@pytest.mark.parametrize(("slot", "mode"), [(s, "success") for s in ["A0", "A1", "B0", "B1"]]
    + [("A0", m) for m in ("overlap", "helpers", "acquisition_error", "failed_final", "artifact_error")]
    + [("B1", mode) for mode in ["proposal", "parameter", "invalid", "slow", "deadline", "tuning_deadline", "final_deadline"]])
def test_core_real_protocol_counts_and_native_selection(tmp_path, monkeypatch, prepared, slot, mode):
    # Given: six two-endpoint axes and the original ordinary parent.
    module = method_module("program")
    initial_cwd = Path.cwd()
    gpu_thread = get_ident()
    protocol = method_module()
    shared, _, cfg = prepared
    cfg.run.seed = 51
    cfg.budgets.trials_per_space = 3
    cfg.budgets.space_expansions_per_candidate = 2
    cfg.gpu.concurrency.enabled = True
    params: dict[str, ParamValue] = {f"x{i}": 1 for i in range(6)}
    shared = shared.model_copy(update={"task": protocol.SLOTS[slot].task,
        "source": f"PARAMS={params!r}\ndef run(): return 1\n",
        "parent": shared.parent.model_copy(update={"params": ParamSet(values=params)})})
    data = shared.model_dump()
    data["space"] = {"params": [{"name": name, "kind": "int", "choices": [1, 2]} for name in params]}
    data["trials"] = [data["parent"]]
    shared = type(shared).model_validate(data)
    if mode == "helpers":
        shared = shared.model_copy(update={"source": "import original_helper\n" + shared.source,
                                           "reference_source": "import original_helper\n" + shared.reference_source})
    roles = []
    handoffs = []
    jobs = []
    observations = []
    clock = [0.0]
    admissions = []
    analysts = Barrier(3)
    unrelated_blocked, first_tune, unrelated_finished = Event(), Event(), Event()
    overlap = []
    original_helper = tmp_path / "original_helper.py"
    original_helper.write_text("VALUE = 17\n")
    for name in ("analyst", "rewriter", "parameterizer"):
        getattr(cfg.agents, name).max_retries = 0
        getattr(cfg.agents, name).max_transport_retries = 0

    def prompt(self, session_id, text, **kwargs):
        assert Path.cwd() == tmp_path / "run/common"
        directory = kwargs["directory"]
        title = kwargs["schema"]["title"]
        roles.append(title)
        if title == "BottleneckReport" and mode not in {"deadline", "acquisition_error"}:
            analysts.wait(timeout=30)
        chain = next(part for part in directory.parts if part.startswith("generation-"))
        if title in {"BottleneckReport", "RewriteResult"}:
            observations.append((chain, title, (directory / "analysis/conditional_responses.md").is_file()))
        if mode == "proposal" and chain == "generation-P" and title == "RewriteResult":
            raise AgentCallError("fixture rewrite rejection")
        if title == "ParameterizationResult":
            if mode == "overlap" and chain == "generation-L":
                unrelated_blocked.set()
                overlap.append(first_tune.wait(timeout=10))
                unrelated_finished.set()
            handoffs.append((chain, (directory / "candidate/source.py").read_bytes(),
                             (directory / "analysis/rewrite_intent.md").is_file()))
            if mode == "parameter" and chain == "generation-P" and sum(h[0] == chain for h in handoffs) == 1:
                raise AgentCallError("fixture parameter rejection")
        number = {"generation-G0": 2, "generation-L": 3, "generation-P": 4}[chain]
        source = f"import triton\nPARAMS={{'z': 7, 'partner': 4}}\n@triton.jit\ndef work(): return {number}\n"
        output_file = "candidate/parameterized.py" if title == "ParameterizationResult" else "child.py"
        if title == "ParameterizationResult":
            source = source.replace("'partner': 4", "'partner': 4, 'STATS_STAGES': 2").replace(f"return {number}", f"return {number + 100}")
        if mode == "helpers":
            source = "import generated_helper\n" + source
        (directory / output_file).parent.mkdir(exist_ok=True)
        (directory / output_file).write_text(source)
        if mode == "helpers":
            (directory / output_file).with_name("generated_helper.py").write_text("VALUE = 23\n")
        answers = {
            "BottleneckReport": {"summary": "fixture", "hypotheses": []},
            "RewriteResult": {"candidates": [{"file": "child.py", "backend": "triton",
                    "change_summary": "fixture", "hypothesis_id": "H1"}]},
            "ParameterizationResult": {"file": output_file, "space": {"params": [
                {"name": "z", "kind": "int", "choices": list(range(80))},
                {"name": "partner", "kind": "int", "choices": [2, 4]},
                {"name": "STATS_STAGES", "kind": "int", "choices": [1, 2]}]}},
        }
        return PromptResult(text="", structured=answers[title], session_id=session_id)

    def worker(self, job, timeout_s, tag, **kwargs):
        assert get_ident() == gpu_thread
        assert Path.cwd() == tmp_path / "run/common"
        admissions.append(clock[0])
        jobs.append((self.jobs_dir, job))
        if mode == "overlap" and "retune" in self.jobs_dir.parts and not first_tune.is_set():
            assert unrelated_blocked.wait(timeout=10)
            overlap.append(not unrelated_finished.is_set())
            first_tune.set()
        if mode == "helpers":
            common = tmp_path / "run/common"
            if "acquisition" in self.jobs_dir.parts:
                assert to_wsl_path(common) in self.cfg.extra_pythonpath
                assert (common / original_helper.name).read_bytes() == original_helper.read_bytes()
            if "retune" in self.jobs_dir.parts:
                staged = self.jobs_dir.parent / "inputs"
                assert (staged / "original_helper.py").read_bytes() == original_helper.read_bytes()
                assert (staged / "generated_helper.py").read_text() == "VALUE = 23\n"
            if "finals" in self.jobs_dir.parts and job.get("num_perf_trials"):
                candidate = Path(job["kernel_src_path"])
                staged = candidate.parent.parent / "inputs" if candidate.parent.name == "report" else common
                assert to_wsl_path(staged) in self.cfg.extra_pythonpath
                assert (staged / "original_helper.py").read_bytes() == original_helper.read_bytes()
        if job["job_type"] == "static_check":
            return {"ok": True}
        if job["job_type"] == "compile_probe":
            paths = [job["kernel_src_path"], *job.get("extra_kernel_src_paths", [])]
            return {"ok": True, "results": {p: {"ok": True, "max_shared": 0} for p in paths}}
        assert job["seed"] == 0
        selected = extract_defaults(Path(job["kernel_src_path"]).read_text())
        value = 4.0 if selected.get("z") == 7 else 8.0
        if mode == "failed_final" and "finals" in self.jobs_dir.parts:
            return {"ok": False, "failure_kind": "correctness_mismatch", "log_tail": "fixture"}
        if "H" in self.jobs_dir.parts and mode == "invalid":
            return {"ok": False, "failure_kind": "correctness_mismatch", "log_tail": "fixture invalid"}
        if mode == "slow" and "retune" in self.jobs_dir.parts:
            value += 10.0
        if mode == "deadline" and "acquisition" in self.jobs_dir.parts:
            clock[0] = 14401.0
        if mode == "tuning_deadline" and sum("retune" in path.parts and j.get("num_perf_trials", 0) > 0
                                             for path, j in jobs) >= 4:
            clock[0] = 14401.0
        if mode == "final_deadline" and "finals" in self.jobs_dir.parts:
            clock[0] = 14401.0
        n = job["num_perf_trials"]
        return {"ok": True, "latency_ms": {"mean": value, "median": value, "min": value,
            "max": value, "std": 0.0, "n": n, "samples": [value] * n}}

    monkeypatch.setattr(OpencodeServer, "start", lambda self: "http://unused.invalid")
    monkeypatch.setattr(OpencodeServer, "stop", lambda self: None)
    monkeypatch.setattr(OpencodeClient, "create_session", lambda self, directory, title: title)
    monkeypatch.setattr(OpencodeClient, "prompt", prompt)
    monkeypatch.setattr(WslGpuWorker, "run_job", worker)
    if mode == "artifact_error":
        from scripts.experiments import c2_retune
        original_copy = c2_retune.shutil.copyfile
        def copy_failure(source, destination):
            copied = original_copy(source, destination)
            if Path(destination).name == "selected.py":
                raise OSError("fixture final artifact copy failure")
            return copied
        monkeypatch.setattr(c2_retune.shutil, "copyfile", copy_failure)
    if mode == "acquisition_error":
        from scripts.experiments.c2_local_inputs import InputError, Responses
        def lost_return(self, shared):
            raise InputError("fixture acquisition envelope failure")
        monkeypatch.setattr(Responses, "validate_for", lost_return)
    reference = tmp_path / "reference.py"
    reference.write_text(shared.reference_source)
    shared_path = tmp_path / "shared.json"
    shared_path.write_text(shared.model_dump_json())
    inputs = protocol.MethodInputs(slot=slot, shared=shared_path, reference=reference,
                                   helpers=(original_helper,) if mode == "helpers" else ())
    # When: the actual core scheduler calls the actual adapters/native tuner.
    result = module.run_core(inputs, cfg, tmp_path / "run", deadline=protocol.Deadline(0, lambda: clock[0]))
    # Then: opportunities, chain counts, acquisitions and full finals are fixed.
    assert all(start < 14400 for start in admissions)
    assert Path.cwd() == initial_cwd and cfg.gpu.concurrency.enabled and cfg.run.seed == 51
    if mode == "acquisition_error":
        assert result.acquisition_calls == 12
        assert result.acquisition_error is not None
        assert all(r.status == "dependency_failed" for r in result.opportunities if r.arm != "G0")
        return
    if mode == "failed_final":
        assert all(r.final_status == "failed" for r in result.opportunities)
        assert method_module("gates").summarize_core([result]).status == "inconclusive"
        return
    if mode == "overlap":
        assert overlap == [True, True]
    if mode in {"tuning_deadline", "final_deadline"}:
        assert all(row.final_status == "censored" for row in result.opportunities)
        assert result.drain_s == 1
        if mode == "tuning_deadline":
            assert all(row.status == "censored" for row in result.opportunities)
        else:
            assert len(result.parent_finals) == 1
        return
    if mode == "deadline":
        assert all(row.status == "censored" for row in result.opportunities)
        assert all("acquisition" in path.parts for path, _ in jobs)
        assert sum(job.get("num_perf_trials", 0) > 0 for _, job in jobs) == 1
        assert result.drain_s == 1
        return
    assert roles.count("BottleneckReport") == roles.count("RewriteResult") == 3
    assert all(present == (chain != "generation-G0") for chain, _, present in observations)
    assert roles.count("ParameterizationResult") == (2 if mode == "proposal" else 4), [r.error for r in result.opportunities]
    ph = [handoff for handoff in handoffs if handoff[0] == "generation-P"]
    if mode != "proposal":
        assert len(ph) == 2 and ph[0][1] == ph[1][1] and [h[2] for h in ph] == [False, True]
        assert len({handoff[1] for handoff in handoffs}) == 3
    assert result.acquisition_calls == 12
    assert all(item.provider_cost is None and item.provider_tokens is None for item in result.costs)
    assert [row.arm for row in result.opportunities] == list(protocol.SLOTS[slot].order)
    by_arm = {row.arm: row for row in result.opportunities}
    failed = {"proposal": {"P", "H"}, "parameter": {"P"}, "invalid": {"H"},
              "artifact_error": {"G0", "L", "P", "H"}}.get(mode, set())
    if mode == "artifact_error":
        assert all(r.status == "failed" and r.error == r.retune.error for r in result.opportunities)
    assert all(row.asked == 40 and row.status == "complete" for row in result.opportunities if row.arm not in failed)
    assert all(row.retune.final_blocks_requested == 0 and not row.retune.finals
               for row in result.opportunities if row.retune is not None)
    assert all(row.selected == ("parent" if row.arm in failed or mode == "slow" else "child")
               and len(row.finals) == 3 for row in result.opportunities)
    if mode == "proposal":
        assert by_arm["P"].status == "failed" and by_arm["H"].status == "dependency_failed"
    if mode == "invalid":
        assert by_arm["H"].status == "invalid" and by_arm["H"].retune.rejection is not None
    assert len(result.parent_finals) == 3
    finals = [job for _, job in jobs if job.get("num_perf_trials") == 100]
    assert len(finals) == 15
    assert all(job["num_correct_trials"] == 5 for job in finals)
    from kernel_optimizer.store.run_store import RunStore
    for row in result.opportunities:
        if row.retune is not None:
            native_store = RunStore.open(tmp_path / "run" / row.arm / "retune")
            import json
            manifest = json.loads((native_store.run_dir / "manifest.json").read_text())
            assert manifest["gpu"]["concurrency"]["enabled"] is True
            assert manifest["inputs"]["space_expansions_per_candidate"] == 0
            assert len([e for e in native_store.iter_events() if e.type == "RETUNE_VALIDATION"]) == 1
            if row.retune.rejection is None:
                accepted = next(e.payload["result"] for e in native_store.iter_events() if e.type == "RETUNE_VALIDATION")
                witnesses = accepted["witnesses"]
                assert len(witnesses) == 2 and witnesses[0]["params"]["values"] == {"z": 7, "partner": 4, "STATS_STAGES": 2}
                assert [t.params.model_dump() for t in row.retune.trials[:2]] == [w["params"] for w in witnesses]
            if row.retune.selected_trial is not None:
                trial = row.retune.selected_trial
                measured = native_store.candidate_dir(trial.candidate_id) / "trials" / f"{trial.trial_id}.py"
                assert (native_store.run_dir / "report/selected.py").read_bytes() == measured.read_bytes()
                assert {t.params.values["z"] for t in row.retune.trials[:2]} >= {7}
    events = RunStore.open(tmp_path / "run/finals").iter_events()
    assert [e.payload["phase"] for e in events if e.type == "LOCAL_EVAL_STARTED"] == [
        f"final_{name}_{block}" for block, order in enumerate(protocol.final_orders(slot)) for name in order]
    native_order = list(dict.fromkeys(path.parent.parent.name for path, _ in jobs if "retune" in path.parts))
    skipped = {"proposal": {"P", "H"}, "parameter": {"P"}}.get(mode, set())
    assert native_order == [arm for arm in protocol.SLOTS[slot].order if arm not in skipped]
