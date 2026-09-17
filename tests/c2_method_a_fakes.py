"""CPU external boundaries shared by trajectory and terminal protocol tests."""

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from kernel_optimizer.agents.runtime import OpencodeClient, OpencodeServer, PromptResult
from kernel_optimizer.config import AppConfig
from kernel_optimizer.gpu.worker_client import WslGpuWorker, to_wsl_path
from kernel_optimizer.models.core import ParamSet
from kernel_optimizer.paramspace.materializer import extract_defaults
from scripts.experiments.c2_local_inputs import Shared
from tests.test_c2_information_inputs import prepared
from tests.test_c2_method_program import core_rows


@dataclass(frozen=True, slots=True)
class AFixture:
    root: Path
    cfg: AppConfig
    shared: Shared
    roles: list[tuple[int, str, bool]]
    histories: list[list]
    jobs: list[tuple[int, str, int]]
    clock: list[float]
    mode: list[str]

    def wrapper(self, slot: str) -> Path:
        task = "level3:43" if slot in {"A0", "B1"} else "level3:21"
        shared = self.shared.model_copy(update={"task": task})
        snapshot = self.root / f"{slot}-shared.json"
        snapshot.write_text(shared.model_dump_json())
        reference = self.root / "reference.py"
        reference.write_text(shared.reference_source)
        core = []
        for row in core_rows():
            path = self.root / f"core-{slot}-{row.slot}.json"
            path.write_text(row.model_copy(update={"shared_id": shared.identity() if row.task == task else row.shared_id}).model_dump_json())
            core.append(str(path))
        wrapper = self.root / f"{slot}-inputs.json"
        wrapper.write_text(json.dumps({"slot": slot, "shared": str(snapshot), "reference": str(reference),
            "helpers": [], "core_results": core, "source_fidelity": "failed",
            "execution_kind": "bounded_exploratory_confirmation"}))
        return wrapper


@pytest.fixture
def a_fixture(tmp_path, monkeypatch, prepared):
    shared, _, cfg = prepared
    params = {f"x{i}": 1 for i in range(6)}
    data = shared.model_dump()
    data.update(source=f"PARAMS={params!r}\ndef run(): return 1\n", space={"params": [
        {"name": key, "kind": "int", "choices": [1, 2]} for key in params]})
    data["parent"]["params"] = ParamSet(values=params).model_dump()
    data["trials"] = [data["parent"]]
    shared = Shared.model_validate(data)
    fixture = AFixture(tmp_path, cfg, shared, [], [], [], [0.0], ["success"])
    original_write = Path.write_text
    def write_text(path, data, *args, **kwargs):
        if fixture.mode[0] == "native_failure" and path.name == "result.json" and path.parent.name == "retune":
            raise OSError("fixture native result write failed")
        return original_write(path, data, *args, **kwargs)
    monkeypatch.setattr(Path, "write_text", write_text)
    for name in ("analyst", "rewriter", "parameterizer"):
        getattr(cfg.agents, name).max_retries = 0
        getattr(cfg.agents, name).max_transport_retries = 0

    def number(path):
        return next((int(p[-1]) for p in path.parts if p in {"round-1", "round-2"}), 0)

    def prompt(self, session_id, text, **kwargs):
        directory, title = kwargs["directory"], kwargs["schema"]["title"]
        round_id = number(directory)
        fixture.roles.append((round_id, title, (directory / "analysis/conditional_responses.md").exists()))
        if title == "RewriteResult":
            fixture.histories.append(json.loads((directory / "history/failed_hypotheses.json").read_text()))
        prefix, default = ("z", 7) if round_id == 1 else ("w", 9)
        values = {f"{prefix}{i}": default if i == 0 else 1 for i in range(2 if fixture.mode[0] == "short" else 6)}
        if title == "ParameterizationResult":
            assert (directory / "analysis/rewrite_intent.md").is_file()
            output = directory / "candidate/parameterized.py"
            output.write_text(f"import triton\nimport source\nPARAMS={values!r}\n@triton.jit\ndef work(): return source.work()+{round_id}\n")
        else:
            output = directory / "rewrite.py"
            output.write_text("import triton\nPARAMS={'old_axis': 1}\n@triton.jit\ndef work(): return 10\n")
        answers = {
            "BottleneckReport": {"summary": "fixture", "hypotheses": []},
            "RewriteResult": {"candidates": [{"file": "rewrite.py", "backend": "triton", "hypothesis_id": f"H{round_id}", "change_summary": "fixture"}]},
            "ParameterizationResult": {"file": "candidate/parameterized.py", "space": {"params": [
                {"name": key, "kind": "int", "choices": ([default - 1, default] if i == 0 else [1, 2])
                 if fixture.mode[0] == "short" else list(range(80)) if i == 0 else [1, 2]}
                for i, key in enumerate(values)]}},
        }
        return PromptResult(text="", structured=answers[title], session_id=session_id)

    def worker(self, job, timeout_s, tag, **kwargs):
        round_id = number(self.jobs_dir)
        phase = "acquisition" if "acquisition" in self.jobs_dir.parts else "parent" if "parent" in self.jobs_dir.parts else "child"
        if job["job_type"] == "static_check":
            return {"ok": True}
        if job["job_type"] == "compile_probe":
            paths = [job["kernel_src_path"], *job.get("extra_kernel_src_paths", [])]
            return {"ok": True, "results": {p: {"ok": True, "max_shared": 0} for p in paths}}
        assert job["seed"] == 0
        n = job["num_perf_trials"]
        fixture.jobs.append((round_id, phase, n))
        if n == 100:
            assert job["num_correct_trials"] == 5
        if fixture.mode[0] == "deadline":
            fixture.clock[0] = 9001
        if fixture.mode[0] == "invalid" and round_id == 1 and phase == "child":
            return {"ok": False, "failure_kind": "correctness_mismatch", "log_tail": "fixture"}
        params = extract_defaults(Path(job["kernel_src_path"]).read_text())
        if round_id == 0:
            arm = Path(job["kernel_src_path"]).parent.name
            other = "H" if arm == "G0" else "G0"
            assert to_wsl_path(self.jobs_dir.parent / other / "imports") not in self.cfg.extra_pythonpath
        value = 3.0 if "w0" in params else 4.0 if "z0" in params else 5.0
        if fixture.mode[0] == "fresh_baseline" and round_id == 1 and phase == "parent":
            value = 2.0
        if fixture.mode[0] == "slow" and round_id == 1 and phase == "child":
            value = 8.0
        if fixture.mode[0] == "final_metadata" and n == 100:
            return {"ok": True, "latency_ms": {"mean": value, "min": value, "max": value, "std": 0.0, "n": n}}
        return {"ok": True, "latency_ms": {"mean": value, "median": value, "min": value, "max": value,
            "std": 0.0, "n": n, "samples": [value] * n}}

    monkeypatch.setattr(OpencodeServer, "start", lambda self: "http://unused.invalid")
    monkeypatch.setattr(OpencodeServer, "stop", lambda self: None)
    monkeypatch.setattr(OpencodeClient, "create_session", lambda self, directory, title: title)
    monkeypatch.setattr(OpencodeClient, "prompt", prompt)
    monkeypatch.setattr(WslGpuWorker, "run_job", worker)
    return fixture
