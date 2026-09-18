"""CPU provider/worker boundaries for the fixed two-wave opportunity program."""

import importlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pytest

from kernel_optimizer.agents.runtime import AgentCallError, OpencodeClient, OpencodeServer, PromptResult
from kernel_optimizer.config import AppConfig
from kernel_optimizer.gpu.worker_client import WslGpuWorker
from kernel_optimizer.models.core import ParamSet
from kernel_optimizer.paramspace.materializer import extract_defaults
from scripts.experiments.c2_local_inputs import Shared
from tests.test_c2_information_inputs import prepared


def opportunity_module(name: str = "program"):
    assert (Path(__file__).parents[1] / "scripts/experiments" / f"c2_opportunity_{name}.py").is_file()
    return importlib.import_module(f"scripts.experiments.c2_opportunity_{name}")


@dataclass(frozen=True, slots=True)
class CampaignFixture:
    root: Path
    cfg: AppConfig
    parents: dict[str, Shared]
    clock: list[float]
    mode: list[str]
    roles: list[tuple[str, bool, bool]]
    jobs: list[tuple[str, int]]
    active: list[bool]

    def inputs(self, wave: int, slot: str):
        task = "level3:43" if (wave == 1 and slot.startswith("A")) or (wave == 2 and slot.startswith("B")) else "level3:21"
        return opportunity_module("inputs").OpportunityInputs(wave=wave, slot=slot,
            shared=self.root / task.replace(":", "-") / "shared.json",
            reference=self.root / task.replace(":", "-") / "reference.py")

    def run(self, wave: int, slot: str):
        module = opportunity_module()
        return module.run_opportunity(self.inputs(wave, slot), opportunity_module("inputs").CampaignRun(
            self.cfg, self.root / f"wave{wave}" / slot, 15400,
            now=lambda: self.clock[0], clock=lambda: 900000 + self.clock[0]))


@pytest.fixture
def campaign(tmp_path, monkeypatch, prepared):
    base, _, cfg = prepared
    cfg.gpu.compile_screen_enabled = False
    for name in ("analyst", "rewriter", "parameterizer"):
        getattr(cfg.agents, name).max_retries = 0
        getattr(cfg.agents, name).max_transport_retries = 0
    parents = {}
    selected = {f"k{i}": 2 for i in range(6)}
    defaults = {key: 0 for key in selected}
    for task, candidate, trial in (("level3:43", "cand-b67a1cb4", "tr-6fb84baa"),
                                    ("level3:21", "cand-4c96b8c4", "tr-6afd5e56")):
        data = base.model_dump()
        data.update(task=task, source=f"PARAMS={defaults!r}\ndef run(): return 1\n",
            space={"params": [{"name": k, "kind": "int", "choices": [0, 2, 4]} for k in selected]})
        data["parent"].update(candidate_id=candidate, trial_id=trial, params=ParamSet(values=selected).model_dump(),
                              latency_ms={"mean": 100, "median": 100, "min": 100, "max": 100, "std": 0, "n_samples": 20})
        data["trials"] = [data["parent"]]
        shared = Shared.model_validate(data)
        folder = tmp_path / task.replace(":", "-")
        folder.mkdir()
        (folder / "shared.json").write_text(shared.model_dump_json(), encoding="utf-8")
        (folder / "reference.py").write_text(shared.reference_source, encoding="utf-8")
        parents[task] = shared
    fixture = CampaignFixture(tmp_path, cfg, parents, [1000.0], ["expand"], [], [], [])
    full_counts = Counter()

    def start(self):
        fixture.active.append(True)
        return "http://unused.invalid"

    def prompt(self, session_id, text, **kwargs):
        folder, title = kwargs["directory"], kwargs["schema"]["title"]
        assert fixture.active and not self._http.is_closed
        assert any((p / "responses.json").is_file() for p in folder.parents)
        expansion = "retune" in folder.parts
        fresh = (folder / "analysis/conditional_responses.md").exists()
        fixture.roles.append((title, expansion, fresh))
        if fixture.mode[0] == "service_error":
            raise AgentCallError("fixture service failure")
        if title in {"BottleneckReport", "RewriteResult"}:
            point = json.loads((folder / "tuning/selected_params.json").read_text())
            assert point == {"values": selected}
            path = folder / ("candidate/source.py" if title == "BottleneckReport" else "candidate/best.py")
            assert extract_defaults(path.read_text()) == (defaults if title == "BottleneckReport" else selected)
        source = "import triton\nPARAMS={'x': 7}\n@triton.jit\ndef work(): return PARAMS['x']\n"
        if expansion:
            source = source.replace("'x': 7", "'x': 59")
            if fixture.mode[0] == "changed":
                source = source.replace("return PARAMS['x']", "return PARAMS['x'] + 100")
        file = "candidate/parameterized.py" if title == "ParameterizationResult" else "rewrite.py"
        (folder / file).write_text(source, encoding="utf-8")
        if title == "ParameterizationResult":
            assert (folder / "analysis/rewrite_intent.md").is_file()
        answers = {
            "BottleneckReport": {"summary": "fixture", "hypotheses": []},
            "RewriteResult": {"candidates": [{"file": file, "backend": "triton", "hypothesis_id": "H1", "change_summary": "fixture"}]},
            "ParameterizationResult": {"file": file, "space": {"params": [{"name": "wrong" if expansion and fixture.mode[0] == "reject" else "x",
                "kind": "int", "choices": [-1, *range(60)] if expansion else list(range(60))}]}},
        }
        return PromptResult(text="", structured=answers[title], session_id=session_id)

    def worker(self, job, timeout_s, tag, **kwargs):
        if job["job_type"] == "static_check":
            return {"ok": True}
        assert job["seed"] == 0 and fixture.clock[0] < 15400
        path = Path(job["kernel_src_path"])
        params = extract_defaults(path.read_text(encoding="utf-8"))
        heldout = "heldout" in self.jobs_dir.parts
        acquisition = "acquisition" in self.jobs_dir.parts
        confirmation = "confirmation" in self.jobs_dir.parts
        role = path.parent.name if heldout else "acquisition" if acquisition else "confirmation" if confirmation else "screen" if job["num_perf_trials"] == 100 else "tuning"
        fixture.jobs.append((role, job["num_perf_trials"]))
        if job["num_perf_trials"] == 100:
            assert job["num_correct_trials"] == 5
        if acquisition:
            assert sum(params[k] != selected[k] for k in selected) <= 1
            if params["k0"] == 4:
                return {"ok": False, "failure_kind": "runtime_error", "log_tail": "fixture endpoint failure"}
        if fixture.mode[0] == "invalid" and "retune" in self.jobs_dir.parts:
            return {"ok": False, "failure_kind": "correctness_mismatch", "log_tail": "fixture"}
        if fixture.mode[0] == "heldout_deadline" and heldout:
            fixture.clock[0] = 15401
        if fixture.mode[0] == "heldout_invalid" and heldout and role == "C2":
            return {"ok": False, "failure_kind": "correctness_mismatch", "log_tail": "fixture"}
        value = 100.0 if "k0" in params else 10.0 + int(params["x"])
        if fixture.mode[0] == "flat" and "x" in params:
            value = 10.0
        if "+ 100" in path.read_text(encoding="utf-8"):
            value += 100
        if job["num_perf_trials"] == 100:
            value = {"parent": 101.0, "G0": 90.0, "C2": 80.0}[role] if heldout else 100.0 if "k0" in params else 90.0
            if fixture.mode[0] == "confirm" and not heldout:
                value = 101.0 if "k0" in params else 100.0
                if not confirmation:
                    value = 100.0 + full_counts[self.jobs_dir] if "k0" in params else 100.5
                    full_counts[self.jobs_dir] += 1
        n = job["num_perf_trials"]
        return {"ok": True, "latency_ms": {"mean": value, "median": value, "min": value, "max": value,
            "std": 0.0, "n": n, "samples": [value] * n}}

    monkeypatch.setattr(OpencodeServer, "start", start)
    monkeypatch.setattr(OpencodeServer, "stop", lambda self: fixture.active.pop())
    monkeypatch.setattr(OpencodeClient, "create_session", lambda self, directory, title: title)
    monkeypatch.setattr(OpencodeClient, "prompt", prompt)
    monkeypatch.setattr(WslGpuWorker, "run_job", worker)
    return fixture
