"""Small provider-boundary captures; no provider, GPU, or Git execution."""

from contextlib import closing
from pathlib import Path
import shlex
from typing import Final, Literal, assert_never, override

import pytest
from pydantic import BaseModel, ConfigDict, Field, JsonValue, TypeAdapter

from kernel_optimizer import wiring
from kernel_optimizer.agents.modules import RewriterInputs
from kernel_optimizer.agents.runtime import OpencodeClient, PromptResult
from kernel_optimizer.agents.task_rewriter import TaskRewriteInputs
from kernel_optimizer.config import AppConfig
from kernel_optimizer.control.direct_task import TaskSpace
from kernel_optimizer.evaluation.benchmark import Benchmarker
from kernel_optimizer.evaluation.correctness import CorrectnessEvaluator
from kernel_optimizer.evaluation.profilerx import LightProfiler
from kernel_optimizer.gpu.worker_client import WslGpuWorker, to_wsl_path
from kernel_optimizer.models.core import ParamSet, TaskSpec, sha256_text
from kernel_optimizer.models.reports import BottleneckReport
from kernel_optimizer.store.run_store import RunStore
from kernel_optimizer.tuning.objective import Objective

SOURCE: Final = "PARAMS = {'tile': 2}\ndef run(x): return x * 2\n"
REFERENCE: Final = "def reference(x): return x * 2\n"
FIXTURES: Final = Path(__file__).parent / "fixtures/c2_contracts"
Route = Literal["generic", "bundle", "legacy"]


@pytest.fixture(autouse=True)
def cpu_stack(monkeypatch: pytest.MonkeyPatch) -> None:
    def stack(cfg: AppConfig, store: RunStore) -> tuple[WslGpuWorker, CorrectnessEvaluator, Benchmarker, LightProfiler]:
        worker = WslGpuWorker(cfg.wsl, cfg.gpu.concurrency, jobs_dir=store.run_dir / "jobs")
        evaluator = CorrectnessEvaluator(worker, cfg.evaluation, cfg.gpu.concurrency, seed=cfg.run.seed)
        return worker, evaluator, Benchmarker(worker, evaluator, cfg.evaluation), LightProfiler()

    def reject_job(*args: JsonValue, **kwargs: JsonValue) -> None:
        pytest.fail("T1 must not submit a GPU job")

    monkeypatch.setattr(wiring, "build_gpu_stack", stack)
    monkeypatch.setattr(WslGpuWorker, "run_job", reject_job)


class Captured(Exception):
    """Deliberate stop after recording the real provider arguments."""


class RequestCapture(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    prompt: str
    output_schema: dict[str, JsonValue]
    files: dict[str, str]
    model: str
    agent: str
    sequence: tuple[str, ...]
    normalization_map: dict[str, str] = Field(default_factory=dict, exclude=True)


class RecordingProvider(OpencodeClient):
    """Accumulate requests, stopping before any external provider operation."""

    def __init__(self, root: Path) -> None:
        super().__init__("http://127.0.0.1:0")
        self.root = root
        self.requests: list[RequestCapture] = []
        self.sequence: list[str] = []

    @override
    def create_session(self, directory: Path, title: str) -> str:
        self.sequence.append("create_session")
        return title

    @override
    def prompt(self, session_id: str, text: str, *, model: str, agent: str = "build",
               schema: dict[str, JsonValue] | None = None, directory: Path | None = None,
               system: str | None = None) -> PromptResult:
        # The parameter list intentionally matches the existing provider API.
        assert directory is not None and schema is not None
        self.sequence.append("prompt")
        mappings = ((str(directory.resolve()), "<SANDBOX>"), (directory.resolve().as_posix(), "<SANDBOX>"),
                    (str(directory), "<SANDBOX>"), (directory.as_posix(), "<SANDBOX>"),
                    (str(self.root.resolve()), "<CASE>"), (self.root.resolve().as_posix(), "<CASE>"),
                    (str(self.root), "<CASE>"), (self.root.as_posix(), "<CASE>"),
                    (str(Path(wiring.__file__).resolve().parents[1]), "<SOURCE>"),
                    (Path(wiring.__file__).resolve().parents[1].as_posix(), "<SOURCE>"))
        source_argument = "/fixture/kernelbench/src:" + to_wsl_path(Path(wiring.__file__).resolve().parents[1])
        replacements = {shlex.quote(source_argument): shlex.quote("/fixture/kernelbench/src:<SOURCE>")}
        for original, replacement in mappings:
            replacements[original.replace("\\", "\\\\")] = replacement
            replacements[original] = replacement
            if len(original) > 2 and original[1] == ":":
                wsl = "/mnt/" + original[0].lower() + original[2:].replace("\\", "/")
                replacements[wsl] = replacement
                replacements[wsl.replace("/", "\\").replace("\\", "\\\\")] = replacement
                replacements[wsl.replace("/", "\\")] = replacement

        def normalized(value: str) -> str:
            for original, replacement in replacements.items():
                value = value.replace(original, replacement)
            return value

        files = {path.relative_to(directory).as_posix(): normalized(path.read_text(encoding="utf-8"))
                 for path in sorted(directory.rglob("*")) if path.is_file()}
        self.requests.append(RequestCapture(prompt=normalized(text), output_schema=schema,
            files=files, model=model, agent=agent, sequence=tuple(self.sequence), normalization_map=replacements))
        raise Captured


def request_at(root: Path) -> TaskRewriteInputs:
    root.mkdir(parents=True, exist_ok=True)
    (root / "parent.py").write_text(SOURCE, encoding="utf-8")
    (root / "reference.py").write_text(REFERENCE, encoding="utf-8")
    return TaskRewriteInputs(project_root=root, candidate_id="parent", candidate_path=root / "parent.py",
        source_paths=(Path("reference.py"),), goal="Minimize arbitrary J while preserving outputs",
        context={"cases": [{"shape": [2, 4], "dtype": "float32"}],
                 "history": [{"params": {"tile": 2}, "score": 4.0}]},
        objective=Objective(direction="minimize", label="J", unit="ms"),
        params=ParamSet(values={"tile": 2}), space=TaskSpace.model_validate({"params": [
            {"name": "tile", "kind": "int", "choices": [1, 2, 4]}]}))


def capture_at(root: Path, route: Route) -> RequestCapture:
    inputs = request_at(root)
    cfg = AppConfig()
    cfg.wsl.venv = "/fixture/venv"
    cfg.wsl.kernelbench_src = "/fixture/kernelbench/src"
    store = RunStore(root / "run")
    with closing(RecordingProvider(root)) as provider:
        runtime = wiring.Runtime(cfg)
        runtime.client = provider
        match route:
            case "legacy":
                task = TaskSpec(level=3, problem_id=21, name="finite-fixture",
                    ref_path=root / "reference.py", ref_src_sha=sha256_text(REFERENCE))
                agent = wiring.build_orchestrator(cfg, store, task, runtime).deps.rewriter
                legacy = RewriterInputs(task=task, best_source=SOURCE,
                    report=BottleneckReport(summary="finite fixture", hypotheses=[]),
                    failed_hypotheses=[{"id": "H0", "outcome": "no_gain"}], device=cfg.device,
                    n_candidates=1, reference_source=REFERENCE, selected_params=inputs.params,
                    source_materialized=True, eval_semantics={"training": True})
                with pytest.raises(Captured):
                    agent.invoke(legacy)
            case "generic" | "bundle":
                if route == "bundle":
                    inputs = inputs.model_copy(update={"bundle_sources": {"operators.py": SOURCE},
                        "bundle_document": {"entry": "operators.py", "files": ["operators.py"], "helpers": []}})
                with pytest.raises(Captured):
                    wiring.build_task_rewriter(cfg, store, runtime).invoke(inputs)
            case unreachable:
                assert_never(unreachable)
        return provider.requests[0]


def load_anchor(name: str) -> RequestCapture:
    return RequestCapture.model_validate_json((FIXTURES / f"{name}.json").read_bytes())


def structured_files(capture: RequestCapture) -> dict[str, JsonValue]:
    """Compare machine inputs; complete prose files remain reviewable capture evidence."""
    adapter = TypeAdapter(JsonValue)
    return {name: adapter.validate_json(text) for name, text in capture.files.items() if name.endswith(".json")}
