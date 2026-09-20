"""Numerical CPU device/provider boundaries for the real fast workflow, never native proof."""

import json
from contextlib import closing
from pathlib import Path
from time import time

from kernel_optimizer.agents.runtime import AgentCallError, OpencodeClient, PromptResult
from kernel_optimizer.config import AppConfig
from kernel_optimizer.store.run_store import RunStore
from kernel_optimizer.wiring import Runtime, build_task_rewriter
from examples.c3_qwen3.device_profile import ProfileRequest, profile_goal
from examples.c3_qwen3.device_records import LocalRuntime, StageIdentity
from examples.c3_qwen3.fast_artifacts import framework_id, identify_framework
from examples.c3_qwen3.fast_budget import BudgetSpec, StageBudget
from examples.c3_qwen3.fast_evaluation import EvaluationFiles, FastEvaluator
from examples.c3_qwen3.fast_generation import DraftGenerator, GenerationContext
from examples.c3_qwen3.fast_prepare import FastAccess, prepare_fast
from examples.c3_qwen3.fast_target import build_target
from examples.c3_qwen3.fast_workflow import scoped_config
from examples.c3_qwen3.model_binding import load_bundle
from examples.c3_qwen3.search_bundle import baseline_bundle
from tests.c3_device_fakes import Device, Trace, case, tensor_backend


class CudaBoundary(Trace):
    def records(self):
        return {"site_kernels": [{"scope": s, "kernel": "numerical-device-boundary", "cuda_duration_us": 2.0}
                                 for s in self.regions],
                "cuda_intervals": [{"kernel": "numerical-device-boundary", "start_us": 0, "end_us": 2},
                                   {"kernel": "other-work", "start_us": 2, "end_us": 10}]}


class StableDevice(Device):
    def time_us(self, operation):
        operation()
        return 5.0


class Provider(OpencodeClient):
    def __init__(self, mode="good"):
        super().__init__("http://127.0.0.1:0")
        self.mode, self.requests, self.sessions = mode, [], []

    def create_session(self, directory, title):
        self.sessions.append(title)
        return title

    def prompt(self, session_id, text, **kwargs):
        folder = kwargs["directory"]
        data = json.loads((folder / "analysis/task_response.json").read_text())
        self.requests.append(data)
        if self.mode == "transport":
            raise AgentCallError("CPU fixture transport failure")
        context = data["context"]
        wrong = self.mode == "wrong_then_repair" and context["repair_feedback"] is None
        source = ("def replace(site, call, params):\n    return call.args[0] * " + ("3" if wrong else "2") +
                  "\ndef kernel(x, output):\n    return None\n")
        (folder / "operators.py").write_bytes(source.encode())
        site = context["operator_brief"]["site_id"]
        bundle = {**data["bundle_document"], "parent_bundle": context["parent_bundle_sha256"],
                  "sites": [{"site_id": site, "replacement_callable": "replace"}]}
        (folder / "bundle.json").write_text(json.dumps(bundle))
        return PromptResult(text="", session_id=session_id, structured={"candidate_file": "operators.py",
            "bundle_file": "bundle.json", "site_groups": {site: context["operator_brief"]["module_paths"]},
            "device_kernels": [{"backend": "triton", "source_file": "operators.py", "entry": "kernel", "output_arg_indices": [1]}]})


def workflow_case(root: Path, *, mode="good", phase="development", revision="F0", budget_path=None, target=None):
    root.mkdir(parents=True, exist_ok=True)
    runtime, _, _ = case(root)
    tensor_backend(runtime)
    runner = runtime.runner
    setup = Path(__file__).parents[1] / "results/c3-fast-kernel-iteration/setup"
    prepared = prepare_fast(setup / "contract.json", setup / "preflight.json",
        FastAccess(profile="c3_fast_device", phase=phase))
    runner.prepared = prepared.task
    runner.binding.restore()
    runner.binding.sites.clear()
    runner.binding.fixtures.clear()
    runner.binding.fixture_bytes = 0
    baseline = baseline_bundle(root / "baseline")
    runner.bind(load_bundle(baseline, {}))
    device = StableDevice()
    cfg = scoped_config(AppConfig())
    frame = identify_framework(revision, cfg, prepared)
    stage = StageIdentity(profile="c3_fast_device", stage=phase, framework_id=framework_id(frame))
    if target is None:
        setup_stage = stage.model_copy(update={"stage": "setup"})
        setup_budget = StageBudget(root / "setup-budget.json", BudgetSpec(lane="setup", stage=setup_stage,
            started_unix_s=time()-1, deadline_unix_s=time()+60))
        profile = profile_goal(ProfileRequest(setup_stage, "single"), LocalRuntime(runner, device, setup_budget), trace_factory=CudaBoundary)
        target = build_target(profile, "scale")
    oracle = runner.create_oracles(("calibration-calibration-00", "calibration-calibration-01"))
    started = time()-1
    spec = BudgetSpec(lane=phase, stage=stage, started_unix_s=started, deadline_unix_s=started+5400)
    if budget_path is not None:
        from examples.c3_qwen3.fast_budget import BudgetState
        spec = BudgetState.model_validate_json(budget_path.read_bytes()).spec
    budget = StageBudget(budget_path or root / "budget.json", spec)
    evaluator = FastEvaluator(LocalRuntime(runner, device, budget), target, EvaluationFiles(baseline, oracle, root / "evaluation"))
    provider = Provider(mode)
    store = RunStore.create(root, "agent", {})
    service = Runtime.__new__(Runtime)
    service.client = provider
    agent = build_task_rewriter(cfg, store, service, execution_profile="model_operator")
    generator = DraftGenerator(agent, evaluator, GenerationContext(prepared, frame, "Improve this actual operator's full-model goal."))
    return generator, provider, frame, target
