"""Composition root: config -> concrete instances -> Orchestrator wiring."""

from __future__ import annotations

from pathlib import Path

from kernel_optimizer.agents.modules import (
    BottleneckAnalystAgent,
    CandidateGeneratorAgent,
    NoveltyGeneratorAgent,
    ParameterizerAgent,
    RepairAgent,
    StructureRewriterAgent,
)
from kernel_optimizer.agents.runtime import OpencodeClient, OpencodeServer
from kernel_optimizer.agents.sandbox import PermissionAutoResponder, SandboxFactory
from kernel_optimizer.config import AppConfig
from kernel_optimizer.control.convergence import ConvergencePolicy
from kernel_optimizer.control.families import FamilyManager
from kernel_optimizer.control.orchestrator import Orchestrator, Wiring
from kernel_optimizer.evaluation.benchmark import Benchmarker
from kernel_optimizer.evaluation.correctness import CorrectnessEvaluator
from kernel_optimizer.evaluation.profilerx import LightProfiler
from kernel_optimizer.gpu.worker_client import WslGpuWorker
from kernel_optimizer.models.core import TaskSpec
from kernel_optimizer.paramspace.validation import SpaceValidator
from kernel_optimizer.store.run_store import RunStore
from kernel_optimizer.tasks.kernelbench import KernelBenchAdapter
from kernel_optimizer.tuning.stats import TuningStatsAnalyzer


class Runtime:
    """Owns the opencode server + client lifecycle around an orchestrated run."""

    def __init__(self, cfg: AppConfig, log_dir: Path | None = None):
        self.cfg = cfg
        log_path = (log_dir / "opencode-server.log") if log_dir else None
        self.server = OpencodeServer(cfg.opencode, log_path=log_path)
        self.client: OpencodeClient | None = None
        self.responder: PermissionAutoResponder | None = None

    def __enter__(self) -> "Runtime":
        base_url = self.server.start()
        self.client = OpencodeClient(base_url, timeout_s=self.cfg.opencode.request_timeout_s)
        if self.cfg.opencode.permission_mode == "sse_auto_approve":
            self.responder = PermissionAutoResponder(base_url)
            self.responder.start()
        return self

    def __exit__(self, *exc) -> None:
        if self.responder:
            self.responder.stop()
        if self.client:
            self.client.close()
        self.server.stop()


def build_gpu_stack(cfg: AppConfig, store: RunStore):
    worker = WslGpuWorker(cfg.wsl, cfg.gpu.concurrency, jobs_dir=store.run_dir / "jobs")
    evaluator = CorrectnessEvaluator(worker, cfg.evaluation, cfg.gpu.concurrency,
                                     seed=cfg.run.seed)
    benchmarker = Benchmarker(worker, evaluator, cfg.evaluation)
    profiler = LightProfiler()
    return worker, evaluator, benchmarker, profiler


def build_orchestrator(cfg: AppConfig, store: RunStore, task: TaskSpec,
                       runtime: Runtime) -> Orchestrator:
    assert runtime.client is not None
    _, evaluator, benchmarker, profiler = build_gpu_stack(cfg, store)
    validator = SpaceValidator(evaluator, cfg.device, cfg.evaluation, seed=cfg.run.seed)
    stats_analyzer = TuningStatsAnalyzer(cfg.device)
    families = FamilyManager(
        max_families_active=cfg.budgets.max_families_active,
        max_families_total=cfg.budgets.max_families_total,
        max_families_total_hard=cfg.budgets.max_families_total_hard,
    )
    convergence = ConvergencePolicy(cfg.budgets)
    sandboxes = SandboxFactory(store.run_dir / "sandboxes")

    def agent(cls, name: str):
        module_cfg = cfg.agents.module(name)
        return cls(runtime.client, sandboxes, store, module_cfg,
                   agent_name=cfg.opencode.agent)

    deps = Wiring(
        evaluator=evaluator,
        benchmarker=benchmarker,
        profiler=profiler,
        validator=validator,
        stats_analyzer=stats_analyzer,
        families=families,
        convergence=convergence,
        generator=agent(CandidateGeneratorAgent, "generator"),
        parameterizer=agent(ParameterizerAgent, "parameterizer"),
        analyst=agent(BottleneckAnalystAgent, "analyst"),
        rewriter=agent(StructureRewriterAgent, "rewriter"),
        novelty=agent(NoveltyGeneratorAgent, "novelty"),
        repair=agent(RepairAgent, "repair"),
    )
    return Orchestrator(deps, cfg, store, task)


def load_task(cfg: AppConfig, level: int, problem_id: int) -> TaskSpec:
    adapter = KernelBenchAdapter(Path(cfg.kernelbench_root))
    return adapter.load(level, problem_id)
