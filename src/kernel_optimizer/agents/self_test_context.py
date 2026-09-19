"""Per-call, evaluation-only context for voluntary formal agent self-tests."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
import hashlib
import json
import os
from pathlib import Path
import shlex
import re
from typing import TYPE_CHECKING, ClassVar, Final, Literal

from pydantic import BaseModel, ConfigDict, Field

from kernel_optimizer.agents.self_test_artifacts import copy_dependencies
from kernel_optimizer.config import EvalConfig, GpuConcurrencyConfig, WslConfig
from kernel_optimizer.evaluation.correctness import CorrectnessEvaluator
from kernel_optimizer.gpu.worker_client import to_wsl_path
from kernel_optimizer.models.core import TaskSpec

if TYPE_CHECKING:
    from kernel_optimizer.agents.sandbox import Sandbox


FORMAL_HELPER_CUTOFF: Final[ContextVar[float | None]] = ContextVar("formal_helper_cutoff", default=None)


@contextmanager
def formal_helper_cutoff(cutoff: float | None) -> Iterator[None]:
    token = FORMAL_HELPER_CUTOFF.set(cutoff)
    try:
        yield
    finally:
        FORMAL_HELPER_CUTOFF.reset(token)


class SelfTestContext(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")

    label: Literal["agent_self_test"] = "agent_self_test"
    task: TaskSpec
    evaluation: EvalConfig
    wsl: WslConfig
    concurrency: GpuConcurrencyConfig
    seed: int
    jobs_dir: Path
    worker_main_path: Path
    configured_worker_python: str
    source_path: str
    reference_dependencies: Path
    input_mode: str = "reference get_init_inputs/get_inputs; shape defined by reference snapshot"
    eval_semantics: str = "unknown; use the reference and formal worker semantics"
    formal_cutoff_unix_s: float | None = Field(default=None, gt=0, allow_inf_nan=False)


def self_test_context_factory(task: TaskSpec, evaluator: CorrectnessEvaluator) -> Callable[[Sandbox], str]:
    """Capture the live evaluator, never an AppConfig or a tuning sampler seed."""
    configured = task.ref_path.resolve()
    cutoff = FORMAL_HELPER_CUTOFF.get()
    expected_hash = task.ref_src_sha.lower() if re.fullmatch(r"[0-9a-fA-F]{64}", task.ref_src_sha) else None

    def seed_context(sb: Sandbox) -> str:
        seeded = sb.root / "task/ref.py"
        sources = [configured, seeded] if task.ref_path.is_absolute() or expected_hash else [seeded]
        if expected_hash is None and not task.ref_path.is_absolute() and configured.is_file() and seeded.is_file():
            if configured.read_text(encoding="utf-8") == seeded.read_text(encoding="utf-8"):
                sources.insert(0, configured)
        source: Path | None = None
        mismatched = False
        for path in sources:
            if not path.is_file():
                continue
            content = path.read_bytes()
            # TaskSpec loaders hash decoded text; the helper snapshot itself hashes exact bytes.
            hashes = {hashlib.sha256(content).hexdigest(),
                      hashlib.sha256(content.decode("utf-8").replace("\r\n", "\n").encode()).hexdigest()}
            if expected_hash is not None and expected_hash not in hashes:
                mismatched = True
                continue
            source = path
            break
        if source is None:
            reason = "reference_hash_mismatch" if mismatched else "reference_unavailable"
            sb.write_input("task/self_test.json", json.dumps({
                "label": "agent_self_test", "available": False, "reason": reason}))
            (sb.root / "task/self_test_usage.md").unlink(missing_ok=True)
            return (f"\n\nOptional formal self-test unavailable ({reason}); capability is unknown. "
                    "This does not invalidate the candidate; bash/private experiments remain allowed.")
        worker = evaluator.worker
        copied = copy_dependencies(source, sb.root / "task/self_test_reference_files")
        reference = sb.write_input("task/self_test_reference.py", copied.read_bytes())
        wsl = worker.cfg.model_copy(deep=True)
        context = SelfTestContext(
            task=task.model_copy(update={"ref_path": Path(to_wsl_path(reference)),
                "ref_src_sha": hashlib.sha256(reference.read_bytes()).hexdigest()}),
            evaluation=evaluator.cfg.model_copy(deep=True), wsl=wsl,
            concurrency=worker.conc.model_copy(deep=True), seed=evaluator.seed,
            jobs_dir=Path(to_wsl_path(worker.jobs_dir)),
            worker_main_path=Path(to_wsl_path(worker.worker_main_path)),
            configured_worker_python=f"{os.path.expanduser(worker.cfg.venv)}/bin/python",
            source_path=to_wsl_path(Path(__file__).resolve().parents[2]),
            reference_dependencies=Path(to_wsl_path(copied.parent)),
            formal_cutoff_unix_s=cutoff,
            eval_semantics=(sb.read_output("task/eval_semantics.md")
                            if sb.exists("task/eval_semantics.md") else "unknown; see reference"),
        )
        sb.write_input("task/self_test.json", context.model_dump_json(indent=2,
            exclude={"formal_cutoff_unix_s"} if cutoff is None else None))
        pythonpath = ":".join(filter(None, [os.path.expanduser(wsl.kernelbench_src),
                                          wsl.extra_pythonpath, context.source_path]))
        command = (f"PYTHONPATH={shlex.quote(pythonpath)} "
                   f"{shlex.quote(context.configured_worker_python)} "
                   "-m kernel_optimizer.agents.self_test --context task/self_test.json "
                   "--candidate PATH --mode quick --output NEWDIR")
        sb.write_input("task/self_test_usage.md", command + "\n")
        return ("\n\nOptional formal self-test (bash/private experiments remain allowed): "
                f"`{command}`. Add `--params PATH` for a full native ParamSet JSON; "
                "use `--mode full` for final reevaluation. Prefer this protocol for comparable "
                "claims. Results are agent_self_test, not native B40 trials or reusable TPE scores.")

    return seed_context
