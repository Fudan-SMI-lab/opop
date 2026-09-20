"""Framework seals, static source admission and byte-preserved candidate artifacts."""

import ast
import inspect
import json
from pathlib import Path

from pydantic import Field

from kernel_optimizer.agents.model_operator_rewriter import ModelOperatorRewriterAgent, ModelOperatorRewriteResult
from kernel_optimizer.config import AppConfig
from kernel_optimizer.models.core import DeviceLimits, ParamSet, sha256_text
from kernel_optimizer.paramspace.guard import check_config
from .device_records import LocalRequest, SourceEligibility
from .fast_prepare import FastPrepared
from .fast_records import Artifact, Framework, TargetSpec
from .manual_data import fingerprint
from .model_binding import load_bundle
from .runner_records import FrozenRecord, RunnerError
from .search_bundle import parameter_space, validate_child


class GateRecord(FrozenRecord):
    eligible: bool
    bundle_sha256: str = ""
    params_sha256: str = ""
    source_hashes: dict[str, str] = Field(default_factory=dict)
    detail: str | None = None
    device_proof_status: str = "not_run"


def framework_id(framework: Framework) -> str:
    return sha256_text(framework.model_dump_json())


def identify_framework(revision: str, cfg: AppConfig, prepared: FastPrepared) -> Framework:
    root = Path(__file__).resolve().parents[2]
    names = ("fast_prepare", "fast_model", "local_eval", "local_device", "device_gate", "device_capture",
             "device_profile", "device_records", "device_ir", "cuda_trace", "triton_observer", "pure_tensors",
             "model_runner", "model_binding", "binding_runtime", "runner_quality", "runner_measure", "search_helper")
    files = {Path(__file__).with_name(name + ".py") for name in names}
    files.update(Path(__file__).parent.glob("fast_*.py"))
    files.update(root / "src/kernel_optimizer" / name for name in (
        "agents/model_operator_rewriter.py", "agents/task_rewriter.py", "agents/base.py",
        "models/device_operator.py", "wiring.py", "tuning/tpe.py"))
    hashes = {p.relative_to(root).as_posix(): sha256_text(p.read_text(encoding="utf-8")) for p in sorted(files)}
    return Framework(revision=revision, implementation_sha256=sha256_text(json.dumps(hashes, sort_keys=True)),
        schema_sha256=sha256_text(json.dumps(ModelOperatorRewriteResult.model_json_schema(), sort_keys=True)),
        prompt_sha256=sha256_text(inspect.getsource(ModelOperatorRewriterAgent.render_prompt)),
        agent_config_sha256=sha256_text(json.dumps({"module": cfg.agents.module("rewriter").model_dump(mode="json"),
            "agent": cfg.opencode.agent, "timeout_s": cfg.opencode.request_timeout_s}, sort_keys=True)),
        contract_sha256=prepared.task.contract_sha256)


def source_gate(request: LocalRequest, baseline: Path, target: TargetSpec) -> GateRecord:
    bundle_sha, params_sha, hashes = "", "", {}
    try:
        bundle = load_bundle(request.bundle_path, request.params)
        bundle_sha, params_sha, hashes = bundle.bundle_sha256, bundle.params_sha256, dict(bundle.source_hashes)
        validate_child(bundle, baseline)
        if request.site_id != target.brief.site_id or tuple(s.site_id for s in bundle.document.sites) != (target.brief.site_id,):
            raise RunnerError("candidate scope differs from the one selected pure operator")
        params = ParamSet.model_validate({"values": request.params})
        rejected = check_config(parameter_space(bundle), params, DeviceLimits())
        if rejected:
            raise RunnerError(f"invalid effective params: {rejected.reason}: {rejected.detail}")
        for kernel in request.kernels:
            if kernel.backend != "triton" or not kernel.output_arg_indices:
                raise RunnerError("current adapter requires Triton and explicit direct returned-output operands")
            if kernel.source_file not in bundle.sources:
                raise RunnerError("kernel source is outside the declared bundle")
            tree = ast.parse(bundle.sources[kernel.source_file].decode("utf-8"))
            if not any(isinstance(n, ast.FunctionDef) and n.name == kernel.entry.split(".")[-1] for n in ast.walk(tree)):
                raise RunnerError("declared kernel entry has no authored function definition in its source")
        return GateRecord(eligible=True, bundle_sha256=bundle_sha, params_sha256=params_sha, source_hashes=hashes)
    except (OSError, ValueError, RuntimeError, SyntaxError, TypeError, ArithmeticError) as exc:
        return GateRecord(eligible=False, bundle_sha256=bundle_sha, params_sha256=params_sha,
                          source_hashes=hashes, detail=f"{type(exc).__name__}: {exc}")


def preserve_bundle(path: Path, destination: Path) -> Path:
    bundle = load_bundle(path, {})
    destination.mkdir(parents=True, exist_ok=False)
    for name, source in bundle.sources.items():
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source)
    result = destination / "bundle.json"
    result.write_bytes(bundle.path.read_bytes())
    return result.resolve()


def write_artifact(artifact: Artifact, destination: Path) -> Artifact:
    current = load_bundle(artifact.bundle, artifact.params.values)
    if (current.bundle_sha256, dict(current.source_hashes), current.params_sha256) != (
            artifact.bundle_sha256, artifact.source_hashes, artifact.params_sha256):
        raise RunnerError("candidate bytes/params changed after evaluation")
    path = preserve_bundle(artifact.bundle, destination)
    result = artifact.model_copy(update={"bundle": path})
    (destination / "artifact.json").write_text(result.model_dump_json(indent=2), encoding="utf-8")
    return result


def read_artifact(path: Path) -> Artifact:
    artifact = Artifact.model_validate_json(path.read_bytes())
    bundle = load_bundle(path.parent / "bundle.json", artifact.params.values)
    if (bundle.bundle_sha256, dict(bundle.source_hashes), bundle.params_sha256) != (
            artifact.bundle_sha256, artifact.source_hashes, artifact.params_sha256):
        raise RunnerError("frozen exported candidate identity mismatch")
    return artifact.model_copy(update={"bundle": bundle.path})
