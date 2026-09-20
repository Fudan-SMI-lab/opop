"""Build a narrow operator pack from observed CUDA scopes, never from goal-name heuristics."""

import ast
import json
from collections.abc import Mapping
from pathlib import Path
from statistics import median

from pydantic import JsonValue

from kernel_optimizer.agents.model_operator_rewriter import OperatorBrief, RepresentativeCall, TensorArgument
from kernel_optimizer.models.core import sha256_text
from .device_records import LocalReport, TensorDescription
from .manual_data import fingerprint
from .runner_records import FrozenRecord, GoalId, RunnerError
from .fast_records import TargetSpec


class ProfileCall(FrozenRecord):
    scope: str
    module_path: str
    phase: str
    decode_step: int
    tensors: tuple[TensorDescription, ...]
    weights: tuple[TensorDescription, ...]
    module_attributes: dict[str, JsonValue]
    source_file: str | None
    source: str
    module_source: str


class KernelLink(FrozenRecord):
    scope: str
    kernel: str
    cuda_duration_us: float


class CudaInterval(FrozenRecord):
    kernel: str
    start_us: float
    end_us: float


class ProfileData(FrozenRecord):
    valid: bool
    goal: GoalId
    trace_windows: int
    calls: tuple[ProfileCall, ...]
    site_kernels: tuple[KernelLink, ...]
    cuda_intervals: tuple[CudaInterval, ...]


class DeviceProfile(FrozenRecord):
    data: ProfileData


def canonical_site(paths: tuple[str, ...]) -> str:
    return "pure-" + sha256_text(json.dumps(sorted(paths), separators=(",", ":")))


def family_key(call: ProfileCall) -> str:
    weights = [w.model_dump(exclude={"device"}) for w in call.weights]
    return sha256_text(json.dumps({"source": call.module_source, "weights": weights,
                                  "attributes": call.module_attributes}, sort_keys=True))


def interval_union(intervals: tuple[CudaInterval, ...]) -> float:
    end, total = float("-inf"), 0.0
    for row in sorted(intervals, key=lambda r: r.start_us):
        total += max(0, row.end_us - max(end, row.start_us))
        end = max(end, row.end_us)
    return total


def build_target(profile_path: Path, module_path: str) -> TargetSpec:
    profile = DeviceProfile.model_validate_json(profile_path.read_bytes()).data
    if not profile.valid or profile.trace_windows != 1 or not profile.cuda_intervals:
        raise RunnerError("operator selection needs an actual valid short CUDA profile")
    observed = {c.module_path for c in profile.calls}
    if any(p.startswith(module_path + ".") for p in observed):
        raise RunnerError("choose a narrow leaf pure-tensor boundary, not a decoder/model wrapper")
    chosen = next((c for c in profile.calls if c.module_path == module_path and c.phase == "prefill"), None)
    if chosen is None:
        raise RunnerError("selected module has no observed prefill")
    family = family_key(chosen)
    paths = tuple(sorted({c.module_path for c in profile.calls if c.phase == "prefill" and family_key(c) == family
                          and not any(p.startswith(c.module_path + ".") for p in observed)}))
    calls = [c for c in profile.calls if c.module_path in paths]
    geometries: dict[str, ProfileCall] = {}
    for call in calls:
        key = json.dumps({"phase": call.phase, "tensors": [t.model_dump() for t in call.tensors]}, sort_keys=True)
        geometries.setdefault(key, call)
    required = 1 if profile.goal == "ttft" else 2
    representatives = tuple(geometries.values())
    if len(representatives) != required or {c.decode_step for c in representatives} != ({0} if required == 1 else {0, 127}):
        raise RunnerError("operator family needs narrower unambiguous prefill/late-decode representatives")
    scopes = {c.scope for c in calls}
    links = [k for k in profile.site_kernels if k.scope in scopes]
    names = {k.kernel for k in links}
    associated = sum(k.cuda_duration_us for k in links)
    if not names or associated <= 0:
        raise RunnerError("selected operator has no measured associated CUDA work")
    others = {k.kernel for k in profile.site_kernels if k.scope not in scopes}
    selected_intervals = tuple(i for i in profile.cuda_intervals if i.kernel in names)
    fraction = None
    if not names.intersection(others) and selected_intervals:
        fraction = interval_union(selected_intervals) / interval_union(profile.cuda_intervals)
        if fraction >= 1:
            fraction = None
    reference = chosen.module_source + "\n\n# Observed forward\n" + chosen.source
    if chosen.source_file and Path(chosen.source_file).is_file():
        tree = ast.parse(Path(chosen.source_file).read_text(encoding="utf-8"))
        definitions = {n.name: n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
        pending = {n.id for n in ast.walk(ast.parse(reference)) if isinstance(n, ast.Name)}
        seen = set()
        while pending - seen:
            name = sorted(pending - seen)[0]
            seen.add(name)
            if name in definitions and name not in {n.name for n in ast.parse(chosen.module_source).body if isinstance(n, ast.ClassDef)}:
                node = definitions[name]
                reference += "\n\n" + ast.unparse(node)
                pending.update(n.id for n in ast.walk(node) if isinstance(n, ast.Name))
    metadata = json.dumps({"weights": [w.model_dump() for w in chosen.weights],
                           "module_attributes": chosen.module_attributes}, sort_keys=True)
    brief = OperatorBrief(site_id=canonical_site(paths), module_paths=paths, reference_source=reference,
        callable_interface="replacement(site, call, params): site.module is the actual original instance; "
            "site.weights are read-only resident references; call.args/call.kwargs are the exact invocation. "
            "Current local adapter supports TRITON ONLY and direct returned-output operands. "
            "Declare the actual defining Python source_file/JIT entry and output_arg_indices; not CUDA C++ or indirect outputs.",
        numerical_semantics=("Preserve every intermediate cast/rounding/reduction and operation order in the complete reference.",
            "Preserve dtype, shape, strides, aliasing, epsilon and required side effects; do not mutate weights or cache answers.",
            "Observed module/weight metadata: " + metadata),
        representative_calls=tuple(RepresentativeCall(phase=c.phase,
            tensors=tuple(TensorArgument(**t.model_dump(exclude={"path"})) for t in c.tensors)) for c in representatives))
    return TargetSpec(goal=profile.goal, brief=brief, profile=profile_path.resolve(), profile_sha256=fingerprint(profile_path),
        family_sha256=family, cuda_kernel_names=tuple(sorted(names)), associated_cuda_us=associated,
        exclusive_fraction=fraction, headroom_assumptions=(
            "Associated CUDA durations are observed; inclusive or same-name correlations can overlap.",
            "Exclusive fraction is unavailable when kernel-name attribution is ambiguous; no unproven Amdahl bound is claimed."),
        representatives=tuple((c.module_path, c.decode_step) for c in representatives))


def match_noise(target: TargetSpec, reports: tuple[LocalReport, LocalReport], output: Path) -> TargetSpec:
    a, b = reports
    if (not a.valid or not b.valid or a.device_identity != b.device_identity
            or [f.fixture_identity for f in a.fixtures] != [f.fixture_identity for f in b.fixtures]):
        raise RunnerError("local A/A evidence must be valid and share device/fixture identity")
    noise = max(abs(median(x.reference_us) - median(y.reference_us)) for x, y in zip(a.fixtures, b.fixtures, strict=True))
    output.write_text(json.dumps({"reports": [str(a.raw_path), str(b.raw_path)], "reference_noise_us": noise,
                                  "fixture_ids": [f.fixture_identity for f in a.fixtures]}, indent=2), encoding="utf-8")
    return target.model_copy(update={"reference_noise_us": noise, "noise_evidence": output.resolve(),
                                     "noise_evidence_sha256": fingerprint(output)})


def require_target(target: TargetSpec, *, noise: bool = True) -> None:
    if fingerprint(target.profile) != target.profile_sha256:
        raise RunnerError("frozen operator profile changed")
    if target.brief.site_id != canonical_site(target.brief.module_paths):
        raise RunnerError("operator site ID is not canonical for its actual module set")
    if not noise:
        return
    if target.reference_noise_us is None or target.noise_evidence is None or target.noise_evidence_sha256 is None:
        raise RunnerError("candidate admission requires measured local A/A noise evidence")
    if fingerprint(target.noise_evidence) != target.noise_evidence_sha256:
        raise RunnerError("frozen A/A evidence changed")
