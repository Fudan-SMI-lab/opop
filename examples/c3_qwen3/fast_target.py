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


def type_name(call: ProfileCall) -> str:
    """The class name from a recorded `module_source`, for labelling a child in the brief."""
    for node in ast.parse(call.module_source).body:
        if isinstance(node, ast.ClassDef):
            return node.name
    return "unknown"


def interval_union(intervals: tuple[CudaInterval, ...]) -> float:
    end, total = float("-inf"), 0.0
    for row in sorted(intervals, key=lambda r: r.start_us):
        total += max(0, row.end_us - max(end, row.start_us))
        end = max(end, row.end_us)
    return total


def subtree_is_pure(profile: ProfileData, module_path: str) -> tuple[bool, str | None]:
    """Can a composite module be a site at all -- i.e. would its call survive fixture capture?

    The leaf rule is only the first of four gates a site must pass, and it is the only one that is
    merely conservative. The load-bearing one is purity: `pure_tensors.tensor_leaves` RAISES on a
    `CacheValue`, so a module whose forward receives the KV cache can never be captured as a fixture,
    and with no fixture there is no local proof, no `validate_local`, and therefore no device-proven
    row. Relaxing the leaf rule for such a module would not widen the search; it would produce a site
    that fails later, further from the cause.

    The profile strips `CacheValue` arguments before describing tensors, so the cache cannot be seen in
    the recorded shapes. It IS visible in the module's own observed forward source, which the profile
    records verbatim -- that is what this reads. Measured on the frozen Qwen3 profile: the MLP block is
    clean, while `self_attn`, the decoder layer and `model` all name a cache parameter.
    """
    calls = [c for c in profile.calls if c.module_path == module_path]
    if not calls:
        return False, "selected module was never observed"
    source = calls[0].source or ""
    if not source:
        return False, "selected module has no recorded forward source to check for cache arguments"
    for needle in ("past_key_value", "layer_past", "Cache |", "Cache]", ": Cache", "cache_position"):
        if needle in source:
            return False, ("selected composite module receives a KV cache, which cannot be captured as "
                           "a pure-tensor fixture; no fixture means no local proof and no device proof")
    return True, None


def load_profile(profile_path: Path) -> ProfileData:
    """Parse a recorded device profile. Separate from `build_target` so it can be reused.

    These files are large (4.86 MB for the Qwen3 prefill profile) and validating one is the dominant
    cost of building a target. Ranking every observed module as a candidate site calls `build_target`
    hundreds of times, so re-reading the file each time turned a CPU-only ranking into minutes of
    parsing. Callers that ask about many sites in one profile parse it once and pass it in.
    """
    return DeviceProfile.model_validate_json(profile_path.read_bytes()).data


def build_target(profile_path: Path, module_path: str, *, allow_composite: bool = False,
                 profile: ProfileData | None = None) -> TargetSpec:
    """Build the operator pack for one site.

    `allow_composite` is the opt-in that admits a WRAPPER module -- one with observed children -- as the
    site, so the replacement covers the wrapper's whole subtree and a candidate may fuse across operator
    KINDS rather than only across one family's repeated stages. It is off by default, which keeps the
    behaviour of every existing target byte-identical, and it is reached only from C3's own selection
    step (nothing on the C2/KernelBench path imports this module).

    Admitting a wrapper is not just skipping a check: the site's children must be EXCLUDED from the
    group, or the binding would install a replacement on the wrapper and on its own children, and the
    per-module coverage check (`require_coverage`) would then see the children's originals never called.
    Purity is verified up front (`subtree_is_pure`) rather than left to fail at capture time.

    `profile` is an optional already-parsed copy of `profile_path`'s contents, for callers that build
    many targets from one profile. Re-reading the file to verify it would defeat the purpose, so the
    caller is responsible for parsing the SAME path it passes -- `fast_sites` does both from one
    variable. The returned spec still fingerprints `profile_path` itself, so the recorded
    `profile_sha256` always describes the file on disk and `require_target` re-checks it.
    """
    profile = load_profile(profile_path) if profile is None else profile
    if not profile.valid or profile.trace_windows != 1 or not profile.cuda_intervals:
        raise RunnerError("operator selection needs an actual valid short CUDA profile")
    observed = {c.module_path for c in profile.calls}
    composite = any(p.startswith(module_path + ".") for p in observed)
    if composite and not allow_composite:
        raise RunnerError("choose a narrow leaf pure-tensor boundary, not a decoder/model wrapper")
    if composite:
        pure, why = subtree_is_pure(profile, module_path)
        if not pure:
            raise RunnerError(why or "composite site is not a pure-tensor boundary")
    chosen = next((c for c in profile.calls if c.module_path == module_path and c.phase == "prefill"), None)
    if chosen is None:
        raise RunnerError("selected module has no observed prefill")
    family = family_key(chosen)
    # A composite site groups the wrapper instances themselves and must not also contain their
    # children; a leaf site keeps the original rule (only modules that are themselves leaves).
    def eligible(call: ProfileCall) -> bool:
        if composite:
            return True
        return not any(p.startswith(call.module_path + ".") for p in observed)

    paths = tuple(sorted({c.module_path for c in profile.calls if c.phase == "prefill"
                          and family_key(c) == family and eligible(c)}))
    if composite and any(p != q and p.startswith(q + ".") for p in paths for q in paths):
        raise RunnerError("composite site would nest one member inside another")
    calls = [c for c in profile.calls if c.module_path in paths]
    geometries: dict[str, ProfileCall] = {}
    for call in calls:
        key = json.dumps({"phase": call.phase, "tensors": [t.model_dump() for t in call.tensors]}, sort_keys=True)
        geometries.setdefault(key, call)
    required = 1 if profile.goal == "ttft" else 2
    representatives = tuple(geometries.values())
    if len(representatives) != required or {c.decode_step for c in representatives} != ({0} if required == 1 else {0, 127}):
        raise RunnerError("operator family needs narrower unambiguous prefill/late-decode representatives")
    # The kernels a replacement of this site would subsume. For a leaf site that is the site's own
    # scopes; for a COMPOSITE site it must also include the scopes of the children being subsumed,
    # because the wrapper's own scope is attributed only the kernels its own body launched (for an MLP
    # wrapper: the elementwise multiply) while the children's GEMMs sit under their own scopes. Without
    # this, a composite site reports LESS associated CUDA time than the leaf it replaces -- measured
    # 8582 us for the MLP wrapper against 14041 us for the RMSNorm leaf, which inverts the ranking the
    # number exists to provide.
    def in_site(scope: str) -> bool:
        tail = scope.split(":")[-1]
        return tail in paths or (composite and any(tail.startswith(p + ".") for p in paths))

    scopes = {k.scope for k in profile.site_kernels if in_site(k.scope)}
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
    if composite:
        # A wrapper's own forward only names its children (`self.down_proj(self.act_fn(...))`), so the
        # reference is not self-contained: a candidate cannot reproduce the computation, let alone fuse
        # it, without the children's definitions and weight shapes.
        #
        # The children are listed first, each with its OWN observed metadata (weight shapes differ:
        # gate/up project 2560->9728 while down projects back), and each class body is emitted ONCE.
        # Three `nn.Linear` children would otherwise repeat torch's whole `Linear` class three times --
        # measured at 200 redundant lines of a 341-line brief, which is prompt budget spent on nothing
        # and, on the first live attempt, a 1500 s transport timeout.
        children = sorted({c.module_path: c for c in profile.calls
                           if c.phase == chosen.phase
                           and c.module_path.startswith(module_path + ".")}.values(),
                          key=lambda c: c.module_path)
        roster = []
        bodies: dict[str, str] = {}
        for child in children:
            suffix = child.module_path[len(module_path) + 1:]
            cls = type_name(child)
            child_meta = json.dumps({"weights": [w.model_dump() for w in child.weights],
                                     "attributes": child.module_attributes,
                                     "input_tensors": [t.model_dump() for t in child.tensors]},
                                    sort_keys=True)
            roster.append(f"#   self.{suffix}: {cls}  {child_meta}")
            bodies.setdefault(cls, child.module_source + "\n\n# Observed forward\n" + child.source)
        if roster:
            reference += ("\n\n# Child submodules this replacement must also subsume, with their own "
                          "observed metadata:\n" + "\n".join(roster))
            for cls, body in bodies.items():
                reference += f"\n\n# Definition of {cls}, used by the children listed above\n" + body
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
