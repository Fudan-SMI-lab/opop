from collections.abc import Callable, Mapping
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

from pydantic import Field, JsonValue

from kernel_optimizer.models.device_operator import DeviceKernelDeclaration
from .model_binding import BundleSpec
from .model_runner import ResidentRunner
from .operator_types import FixtureCodec, OperatorCall, OperatorFixture, Value
from .runner_records import FrozenRecord, GoalId, Phase

type DeviceStage = Literal["setup", "development", "formal_search", "final"]
type DeviceOperation = Literal["local", "capture", "profile", "model"]


class StageIdentity(FrozenRecord):
    profile: Literal["c3_fast_device"]
    stage: DeviceStage
    framework_id: str = Field(min_length=1)


class Admission(Protocol):
    def admit(self, stage: StageIdentity, operation: DeviceOperation) -> int | None: ...


class TensorDescription(FrozenRecord):
    path: str
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    dtype: str
    device: str


class FixtureDescription(FrozenRecord):
    site_id: str
    module_path: str
    goal: GoalId
    phase: Phase
    decode_step: int = Field(ge=0, le=127)
    batch_size: int
    prompt_length: int
    tensors: tuple[TensorDescription, ...]
    weights: tuple[TensorDescription, ...] = ()
    module_attributes: dict[str, JsonValue] = Field(default_factory=dict)
    source: str
    source_sha256: str
    contract_sha256: str
    corpus_sha256: str


@dataclass(frozen=True, slots=True)
class PureFixture:
    description: FixtureDescription
    original: OperatorFixture
    identity: str
    resident_bytes: int


class SourceEligibility(FrozenRecord):
    eligible: bool
    bundle_sha256: str
    params_sha256: str


class LocalRequest(FrozenRecord):
    mode: Literal["fixture_only"]
    stage: StageIdentity
    bundle_path: Path
    params: dict[str, JsonValue]
    site_id: str
    kernels: tuple[DeviceKernelDeclaration, ...] = Field(min_length=1)
    source_gate: SourceEligibility


class KernelEvidence(FrozenRecord):
    backend: Literal["triton", "cuda"]
    source_file: str
    entry: str
    compiled: bool
    kernel_name: str
    launches: int
    cuda_names: tuple[str, ...]
    output_used: bool
    stored_output_args: tuple[int, ...]
    loaded_input_args: tuple[int, ...]
    skip_control_rejected: bool
    local_quality_passed: bool | None = None
    quality_detail: str | None = None
    compile_calls: int = 0
    compile_wall_ms: float = 0.0
    jit_calls: int = 0
    n_regs: int | None = None
    shared_bytes: int | None = None


class FixtureResult(FrozenRecord):
    fixture_identity: str
    description: FixtureDescription
    quality_passed: bool
    reference_us: tuple[float, ...] = ()
    candidate_us: tuple[float, ...] = ()
    pair_order: tuple[str, ...] = ()
    device_evidence: tuple[KernelEvidence, ...] = ()


class LocalReport(FrozenRecord):
    mode: Literal["fixture_only"] = "fixture_only"
    stage: StageIdentity
    valid: bool
    failure_stage: str | None = None
    detail: str | None = None
    official_model_score: None = None
    latency_us: float | None = None
    bundle_sha256: str
    params_sha256: str
    source_hashes: dict[str, str]
    declared_kernels: tuple[DeviceKernelDeclaration, ...] = ()
    device_identity: dict[str, JsonValue] = Field(default_factory=dict)
    fixtures: tuple[FixtureResult, ...]
    counts: dict[str, int]
    wall_ms: float
    raw_path: Path


class PureTools(FixtureCodec, Protocol):
    def describe_tensors(self, call: OperatorCall) -> tuple[TensorDescription, ...]: ...
    def digest(self, call: OperatorCall) -> str: ...
    def require_pure(self, call: OperatorCall) -> None: ...


class LocalDevice(Protocol):
    tools: PureTools
    counters: dict[str, int]
    def ready(self) -> None: ...
    def execution(self) -> AbstractContextManager: ...
    def identity(self) -> dict[str, JsonValue]: ...
    def prove(self, runner: ResidentRunner, declaration: DeviceKernelDeclaration,
              invocation: "LocalInvocation") -> KernelEvidence: ...
    def time_us(self, call: Callable[[], Value]) -> float: ...


@dataclass(frozen=True, slots=True)
class LocalInvocation:
    bundle: BundleSpec
    fixture: PureFixture
    candidate: Callable[[OperatorCall], Value]


@dataclass(frozen=True, slots=True)
class LocalRuntime:
    runner: ResidentRunner
    device: LocalDevice
    admission: Admission


class CaptureSpec(FrozenRecord):
    stage: StageIdentity
    goal: GoalId
    site_id: str
    module_path: str
    decode_step: int = Field(ge=0, le=127)
