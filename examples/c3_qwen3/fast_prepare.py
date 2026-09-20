import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Final, Literal, assert_never

from pydantic import JsonValue, TypeAdapter

from .model_runner import ResidentRunner, load_prepared
from .runner_backend import ModelBackend
from .runner_prepare import Corpus, Preflight, REVISION
from .runner_records import AssetSpec, BindingReceipt, Contract, FrozenRecord, GoalId, PreparedTask, RunnerError

CONTRACT_SHA: Final = "2c0e2fc70c76d4b780cb1301addcf2d723ec5f19f106af7ac3919fbc86996bd8"
INTERFACE_SHA: Final = "9b8859a327614bb87b010c9d49ae392a773e8297959e89648121c4f69ecdbbca"
PREFLIGHT_SHA: Final = "0796cfe011f18a2169ebdf49ea3621bdc431352348c7ade1e983f79ca1586c1a"
FACET_SHA: Final = "0c406df28453b2a92810dc897b6bf42f25b38aac8be9c7f64c2da29ca9c78c5e"
DEV_SHA: Final = "5d6a568321c96ff43ea90474b49c5d62aed81df7c8bf8bda5fe8ccad5703092d"
FINAL_SHA: Final = "bcb3d7693c0956fb2e693dd1a805526f771b47096d1e16699bd2a23653da25ed"
type AccessPhase = Literal["development", "formal_search", "final"]


class FastAccess(FrozenRecord):
    profile: Literal["c3_fast_device"]
    phase: AccessPhase
    task_kind: Literal["model_project_operator"] = "model_project_operator"
    formal_complete: bool = False


def checked_bytes(path: Path, sha: str) -> bytes:
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != sha:
        raise RunnerError(f"frozen fast input hash mismatch: {path.name}")
    return content


@dataclass(frozen=True, slots=True)
class FastPrepared:
    task: PreparedTask
    access: FastAccess
    facet_json: bytes

    def agent_context(self, goal_id: GoalId) -> dict[str, JsonValue]:
        if self.access.phase == "final":
            raise RunnerError("final inputs cannot enter an agent context")
        facet = TypeAdapter(dict[str, JsonValue]).validate_json(self.facet_json)
        goals = facet.pop("goals")
        if not isinstance(goals, list):
            raise RunnerError("invalid pinned agent facet")
        selected = [g for g in goals if isinstance(g, dict) and g.get("id") == goal_id]
        if len(selected) != 1:
            raise RunnerError("unknown goal in agent facet")
        facet["goal"] = selected[0]
        budgets = facet.pop("stage_budgets")
        if not isinstance(budgets, dict):
            raise RunnerError("invalid pinned budgets")
        key = "formal" if self.access.phase == "formal_search" else "development"
        facet["stage_budget"] = budgets[key]
        facet["execution_profile"] = self.access.profile
        facet["fixture_feedback"] = {"supported_backends": ["triton"], "direct_output_operands_required": True,
                                      "warmups_per_side": 3, "cuda_event_repeats_per_side": 20,
                                      "official_model_score": None}
        return facet


def prepare_fast(contract_path: Path, assets_manifest: Path, access: FastAccess) -> FastPrepared:
    match access.phase:
        case "development" | "formal_search":
            corpus_name, corpus_sha, count = "development-corpus.json", DEV_SHA, 16
        case "final":
            if not access.formal_complete:
                raise RunnerError("sealed final access requires formal completion")
            corpus_name, corpus_sha, count = "heldout/corpus.json", FINAL_SHA, 58
        case unreachable:
            assert_never(unreachable)
    contract_path, assets_manifest = contract_path.resolve(), assets_manifest.resolve()
    root = contract_path.parent
    contract = Contract.model_validate_json(checked_bytes(contract_path, CONTRACT_SHA))
    checked_bytes(root / "interface.md", INTERFACE_SHA)
    facet = checked_bytes(root / "agent-input-contract.json", FACET_SHA)
    manifest = Preflight.model_validate_json(checked_bytes(assets_manifest, PREFLIGHT_SHA))
    if manifest.contract_sha256 != CONTRACT_SHA or set(manifest.hosts) != {"A", "B"}:
        raise RunnerError("fast manifest contract/hosts mismatch")
    for host in manifest.hosts.values():
        assets = host.assets
        if (assets.revision != REVISION or assets.repo_id != "Qwen/Qwen3-4B"
                or not assets.assets_complete or assets.asset_path != contract.assets.local_path_both_hosts
                or len(assets.files) != 10 or not all(f.official_integrity_match for f in assets.files)):
            raise RunnerError("incompatible pinned model assets")
    if manifest.hosts["A"].assets.files != manifest.hosts["B"].assets.files:
        raise RunnerError("host asset identities differ")
    corpus = Corpus.model_validate_json(checked_bytes(root / corpus_name, corpus_sha))
    prompts = {p.id: p for p in corpus.prompts}
    if len(prompts) != count or corpus.revision != REVISION or corpus.enable_thinking:
        raise RunnerError("fast corpus identity/count mismatch")
    for p in prompts.values():
        if (len(p.input_ids) != p.input_tokens
                or hashlib.sha256(p.rendered_text.encode()).hexdigest() != p.rendered_sha256
                or hashlib.sha256(json.dumps(p.input_ids, separators=(",", ":")).encode()).hexdigest() != p.token_ids_sha256):
            raise RunnerError("frozen prompt bytes/count mismatch")
    if access.phase != "final":
        if any(p.split == "heldout" for p in prompts.values()):
            raise RunnerError("heldout record in development corpus")
        contract = contract.model_copy(update={"goals": tuple(
            g.model_copy(update={"final_prompt_groups": ()}) for g in contract.goals)})
    assets = contract.assets.model_copy(update={"corpus_file": corpus_name, "corpus_sha256": corpus_sha})
    contract = contract.model_copy(update={"assets": assets})
    spec = AssetSpec(assets.local_path_both_hosts, REVISION, contract_path, assets_manifest,
                     MappingProxyType({f.name: f.size for f in manifest.hosts["A"].assets.files}))
    return FastPrepared(PreparedTask(contract, spec, MappingProxyType(prompts), CONTRACT_SHA, corpus_sha), access, facet)


def load_fast(prepared: FastPrepared, device: str, *, backend_factory: Callable[[AssetSpec, str], ModelBackend] | None = None) -> ResidentRunner:
    spec = prepared.task.asset_spec
    fresh = prepare_fast(spec.contract_path, spec.assets_manifest, prepared.access)
    return load_prepared(fresh.task, device, backend_factory=backend_factory)


def attach_fast(runner: ResidentRunner, prepared: FastPrepared) -> None:
    old, new = runner.prepared.asset_spec, prepared.task.asset_spec
    if runner.closed or (old.local_path, old.revision, old.file_sizes) != (new.local_path, new.revision, new.file_sizes):
        raise RunnerError("cannot reuse incompatible or closed resident weights")
    runner.binding.restore()
    runner.reset()
    runner.prepared = prepared.task
    runner.binding.fixtures.clear()
    runner.binding.sites.clear()
    runner.binding.fixture_bytes = 0
    runner.receipt = BindingReceipt("baseline", MappingProxyType({}), "baseline", (), True)
    runner.warmed.clear()
    runner.quality_identity = None


def require_feedback_inputs(runner: ResidentRunner) -> None:
    if runner.prepared.contract_sha256 != CONTRACT_SHA or runner.prepared.corpus_sha256 != DEV_SHA:
        raise RunnerError("device feedback requires explicit fast development/formal-search preparation")
