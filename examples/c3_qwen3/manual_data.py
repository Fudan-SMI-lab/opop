"""Frozen file and workload admission; no model/backend imports."""

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Final, assert_never

from pydantic import JsonValue

from kernel_optimizer.evaluation.task_eval import TaskEvaluationError

from .manual_types import Context
from examples.c3_qwen3.model_binding import BundleSpec, load_bundle
from examples.c3_qwen3.runner_prepare import prepare
from examples.c3_qwen3.runner_records import GoalSpec, OracleManifest, Prompt

CONTRACT_SHA: Final = "9d1ef6a79bf99eb6ac0e639d82e894227b403124764ba4018531c9ddf179eaf8"
INTERFACE_SHA: Final = "ffddc3eb03c181194b57128eee085946cd4f1f1b701ca4da14ffc97e8d495e7f"
CORPUS_SHA: Final = "d60e5c108e79c8327a74199d5c093b117f9e64661cff4356e33a723dceb30a02"
CALIBRATION_IDS: Final = ("calibration-calibration-00", "calibration-calibration-01")


def fingerprint(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


MANUAL_HASHES: Final = MappingProxyType({
    Path(__file__).with_name(name).resolve(): fingerprint(Path(__file__).with_name(name))
    for name in ("manual_task.py", "manual_types.py", "manual_data.py")
})


def verify_frozen(hashes: Mapping[Path, str]) -> None:
    if any(not path.is_absolute() or fingerprint(path) != sha for path, sha in hashes.items()):
        raise TaskEvaluationError("frozen task/source/oracle files changed")


@dataclass(frozen=True, slots=True)
class Admitted:
    goal: GoalSpec
    quality_ids: tuple[str, ...]
    quality_requests: tuple[Prompt, ...]
    requests: tuple[Prompt, ...]
    bundle: BundleSpec
    hashes: Mapping[Path, str]
    source_hashes: Mapping[str, str]
    site_ids: tuple[str, ...]


def admit_files(context: Context, candidate: Path, params: Mapping[str, JsonValue]) -> Admitted:
    paths = (context.contract, context.assets_manifest, context.bundle_path, context.output_dir, candidate)
    if not all(path.is_absolute() for path in paths):
        raise TaskEvaluationError("runtime paths must be absolute")
    folder = context.contract.parent
    verify_frozen(context.frozen_files)
    hashes = {**context.frozen_files, **MANUAL_HASHES,
              context.contract: CONTRACT_SHA, folder / "interface.md": INTERFACE_SHA,
              folder / "proposed-prompts.json": CORPUS_SHA}
    verify_frozen(hashes)
    prepared = prepare(context.contract, context.assets_manifest)
    goal = next(goal for goal in prepared.contract.goals if goal.id == context.goal_id)
    prompts = prepared.prompts_by_id
    match context.split:
        case "search":
            groups = (goal.search_prompt_ids,)
            quality_ids = CALIBRATION_IDS
        case "heldout":
            groups = goal.final_prompt_groups
            quality_ids = context.prompt_ids
        case unreachable:
            assert_never(unreachable)
    if context.prompt_ids not in groups or len(context.prompt_ids) != goal.requests:
        raise TaskEvaluationError("prompt IDs/count do not match the frozen goal/split")
    if any(prompts[pid].goal != goal.id or prompts[pid].split != context.split
           or len(prompts[pid].input_ids) != goal.input_tokens for pid in context.prompt_ids):
        raise TaskEvaluationError("corpus token counts or split mismatch")
    oracle = context.oracle_refs
    if not oracle.is_absolute() or not oracle.is_file():
        raise TaskEvaluationError("missing frozen oracle manifest")
    manifest = OracleManifest.model_validate_json(oracle.read_bytes())
    if (manifest.contract_sha256 != CONTRACT_SHA or manifest.corpus_sha256 != CORPUS_SHA
            or not set(quality_ids) <= {prompt.prompt_id for prompt in manifest.prompts}):
        raise TaskEvaluationError("oracle identity or prompt coverage mismatch")
    hashes.setdefault(oracle, fingerprint(oracle))
    hashes.setdefault(context.assets_manifest, fingerprint(context.assets_manifest))
    hashes.setdefault(context.bundle_path, fingerprint(context.bundle_path))
    bundle = load_bundle(context.bundle_path, params)
    root = context.bundle_path.parent
    if candidate.resolve() != (root / bundle.document.entry).resolve():
        raise TaskEvaluationError("candidate_path must equal the declared bundle entry")
    source_hashes = bundle.source_hashes
    hashes.update({(root / name).resolve(): sha for name, sha in source_hashes.items()})
    quality_requests = tuple(prompts[pid] for pid in quality_ids)
    requests = tuple(prompts[pid] for pid in context.prompt_ids)
    return Admitted(goal, quality_ids, quality_requests, requests, bundle, hashes,
                    source_hashes, tuple(site.site_id for site in bundle.document.sites))
