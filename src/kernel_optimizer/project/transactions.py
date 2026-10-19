"""Immutable full-project revisions with one atomic accepted-version pointer."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any, TypeVar

from pydantic import BaseModel

from kernel_optimizer.models.contract_common import FileRef
from kernel_optimizer.models.project import (
    CandidateBundle, ExecutionPlan, GroupRegistry, ProjectContract, ProjectEvaluation, ProjectRevision,
)
from kernel_optimizer.runtime.execution import DeviceLease, write_json
from kernel_optimizer.store.run_store import RunStore

T = TypeVar("T", bound=BaseModel)


def canonical(value: Any) -> bytes:
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    def encode(item):
        if isinstance(item, BaseModel):
            return item.model_dump(mode="json")
        raise TypeError(f"not JSON serializable: {type(item).__name__}")
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False, default=encode).encode()


def identity(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def _relative(name: str) -> str:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or "\\" in name or ":" in name or not path.parts:
        raise ValueError(f"project file must be a relative POSIX path: {name}")
    return str(path)


class RevisionStore:
    def __init__(self, store: RunStore):
        self.store = store
        self.root = store.run_dir / "project"
        for sub in ("revisions", "bundles", "transactions"):
            (self.root / sub).mkdir(parents=True, exist_ok=True)
        self.pointer = self.root / "current.json"

    def put(self, value: BaseModel) -> str:
        return self.store.put_artifact(canonical(value))

    def load(self, ref: str, model: type[T]) -> T:
        raw = self.store.get_artifact(ref)
        if "sha256:" + hashlib.sha256(raw).hexdigest() != ref:
            raise ValueError("artifact content no longer matches its identity")
        return model.model_validate_json(raw)

    def revision(self, ident: str) -> ProjectRevision:
        return ProjectRevision.model_validate_json((self.root / "revisions" / f"{ident}.json").read_bytes())

    def current(self) -> ProjectRevision:
        return self.revision(json.loads(self.pointer.read_text())["revision"])

    def bundle(self, ident: str) -> CandidateBundle:
        return CandidateBundle.model_validate_json((self.root / "bundles" / f"{ident}.json").read_bytes())

    def _save_revision(self, values: dict) -> ProjectRevision:
        ident = identity(values)
        revision = ProjectRevision(id=ident, **values)
        write_json(self.root / "revisions" / f"{ident}.json", revision.model_dump(mode="json"))
        return revision

    def initialize(self, source: Path, files: list[str], contract: ProjectContract,
                   plan: ExecutionPlan, groups: GroupRegistry) -> ProjectRevision:
        if self.pointer.exists():
            raise FileExistsError("project already initialized")
        self._check_registry(plan, groups)
        refs = []
        for name in sorted(set(files)):
            name = _relative(name)
            raw = (source / name).read_bytes()
            refs.append(FileRef(path=name, content_ref=self.store.put_artifact(raw, name),
                                sha256=hashlib.sha256(raw).hexdigest()))
        if not refs:
            raise ValueError("project snapshot must contain source files")
        contract_ref = self.put(contract)
        revision = self._save_revision(dict(parent_id=None, anchor_id=identity({"contract": contract_ref, "files": refs}),
            contract_ref=contract_ref, plan_ref=self.put(plan), registry_ref=self.put(groups), files=tuple(refs)))
        write_json(self.root / "anchor.json", {"revision": revision.id, "contract_ref": contract_ref})
        write_json(self.pointer, {"revision": revision.id})
        self.store.append("PROJECT_INITIALIZED", {"revision": revision.id})
        return revision

    @staticmethod
    def _check_registry(plan: ExecutionPlan, registry: GroupRegistry) -> None:
        if registry.plan_id != plan.id:
            raise ValueError("registry belongs to another execution plan")
        calls = {call.id for call in plan.calls}
        if any(set(group.calls) - calls for group in registry.groups):
            raise ValueError("group refers to an absent call")

    def materialize(self, revision: ProjectRevision, destination: Path) -> Path:
        destination.mkdir(parents=True, exist_ok=False)
        for file in revision.files:
            raw = self.store.get_artifact(file.content_ref)
            if hashlib.sha256(raw).hexdigest() != file.sha256:
                raise ValueError("snapshot hash mismatch")
            target = destination / _relative(file.path)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(raw)
        write_json(destination / ".opop-revision.json", {"revision": revision.id, "bundles": revision.bundles,
                   "bundle_hashes": {ident: identity(self.bundle(ident)) for ident in revision.bundles},
                   "files": {file.path: file.sha256 for file in revision.files}})
        return destination

    def stage(self, bundle: CandidateBundle) -> ProjectRevision:
        # Exploratory parents need not be the accepted current version. Publication uses CAS.
        parent = self.revision(bundle.parent_revision)
        contract = self.load(parent.contract_ref, ProjectContract)
        if bundle.contract_ref != parent.contract_ref or bundle.scope not in contract.allowed_scopes:
            raise ValueError("candidate contract or scope differs from frozen task")
        if not set(contract.invariants) <= set(bundle.external_invariants):
            raise ValueError("candidate omits external invariants")
        if not set(bundle.requires) <= set(parent.bundles):
            raise ValueError("candidate dependencies are not installed")
        if not set(bundle.replaces) <= set(parent.bundles):
            raise ValueError("candidate replaces an absent bundle")
        registry = self.load(parent.registry_ref, GroupRegistry)
        plan = self.load(parent.plan_ref, ExecutionPlan)
        if set(bundle.groups) - {g.id for g in registry.groups} or set(bundle.calls) - {c.id for c in plan.calls}:
            raise ValueError("candidate refers to unknown groups or calls")
        files = {file.path: file for file in parent.files}
        for edit in bundle.edits:
            name = _relative(edit.path)
            old = files.get(name)
            if edit.before_sha256 != (old.sha256 if old else None):
                raise ValueError("stale file edit")
            if "file:" + name not in bundle.write_set:
                raise ValueError("file edit absent from declared write set")
            if edit.content_ref is None:
                if old is None:
                    raise ValueError("cannot delete absent file")
                del files[name]
            else:
                raw = self.store.get_artifact(edit.content_ref)
                if edit.content_ref != "sha256:" + hashlib.sha256(raw).hexdigest():
                    raise ValueError("candidate artifact hash mismatch")
                files[name] = FileRef(path=name, content_ref=edit.content_ref, sha256=hashlib.sha256(raw).hexdigest())
        plan_ref, registry_ref = bundle.plan_ref or parent.plan_ref, bundle.registry_ref or parent.registry_ref
        self._check_registry(self.load(plan_ref, ExecutionPlan), self.load(registry_ref, GroupRegistry))
        existing = tuple(ident for ident in parent.bundles if ident not in bundle.replaces)
        for ident in existing:
            if set(self.bundle(ident).requires) & set(bundle.replaces):
                raise ValueError("replacement would break an inherited bundle dependency")
        bundle_path = self.root / "bundles" / f"{bundle.id}.json"
        if bundle_path.exists() and canonical(self.bundle(bundle.id)) != canonical(bundle):
            raise ValueError("bundle identity was reused for different content")
        write_json(bundle_path, bundle.model_dump(mode="json"))
        revision = self._save_revision(dict(parent_id=parent.id, anchor_id=parent.anchor_id,
            contract_ref=parent.contract_ref, plan_ref=plan_ref, registry_ref=registry_ref,
            files=tuple(files[name] for name in sorted(files)), bundles=existing + (bundle.id,),
            retired_bundles=parent.retired_bundles + bundle.replaces))
        write_json(self.root / "transactions" / f"{revision.id}.json",
                   {"state": "staged", "parent": parent.id, "revision": revision.id, "bundle": bundle.id})
        self.store.append("PROJECT_STAGED", {"revision": revision.id, "parent": parent.id, "bundle": bundle.id})
        return revision

    def evaluate(self, report: ProjectEvaluation) -> dict:
        revision = self.revision(report.revision_id)
        bundle = self.bundle(report.bundle_id)
        contract = self.load(revision.contract_ref, ProjectContract)
        if (report.parent_id, report.contract_ref, bundle.parent_revision) != (
                revision.parent_id, revision.contract_ref, revision.parent_id):
            raise ValueError("evaluation belongs to another parent or contract")
        if bundle.id not in revision.bundles:
            raise ValueError("evaluation bundle is not installed")
        reasons = list(report.reasons)
        if report.quality != "pass" or not report.feasible:
            reasons.append("quality_or_constraint_failure")
        required = set(bundle.calls)
        if set(report.required_calls) != required or not required <= set(report.executed_calls):
            reasons.append("candidate_coverage_missing")
        expected_ranks = {str(rank) for rank in range(contract.world_size)}
        if set(map(str, report.required_ranks)) != expected_ranks or set(report.rank_bundle_hashes) != expected_ranks or any(
                digest != identity(bundle) for digest in report.rank_bundle_hashes.values()):
            reasons.append("rank_bundle_mismatch")
        if report.objective is None or report.reference_objective is None:
            reasons.append("objective_missing")
        else:
            gain = report.reference_objective - report.objective
            if contract.objective.direction == "maximize":
                gain = -gain
            if gain <= contract.objective.improvement_threshold:
                reasons.append("objective_not_improved")
        result = {"state": "evaluated", "revision": revision.id, "parent": revision.parent_id,
                  "bundle": bundle.id, "accepted": not reasons, "reasons": reasons,
                  "report_ref": self.put(report)}
        write_json(self.root / "transactions" / f"{revision.id}.json", result)
        self.store.append("PROJECT_EVALUATED", result)
        return result

    def commit(self, revision_id: str) -> ProjectRevision:
        with DeviceLease(self.root / "locks", "accepted-project-version"):
            path = self.root / "transactions" / f"{revision_id}.json"
            state = json.loads(path.read_text())
            revision = self.revision(revision_id)
            if state["state"] != "evaluated" or not state["accepted"]:
                raise ValueError("only a passing evaluated revision can be committed")
            if self.current().id != revision.parent_id:
                raise ValueError("accepted parent changed; candidate must be rebased and reevaluated")
            write_json(self.pointer, {"revision": revision.id})
            state["state"] = "committed"
            write_json(path, state)
            self.store.append("PROJECT_COMMITTED", {"revision": revision.id, "parent": revision.parent_id})
            return revision

    def abort(self, revision_id: str, reason: str) -> None:
        path = self.root / "transactions" / f"{revision_id}.json"
        state = json.loads(path.read_text())
        if self.current().id == revision_id:
            raise ValueError("use rollback for an already committed revision")
        state.update(state="aborted", reason=reason)
        write_json(path, state)
        self.store.append("PROJECT_ABORTED", state)

    def rollback(self, ancestor_id: str, reason: str) -> ProjectRevision:
        with DeviceLease(self.root / "locks", "accepted-project-version"):
            previous = cursor = self.current()
            while cursor.id != ancestor_id and cursor.parent_id is not None:
                cursor = self.revision(cursor.parent_id)
            if cursor.id != ancestor_id:
                raise ValueError("rollback target must be an ancestor")
            write_json(self.pointer, {"revision": cursor.id})
            self.store.append("PROJECT_ROLLED_BACK", {"previous": previous.id, "revision": cursor.id, "reason": reason})
            return cursor

    def recover(self) -> list[dict]:
        # The atomic pointer is authoritative after interruption between pointer and journal writes.
        accepted = self.current()
        chain = {accepted.id}
        while accepted.parent_id is not None:
            accepted = self.revision(accepted.parent_id)
            chain.add(accepted.id)
        recovered = []
        for path in sorted((self.root / "transactions").glob("*.json")):
            state = json.loads(path.read_text())
            if state["state"] in {"staged", "evaluated"}:
                state["state"] = "committed" if state["revision"] in chain else "aborted"
                state["recovery_reason"] = "accepted pointer reconciliation"
                write_json(path, state)
                recovered.append(state)
        self.store.append("PROJECT_RECOVERED", {"transactions": recovered})
        return recovered
