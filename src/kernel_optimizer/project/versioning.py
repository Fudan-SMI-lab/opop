"""Import and independently evaluate a complete exploratory project version."""
from pathlib import Path
import json
import os
import shutil
from uuid import uuid4

from kernel_optimizer.models.project import ExecutionPlan, GroupRegistry, ProjectContract
from kernel_optimizer.project.aggregation import rank_requirements
from kernel_optimizer.project.measurement import paired_decision, valid_run
from kernel_optimizer.project.quality_report import quality_and_coverage
from kernel_optimizer.project.transactions import RevisionStore, canonical, identity
from kernel_optimizer.runtime.execution import write_json
from kernel_optimizer.store.run_store import RunStore


def stage_version(destination, source_output, revision_id=None):
    """Preserve candidate bytes, order and identities; only rebase the whole snapshot."""
    source = RevisionStore(RunStore.open(Path(source_output)))
    incoming = source.revision(revision_id) if revision_id else source.current()
    parent = destination.current()
    if (incoming.anchor_id, incoming.contract_ref) != (parent.anchor_id, parent.contract_ref):
        raise ValueError("complete versions must share the same original anchor and frozen contract")
    contract = destination.load(parent.contract_ref, ProjectContract)
    old_files, new_files = ({f.path: f for f in r.files} for r in (parent, incoming))
    if any(old_files.get(name) != new_files.get(name) for name in contract.frozen_files):
        raise ValueError("complete version changed a frozen evaluator file")
    seen, writes = set(), set()
    for ident in incoming.bundles:
        bundle = source.bundle(ident)
        if bundle.contract_ref != incoming.contract_ref or not set(bundle.requires) <= seen:
            raise ValueError("complete version has an inconsistent ordered bundle dependency")
        if writes & set(bundle.write_set):
            raise ValueError("complete version contains conflicting active writes")
        seen.add(ident)
        writes.update(bundle.write_set)
    # Supporting revisions hold each bundle's original rank-local binding plan.
    # Mutable artifact indexes and source transaction states are never imported.
    for path in source.store.artifacts_dir.iterdir():
        if not path.is_file() or len(path.name) != 64 or any(c not in "0123456789abcdef" for c in path.name):
            continue
        target = destination.store.artifacts_dir/path.name
        if target.exists():
            if target.read_bytes() != path.read_bytes():
                raise ValueError("artifact identity collision")
        else:
            try:
                os.link(path, target)
            except OSError:
                shutil.copyfile(path, target)
    for folder in ("revisions", "bundles"):
        for path in (source.root/folder).glob("*.json"):
            target = destination.root/folder/path.name
            if target.exists():
                if canonical(json.loads(target.read_bytes())) != canonical(json.loads(path.read_bytes())):
                    raise ValueError("revision or bundle identity collision")
            else:
                shutil.copyfile(path, target)
    destination._check_registry(source.load(incoming.plan_ref, ExecutionPlan),
                                source.load(incoming.registry_ref, GroupRegistry))
    retired = (set(parent.retired_bundles) | set(parent.bundles) | set(incoming.retired_bundles)) - set(incoming.bundles)
    staged = destination._save_revision(dict(parent_id=parent.id, anchor_id=parent.anchor_id,
        contract_ref=parent.contract_ref, plan_ref=incoming.plan_ref, registry_ref=incoming.registry_ref,
        files=incoming.files, bundles=incoming.bundles, retired_bundles=tuple(sorted(retired))))
    state = {"state": "staged", "kind": "complete_version", "parent": parent.id, "revision": staged.id,
             "source_output": str(Path(source_output).resolve()), "source_revision": incoming.id}
    write_json(destination.root/"transactions"/(staged.id+".json"), state)
    destination.store.append("PROJECT_VERSION_STAGED", state)
    return staged


def coverage_errors(revisions, candidate, runs):
    """Require every active bundle on every endpoint and its own observed ranks."""
    contract = revisions.load(candidate.contract_ref, ProjectContract)
    expected_ranks = set(range(contract.world_size))
    errors = []
    for endpoint, run in enumerate(runs):
        if {r["rank"] for r in run.get("ranks", [])} != expected_ranks:
            errors.append({"endpoint": endpoint, "reason": "version_rank_missing"})
        for rank in run.get("ranks", []):
            result = rank.get("result", {})
            if set(result.get("bundle_hashes", {})) != set(candidate.bundles):
                errors.append({"endpoint": endpoint, "rank": rank["rank"], "reason": "version_bundle_set_mismatch"})
            for ident in candidate.bundles:
                bundle = revisions.bundle(ident)
                plan = revisions.load(revisions.revision(bundle.parent_revision).plan_ref, ExecutionPlan)
                required = set(rank_requirements(bundle, plan, contract.world_size)[str(rank["rank"])])
                actual = result.get("coverage", {}).get(ident, {})
                executed = {c for c, n in actual.get("counts", {}).items() if n > 0}
                if result.get("bundle_hashes", {}).get(ident) != identity(bundle):
                    errors.append({"endpoint": endpoint, "rank": rank["rank"], "bundle": ident,
                                   "reason": "version_bundle_hash_mismatch"})
                if set(actual.get("required", [])) != required or not required <= executed:
                    errors.append({"endpoint": endpoint, "rank": rank["rank"], "bundle": ident,
                                   "reason": "version_bundle_coverage_missing", "calls": sorted(required-executed)})
    return errors


def promote_version(search, source_output, revision_id=None, *, publish=True):
    """Compare the full branch against the current main version, then atomic CAS."""
    revisions = search.revisions
    candidate = stage_version(revisions, source_output, revision_id)
    parent = revisions.revision(candidate.parent_id)
    trial_id = "version-"+uuid4().hex[:12]
    pair_count = int(search.request.objective.get("paired_repeats", 2))
    if pair_count < 2:
        raise ValueError("complete version promotion requires AB and BA pairs")
    pairs = []
    for index in range(pair_count):
        pair = {}
        order = (("parent", parent), ("candidate", candidate))
        for role, revision in order if index % 2 == 0 else reversed(order):
            pair[role] = search.execute(revision, f"{trial_id}-pair-{index}-{role}", "evaluate")
            if not valid_run(pair[role]):
                break
        pairs.append(pair)
        if len(pair) != 2 or not all(valid_run(run) for run in pair.values()):
            break
    complete = len(pairs) == pair_count and all(len(p) == 2 for p in pairs)
    measurement = paired_decision(pairs, search.request.objective, search.prepared["evaluation"]) if complete else {
        "valid": False, "objective": None, "reference": None, "reasons": ["paired_endpoint_failed"]}
    runs = [p["candidate"] for p in pairs if "candidate" in p]
    quality, missing = quality_and_coverage(runs)
    errors = coverage_errors(revisions, candidate, runs)
    reasons = set(measurement["reasons"]) | {e["reason"] for e in errors}
    if quality != "pass" or not measurement["valid"]:
        reasons.add("version_quality_or_execution_failure")
    report = {"schema_version": "project.version-evaluation.v1", "id": trial_id,
        "revision": candidate.id, "parent": parent.id, "contract_ref": candidate.contract_ref,
        "bundles": list(candidate.bundles), "quality": quality, "missing_coverage": missing,
        "coverage_errors": errors, "measurement": measurement, "pairs": pairs,
        "attribution_status": "joint_only", "source_output": str(source_output),
        "source_revision": revision_id or RevisionStore(RunStore.open(Path(source_output))).current().id}
    decision = {"state": "evaluated", "kind": "complete_version", "revision": candidate.id,
        "parent": parent.id, "accepted": not reasons, "reasons": sorted(reasons),
        "report_ref": revisions.store.put_artifact(canonical(report))}
    write_json(revisions.root/"transactions"/(candidate.id+".json"), decision)
    search.store.append("PROJECT_VERSION_EVALUATED", decision)
    if decision["accepted"] and publish:
        revisions.commit(candidate.id)
    elif not decision["accepted"]:
        revisions.abort(candidate.id, "; ".join(decision["reasons"]))
    result = {**report, "trial": trial_id, "decision": decision,
        "publication": "committed" if decision["accepted"] and publish else "exploration" if decision["accepted"] else "rejected"}
    write_json(search.output/"version-trials"/trial_id/"result.json", result)
    write_json(search.output/"latest-feedback.json", result)
    return result
