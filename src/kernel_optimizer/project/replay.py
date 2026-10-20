"""Transfer frozen candidate code to a separately validated goal/workload variant.

This is an experiment operation, not a successful optimization. Call identity must
survive discovery; unknown/ambiguous sites require a new generated candidate.
"""
import hashlib
import json
from pathlib import Path
from uuid import uuid4

from kernel_optimizer.models.project import CandidateBundle, FileEdit, ProjectContract
from kernel_optimizer.project.transactions import RevisionStore
from kernel_optimizer.store.run_store import RunStore
from kernel_optimizer.runtime.execution import write_json


def transfer_bundle(source_output, source_bundle, search, *, parent=None, discovery=None,
                    dependency_map=None, allow_absent_replacements=False):
    source_output = Path(source_output)
    source = RevisionStore(RunStore.open(source_output))
    original = source.bundle(source_bundle)
    parent = parent or search.revisions.current()
    discovery = discovery or json.loads((search.output / "discovery.json").read_text())
    if discovery["revision"] != parent.id:
        raise ValueError("replay requires discovery belonging to its target parent")
    source_prepared = json.loads((source_output / "prepared.json").read_text())
    if source_prepared["adapter_sha256"] != search.prepared["adapter_sha256"]:
        raise ValueError("replay requires the same frozen evaluator")
    # Same evaluator is insufficient when its inputs describe a different project.
    old_request = json.loads((source_output / "request.json").read_text())
    for key in ("entry", "files", "assets", "quality", "continuous_steps", "distributed_backend"):
        if old_request.get(key) != search.request.portable().get(key):
            raise ValueError(f"replay changed fixed project semantics: {key}")
    call_lookup = {c["id"]: c for c in discovery["calls"]}
    missing = set(original.calls) - call_lookup.keys()
    if missing:
        raise ValueError(f"frozen candidate sites absent in variant: {sorted(missing)}")
    old_targets_path = str(Path(original.installer).with_suffix(".targets.json")).replace("\\", "/")
    edits = {edit.path: edit for edit in original.edits}
    if set(edits) != {original.installer, old_targets_path}:
        raise ValueError("replay supports installer bundles; additional file/plan edits need explicit migration")
    old_targets = {c["id"]: c for c in json.loads(source.store.get_artifact(edits[old_targets_path].content_ref))}
    for ident in original.calls:
        for key in ("kind", "target", "phase"):
            if old_targets[ident].get(key) != call_lookup[ident].get(key):
                raise ValueError(f"call identity changed meaning: {ident} {key}")
    mapping = dependency_map or {}
    # A replacement removes a source-run predecessor; it does not necessarily
    # depend on that predecessor's implementation. An explicit replay onto a
    # pristine variant can omit those removals because nothing is installed
    # there. True `requires` relations and nonempty target versions still need
    # their ordinary identities. Candidate code and parameters stay immutable.
    omitted_replacements = set()
    if allow_absent_replacements:
        if parent.bundles:
            raise ValueError("omitting source replacements requires an empty target version")
        omitted_replacements = set(original.replaces)
    dependencies = set(original.requires) | (set(original.replaces) - omitted_replacements)
    if dependencies - mapping.keys():
        raise ValueError("replay requires explicit identities for inherited dependencies/replacements")
    affected = set(original.calls)
    while True:
        closure = affected | {c["id"] for c in discovery["calls"] if set(c["parents"]) & affected}
        if closure == affected:
            break
        affected = closure
    groups = tuple(g["id"] for g in discovery["groups"] if set(g["calls"]) & set(original.calls))
    if original.scope == "group" and len(groups) != 1:
        raise ValueError("original single group no longer exists as one group")
    ident = "replay-" + uuid4().hex[:12]
    installer = f"opop_candidates/{ident}.py"
    targets_path = str(Path(installer).with_suffix(".targets.json")).replace("\\", "/")
    code = source.store.get_artifact(edits[original.installer].content_ref)
    contract = search.revisions.load(parent.contract_ref, ProjectContract)
    bundle = CandidateBundle(id=ident, parent_revision=parent.id, contract_ref=parent.contract_ref,
        scope=original.scope, implementation=original.implementation, groups=groups,
        calls=original.calls, affected_calls=tuple(sorted(affected)),
        edits=(FileEdit(path=installer, before_sha256=None, content_ref=search.store.put_artifact(code)),
               FileEdit(path=targets_path, before_sha256=None, content_ref=search.store.put_artifact(
                   json.dumps([call_lookup[c] for c in sorted(affected)])))),
        read_set=tuple("call:"+c for c in sorted(affected)),
        write_set=("file:"+installer, "file:"+targets_path, *("call:"+c for c in original.calls)),
        requires=tuple(mapping[c] for c in original.requires),
        replaces=tuple(mapping[c] for c in original.replaces if c not in omitted_replacements),
        external_invariants=contract.invariants, installer=installer, parameters=original.parameters)
    provenance = {"trial": ident, "replay": {"source_output": str(source_output), "source_bundle": source_bundle,
        "source_parent": original.parent_revision, "target_parent": parent.id,
        "candidate_code_sha256": hashlib.sha256(code).hexdigest(), "code_changed": False,
        "parameters_changed": False, "binding": "same call identities, current observed signatures",
        "new_generation": False,
        "source_replaces": list(original.replaces),
        "omitted_replacements": sorted(omitted_replacements),
        "replacement_migration": "explicit empty target" if omitted_replacements else "identity mapping"}}
    write_json(search.output / "trials" / ident / "replay.json", {**provenance, "bundle": bundle.model_dump(mode="json")})
    return bundle, provenance
