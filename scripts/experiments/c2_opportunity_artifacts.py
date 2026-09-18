"""Export only the full-policy incumbent with its actual native space and helper namespace."""

import hashlib
import shutil
from pathlib import Path

from kernel_optimizer.models.core import Candidate, ParameterSpace, sha256_text
from kernel_optimizer.paramspace.materializer import extract_defaults
from kernel_optimizer.store.run_store import RunStore
from scripts.experiments.c2_information_inputs import InformationResult
from scripts.experiments.c2_local_inputs import InputError, Shared
from scripts.experiments.c2_method_files import copy_helpers
from scripts.experiments.c2_opportunity_records import Incumbent


def export_incumbent(shared: Shared, result: InformationResult, root: Path) -> Incumbent:
    opportunity = root / "information"
    source = opportunity / result.selected_artifact
    params, backend = shared.parent.params, shared.backend
    space = ParameterSpace(candidate_id=shared.parent.candidate_id, space_id=shared.parent.space_id,
        source_sha=sha256_text(shared.source), domains=shared.space.params, constraints=shared.space.constraints)
    helper_root = opportunity / "common/imports"
    if result.selected == "child":
        child = result.child
        if child is None or child.selected is None or child.selected_trial is None or child.selected_space is None:
            raise InputError("promoted child lacks a native selected record/space")
        params, space = child.selected.params, child.selected_space
        candidates = [Candidate.model_validate(e.payload["candidate"]) for e in RunStore.open(opportunity / "retune").iter_events()
                      if e.type == "CANDIDATE_REGISTERED"]
        candidate = next((c for c in candidates if c.candidate_id == child.selected.candidate_id), None)
        if candidate is None:
            raise InputError("selected child lacks a registered backend")
        backend = candidate.backend
        helper_root = opportunity / "retune/inputs"
    if extract_defaults(source.read_text(encoding="utf-8")) != params.values:
        raise InputError("selected artifact does not contain selected native parameters")
    export = root / "export"
    export.mkdir()
    selected = export / "selected.py"
    shutil.copyfile(source, selected)
    helpers = copy_helpers(tuple(helper_root.rglob("*.py")), helper_root, export / "imports")
    return Incumbent(selected=result.selected, source=selected, source_sha256=hashlib.sha256(selected.read_bytes()).hexdigest(),
        params=params, backend=backend, space=space, helper_root=export / "imports", helpers=helpers,
        helper_sha256={p.relative_to(export / "imports").as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in helpers})


def stage_incumbent(incumbent: Incumbent, target: Path) -> Path:
    if hashlib.sha256(incumbent.source.read_bytes()).hexdigest() != incumbent.source_sha256:
        raise InputError("heldout source differs from frozen incumbent")
    if extract_defaults(incumbent.source.read_text(encoding="utf-8")) != incumbent.params.values:
        raise InputError("heldout parameters differ from frozen incumbent")
    hashes = {p.relative_to(incumbent.helper_root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in incumbent.helpers}
    if hashes != incumbent.helper_sha256:
        raise InputError("heldout helper bytes differ from frozen incumbent")
    target.mkdir(parents=True)
    source = target / "selected.py"
    shutil.copyfile(incumbent.source, source)
    copy_helpers(incumbent.helpers, incumbent.helper_root, target / "imports")
    return source
