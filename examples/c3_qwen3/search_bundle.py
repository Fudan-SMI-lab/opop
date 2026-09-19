"""Standalone cumulative bundles: structural identity, legal anchors and frozen export."""

import ast
from pathlib import Path
from collections.abc import Mapping
from typing import assert_never

from kernel_optimizer.control.direct_task import TaskSpace
from kernel_optimizer.control.families import structural_signature
from kernel_optimizer.models.core import DeviceLimits, ParameterSpace, ParamSet
from kernel_optimizer.paramspace.guard import check_config
from .model_binding import BundleDocument, BundleSpec, load_bundle
from .runner_records import RunnerError
from .search_records import Selection


class WithoutParams(ast.NodeTransformer):
    def visit_Assign(self, node: ast.Assign) -> ast.Assign | None:
        if any(isinstance(target, ast.Name) and target.id == "PARAMS" for target in node.targets):
            return None
        self.generic_visit(node)
        return node

    def visit_AnnAssign(self, node: ast.AnnAssign) -> ast.AnnAssign | None:
        match node.target:
            case ast.Name(id="PARAMS"):
                return None
            case ast.expr():
                self.generic_visit(node)
                return node
            case unreachable:
                assert_never(unreachable)


def baseline_bundle(directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "operators.py").write_text("# Original model dispatch; no replacements.\n", encoding="utf-8")
    document = BundleDocument(entry="operators.py", sites=(), files=("operators.py",), helpers=(),
        params={}, space={"params": [], "constraints": []}, parent_bundle=None,
        cumulative_from_original_baseline=True, source_body_rewrite_required=True)
    path = directory / "bundle.json"
    path.write_text(document.model_dump_json(indent=2), encoding="utf-8")
    return path.resolve()


def bundle_structure(path: Path) -> Mapping[str, str]:
    bundle = load_bundle(path, {})
    return {name: structural_signature(ast.unparse(WithoutParams().visit(ast.parse(source.decode("utf-8")))))
            for name, source in bundle.sources.items()}


def validate_child(bundle: BundleSpec, parent: Path) -> TaskSpace:
    ancestor = load_bundle(parent, {})
    if bundle.document.parent_bundle != ancestor.bundle_sha256:
        raise RunnerError("candidate parent_bundle differs from this opportunity's incumbent")
    if not set(s.site_id for s in ancestor.document.sites) <= {s.site_id for s in bundle.document.sites}:
        raise RunnerError("candidate dropped an accepted cumulative site")
    if not bundle.document.sites or bundle_structure(bundle.path) == bundle_structure(parent):
        raise RunnerError("candidate has no non-PARAMS bundle computation change")
    for name, source in bundle.sources.items():
        compile(source, name, "exec")
    return TaskSpace.model_validate(bundle.document.space)


def parameter_space(bundle: BundleSpec) -> ParameterSpace:
    space = TaskSpace.model_validate(bundle.document.space)
    return ParameterSpace(space_id=bundle.bundle_sha256, candidate_id=bundle.bundle_sha256,
        source_sha=bundle.bundle_sha256, domains=space.params, constraints=space.constraints)


def anchors(bundle: BundleSpec, recommendations: tuple[ParamSet, ...], device: DeviceLimits) -> tuple[ParamSet, ...]:
    proposed = (ParamSet.model_validate({"values": bundle.document.params}), *recommendations)
    accepted: list[ParamSet] = []
    for params in proposed:
        if check_config(parameter_space(bundle), params, device) is None and params not in accepted:
            accepted.append(params)
    return tuple(accepted)


def export_selection(selection: Selection, directory: Path) -> Selection:
    bundle = load_bundle(selection.bundle, selection.params.values)
    if (bundle.bundle_sha256 != selection.bundle_sha256 or dict(bundle.source_hashes) != selection.source_hashes
            or bundle.params_sha256 != selection.params_sha256):
        raise RunnerError("selected bundle bytes or effective params changed before export")
    directory.mkdir(parents=True, exist_ok=False)
    for name, source in bundle.sources.items():
        target = directory / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source)
    path = directory / "bundle.json"
    path.write_bytes(bundle.path.read_bytes())
    result = selection.model_copy(update={"bundle": path.resolve()})
    (directory / "selection.json").write_text(result.model_dump_json(indent=2), encoding="utf-8")
    return result
