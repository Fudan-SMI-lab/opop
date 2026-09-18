"""Fresh source snapshots with statically reachable local Python dependencies."""

from __future__ import annotations

import ast
from pathlib import Path

from kernel_optimizer.models.core import ParamSet
from kernel_optimizer.paramspace.materializer import extract_defaults, materialize


def copy_dependencies(source: Path, destination: Path, root: Path | None = None) -> Path:
    """Preserve local import layout without executing or copying unrelated modules."""
    source = source.resolve()
    root = (root or source.parent).resolve()
    pending = [source]
    visited: set[Path] = set()
    while pending:
        current = pending.pop()
        if current in visited:
            continue
        visited.add(current)
        content = current.read_bytes()
        target = destination / current.relative_to(root)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        for node in ast.walk(ast.parse(content)):
            names: list[str] = []
            bases = [current.parent, root]
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                names = [module, *(f"{module}.{alias.name}".strip(".") for alias in node.names)]
                if node.level:
                    bases = [current.parents[node.level - 1]]
            for base in bases:
                for name in names:
                    path = base.joinpath(*name.split("."))
                    for dependency in (path.with_suffix(".py"), path / "__init__.py"):
                        if dependency.is_file() and dependency.resolve().is_relative_to(root):
                            pending.append(dependency.resolve())
                            for parent in dependency.parents:
                                if parent == root:
                                    break
                                init = parent / "__init__.py"
                                if init.is_file():
                                    pending.append(init.resolve())
    return destination / source.relative_to(root)


def stage_candidate(candidate: Path, destination: Path, params: ParamSet | None) -> tuple[Path, ParamSet]:
    """Materialize only this candidate's native PARAMS, or preserve exact default bytes."""
    source = candidate.read_text(encoding="utf-8")
    selected = params if params is not None else ParamSet(values=extract_defaults(source))
    rendered = materialize(source, selected) if params is not None else None
    root = next((p for p in candidate.resolve().parents
                 if (p / "task/self_test.json").is_file()), candidate.resolve().parent)
    staged = copy_dependencies(candidate, destination, root)
    if rendered is not None:
        staged.write_bytes(rendered.encode("utf-8"))
    return staged, selected
