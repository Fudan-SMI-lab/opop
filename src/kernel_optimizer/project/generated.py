"""Machine-readable contracts for generated artifacts, before GPU execution."""
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class Selection(BaseModel):
    model_config = ConfigDict(extra="allow")
    scope: Literal["group", "joint", "region", "execution_plan"]
    groups: list[str] = Field(min_length=1)
    affected_calls: list[str] = Field(min_length=1)
    rationale: str
    hypothesis: str


class Proposal(BaseModel):
    model_config = ConfigDict(extra="allow")
    implementation: Literal["kernel", "library", "composite", "distributed", "strategy"]
    replacement_calls: list[str] = Field(min_length=1)
    rationale: str
    setup_work: str | dict | list
    quality_risks: str | list[str]
    eliminated_calls: list[str] = Field(default_factory=list)
    requires: list[str] = Field(default_factory=list)
    replaces: list[str] = Field(default_factory=list)
    parameter_space: dict[str, Any] = Field(default_factory=dict)


class CollectionPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")
    collectors: list[str]
    requested_fields: list[str]
    unavailable_fields: dict[str, str]
    rationale: str


def validate_proposal(directory: Path):
    import ast
    import json
    import math
    from kernel_optimizer.control.direct_task import TaskSpace
    from kernel_optimizer.models.core import DeviceLimits
    proposal = Proposal.model_validate_json((directory/"proposal.json").read_bytes())
    space = TaskSpace.model_validate(proposal.parameter_space)
    names = [p.name for p in space.params]
    if len(set(names)) != len(names) or any(not p.choices for p in space.params):
        raise ValueError("parameter names must be unique and domains nonempty")
    for parameter in space.params:
        expected = {"int": int, "float": (int, float), "str": str}[parameter.kind]
        if any(isinstance(v, bool) or not isinstance(v, expected) or isinstance(v, float) and not math.isfinite(v)
               for v in parameter.choices):
            raise ValueError(f"invalid {parameter.kind} choices for {parameter.name}")
    allowed_names = set(names) | DeviceLimits().as_env().keys()
    # Match the existing expression language without changing the legacy guard.
    allowed = (ast.Expression, ast.Constant, ast.Name, ast.Load, ast.UnaryOp, ast.USub, ast.Not,
        ast.BinOp, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow,
        ast.Compare, ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Eq, ast.NotEq, ast.BoolOp, ast.And, ast.Or)
    for constraint in space.constraints:
        tree = ast.parse(constraint.expr, mode="eval")
        for node in ast.walk(tree):
            if not isinstance(node, allowed):
                raise ValueError(f"unsupported constraint expression {type(node).__name__}; use arithmetic, comparisons and and/or/not; rewrite implications as (not A) or B")
            if isinstance(node, ast.Name) and node.id not in allowed_names:
                raise ValueError(f"unknown constraint parameter/device field {node.id}")
            if isinstance(node, ast.Constant) and not isinstance(node.value, (str, int, float, bool)):
                raise ValueError("unsupported constraint constant")
    selection = directory/"inputs/selection.json"
    if selection.exists():
        allowed_calls = set(json.loads(selection.read_text())["affected_calls"])
        if set(proposal.replacement_calls)-allowed_calls:
            raise ValueError("replacement_calls must belong to the supplied selection effect closure")
    search = directory/"inputs/search.json"
    if search.exists():
        installed = {b["id"]: b for b in json.loads(search.read_text()).get("installed_bundles", [])}
        if (set(proposal.requires) | set(proposal.replaces))-installed.keys() or set(proposal.requires) & set(proposal.replaces):
            raise ValueError("requires/replaces must reference installed bundles and cannot overlap")
        for ident, bundle in installed.items():
            if ident not in proposal.replaces and set(bundle["calls"]) & set(proposal.replacement_calls):
                raise ValueError(f"overlapping replacement calls require explicit replaces for installed bundle {ident}")
    compile((directory/"candidate.py").read_bytes(), "candidate.py", "exec")
