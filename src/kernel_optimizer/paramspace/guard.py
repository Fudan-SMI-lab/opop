"""Config legality guard: domain membership + restricted constraint expressions."""

from __future__ import annotations

import ast
from typing import Literal

from pydantic import BaseModel

from kernel_optimizer.models.core import (
    DeviceLimits,
    ParameterSpace,
    ParamSet,
    ParamValue,
)


class GuardRejection(BaseModel):
    reason: Literal[
        "unknown_param", "missing_param", "type_mismatch", "not_in_choices",
        "constraint_violated", "constraint_invalid",
    ]
    detail: str


_ALLOWED_BINOPS = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow)
_ALLOWED_CMPOPS = (ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Eq, ast.NotEq)


class ConstraintError(Exception):
    pass


def _eval_node(node: ast.expr, env: dict[str, ParamValue]) -> ParamValue | bool:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float, str, bool)):
            return node.value
        raise ConstraintError(f"constant {node.value!r} not allowed")
    if isinstance(node, ast.Name):
        if node.id not in env:
            raise ConstraintError(f"unknown name {node.id!r}")
        return env[node.id]
    if isinstance(node, ast.UnaryOp):
        operand = _eval_node(node.operand, env)
        if isinstance(node.op, ast.USub) and isinstance(operand, (int, float)):
            return -operand
        if isinstance(node.op, ast.Not):
            return not operand
        raise ConstraintError("unary op not allowed")
    if isinstance(node, ast.BinOp) and isinstance(node.op, _ALLOWED_BINOPS):
        left = _eval_node(node.left, env)
        right = _eval_node(node.right, env)
        if not (isinstance(left, (int, float)) and isinstance(right, (int, float))):
            raise ConstraintError("arithmetic on non-numeric values")
        op = node.op
        if isinstance(op, ast.Add):
            return left + right
        if isinstance(op, ast.Sub):
            return left - right
        if isinstance(op, ast.Mult):
            return left * right
        if isinstance(op, ast.Div):
            return left / right
        if isinstance(op, ast.FloorDiv):
            return left // right
        if isinstance(op, ast.Mod):
            return left % right
        return left**right
    if isinstance(node, ast.Compare):
        left = _eval_node(node.left, env)
        result = True
        for op, comparator in zip(node.ops, node.comparators):
            if not isinstance(op, _ALLOWED_CMPOPS):
                raise ConstraintError(
                    f"comparison op {type(op).__name__} not allowed "
                    "(only < <= > >= == !=; rewrite `in`/`is` as a disjunction of `==`)")
            right = _eval_node(comparator, env)
            checks = {
                ast.Lt: lambda a, b: a < b,
                ast.LtE: lambda a, b: a <= b,
                ast.Gt: lambda a, b: a > b,
                ast.GtE: lambda a, b: a >= b,
                ast.Eq: lambda a, b: a == b,
                ast.NotEq: lambda a, b: a != b,
            }
            result = result and checks[type(op)](left, right)
            left = right
        return result
    if isinstance(node, ast.BoolOp):
        values = [_eval_node(v, env) for v in node.values]
        return all(values) if isinstance(node.op, ast.And) else any(values)
    raise ConstraintError(f"expression node {type(node).__name__} not allowed")


def eval_constraint(expr: str, env: dict[str, ParamValue]) -> bool:
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        raise ConstraintError(f"syntax error in constraint {expr!r}: {exc}") from exc
    return bool(_eval_node(tree.body, env))


def check_config(
    space: ParameterSpace, params: ParamSet, device: DeviceLimits
) -> GuardRejection | None:
    declared = set(space.param_names())
    given = set(params.values)
    if given - declared:
        return GuardRejection(reason="unknown_param", detail=str(sorted(given - declared)))
    if declared - given:
        return GuardRejection(reason="missing_param", detail=str(sorted(declared - given)))

    kind_types = {"int": int, "float": (int, float), "str": str}
    for domain in space.domains:
        value = params.values[domain.name]
        expected = kind_types[domain.kind]
        if isinstance(value, bool) or not isinstance(value, expected):
            return GuardRejection(
                reason="type_mismatch",
                detail=f"{domain.name}={value!r} is not {domain.kind}",
            )
        if value not in domain.choices:
            return GuardRejection(
                reason="not_in_choices",
                detail=f"{domain.name}={value!r} not in {domain.choices}",
            )

    env: dict[str, ParamValue] = {**device.as_env(), **params.values}
    for constraint in space.constraints:
        try:
            ok = eval_constraint(constraint.expr, env)
        except ConstraintError as exc:
            return GuardRejection(reason="constraint_invalid", detail=str(exc))
        if not ok:
            return GuardRejection(reason="constraint_violated", detail=constraint.expr)
    return None
