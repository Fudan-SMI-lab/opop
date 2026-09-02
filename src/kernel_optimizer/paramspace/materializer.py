"""PARAMS-dict materialization: AST-located literal span replacement.

The candidate contract requires exactly one module-level `PARAMS = {...}` dict
of string keys to int/float/str literals. Materialization replaces only that
byte span and verifies the rest of the file is untouched.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass

from kernel_optimizer.models.core import ParamSet, ParamValue

ERROR_KINDS = (
    "NO_PARAMS_DICT",
    "MULTIPLE_PARAMS",
    "NON_LITERAL_VALUE",
    "NON_STRING_KEY",
    "KEY_MISMATCH",
    "REPARSE_FAILED",
    "SYNTAX_ERROR",
)


class MaterializeError(Exception):
    def __init__(self, kind: str, detail: str):
        self.kind = kind
        self.detail = detail
        super().__init__(f"{kind}: {detail}")


@dataclass(frozen=True)
class ParamsSpan:
    start: int  # byte offset of the dict literal '{'
    end: int  # byte offset one past the closing '}'
    defaults: dict[str, ParamValue]


def _literal_value(node: ast.expr) -> ParamValue:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float, str)):
        if isinstance(node.value, bool):
            raise MaterializeError("NON_LITERAL_VALUE", "bool values are not allowed in PARAMS")
        return node.value
    if (
        isinstance(node, ast.UnaryOp)
        and isinstance(node.op, ast.USub)
        and isinstance(node.operand, ast.Constant)
        and isinstance(node.operand.value, (int, float))
    ):
        return -node.operand.value
    raise MaterializeError(
        "NON_LITERAL_VALUE",
        f"PARAMS values must be int/float/str literals, got {ast.dump(node)[:120]}",
    )


def _line_offsets(source: str) -> list[int]:
    offsets = [0]
    for line in source.splitlines(keepends=True):
        offsets.append(offsets[-1] + len(line))
    return offsets


def find_params_span(source: str) -> ParamsSpan:
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise MaterializeError("SYNTAX_ERROR", str(exc)) from exc

    assigns: list[tuple[ast.Assign, ast.Dict]] = []
    for node in tree.body:  # module level only
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "PARAMS":
                    if not isinstance(node.value, ast.Dict):
                        raise MaterializeError(
                            "NO_PARAMS_DICT", "PARAMS must be assigned a dict literal"
                        )
                    assigns.append((node, node.value))
    if not assigns:
        raise MaterializeError("NO_PARAMS_DICT", "no module-level `PARAMS = {...}` found")
    if len(assigns) > 1:
        raise MaterializeError("MULTIPLE_PARAMS", "multiple module-level PARAMS assignments")

    _, dict_node = assigns[0]
    defaults: dict[str, ParamValue] = {}
    for key_node, value_node in zip(dict_node.keys, dict_node.values):
        if not (isinstance(key_node, ast.Constant) and isinstance(key_node.value, str)):
            raise MaterializeError("NON_STRING_KEY", "PARAMS keys must be string literals")
        defaults[key_node.value] = _literal_value(value_node)

    offsets = _line_offsets(source)
    start = offsets[dict_node.lineno - 1] + dict_node.col_offset
    end = offsets[dict_node.end_lineno - 1] + dict_node.end_col_offset
    return ParamsSpan(start=start, end=end, defaults=defaults)


def extract_defaults(source: str) -> dict[str, ParamValue]:
    return find_params_span(source).defaults


def _render_dict(values: dict[str, ParamValue]) -> str:
    parts = [f"    {key!r}: {value!r}," for key, value in values.items()]
    return "{\n" + "\n".join(parts) + "\n}"


def materialize(source: str, params: ParamSet) -> str:
    span = find_params_span(source)
    if set(span.defaults) != set(params.values):
        raise MaterializeError(
            "KEY_MISMATCH",
            f"PARAMS keys {sorted(span.defaults)} != requested {sorted(params.values)}",
        )
    # Preserve the source key order.
    ordered = {key: params.values[key] for key in span.defaults}
    new_source = source[: span.start] + _render_dict(ordered) + source[span.end :]

    # Verify: re-parse, re-extract, and confirm the remainder is untouched.
    try:
        new_defaults = extract_defaults(new_source)
    except MaterializeError as exc:
        raise MaterializeError("REPARSE_FAILED", f"materialized source invalid: {exc}") from exc
    if new_defaults != ordered:
        raise MaterializeError("REPARSE_FAILED", "re-extracted PARAMS do not match request")
    new_span = find_params_span(new_source)
    if (source[: span.start] != new_source[: new_span.start]
            or source[span.end :] != new_source[new_span.end :]):
        raise MaterializeError("REPARSE_FAILED", "bytes outside the PARAMS span changed")
    return new_source
