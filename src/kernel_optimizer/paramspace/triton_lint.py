"""Zero-cost static lint for Triton kernel sources (host-side, no GPU).

Catches a small set of *certain* compile-failure patterns before a candidate is
sent to the GPU, so the agent's own retry loop can fix them cheaply instead of
burning a WSL eval. Only patterns that ALWAYS fail to compile are hard errors;
anything merely suspicious is a warning that is surfaced but never blocks.
"""

from __future__ import annotations

import ast


class _JitBodyVisitor(ast.NodeVisitor):
    """Walk the bodies of @triton.jit-decorated functions and flag certain-failure
    constructs. We only descend into jit kernels — host code may legitimately call
    triton.next_power_of_2(), etc."""

    def __init__(self) -> None:
        self.hard_errors: list[str] = []
        self.warnings: list[str] = []
        self._in_jit = False

    @staticmethod
    def _is_jit(node: ast.FunctionDef) -> bool:
        for dec in node.decorator_list:
            # matches @triton.jit, @jit, @triton.jit(...) forms
            target = dec.func if isinstance(dec, ast.Call) else dec
            if isinstance(target, ast.Attribute) and target.attr == "jit":
                return True
            if isinstance(target, ast.Name) and target.id == "jit":
                return True
        return False

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        if self._is_jit(node):
            prev = self._in_jit
            self._in_jit = True
            self.generic_visit(node)
            self._in_jit = prev
        else:
            self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if self._in_jit and isinstance(node.func, ast.Attribute):
            # tl.next_power_of_2(...) / triton.next_power_of_2(...) inside device code
            # is not valid Triton device code and always fails to compile.
            if node.func.attr == "next_power_of_2":
                self.hard_errors.append(
                    "`next_power_of_2` is called inside a @triton.jit kernel body. It "
                    "is a host helper and is not valid device code — compute it on the "
                    "host (triton.next_power_of_2(D)) and pass the result in as a "
                    "tl.constexpr argument, then use tl.arange(0, THAT_CONSTEXPR)."
                )
        self.generic_visit(node)


def lint_triton_source(source: str) -> tuple[list[str], list[str]]:
    """Return (hard_errors, warnings). hard_errors are certain compile failures."""
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return ([f"source does not parse: {exc}"], [])
    visitor = _JitBodyVisitor()
    visitor.visit(tree)
    return (visitor.hard_errors, visitor.warnings)
