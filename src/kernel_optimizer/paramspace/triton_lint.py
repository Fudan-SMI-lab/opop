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


# dtype tokens that indicate a low-precision tensor-core path (improvement L).
_LOWP_DTYPE_TOKENS = ("float16", "bfloat16")
# values a dtype/precision PARAMS knob would carry (name-agnostic detection).
_DTYPE_KNOB_VALUES = {"fp16", "bf16", "tf32", "ieee", "float16", "bfloat16", "float32"}


def _has_hardcoded_lowp_cast(tree: ast.Module) -> bool:
    """True if the source casts to fp16/bf16 anywhere (e.g. `.to(tl.float16)`,
    `x.to(torch.float16)`, `tl.float16` literal). Name-agnostic to the exact call
    form — we only care that a low-precision dtype token appears in the code."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr in _LOWP_DTYPE_TOKENS:
            return True
        # bare Name (e.g. `float16` imported directly) — rare but cheap to catch
        if isinstance(node, ast.Name) and node.id in _LOWP_DTYPE_TOKENS:
            return True
    return False


def _params_has_dtype_knob(tree: ast.Module) -> bool:
    """True if the module-level PARAMS dict has a knob whose VALUE looks like a
    dtype/precision selector (name-agnostic: matches on the value, not the key, so
    DOT_PRECISION / COMPUTE_DTYPE / any name is recognized)."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
        if "PARAMS" not in targets or not isinstance(node.value, ast.Dict):
            continue
        for v in node.value.values:
            if isinstance(v, ast.Constant) and isinstance(v.value, str):
                if v.value.strip().lower() in _DTYPE_KNOB_VALUES:
                    return True
    return False


def lint_triton_source(source: str) -> tuple[list[str], list[str]]:
    """Return (hard_errors, warnings). hard_errors are certain compile failures."""
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return ([f"source does not parse: {exc}"], [])
    visitor = _JitBodyVisitor()
    visitor.visit(tree)
    warnings = list(visitor.warnings)

    # Improvement L: if the kernel hardcodes a low-precision (fp16/bf16) cast but
    # exposes NO dtype/precision PARAMS knob, the tuner cannot compare precisions and
    # a fp16 path is never evaluated in isolation. This is a WARNING only (never
    # blocks): a false positive costs at most a little agent attention, and whether
    # fp16 actually helps is judged by measured tuning data, not this lint.
    if _has_hardcoded_lowp_cast(tree) and not _params_has_dtype_knob(tree):
        warnings.append(
            "This kernel casts to fp16/bf16 (a tensor-core precision choice) but its "
            "PARAMS has no dtype/precision knob. Expose the compute dtype as a PARAMS "
            'entry (e.g. "COMPUTE_DTYPE": "fp16" with choices like '
            '["fp16","bf16","tf32","ieee"]) that drives BOTH the input cast and the '
            "tl.dot precision, so the tuner can compare precisions on real measurements "
            "instead of leaving fp16 hardcoded and unmeasured (accumulator stays fp32)."
        )
    return (visitor.hard_errors, warnings)
