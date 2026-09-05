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


def _mode_gated_kernel_branches(tree: ast.Module) -> list[str]:
    """Improvement M: flag `if <...>.training:` branches that launch a Triton kernel on
    either side. The harness never calls `.eval()` or `.train()`, so the reference's
    mode is FIXED for the whole run and exactly one side of such a branch is ever
    executed — the other side is dead code. That is dangerous rather than merely
    wasteful, because a candidate whose fast path sits on the dead side still passes
    correctness (the live fallback is correct) and still produces timings, so nothing
    in the pipeline reports a problem while every trial measures the unoptimized path.

    Observed on L3:21 `cand-c0b3b7cd`: 31 trials, all `complete`, best 25.1 ms, and
    every one launched only `_depthwise_kernel` — the train-mode fallback — while the
    advertised fused `_depthwise_bn_relu6_kernel` sat in the `else` branch and never
    ran. Both sides launch exactly one kernel, so an asymmetry test would miss it; the
    branch existing at all is the signal.

    A `.training` branch that launches NO Triton kernel on either side is the benign
    pattern (choosing between two torch formulations) and is not reported: 16 of the
    33 such branches on disk are that case. Branching on `.training` is legal, so this
    is a WARNING and never a hard error.
    """
    jit_kernels = jit_kernel_names(tree)
    findings: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        test = node.test
        # `if m.training:` / `if not m.training:` / `if self.bn.training:`
        if isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not):
            test = test.operand
        if not (isinstance(test, ast.Attribute) and test.attr == "training"):
            continue
        launched_if = _launched_kernels(node.body, jit_kernels)
        launched_else = _launched_kernels(node.orelse, jit_kernels)
        if not launched_if and not launched_else:
            continue  # benign: a pure-torch mode choice, no kernel is stranded
        findings.append(
            f"this file launches Triton kernels inside an `if ...training:` branch "
            f"(`if` side: {sorted(launched_if) or 'none'}; `else` side: "
            f"{sorted(launched_else) or 'none'}). The harness evaluates the reference in "
            "ONE fixed mode and never calls `.eval()`/`.train()`, so only one side of "
            "this branch EVER runs and the kernels on the other side are dead code. "
            "That is the worst case to get wrong: correctness still passes (the live "
            "branch is correct) and timings are still produced, so nothing reports an "
            "error while every trial measures the unoptimized path. Read "
            "`task/eval_semantics.md` for the mode the harness actually uses and put "
            "your optimized kernel on THAT side. In TRAIN mode BatchNorm uses the "
            "current batch's mean/var, so its scale/shift are unknown until the batch "
            "has been reduced and CANNOT be folded into preceding weights — reduce the "
            "batch statistics first (two-pass or partial reduction), then fuse."
        )
    return findings


def jit_kernel_names(tree: ast.Module) -> set[str]:
    """Names of module-level functions decorated with `@triton.jit` (or `@jit`)."""
    names: set[str] = set()
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef):
            continue
        for dec in node.decorator_list:
            target = dec.func if isinstance(dec, ast.Call) else dec  # @triton.autotune(...)
            attr = target.attr if isinstance(target, ast.Attribute) else (
                target.id if isinstance(target, ast.Name) else ""
            )
            if attr in ("jit", "autotune", "heuristics"):
                names.add(node.name)
                break
    return names


def _launched_kernels(body: list[ast.stmt], jit_kernels: set[str]) -> set[str]:
    """Kernel names launched as `kernel[grid](...)` in a statement list. Matches only
    names known to be `@triton.jit` functions, so ordinary subscript calls like
    `self.layers[2](x)` are not mistaken for launches."""
    found: set[str] = set()
    for stmt in body:
        for node in ast.walk(stmt):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Subscript)):
                continue
            base = node.func.value
            name = base.id if isinstance(base, ast.Name) else (
                base.attr if isinstance(base, ast.Attribute) else ""
            )
            if name in jit_kernels:
                found.add(name)
    return found


def device_helper_names(tree: ast.Module, jit_names: set[str]) -> set[str]:
    """Of `jit_names`, those CALLED BY NAME from inside another `@triton.jit` body.

    Triton inlines a jit function invoked as `f(...)` (no `[grid]`) into its caller, so
    it produces no separate compiled kernel and never appears in a trial's
    `kernel_names` — even though it executes on every trial. Anything that treats
    "absent from kernel_names" as "never ran" must exclude these, or every candidate
    factored into device helpers is falsely reported as carrying dead code (L3:43
    `cand-d257924a`: `_qk_scores` is inlined into two kernels that launched on all 76
    trials).
    """
    helpers: set[str] = set()
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef) or node.name not in jit_names:
            continue
        for inner in ast.walk(node):
            # a plain call `f(...)`; a launch is Call(func=Subscript(...)) instead
            if isinstance(inner, ast.Call) and isinstance(inner.func, ast.Name):
                if inner.func.id in jit_names and inner.func.id != node.name:
                    helpers.add(inner.func.id)
    return helpers


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
    warnings.extend(_mode_gated_kernel_branches(tree))
    return (visitor.hard_errors, warnings)
