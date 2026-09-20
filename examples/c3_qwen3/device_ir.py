import re
from collections.abc import Mapping
from typing import Final

SSA: Final = r"%[\w.$-]+(?:#\d+)?"


def _parenthesized(text: str, start: int) -> str | None:
    depth = 0
    for index in range(start, len(text)):
        depth += int(text[index] == "(") - int(text[index] == ")")
        if depth == 0:
            return text[start + 1:index]
    return None


def _roots(text: str, argument_indices: Mapping[str, int] | None) -> dict[str, set[int]]:
    signatures = list(re.finditer(r"(?m)^\s*tt\.func\b[^\n(]*\(", text))
    if not signatures:
        # Historical fragment probes have no signature; production supplies a JIT mapping.
        return {arg: {int(arg[4:])} for arg in re.findall(r"%arg\d+\b", text)} if argument_indices is None else {}
    if len(signatures) != 1:
        return {}
    signature = _parenthesized(text, signatures[0].end() - 1)
    if signature is None:
        return {}
    names = tuple(re.findall(rf"({SSA})\s*:", signature))
    if argument_indices is None:
        return {name: {i} for i, name in enumerate(names)}
    if all(name[1:] in argument_indices for name in names):
        return {name: {argument_indices[name[1:]]} for name in names}
    if names == tuple(f"%arg{i}" for i in range(len(argument_indices))):
        return {name: {slot} for name, slot in zip(names, sorted(argument_indices.values()), strict=True)}
    return {}


def stored_dependencies(ttir: str, argument_indices: Mapping[str, int] | None = None) -> dict[int, frozenset[int]]:
    """Single-function TTIR provenance; optional mapping returns original JIT argument slots."""
    # Keep quoted operation spellings, but never interpret SSA-like text inside locations/strings.
    text = re.sub(r'"(?:\\.|[^"\\])*"',
                  lambda m: m[0][1:-1] if m[0] in ('"tt.load"', '"tt.store"', '"tt.reduce"') else "", ttir)
    text = re.sub(r"//[^\n]*", "", text)
    dependencies = _roots(text, argument_indices)
    loaded: dict[str, set[int]] = {}
    stores: dict[int, set[int]] = {}
    lines = iter(text.splitlines())
    for line in lines:
        assignment = re.match(rf"^\s*({SSA})\s*=\s*(.+)", line)
        store = re.match(r"^\s*tt\.store\b(.*)", line)
        if assignment is None and store is None:
            continue
        right = assignment[2] if assignment else line.strip()
        generic = re.match(r"[\w.]+\s*\(", right)
        if generic:
            operand_text = _parenthesized(right, generic.end() - 1)
            while operand_text is None:
                continuation = next(lines, None)
                if continuation is None:
                    return {}
                right += " " + continuation
                operand_text = _parenthesized(right, generic.end() - 1)
        else:
            operand_text = re.split(r"\bloc\s*\(", right.split(":", 1)[0], maxsplit=1)[0]
        operands = re.findall(SSA, operand_text)
        if store:
            if len(operands) >= 2:
                for output in dependencies.get(operands[0], set()):
                    stores.setdefault(output, set()).update(loaded.get(operands[1], set()))
            continue
        origin = set().union(*(dependencies.get(v, set()) for v in operands))
        reads = set().union(*(loaded.get(v, set()) for v in operands))
        if re.match(r"tt\.load\b", right) and operands:
            reads.update(dependencies.get(operands[0], set()))
        if assignment:
            dependencies[assignment[1]] = origin
            loaded[assignment[1]] = reads
    return {output: frozenset(inputs) for output, inputs in stores.items() if inputs}
