import re


def stored_dependencies(ttir: str) -> dict[int, frozenset[int]]:
    """Conservative straight-line TTIR store/load provenance; unknown flows do not prove use."""
    dependencies: dict[str, set[int]] = {}
    loaded: dict[str, set[int]] = {}
    stores: dict[int, set[int]] = {}
    for arg in re.findall(r"%arg\d+", ttir):
        dependencies[arg] = {int(arg[4:])}
    for line in ttir.splitlines():
        if "tt.store " in line:
            operands = re.findall(r"%[\w.$]+(?:#\d+)?", line.split("tt.store", 1)[1].split(":", 1)[0])
            if len(operands) >= 2:
                for output in dependencies.get(operands[0], set()):
                    stores.setdefault(output, set()).update(loaded.get(operands[1], set()))
        if "=" not in line:
            continue
        left, right = line.split("=", 1)
        results = re.findall(r"%[\w.$]+(?:#\d+)?", left)
        operands = re.findall(r"%[\w.$]+(?:#\d+)?", right.split(":", 1)[0])
        origin = set().union(*(dependencies.get(v, set()) for v in operands))
        reads = set().union(*(loaded.get(v, set()) for v in operands))
        if "tt.load " in right and operands:
            reads.update(dependencies.get(operands[0], set()))
        for result in results:
            dependencies[result] = origin
            loaded[result] = reads
    return {output: frozenset(inputs) for output, inputs in stores.items() if inputs}
