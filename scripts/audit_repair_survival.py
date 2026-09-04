"""Does the parameterizer routinely discard the repair it was handed?

Loop A is: repair(broken) -> fixed -> parameterizer(fixed) -> parameterized -> validate.
The thing VALIDATED is the parameterizer's output, so a parameterizer that rewrites from
scratch silently reverts the repair. Measure how often, across runs, by comparing each
repair's `fixed.py` against the `parameterized.py` that consumed it.
"""

import hashlib
import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def sig(text: str) -> tuple:
    """Cheap structural signature: size class, dot count, kernel count."""
    return (len(text), text.count("tl.dot("), text.count("@triton.jit"))


print(f"{'run':32}{'repair sandbox':22}{'fixed.py':>22}  ->  consumed by, emitted")
total = reverted = 0
for run in sorted((REPO / "runs").glob("run-l3-*")):
    sb = run / "sandboxes"
    if not sb.exists():
        continue
    # Map every parameterizer sandbox by the sha of the source it was GIVEN.
    given: dict[str, tuple[str, str]] = {}
    for d in sorted(sb.glob("parameterizer-*")):
        src = d / "candidate" / "source.py"
        out = next((p for p in (d / "candidate" / "parameterized.py",) if p.exists()), None)
        if not src.exists() or out is None:
            continue
        s = src.read_text(encoding="utf-8")
        o = out.read_text(encoding="utf-8")
        given[hashlib.sha256(s.encode()).hexdigest()] = (d.name, o)

    for d in sorted(sb.glob("repair-*")):
        fx = d / "candidate" / "fixed.py"
        if not fx.exists():
            continue
        t = fx.read_text(encoding="utf-8")
        h = hashlib.sha256(t.encode()).hexdigest()
        if h not in given:
            continue
        pname, emitted = given[h]
        total += 1
        a, b = sig(t), sig(emitted)
        bad = b[1] < a[1] or abs(b[0] - a[0]) > max(200, 0.1 * a[0])
        if bad:
            reverted += 1
        flag = "  <-- REVERTED" if bad else ""
        print(f"{run.name:32}{d.name:22}{str(a):>22}  ->  {pname[:18]:20}{str(b)}{flag}")

print(f"\n{reverted} of {total} repair->parameterize hand-offs lost the repair's structure")
print("signature = (bytes, tl.dot count, @triton.jit count)")
