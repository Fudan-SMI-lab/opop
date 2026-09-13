"""Render the 2b(2)/2d section against a real run, and check it against the known answer.

This section has a POSITIVE CONTROL available, which is unusual and worth using: the per-dimension
ordering was measured by hand on two boxes before any of this code existed
(`candidate_aten_ops` 88%/100% at the top, `shared_bytes` 36%/20% at the bottom). If the reader
reproduces that ordering on the same kind of data, it is reading the ledger correctly; if it inverts
it or flattens it, the reader is wrong -- and a plausible-looking table would otherwise be believed.

Imports by file path so a module can be checked on a box without touching its installed checkout.
"""
import importlib.util
import json
import os
import sys

mod_path, rd = sys.argv[1], sys.argv[2]
spec = importlib.util.spec_from_file_location("accuracy_under_test", mod_path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

evs = []
with open(os.path.join(rd, "events.jsonl"), encoding="utf-8") as fh:
    for ln in fh:
        ln = ln.strip()
        if ln:
            try:
                evs.append(json.loads(ln))
            except Exception:
                pass

out = mod.accuracy_lines(evs)
print("\n".join(out))
print()
print(f"--- {len(out)} lines from {len(evs)} events")
if not out:
    print("!!! section rendered nothing -- either no round was reconciled, or the reader is broken")
    raise SystemExit(1)
# The control: aten_ops should sit above shared_bytes in the table.
body = "\n".join(out)
i_ops = body.find("candidate_aten_ops")
i_shared = body.find("shared_bytes")
print()
if i_ops < 0 or i_shared < 0:
    print("CONTROL INCONCLUSIVE: one of the two anchor dimensions is absent from this run")
else:
    print(f"CONTROL: aten_ops at char {i_ops}, shared_bytes at {i_shared} -- "
          f"{'consistent with the hand-measured ordering' if i_ops < i_shared else '*** INVERTED vs the hand measurement; check the reader'}")
