"""Render the new A3 report section against a REAL run's events.jsonl.

The unit tests use fixtures, and a fixture can be wrong in a way no test detects -- this project has
recorded that exact failure ("a fixture invented to match the reader proves nothing"). So before
believing the section, run it over the actual log it was written for and read the output.

Imports the module by file path so it can test a wall_report.py that has been scp'd to a box without
touching that box's installed checkout.
"""
import importlib.util
import json
import os
import sys

mod_path, rd = sys.argv[1], sys.argv[2]
spec = importlib.util.spec_from_file_location("wall_report_under_test", mod_path)
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

out = mod.wall_lines(evs)
print("\n".join(out))
print()
print(f"--- {len(out)} lines rendered from {len(evs)} events")
a3 = [ln for ln in out if "A3" in ln or "FREED" in ln or "尚无后代" in ln]
print(f"--- A3 rows: {len(a3)}")
if not a3:
    print("!!! the A3 section rendered NOTHING on a run that has an attributed wall")
