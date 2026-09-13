"""Why was the RESCUED rewrite rejected at the witness gate -- the rescue, or the candidate?

THE QUESTION. `cand-8071c78e` is the control arm's rewrite that came out of the sandbox rescue after a
transport failure, so its `change_summary` is a placeholder and its `hypothesis_id` is empty. It then
failed the witness gate on its first attempt. Two very different causes:

  (a) the RESCUE produced a damaged or truncated source -- which would mean `check_output` did not catch
      what it is there to catch, and the rescue path needs hardening;
  (b) the candidate itself is numerically wrong -- ordinary, and exactly what loop A exists for.

The discriminator is in the rejection detail: a syntax/contract failure or a missing-symbol traceback
points at (a); a correctness mismatch with statistics over real outputs points at (b), because a
truncated file could not have produced outputs at all.

Also prints whether the source on disk parses and has the single module-level PARAMS the contract
requires, since that is the specific thing a truncated rescue would break.

Run on box4:
  /root/autodl-tmp/orch-venv/bin/python /root/probe-clean/rescued_rewrite_rejection.py
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

BASE = Path("/root/autodl-tmp/opop-workspace/opop-glm/runs-v3")
RUN = "run-l3-43-20260913-202332"
ARM = "s7-control"
CAND = "cand-8071c78e"


def main() -> int:
    run = BASE / ARM / RUN
    ev = []
    for line in (run / "events.jsonl").open(encoding="utf-8"):
        line = line.strip()
        if line:
            try:
                ev.append(json.loads(line))
            except ValueError:
                pass

    for e in ev:
        if e.get("type") != "SPACE_REJECTED":
            continue
        p = e.get("payload") or {}
        if str(p.get("candidate_id")) != CAND:
            continue
        print("=" * 78)
        print("SPACE_REJECTED  attempt=%s  reason=%s" % (p.get("attempt"), p.get("reason")))
        print((p.get("detail") or "")[:2200])
        print()

    for e in ev:
        if e.get("type") != "REPAIR_PRODUCED":
            continue
        p = e.get("payload") or {}
        if str(p.get("candidate_id")) != CAND:
            continue
        print("=" * 78)
        print("REPAIR_PRODUCED for %s" % CAND)
        for k in ("diagnosis", "change_summary"):
            if p.get(k):
                print("  %s: %s" % (k, str(p[k])[:900]))

    # Does the candidate's source on disk satisfy the contract? A truncated rescue would fail HERE.
    print()
    print("=" * 78)
    cands = sorted(run.glob("candidates/%s*/**/*.py" % CAND)) or sorted(
        run.glob("candidates/%s/*.py" % CAND))
    if not cands:
        cands = [p for p in run.glob("candidates/**/*.py") if CAND in str(p)]
    if not cands:
        print("no source file found under %s for %s -- cannot check the contract" % (run, CAND))
        return 1
    for f in cands[:3]:
        try:
            src = f.read_text(encoding="utf-8")
        except OSError as exc:
            print("%s: unreadable (%s)" % (f.name, exc))
            continue
        print("%s  %d bytes" % (f.name, len(src)))
        try:
            tree = ast.parse(src)
        except SyntaxError as exc:
            print("  DOES NOT PARSE: %s => the rescue produced a damaged source, and check_output")
            print("  did not catch it. That is cause (a) and the rescue path needs hardening. (%s)"
                  % exc)
            continue
        params = [n for n in tree.body if isinstance(n, ast.Assign)
                  and any(isinstance(t, ast.Name) and t.id == "PARAMS" for t in n.targets)]
        has_model = any(isinstance(n, ast.ClassDef) and n.name == "ModelNew" for n in tree.body)
        print("  parses OK; module-level PARAMS assignments: %d; class ModelNew present: %s"
              % (len(params), has_model))
        if len(params) == 1 and has_model:
            print("  => the contract is INTACT, so the rescue delivered a well-formed candidate and")
            print("     the rejection is about the candidate's own numerics: cause (b), which is what")
            print("     loop A is for. The rescue path does NOT need hardening for this.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
