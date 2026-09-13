"""Did the SOFT wall have any input at all, or did its gate legitimately decline?

WHY THIS MATTERS. `use_soft_wall: true` was chosen for this pair with the cost stated: hard-only is
~4 points/run (0.5% of budget, inside the noise floor, so a null would be uninterpretable), hard+soft
was priced at ~18/run (2.4%). The live treatment arm closed its first space with 4 recomputes, 0 soft
walls and `n_skipped_no_wall: 4`. If soft never fires either, the arm is running the hard-only variant
whose null I explicitly called uninterpretable -- so the reason has to be read, not assumed.

THREE OUTCOMES, and only the last is a defect:
  * the best trial does not spill -- the declared applicability gate, measured at 32 of 56 winners.
    A legitimate "not applicable", not a negative result.
  * the best trial spills but no knob's spill curve rises monotonically into a bound state -- the
    criterion doing its job.
  * `n_spills` is ABSENT from the profile -- then the gate is answering "no measurement" and reads
    identically to "no wall", which is the `a-constant-reading-is-a-broken-probe` shape and would mean
    the soft criterion never had a chance on this box.

`occupancy` is printed too, because its limiter key is `limiter` and it is NESTED -- a flat read turns
"measured" into "unmeasured" (recorded trap).

Run on box4:  /root/autodl-tmp/orch-venv/bin/python /root/probe-clean/soft_wall_input.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, "/root/autodl-tmp/work/opop/src")

from kernel_optimizer.evaluation import soft_wall as soft_wall_mod  # noqa: E402

BASE = Path("/root/autodl-tmp/opop-workspace/opop-glm/runs-v3")


def _robust(lat: dict | None) -> float | None:
    """median-else-mean, reproduced because `LatencyStats.robust_ms` is a @property and is never
    serialized -- reading a `robust_ms` key off the JSON would silently yield None for every trial."""
    if not isinstance(lat, dict):
        return None
    v = lat.get("median")
    if isinstance(v, (int, float)):
        return float(v)
    v = lat.get("mean")
    return float(v) if isinstance(v, (int, float)) else None


def scan(arm: str) -> None:
    runs = sorted(BASE.glob(f"{arm}/run-l3-43-*"))
    if not runs:
        print(f"=== {arm}: no run")
        return
    run = runs[-1]
    done: list[tuple[float, dict, str]] = []
    n_prof = n_spill = 0
    for line in (run / "events.jsonl").open(encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            e = json.loads(line)
        except ValueError:
            continue
        if e.get("type") != "TRIAL_DONE":
            continue
        t = (e.get("payload") or {}).get("trial") or {}
        if t.get("status") != "complete":
            continue
        prof = t.get("profile") if isinstance(t.get("profile"), dict) else {}
        if prof:
            n_prof += 1
            if isinstance(prof.get("n_spills"), (int, float)):
                n_spill += 1
        ms = _robust(t.get("latency_ms"))
        if ms is not None:
            done.append((ms, prof, str(t.get("trial_id"))))

    print(f"=== {arm}  {run.name}")
    print(f"    completed+timed trials {len(done)}   with a profile dict {n_prof}   "
          f"with an n_spills number {n_spill}")
    if not done:
        return
    ms, prof, tid = min(done, key=lambda d: d[0])
    occ = prof.get("occupancy")
    print(f"    BEST {tid}  robust {ms:.4f} ms")
    print(f"      n_spills={prof.get('n_spills')!r}  n_regs={prof.get('n_regs')!r}  "
          f"shared_bytes={prof.get('shared_bytes')!r}")
    # NESTED, and its limiter key is `limiter`. A flat read of this reports "unmeasured".
    print(f"      occupancy={json.dumps(occ)[:200] if occ is not None else None}")
    nz = [d for d in done if isinstance(d[1].get("n_spills"), (int, float))
          and float(d[1]["n_spills"]) > 0.0]
    print(f"    completed trials with n_spills > 0: {len(nz)} of {len(done)}")
    for d in nz[:5]:
        print(f"      spilling: {d[2]} n_spills={d[1].get('n_spills')} at {d[0]:.3f} ms")

    # The shipping criterion's own verdict string, rather than my paraphrase of the gate.
    class _T:
        def __init__(self, ms: float, prof: dict, tid: str, params: dict):
            self.status = "complete"
            self.trial_id = tid
            self.latency_ms = type("L", (), {"robust_ms": ms, "median": ms, "mean": ms})()
            self.profile = type("P", (), prof)() if prof else None
            self.params = type("PS", (), {"values": params})()

    print("    (the criterion's own reason is printed by s7_no_wall_why.py, which builds real "
          "TrialRecords; this probe answers only whether the INPUT exists)")


def main() -> int:
    scan("s7-treatment")
    print()
    scan("s7-control")
    print()
    print("VERDICT KEY. n_spills present and 0 at the best trial => the declared gate (32/56 winners),")
    print("not a defect. n_spills ABSENT => the soft criterion never had input on this box, and the")
    print("pair is effectively running the hard-only variant.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
