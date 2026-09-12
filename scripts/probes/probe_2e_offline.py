"""2e section 8.1 -- answer A1 and A2 OFFLINE on completed runs. Reads only; no production code.

A1  Can a shared-memory wall be attributed to a SINGLE knob?
A2  Does the harness's attribution differ from the analyst agent's own parameter_limits claims?

Method, per candidate wall (knob k, refused value v) found from the run's own
CONFIG_SCREENED_INFEASIBLE records:
  theta*      = the parameters of that candidate's FASTEST completed trial
  theta_probe = theta* with knob k set to v, everything else unchanged
  materialize theta_probe into the candidate's source, then run ONE compile-only probe
  refused  => ATTRIBUTED   (k alone is sufficient to hit the wall at the optimum)
  passes   => NOT ATTRIBUTED (the wall needs k together with other knobs)

Two controls, because an attribution probe that always says "refused" proves nothing:
  NEGATIVE CONTROL  theta* itself must PASS the probe. theta* is the best trial, so it ran; if
                    the probe refuses it, the probe or the materialization is wrong, not the knob.
  SECOND ORIGIN     the same ablation from the space's DEFAULT config, per spec risk 3. If the two
                    origins disagree, the wall's position depends on the other knobs -- itself a
                    finding, and one that would invalidate a single-origin claim.

All variants go through ONE worker process via `extra_kernel_src_paths` (48 variants measured at
11.02 s total, 7 ms marginal), so the whole thing costs one process start.
"""
import json
import os
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, os.environ.get("OPOP_SRC", "src"))
from kernel_optimizer.gpu.jobs import make_compile_probe_job          # noqa: E402
from kernel_optimizer.paramspace import materializer                  # noqa: E402
from kernel_optimizer.store.read import (                             # noqa: E402
    candidate_source, read_events, trial_of,
)

RUN_DIRS = [a for a in sys.argv[1:] if not a.startswith("--")]
REF_PATH = None
for a in sys.argv[1:]:
    if a.startswith("--ref="):
        REF_PATH = a.split("=", 1)[1]
if not RUN_DIRS:
    sys.exit("usage: probe_2e_offline.py <run_dir> [...] --ref=<ref .py>")


def pvals(params):
    """Knob dict from either shape. TRIAL_DONE nests under params.values; the refusal event is flat."""
    if not isinstance(params, dict):
        return {}
    inner = params.get("values")
    return inner if isinstance(inner, dict) else params


def lat_of(t):
    lat = t.get("latency_ms") or {}
    for k in ("median", "mean"):
        v = lat.get(k)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return float(v)
    return None


def as_num(x):
    try:
        return float(x)
    except Exception:
        return None


class Ps:
    """Minimal ParamSet stand-in: materialize() reads `.values`."""

    def __init__(self, values):
        self.values = dict(values)


plan = []       # every variant we will probe, in one batch
meta = []       # parallel metadata
CLAIMS = {}     # (run, cid) -> analyst parameter_limits, for A2

for rd in RUN_DIRS:
    run_name = os.path.basename(rd.rstrip("/"))
    events = read_events(rd)

    refused = defaultdict(list)
    for ev in events:
        if ev.get("type") == "CONFIG_SCREENED_INFEASIBLE":
            p = ev.get("payload") or {}
            if p.get("candidate_id"):
                refused[p["candidate_id"]].append(pvals(p.get("params")))
    ok = defaultdict(list)
    for ev in events:
        if ev.get("type") != "TRIAL_DONE":
            continue
        t = trial_of(ev)
        if t.get("status") != "complete":
            continue
        ms = lat_of(t)
        if ms is not None and t.get("candidate_id"):
            ok[t["candidate_id"]].append((pvals(t.get("params")), ms))

    # the space's default config (choices[0] per domain), for the second origin
    defaults = {}
    for ev in events:
        if ev.get("type") != "SPACE_PUBLISHED":
            continue
        p = ev.get("payload") or {}
        sp = p.get("space") or p
        cid = sp.get("candidate_id")
        d = {}
        for dom in (sp.get("domains") or []):
            ch = dom.get("choices") or []
            if ch:
                d[dom.get("name")] = ch[0]
        if cid and d:
            defaults[cid] = d

    for ev in events:
        if ev.get("type") != "BOTTLENECK_REPORTED":
            continue
        p = ev.get("payload") or {}
        rep = p.get("report") or p
        cid = p.get("candidate_id")
        if cid:
            CLAIMS.setdefault((run_name, cid), []).extend(rep.get("parameter_limits") or [])

    for cid, refs in sorted(refused.items()):
        trials = ok.get(cid)
        if not trials:
            continue
        theta_star, star_ms = min(trials, key=lambda r: r[1])
        try:
            src = candidate_source(rd, cid)
        except Exception as exc:
            print(f"  {cid}: no source ({exc.__class__.__name__}) -- skipped")
            continue

        knobs = sorted({k for p, _ in trials for k in p})
        walls = []
        for k in knobs:
            ran = sorted({as_num(p.get(k)) for p, _ in trials if as_num(p.get(k)) is not None})
            ref_v = sorted({as_num(p.get(k)) for p in refs if as_num(p.get(k)) is not None})
            if len(ran) < 3 or not ref_v:
                continue
            outside = [v for v in ref_v if v > max(ran) or v < min(ran)]
            if not outside:
                continue
            best = {}
            for p, ms in trials:
                x = as_num(p.get(k))
                if x is not None:
                    best[x] = min(best.get(x, 1e18), ms)
            xs = sorted(best)
            hi = [v for v in outside if v > max(ran)]
            side = "high" if hi else "low"
            tail = xs[-3:] if side == "high" else xs[:3][::-1]
            mono = all(best[tail[i + 1]] < best[tail[i]] for i in range(len(tail) - 1)) \
                if len(tail) >= 2 else False
            gain = (best[tail[0]] - best[tail[-1]]) / best[tail[0]] * 100 if len(tail) >= 2 else 0.0
            target = min(hi) if hi else max(v for v in outside if v < min(ran))
            walls.append({"knob": k, "refused_value": target, "ran": xs, "side": side,
                          "monotone": mono, "tail_gain_pct": round(gain, 2)})

        if not walls:
            continue

        def add(values, kind, wall=None, origin=None):
            try:
                text = materializer.materialize(src, Ps(values))
            except Exception as exc:
                meta.append({"run": run_name, "cid": cid, "kind": kind, "wall": wall,
                             "origin": origin,
                             "materialize_error": f"{exc.__class__.__name__}: {exc}"[:160]})
                plan.append(None)
                return
            plan.append(text)
            meta.append({"run": run_name, "cid": cid, "kind": kind, "wall": wall,
                         "origin": origin, "values": dict(values)})

        add(theta_star, "control_theta_star", origin="theta_star")
        for w in walls:
            v = w["refused_value"]
            # keep the literal's type: an int knob written as 512.0 changes the source text
            lit = int(v) if float(v).is_integer() else v
            pv = dict(theta_star)
            pv[w["knob"]] = lit
            add(pv, "ablation", wall=w, origin="theta_star")
            dflt = defaults.get(cid)
            if dflt:
                d2 = dict(dflt)
                d2[w["knob"]] = lit
                add(d2, "ablation", wall=w, origin="default")

        print(f"  {run_name} {cid}: {len(walls)} wall(s), theta*={star_ms:.4f}ms, "
              f"analyst claims={len(CLAIMS.get((run_name, cid), []))}")
        for w in walls:
            print(f"      {w['knob']:18s} ran {w['ran']} -> refused {w['refused_value']}  "
                  f"monotone={w['monotone']}  tail {w['tail_gain_pct']:+.1f}%")

n_real = sum(1 for p in plan if p is not None)
print(f"\nvariants to probe: {n_real} (materialize errors: {sum(1 for p in plan if p is None)})")
if not n_real:
    sys.exit("nothing to probe")
if not REF_PATH:
    sys.exit("need --ref=<reference .py path>")

tmp = Path(tempfile.mkdtemp(prefix="probe2e-"))
paths = []
for i, text in enumerate(plan):
    if text is None:
        paths.append(None)
        continue
    p = tmp / f"v{i:03d}.py"
    p.write_text(text, encoding="utf-8")
    paths.append(p)

real = [p for p in paths if p is not None]
job = make_compile_probe_job(REF_PATH, str(real[0]), backend="triton",
                             extra_kernel_src_paths=[str(p) for p in real[1:]])
job_f, out_f = tmp / "job.json", tmp / "out.json"
job_f.write_text(json.dumps(job), encoding="utf-8")

worker = os.environ.get("OPOP_WORKER", "src/kernel_optimizer/gpu/worker_main.py")
t0 = time.monotonic()
proc = subprocess.run([sys.executable, worker, "--job", str(job_f), "--out", str(out_f)],
                      stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
elapsed = time.monotonic() - t0
print(f"\nworker exit={proc.returncode} in {elapsed:.1f}s for {len(real)} variants "
      f"({elapsed/len(real):.2f}s each)")
if not out_f.is_file():
    print("NO out.json -- worker output tail:")
    print(proc.stdout.decode("utf-8", "replace")[-2000:])
    sys.exit(1)
res = json.loads(out_f.read_text(encoding="utf-8"))
by_path = res.get("results") or {}
print(f"per-variant results: {len(by_path)}")

LIMIT = int(os.environ.get("OPOP_SHARED_LIMIT", "101376"))
print(f"device shared limit used: {LIMIT}")


def verdict_for(path):
    r = by_path.get(str(path)) or {}
    if not r and str(path) == str(real[0]):
        r = res      # single-path shape keeps the primary verdict at the top level
    if not r.get("ok"):
        return None, r.get("reason") or "probe not ok"
    ms = r.get("max_shared")
    if ms is None:
        return None, "no max_shared (no Triton kernel reached)"
    return (ms > LIMIT), ms


print(f"\n{'='*88}\n=== A1: can a wall be attributed to ONE knob? ===")
ctrl_pass = ctrl_fail = attributed = not_attr = unknown = 0
rows = []
for m, p in zip(meta, paths):
    if p is None:
        print(f"  MATERIALIZE ERROR {m['cid']} "
              f"{(m.get('wall') or {}).get('knob')}: {m.get('materialize_error')}")
        continue
    refuse, info = verdict_for(p)
    if m["kind"] == "control_theta_star":
        if refuse is False:
            ctrl_pass += 1
        else:
            ctrl_fail += 1
            print(f"  *** CONTROL FAILED {m['cid']}: theta* itself -> "
                  f"{('refused, shared=' + str(info)) if refuse else info}")
        continue
    w = m["wall"]
    rows.append((m["run"], m["cid"], w["knob"], w["refused_value"], m["origin"],
                 refuse, info, w["monotone"], w["tail_gain_pct"]))
    if refuse is True:
        attributed += 1
    elif refuse is False:
        not_attr += 1
    else:
        unknown += 1

print(f"\nNEGATIVE CONTROL: theta* passed in {ctrl_pass} candidates, FAILED in {ctrl_fail}")
print("  (theta* is the fastest trial, so it ran. A control failure means the probe or the")
print("   materialization is wrong -- every attribution below would then be unreliable.)")
print(f"\nablation verdicts: ATTRIBUTED {attributed}, NOT attributed {not_attr}, "
      f"undecidable {unknown}")
print(f"\n{'run':26s} {'candidate':16s} {'knob':16s} {'->val':>7s} {'origin':10s} "
      f"{'verdict':16s} {'shared':>9s} {'tail':>7s}")
for run, cid, knob, val, origin, refuse, info, mono, gain in rows:
    v = "ATTRIBUTED" if refuse is True else ("not attributed" if refuse is False else "undecidable")
    sh = str(info) if isinstance(info, int) else "-"
    print(f"{run[:26]:26s} {cid:16s} {knob[:16]:16s} {val:>7} {origin:10s} "
          f"{v:16s} {sh:>9s} {gain:+6.1f}%")

pair = defaultdict(dict)
for run, cid, knob, val, origin, refuse, info, mono, gain in rows:
    pair[(run, cid, knob, val)][origin] = refuse
both = {k: v for k, v in pair.items() if len(v) == 2}
agree = sum(1 for v in both.values() if v.get("theta_star") == v.get("default"))
print(f"\nSECOND ORIGIN (spec risk 3): probed from both origins: {len(both)}, "
      f"agreeing: {agree}, disagreeing: {len(both)-agree}")
if len(both) - agree:
    print("  a disagreement means the wall's position depends on the other knobs' values --")
    print("  a single-origin attribution claim is then conditional and must say so.")

print(f"\n{'='*88}\n=== A2: harness attribution vs the analyst's own claims ===")
attr_set = {(run, cid, knob) for run, cid, knob, _, origin, refuse, *_ in rows
            if origin == "theta_star" and refuse is True}
tested = {(run, cid, knob) for run, cid, knob, _, origin, *_ in rows if origin == "theta_star"}
for (run, cid), lims in sorted(CLAIMS.items()):
    if not any(t[0] == run and t[1] == cid for t in tested):
        continue
    print(f"  {run} {cid}: {len(lims)} analyst claim(s)")
    for lim in lims:
        prm, blk = lim.get("param"), lim.get("blocked_by")
        key = (run, cid, prm)
        mark = ("HARNESS CONFIRMS" if key in attr_set
                else ("harness tested and did NOT attribute" if key in tested
                      else "harness found no wall for this knob"))
        print(f"      claim: {str(prm)[:20]:20s} blocked_by={str(blk)[:18]:18s} "
              f"gain={lim.get('predicted_gain_pct')}  -> {mark}")
