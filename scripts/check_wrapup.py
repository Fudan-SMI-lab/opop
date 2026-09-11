"""The three wrap-up checks, pinned in `docs/preflight-control-run.md` BEFORE the runs launched.

Written while the runs are still going, deliberately: a checker improvised after seeing the data
is a checker whose thresholds were chosen to fit it. Run it against one run dir, or two to compare
the control and treatment arms.

    python scripts/check_wrapup.py <control_run_dir> [<treatment_run_dir>]

Every reading goes through the same nesting rules the analysis scripts learned the hard way -- a
`TRIAL_DONE` payload nests under `payload.trial.params.values`, and reading `payload["params"]`
returns None for every trial while printing a plausible all-None table. Each reader asserts it
found something, so a wrong key path fails loudly instead of reporting a clean zero.
"""

from __future__ import annotations

import collections
import glob
import json
import os
import sys
from pathlib import Path


def _events(run_dir: Path):
    p = run_dir / "events.jsonl"
    if not p.exists():
        raise SystemExit("not a run dir (no events.jsonl): %s" % run_dir)
    with p.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


# --- check 1: G27's first production evidence -------------------------------------------------


def check_conversion(run_dir: Path) -> dict:
    """Do the round events carry a `conversion` field?

    A zero here is only a defect once rounds exist. `conversion_verdict` always returns a
    `conversion` key -- "unknown" when latency cannot be compared -- and the orchestrator merges it
    into every FAMILY_ROUND_RECORDED inside `if evaluated:`. So with rounds > 0, missing
    `conversion` is structurally impossible unless something broke, which is exactly the
    distinction the preflight doc asks for: "if still 0 that is a NEW defect, not 'nothing
    converted'".
    """
    rounds = 0
    with_conv = 0
    with_deltas = 0
    verdicts = collections.Counter()
    # `no_conversion` is the INFORMATIVE verdict, not a failure: a resource improved materially and
    # latency did not move, which is evidence that that resource was not the limit for this
    # structure. Counting only "improved" as success would discard exactly the finding G27 exists to
    # produce, so which resources moved is collected alongside.
    resources_improved = collections.Counter()
    notes: list[str] = []
    gains: list[float] = []
    for e in _events(run_dir):
        if e.get("type") != "FAMILY_ROUND_RECORDED":
            continue
        rounds += 1
        p = e.get("payload") or {}
        if "conversion" in p:
            with_conv += 1
            verdicts[p["conversion"]] += 1
        if "resource_deltas" in p:
            with_deltas += 1
        for name in (p.get("resources_improved") or []):
            resources_improved[name] += 1
        g = p.get("latency_gain_pct")
        if isinstance(g, (int, float)):
            gains.append(float(g))
        if p.get("conversion") == "no_conversion" and p.get("conversion_note"):
            notes.append(str(p["conversion_note"])[:200])
    if rounds == 0:
        verdict = "NOT YET DECIDABLE -- 0 rewrite rounds, so 0-of-0 says nothing"
    elif with_conv == 0:
        verdict = ("**NEW DEFECT** -- %d rounds and NONE carries `conversion`. The verdict is "
                   "merged unconditionally, so this is not 'nothing converted'" % rounds)
    elif with_conv < rounds:
        verdict = ("PARTIAL -- %d of %d rounds carry it; a round that skips the verdict is the "
                   "G44 shape (computed, journalled, read zero times)" % (with_conv, rounds))
    else:
        verdict = "PASS -- G27 has production evidence for the first time (%d of %d)" % (
            with_conv, rounds)
    return {"rounds": rounds, "with_conversion": with_conv, "with_resource_deltas": with_deltas,
            "verdicts": dict(verdicts),
            "resources_improved": dict(resources_improved),
            "latency_gains_pct": gains,
            "no_conversion_notes": notes,
            "verdict": verdict}


# --- check 2: J2-5 / J2d-9, the final result must not be worse ---------------------------------


def final_result(run_dir: Path) -> dict:
    """The run's FINAL RE-EVAL latency, never `tuned_ms`.

    `tuned_ms` is systematically optimistic by 1.5-6.7% (measured across the corpus), so comparing
    two arms on it can invert the sign of a small difference.

    Key paths verified against `src/`, not guessed. My first version of this reader looked for a
    `FINAL_REEVAL_DONE` event, which DOES NOT EXIST -- it returned None on a corpus run that has
    the number, i.e. it would have reported "no result" for a finished run. The truth is:

      * `final_reeval_ms` lives in `RUN_FINISHED.payload.summary.best`, written by `_finalize`.
        Note it is `lat.mean`; `final_reeval_median_ms` beside it is the median.
      * `BASELINE_DONE` nests under `payload.baseline` with `kind` and `latency_ms` -- the same
        one-level nesting as TRIAL_DONE, and reading `payload["latency_ms"]` yields nothing.
    """
    best: dict = {}
    trial_best = None
    baseline = {}
    for e in _events(run_dir):
        t = e.get("type")
        p = e.get("payload") or {}
        if t == "RUN_FINISHED":
            b = ((p.get("summary") or {}).get("best")) or {}
            if b:
                best = b
        elif t == "BASELINE_DONE":
            bl = p.get("baseline") or p
            lat = bl.get("latency_ms") or {}
            name = bl.get("kind") or "?"
            v = lat.get("median") or lat.get("mean")
            if isinstance(v, (int, float)):
                baseline[str(name)] = float(v)
        elif t == "TRIAL_DONE":
            tr = p.get("trial") or p
            if tr.get("status") == "complete" and not tr.get("failure_kind"):
                v = (tr.get("latency_ms") or {}).get("median")
                if isinstance(v, (int, float)) and (trial_best is None or v < trial_best):
                    trial_best = float(v)
    reeval = best.get("final_reeval_ms")
    return {"final_reeval_ms": float(reeval) if isinstance(reeval, (int, float)) else None,
            "final_reeval_median_ms": best.get("final_reeval_median_ms"),
            "final_reeval_ok": best.get("final_reeval_ok"),
            "tuned_ms": best.get("tuned_ms"),
            "precision": best.get("precision"),
            "excessive_speedup_flag": best.get("excessive_speedup_flag"),
            "best_trial_median_ms": trial_best,
            "baselines": baseline,
            "used_fallback": not isinstance(reeval, (int, float))}


def _flatten(d: dict, prefix: str = "") -> dict:
    out: dict = {}
    for k, v in (d or {}).items():
        key = "%s.%s" % (prefix, k) if prefix else str(k)
        if isinstance(v, dict):
            out.update(_flatten(v, key))
        else:
            out[key] = v
    return out


# The switches the paired runs are ALLOWED to differ in. Anything else is a confound in a comparison
# whose whole claim is "one thing changed".
_ARM_SWITCHES = ("v3.diagnosis.mode", "v3.diagnosis.expectation_ledger")
# Keys that name WHERE something is rather than WHAT it is. A different interpreter PATH is not a
# different interpreter: measured on the live pair, box 1 ran `orch-venv` and box 2 `kernel-opt-venv`
# and both resolve to torch 2.13.0+cu129 / triton 3.7.1 -- which is why the exemption is conditional
# on the calibration identities matching, below, rather than granted outright. A path exemption with
# no such check would hide a real environment divergence, and G23 already records torch/triton as a
# covarying factor across boxes.
_ARM_PATHLIKE = ("wsl.venv", "wsl.distro", "wsl.kernelbench_src", "wsl.triton_cache_dir",
                 "opencode.launch_cwd", "opencode.server_url", "opencode.sandbox_config_path",
                 "runs_dir", "opencode.node_bin")


def check_arm_parity(control_dir: Path, treatment_dir: Path) -> dict:
    """Do the two arms differ ONLY in the switches under test?

    Reads each run's `manifest.json` config snapshot -- what the run ACTUALLY loaded, not the yaml on
    disk, which may have been edited since. That distinction is the point: a typo in a `v3` key used
    to validate cleanly and leave the field default, so a treatment arm could silently be a SECOND
    CONTROL and the two runs would agree. `extra="forbid"` closed that at the model layer; this
    confirms it on the runs rather than trusting the fix.

    Path-shaped keys are exempted ONLY when both runs recorded the same calibration identity, which
    carries GPU, torch, CUDA and triton versions. A different venv path with the same identity is the
    same environment; a different identity is a real confound no matter what the paths say.
    """
    def snapshot(d: Path) -> tuple[dict, set]:
        try:
            man = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            return {}, set()
        idents = set()
        for e in _events(d):
            if e.get("type") == "DIMENSION_STATE":
                cp = (e.get("payload") or {}).get("compute_ceiling_provenance") or {}
                if cp.get("calibration_identity"):
                    idents.add(cp["calibration_identity"])
        return _flatten(man.get("config") or {}), idents

    c, c_ident = snapshot(control_dir)
    t, t_ident = snapshot(treatment_dir)
    if not c or not t:
        return {"verdict": "CANNOT DECIDE -- a manifest.json could not be read on one or both arms",
                "differing": {}, "calibration_identities": sorted(c_ident | t_ident)}

    differing = {k: (c.get(k, "<absent>"), t.get(k, "<absent>"))
                 for k in sorted(set(c) | set(t)) if c.get(k) != t.get(k)}
    same_env = bool(c_ident) and bool(t_ident) and c_ident == t_ident
    unexpected = {k: v for k, v in differing.items()
                  if k not in _ARM_SWITCHES and not (k in _ARM_PATHLIKE and same_env)}
    switches = {k: differing.get(k) for k in _ARM_SWITCHES}

    missing_switch = [k for k in _ARM_SWITCHES if k not in differing]
    if missing_switch:
        verdict = ("**DEFECT** -- the arms do NOT differ in %s, so this is not a paired comparison: "
                   "both runs are the same arm" % ", ".join(missing_switch))
    elif unexpected:
        verdict = ("**CONFOUND** -- %d key(s) differ beyond the switches under test: %s" % (
            len(unexpected), ", ".join("%s (%r vs %r)" % (k, v[0], v[1])
                                       for k, v in list(unexpected.items())[:4])))
    else:
        note = ""
        pathlike = [k for k in differing if k in _ARM_PATHLIKE]
        if pathlike:
            note = ("  || %d path-shaped key(s) also differ (%s) and are exempt because both arms "
                    "recorded the SAME calibration identity %s -- a different interpreter path is "
                    "not a different interpreter" % (
                        len(pathlike), ", ".join(pathlike), sorted(c_ident)))
        verdict = ("PASS -- the arms differ in exactly the switches under test (%s)%s" % (
            ", ".join("%s: %r vs %r" % (k, v[0], v[1]) for k, v in switches.items() if v), note))
    return {"verdict": verdict, "differing": differing, "switches": switches,
            "unexpected": unexpected,
            "calibration_identities": {"control": sorted(c_ident), "treatment": sorted(t_ident)},
            "same_environment": same_env}


def check_search_effort(run_dir: Path) -> dict:
    """How much search each arm actually spent, grouped BY SPACE.

    Config parity is necessary and not sufficient: `equal-configs-do-not-imply-equal-search` records a
    pair whose configs matched exactly while one arm did 1.76x the trials of the other -- and the ratio
    turned out to be a single agent timeout, not a real difference, which is why this reports the
    distribution rather than a ratio.

    GROUPED BY `space_id`, because a per-candidate count invites a false alarm. Measured on the live
    pair: five candidates on box 1 and five on box 2 each show **80** trials against
    `trials_per_space: 40`, which reads as a 2x budget overrun until the grouping is applied -- every
    space is at exactly 40 and a K-expansion publishes a SECOND space for the same candidate. The
    budget field is literally named `trials_per_space`. So the number to check is the per-space max,
    and a per-candidate max is the wrong reading of a correct run.

    Reports the per-space counts, how many spaces exceeded the configured budget (should be 0), and
    the spaces-per-candidate distribution, which is where a real asymmetry in K-expansion would show.
    """
    budget = None
    try:
        man = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
        budget = ((man.get("config") or {}).get("budgets") or {}).get("trials_per_space")
    except Exception:  # noqa: BLE001
        pass

    per_space: dict[str, int] = {}
    spaces_of: dict[str, set] = {}
    for e in _events(run_dir):
        if e.get("type") != "TRIAL_DONE":
            continue
        t = (e.get("payload") or {}).get("trial") or {}
        sid, cid = t.get("space_id"), t.get("candidate_id")
        if not sid:
            continue
        per_space[sid] = per_space.get(sid, 0) + 1
        spaces_of.setdefault(cid or "", set()).add(sid)

    counts = sorted(per_space.values(), reverse=True)
    over = {s: n for s, n in per_space.items() if budget and n > budget}
    # SPACE_PUBLISHED nests its fields under `payload.space`; read from the emitter, not a guess.
    n_published = n_expanded = 0
    for e in _events(run_dir):
        if e.get("type") == "SPACE_PUBLISHED":
            n_published += 1
        elif e.get("type") == "SPACE_EXPANDED":
            n_expanded += 1
    dist: dict[int, int] = {}
    for cid, s in spaces_of.items():
        dist[len(s)] = dist.get(len(s), 0) + 1

    if not counts:
        verdict = "no trial has recorded a space_id yet"
    elif over:
        verdict = ("**BUDGET OVERRUN** -- %d space(s) exceeded trials_per_space=%s, worst %d. This "
                   "breaks arm parity on search effort" % (len(over), budget, max(over.values())))
    else:
        verdict = ("PASS -- %d space(s), %d trials, max %d per space against trials_per_space=%s"
                   % (len(per_space), sum(counts), counts[0], budget))
    return {"trials": sum(counts), "spaces": len(per_space), "budget": budget,
            "max_per_space": counts[0] if counts else 0,
            "spaces_over_budget": over,
            "spaces_per_candidate": dist,
            "spaces_published": n_published, "spaces_expanded": n_expanded,
            "verdict": verdict}


def compare_arms(control: dict, treatment: dict, noise_floor_pct: float) -> str:
    """J2-5: treatment must not be worse than control beyond the task's measured noise floor."""
    c = control["final_reeval_ms"]
    t = treatment["final_reeval_ms"]
    if c is None or t is None:
        which = [n for n, d in (("control", control), ("treatment", treatment))
                 if d["final_reeval_ms"] is None]
        return ("CANNOT DECIDE -- no final re-eval on: %s. `tuned_ms` is optimistic by 1.5-6.7%%, "
                "so it is not a substitute here" % ", ".join(which))
    allowed = c * (1.0 + noise_floor_pct / 100.0)
    delta = 100.0 * (t - c) / c
    if t > allowed:
        return ("FAIL -- treatment %.4f ms vs control %.4f ms (%+.2f%%), beyond the %.2f%% noise "
                "floor: the extra information made the final result WORSE" % (
                    t, c, delta, noise_floor_pct))
    # J2-5 is ONE-SIDED -- it asks only that the treatment not be worse -- so everything below the
    # allowance passes it. But the three cases below are not the same result, and the first version of
    # this said "within the noise floor" for all of them. On the live pair that printed
    # "treatment 2.84 vs control 3.11 (-8.68%), within the 4.73% noise floor" -- a treatment that is
    # 8.68% BETTER, reported as if the arms were indistinguishable, which is the opposite reading and
    # the one a reader would carry into the paper.
    if delta <= -noise_floor_pct:
        return ("PASS -- treatment %.4f ms vs control %.4f ms (%+.2f%%). J2-5 asks only that the "
                "treatment not be WORSE, and it is BETTER by more than the %.2f%% floor, so this is "
                "a separation in the treatment's favour -- not a tie. Report the direction."
                % (t, c, delta, noise_floor_pct))
    if delta < 0:
        return ("PASS -- treatment %.4f ms vs control %.4f ms (%+.2f%%): better, but by LESS than "
                "the %.2f%% floor, so the arms are indistinguishable on final latency"
                % (t, c, delta, noise_floor_pct))
    return ("PASS -- treatment %.4f ms vs control %.4f ms (%+.2f%%): worse, but within the %.2f%% "
            "floor, so not a regression J2-5 can detect" % (t, c, delta, noise_floor_pct))


# --- check 3: S3 / S4' behaviour on real data --------------------------------------------------


def check_s3_s4(run_dir: Path) -> dict:
    """Four things the preflight names, each of which has a silent-failure twin.

    `precision_mismatch` firing at all, WHICH complementary-slackness state the run is in, whether a
    below-floor reading was REPORTED rather than clamped, and whether any ceiling_provenance carries
    a precision.

    THE SLACKNESS STATE IS DERIVED, NOT JOURNALLED. An earlier version of this function counted
    occurrences of the strings "cannot_run", "no_dimension_judged_slack", "weak_pass" and
    "violation_named" anywhere in a payload -- and not one of those appears anywhere in `src/`. It
    printed `none` on a fully working run, which reads as "S4' produced no states" and is the
    `a-fixture-invented-to-match-the-reader-proves-nothing` shape: the reader and my mental model
    agreed on names the emitter never writes. The real check lives in
    `evaluation/conversion_report.py` and needs two things off the log -- dimensions whose record
    says `verdict == "slack"` and `applicable`, and rounds carrying a `conversion` -- so this calls
    `slackness_violations` itself rather than restating its rule.

    THE PROVENANCE IS A TOP-LEVEL FIELD, not a per-record one. `_do_diagnose` writes
    `compute_ceiling_provenance` beside `records`, because exactly ONE dimension has a precision --
    the compute-pressure denominator, the only one that can be the wrong denominator without
    anything looking wrong (an fp16 kernel scored against a tf32 ceiling read 107.8% of peak). The
    per-record `provenance` blocks are `definitional` / `device_query` for occupancy, registers,
    shared bytes and the rest, and their `precision` is legitimately empty: a hardware limit has no
    precision.

    So counting per-record precisions reports 0 on a run whose S3 field says `precision: "fp16"`
    with a full calibration identity -- the same shape of bug as looking for a `FINAL_REEVAL_DONE`
    event, a clean zero on data that has the number. Both counts are reported separately, because a
    zero means opposite things in the two places.
    """
    prov_with_precision = 0
    prov_total = 0
    record_prov_with_precision = 0
    mismatches = []
    below_floor = 0
    clamped_suspicion = 0
    # WHICH dimensions, not just how many. The two counters print next to each other and on box 3 both
    # read 2 -- from two DIFFERENT dimensions, which invites reading it as one clamped pair. Measured
    # on `run-l3-48-20260911-052647`: `candidate_aten_bytes` is 1.074 GB against a 1.351 GB floor with
    # room -277 MB and `below_floor=True` (reported negative, which is the guard WORKING), while
    # `candidate_aten_ops` is 1.0 = floor 1.0 with room 0.0 and `below_floor=False` (a genuinely
    # fully-fused candidate sitting at its floor, not a clamp).
    #
    # And the below-floor reading is not a physics violation: `candidate_aten_bytes` is an ATEN-LEVEL
    # LOWER BOUND (dimensions.py:316 -- "a fused kernel under-reports"), while the floor is the task's
    # compulsory traffic measured on the REFERENCE. A fully fused Triton kernel moves its bytes inside
    # the kernel where aten cannot see them, so reading below the floor means THE CANDIDATE IS FUSED.
    # Naming the dimension is what makes that legible instead of alarming.
    below_floor_dims = collections.Counter()
    clamped_dims = collections.Counter()
    dims = collections.Counter()
    identities = set()
    unreachable = collections.Counter()
    prompt_modes = collections.Counter()
    # S4' inputs, read from the fields the EMITTER writes rather than from invented state names. An
    # earlier version of this reader counted occurrences of the four strings "cannot_run",
    # "no_dimension_judged_slack", "weak_pass" and "violation_named" anywhere in a payload -- none of
    # which appears anywhere in src/. It printed `none` on a working run, which is the
    # `a-fixture-invented-to-match-the-reader-proves-nothing` shape and would have been read as "S4'
    # produced no states" at wrap-up.
    #
    # What the real check needs (evaluation/conversion_report.py:150-230):
    #   slack dimensions   DIMENSION_STATE.records[] where verdict == "slack" AND applicable
    #   rounds with a verdict  FAMILY_ROUND_RECORDED carrying `conversion`
    # and its four reportable outcomes are DERIVED from those two, not journalled as labels.
    slack_dims: set[str] = set()
    verdicts_seen = 0
    rounds_with_conversion: list[dict] = []
    n_dimension_states = 0
    for e in _events(run_dir):
        t = e.get("type")
        p = e.get("payload") or {}
        if t == "DIMENSION_STATE":
            n_dimension_states += 1
            for rec in (p.get("records") or []):
                dims[rec.get("dimension_id")] += 1
                prov = rec.get("provenance") or {}
                if prov.get("precision"):
                    record_prov_with_precision += 1
                b = rec.get("bound") or {}
                if b.get("below_floor"):
                    below_floor += 1
                    below_floor_dims[rec.get("dimension_id")] += 1
                # A reading at EXACTLY the floor with room 0.0 is what a clamp looks like.
                if b.get("floor") is not None and b.get("room") == 0.0:
                    clamped_suspicion += 1
                    clamped_dims[rec.get("dimension_id")] += 1
                # `applicable` is load-bearing: a dimension with no polarity (threads_launched) is
                # never "slack" in the shadow-price sense, and counting it would manufacture
                # violations out of a dimension the theorem does not apply to.
                if rec.get("verdict") == "slack" and rec.get("applicable"):
                    slack_dims.add(rec.get("dimension_id"))
                if rec.get("verdict"):
                    verdicts_seen += 1
            # The S3 provenance: one per DIMENSION_STATE, not one per record.
            cprov = p.get("compute_ceiling_provenance") or {}
            if cprov:
                prov_total += 1
                if cprov.get("precision"):
                    prov_with_precision += 1
                if cprov.get("calibration_identity"):
                    identities.add(cprov["calibration_identity"])
            for u in (p.get("unreachable_ceilings") or []):
                unreachable[u] += 1
            if p.get("prompt_mode"):
                prompt_modes[p["prompt_mode"]] += 1
            if p.get("precision_mismatch"):
                mismatches.append(p.get("candidate_id"))
        elif t == "FAMILY_ROUND_RECORDED" and p.get("conversion"):
            rounds_with_conversion.append(p)
        blob = json.dumps(p)
        if "precision_mismatch" in blob and t != "DIMENSION_STATE":
            mismatches.append("%s:%s" % (t, p.get("candidate_id")))

    # The four states, DERIVED the way `conversion_lines` derives them, so this reports what the
    # report will say instead of a second opinion that can silently disagree.
    if n_dimension_states == 0:
        slack_state = "cannot_run: no per-dimension state was journalled"
    elif not slack_dims:
        slack_state = ("no_dimension_judged_slack: %d verdict(s) exist but none is an applicable "
                       "`slack`, so the check has nothing to test -- NOT a pass" % verdicts_seen)
    elif not rounds_with_conversion:
        slack_state = ("cannot_run: %d dimension(s) judged slack (%s) but no rewrite round carries a "
                       "conversion verdict yet" % (len(slack_dims), ", ".join(sorted(slack_dims))))
    else:
        try:
            sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
            from kernel_optimizer.evaluation.conversion_report import slackness_violations
            viol = slackness_violations(rounds_with_conversion, slack_dims,
                                        min_improvement_pct=2.0)
        except Exception as exc:  # noqa: BLE001
            slack_state = "cannot_run: %s" % str(exc)[:120]
            viol = None
        if viol is not None:
            if viol:
                slack_state = ("violation_named: %d -- %s" % (
                    len(viol), "; ".join("%s in %s round %s (gain %.2f%%)" % (
                        v.dimension, v.family_id, v.round, v.latency_gain_pct)
                        for v in viol[:3])))
            else:
                slack_state = ("weak_pass: no violation over %d round(s) with a verdict and %d "
                               "slack dimension(s) (%s)" % (
                                   len(rounds_with_conversion), len(slack_dims),
                                   ", ".join(sorted(slack_dims))))
    return {"dimension_records": sum(dims.values()),
            "diagnoses": prov_total,
            "diagnoses_whose_compute_ceiling_names_a_precision": prov_with_precision,
            "per_record_provenance_naming_a_precision": record_prov_with_precision,
            "calibration_identities": sorted(identities),
            "precision_mismatch_fired_on": sorted(set(m for m in mismatches if m)),
            "complementary_slackness_state": slack_state,
            "dimensions_judged_slack": sorted(slack_dims),
            "rounds_with_a_conversion_verdict": len(rounds_with_conversion),
            "below_floor_readings_reported": below_floor,
            "below_floor_dimensions": dict(below_floor_dims),
            "clamp_suspicion_dimensions": dict(clamped_dims),
            "readings_sitting_exactly_on_the_floor": clamped_suspicion,
            "unreachable_ceilings": dict(unreachable),
            "prompt_modes": dict(prompt_modes),
            "dimensions_seen": dict(dims)}


def check_reconciliation(run_dir: Path) -> dict:
    """S2d's expectation ledger: did it reconcile anything, or only produce empty entries?

    Two traps here, both of which look like success from a count alone.

    An `EXPECTATIONS_RECONCILED` event can carry `n_declared: 0` -- the rewriter declared no
    expectations that round, so the entry reconciles nothing. Counting events would report a
    working ledger built entirely of empty entries: the G44 shape (computed, journalled, read zero
    times) with an event stream that looks healthy.

    And it fires in the same `if evaluated:` branch as `conversion`, so 0 with 0 rounds is
    expected and says nothing -- the same distinction check 1 makes.

    `EXPECTATIONS_RECONCILE_FAILED` matters more than its count suggests: the reconciler is
    deliberately wrapped so a diagnostic cannot end a rewrite round, which means a total failure is
    SILENT apart from this event.
    """
    rounds = 0
    entries = 0
    empty = 0
    failed = []
    declared_total = 0
    # The REAL content of a ledger entry. `Reconciliation` (evaluation/reconcile.py) has NO
    # `verdict`/`outcome`/`status` field -- an earlier version of this reader guessed those three
    # names and would have printed "verdict kinds: -" on a fully working ledger, the same
    # clean-zero-on-good-data shape as `FINAL_REEVAL_DONE` and the per-record S3 precision. What it
    # actually carries is a per-dimension table plus counts:
    #   hits / misses / vacuous     -- vacuous is the agent DECLINING to predict, not being wrong
    #   dimensions_unpredicted      -- moved and was not mentioned ("you did not think of shared")
    #   dimensions_unmeasured       -- declared but uncheckable, no reading on one side
    # Those last two are named rather than counted on purpose, and the distinction between a miss
    # (a judgement) and vacuous/unmeasured (no judgement possible) is the whole point of the ledger.
    hits = misses = vacuous = 0
    candidateless = 0
    unpredicted = collections.Counter()
    unmeasured = collections.Counter()
    caveats = []
    per_dim_rows = 0
    for e in _events(run_dir):
        t = e.get("type")
        p = e.get("payload") or {}
        if t == "FAMILY_ROUND_RECORDED":
            rounds += 1
        elif t == "EXPECTATIONS_RECONCILED":
            entries += 1
            if not p.get("candidate_id"):
                candidateless += 1
            n = p.get("n_declared")
            if not n:
                empty += 1
            else:
                declared_total += int(n)
            rec = p.get("reconciliation") or {}
            hits += int(rec.get("hits") or 0)
            misses += int(rec.get("misses") or 0)
            vacuous += int(rec.get("vacuous") or 0)
            per_dim_rows += len(rec.get("per_dimension") or ())
            for d in (rec.get("dimensions_unpredicted") or ()):
                unpredicted[d] += 1
            for d in (rec.get("dimensions_unmeasured") or ()):
                unmeasured[d] += 1
            if rec.get("caveat"):
                caveats.append(str(rec["caveat"])[:200])
        elif t == "EXPECTATIONS_RECONCILE_FAILED":
            failed.append(p.get("error", "")[:120])
    if rounds == 0:
        verdict = "NOT YET DECIDABLE -- 0 rewrite rounds, so 0 ledger entries says nothing"
    elif entries == 0:
        verdict = ("**DEFECT** -- %d rounds and 0 ledger entries. It is journalled "
                   "UNCONDITIONALLY (the switch only gates the PROMPT), so the control arm should "
                   "have them too" % rounds)
    elif empty == entries:
        verdict = ("**EMPTY LEDGER** -- all %d entries carry n_declared=0: the events exist and "
                   "reconcile nothing, which is the G44 shape with a healthy-looking stream"
                   % entries)
    elif empty:
        verdict = "PARTIAL -- %d of %d entries are empty (n_declared=0)" % (empty, entries)
    elif hits + misses == 0 and per_dim_rows:
        # Declarations exist and every row is vacuous/unmeasured: the ledger ran and JUDGED
        # nothing. Distinct from an empty ledger and easy to read as success from a count.
        verdict = ("**NO JUDGEMENT** -- %d entries over %d rounds with %d declaration(s) and %d "
                   "per-dimension row(s), but 0 hits and 0 misses: every row was vacuous or "
                   "unmeasured, so nothing was actually checked" % (
                       entries, rounds, declared_total, per_dim_rows))
    else:
        verdict = ("PASS -- %d entries over %d rounds, %d declarations reconciled, %d hit / %d "
                   "miss / %d vacuous" % (entries, rounds, declared_total, hits, misses, vacuous))
    # ONE ENTRY PER REWRITE CANDIDATE, so entries > rounds is the CORRECT shape and must not read as
    # an anomaly. Measured: every rewrite round in every completed L3 run produced exactly two
    # candidates (9 of 9), and they are asked for different hypotheses, so scoring them against one
    # delta map mixed two agents' claims about two pieces of code -- box 2's round 0 read 6 hits / 6
    # misses pooled against 4/1 and 2/5 apart (docs/result-s2d-pooled-ledger-defect.md). A reader
    # that expected entries == rounds would flag the fix as the defect.
    if rounds and entries:
        verdict += "  || %.1f entries per round (one per rewrite CANDIDATE, so >1 is expected)" % (
            entries / float(rounds))
    if candidateless and entries:
        # Pre-fix entries carry no candidate_id. Worth naming rather than silently mixing, because a
        # pooled entry's hits/misses are not comparable with a per-candidate one's.
        verdict += ("  || %d of %d entries carry NO candidate_id: journalled before per-candidate "
                    "attribution, so their hit/miss counts pool two candidates and are not "
                    "comparable with the rest" % (candidateless, entries))
    if failed:
        verdict += "  || %d RECONCILE_FAILED (silent by design -- a diagnostic must not end a " \
                   "round): %s" % (len(failed), failed[0])
    return {"rounds": rounds, "entries": entries, "empty_entries": empty,
            "entries_without_candidate_id": candidateless,
            "declarations_reconciled": declared_total, "reconcile_failed": len(failed),
            "hits": hits, "misses": misses, "vacuous": vacuous,
            "per_dimension_rows": per_dim_rows,
            "dimensions_unpredicted": dict(unpredicted),
            "dimensions_unmeasured": dict(unmeasured),
            "caveats": caveats[:3],
            "verdict": verdict}


def check_ledger_reach(run_dir: Path) -> dict:
    """Did the ledger ever reach a PROMPT, or only the log?

    S2d(c) is the claim that reconciled predictions improve the NEXT rewrite. The ledger reaches a
    prompt through one line -- `ledger_entries=self.ledger.get(family_id, [])` -- keyed per FAMILY. So
    it arrives only when the SAME family gets a SECOND rewrite round after its first was reconciled.
    That keying is correct: a family must not see another family's predictions, which are about
    different code.

    MEASURED: no family in any run has ever had a second round. Nine families across three finished
    runs plus both live arms, every one with exactly one round -- so `ledger_entries` has been `[]` on
    every rewriter call this project has ever made, in both arms, regardless of the switch.

    The cause is arithmetic and identical on three independent runs: 4 seed families against 3 rewrite
    rounds that fit a 12 h budget, while `active_families()` rule 1 sends every never-rewritten family
    first. The queue never empties, so the rotation never comes back.

    This check exists because `check_reconciliation` reports PASS on exactly this state -- entries
    exist, declarations reconcile, counts look healthy -- and PASS there is about journalling, not
    about delivery. Reporting S2d(c) as tested-and-null when it was never administered would be the
    worst available error: a treatment never given is not a treatment that did not work.

    A round is one rewriter CALL, not one `REWRITE_PRODUCED`: every round in every completed L3 run
    produced exactly two candidates, so counting events would double it.

    The test is ORDERED, not a pair of counts. `k >= 2 and reconciled >= 1` would call a family reached
    when its only reconciliation landed at the close of round 2 -- i.e. after the last prompt that
    could have carried it. The reconciliation for round N is emitted at round N's close, so the flag
    has to be read at the moment each rewriter call was made: a round reached the prompt only if a
    reconciliation for THAT family already existed when the call went out.
    """
    reconciled: collections.Counter = collections.Counter()
    rounds: dict[str, dict] = {}       # family -> {round key: ledger was non-empty at call time}
    for e in sorted(_events(run_dir), key=lambda ev: ev.get("seq") or 0):
        t = e.get("type")
        p = e.get("payload") or {}
        if t == "REWRITE_PRODUCED":
            fid = p.get("family_id") or ""
            key = round(e.get("ts") or 0.0, 0)
            # Both candidates of one round share the call, so the first one to arrive fixes the flag
            # for the round; `setdefault` keeps it rather than letting the second overwrite it.
            rounds.setdefault(fid, {}).setdefault(key, reconciled[fid] >= 1)
        elif t == "EXPECTATIONS_RECONCILED":
            reconciled[p.get("family_id") or ""] += 1
    per_family = {f: len(ks) for f, ks in rounds.items()}
    revisited = sorted(f for f, ks in rounds.items() if any(ks.values()))
    if not per_family:
        verdict = "NOT YET DECIDABLE -- no rewrite round yet"
    elif revisited:
        verdict = ("REACHED -- %d family(ies) made a rewriter call with a non-empty ledger, so S2d(c) "
                   "WAS administered: %s" % (len(revisited), ", ".join(revisited)))
    else:
        verdict = ("**S2d(c) NEVER ADMINISTERED** -- %d family(ies), max %d round(s) each, so no "
                   "rewriter call ever received a ledger. Report S2d(c) as UNTESTED, not as null; "
                   "S2d(a)/(b) are unaffected" % (len(per_family), max(per_family.values())))
    return {"rounds_per_family": per_family,
            "reconciled_per_family": dict(reconciled),
            "families_revisited": revisited,
            "verdict": verdict}


def latency_floor_from_runs(*run_dirs: Path,
                            extra_globs: tuple[str, ...] = ()) -> tuple[float | None, str]:
    """Measure the LATENCY reproducibility floor from finished runs.

    The 2.35% this project has been using as a latency tolerance is `1 - 0.9765`, where 0.9765 is
    the reference's own `frac_within_tol` at two precisions -- a CORRECTNESS quantity, a fraction of
    elements agreeing. J2-5 compares LATENCIES, and there is no reason a numerics figure should
    equal a timing-jitter figure: one is about mantissa bits, the other about clocks, warmup and the
    20-sample median.

    `final_reeval` re-runs theta_best in a FRESH PROCESS, so `|tuned_ms - final_reeval_ms|` is a
    same-kernel, same-config, same-box re-measurement -- the right units.

    MEASURED over every finished run on box 1 (n=5, five tasks across three levels):

        4.73%  run-l3-48-20260907-202457   48_Mamba2ReturnY        1.4800 -> 1.4100  fp16
        4.34%  run-l1-42-20260907-022528   42_Max_Pooling_2D       5.0700 -> 4.8500  ieee_fp32
        2.91%  run-l3-21-20260908-232211   21_EfficientNetMBConv   3.6050 -> 3.5000  fp16
        0.93%  run-l2-37-20260907-064403   37_Matmul_Swish_Sum     0.0108 -> 0.0109  fp16
        0.29%  run-l3-48-20260909-115701   48_Mamba2ReturnY        1.5544 -> 1.5500  fp16

    median 2.91%, widest 4.73% -- so the borrowed 2.35% is BELOW the median of the real spread, and
    a J2-5 verdict against 2.35% alone would be stricter than the measurement supports. That is why
    the caller takes the WIDER of borrowed and measured rather than replacing one with the other.

    `extra_globs` widens the sample beyond the two arms under comparison: with only the two live
    runs the sample was n=2 (0.27% and 2.99%), which straddles 2.35% and is far too thin to bound a
    verdict. Sibling run directories are the same box and the same harness, so they are the right
    population; a glob that matches nothing simply leaves the sample as it was.

    AND THE SIBLING SCAN ALONE IS NOT ENOUGH, which is why the parameter is now implemented rather
    than only documented. It globs each arm's own PARENT, and measured on the live pair the arms sit
    in `runs-v3/` (2 directories, both unfinished) while the 5 finished runs are in `runs-l3/`. So the
    scan found nothing, this returned None, and the caller fell back to the borrowed 2.35% -- which
    is BELOW the 2.91% median of the real spread, i.e. silently stricter than the measurement. A
    parameter that a docstring promises and the signature does not have is worse than an absent
    feature: it reads as covered.

    Returns the widest observed delta and a provenance string naming n, because "inside the noise
    floor" means nothing without the floor and its sample size.
    """
    deltas: list[tuple[float, str]] = []
    seen: set[Path] = set()

    def consider(d: Path) -> float | None:
        if d in seen or not (d / "events.jsonl").exists():
            return None
        seen.add(d)
        fin = final_result(d)
        t, r = fin.get("tuned_ms"), fin.get("final_reeval_ms")
        if not (isinstance(t, (int, float)) and isinstance(r, (int, float)) and t > 0):
            return None
        delta = abs(100.0 * (r - t) / t)
        deltas.append((delta, d.name))
        return delta

    # The ARMS first and separately, because an arm is not merely one more member of the
    # population: its delta is the one being judged, and it must be in the sample even if the
    # sibling scan cannot see it (a run directory moved, or a parent that no longer exists).
    # Keeping the two passes distinct also keeps the sibling widening a widening -- if the arms
    # were left to the sibling glob, disabling this loop would change nothing and the tests would
    # pass on a reader that never looked at its arguments.
    named = [d for d in (consider(Path(x)) for x in run_dirs) if d is not None]
    for d in run_dirs:
        # Siblings of the arms under comparison: same box, same harness, so the same population.
        parent = Path(d).parent
        for sib in sorted(parent.glob("*")) if parent.exists() else []:
            if sib.is_dir():
                consider(sib)
    # And anywhere else the caller names. Separate from the sibling pass because it reaches a
    # DIFFERENT directory -- the finished corpus lives in `runs-l3/` while the arms live in
    # `runs-v3/`, so no amount of sibling globbing finds it. `glob.glob` rather than `Path.glob`
    # because the patterns are absolute paths on the boxes, which `Path().glob` rejects.
    for pattern in extra_globs:
        for hit in sorted(glob.glob(pattern)):
            if os.path.isdir(hit):
                consider(Path(hit))
    if not deltas:
        return None, "no run in the sample has re-evaluated its best kernel yet"
    deltas.sort(reverse=True)
    widest, where = deltas[0]
    med = sorted(d for d, _ in deltas)[len(deltas) // 2]
    return widest, ("same-kernel re-eval delta over n=%d finished run(s), of which %d %s an arm "
                    "under comparison; widest %.2f%% (%s), median %.2f%%" % (
                        len(deltas), len(named), "is" if len(named) == 1 else "are",
                        widest, where, med))


def check_budget_stop(run_dir: Path) -> dict:
    """Did the wall clock cut the run short, and WHERE?

    `WALL_CLOCK_REACHED` is emitted from two places with DIFFERENT payloads, and a reader that
    handles one silently drops the other:

      * `_pipeline_batch` -- between candidates. Payload has `pipelined` / `skipped`. The check is
        `if i and ...`, i.e. per candidate and never mid-candidate, so the batch always finishes the
        one it started and the skipped candidates stay REGISTERED for a resume with a larger budget.
        Nothing is lost.
      * `_rewrite_round` -- between families inside a round. Payload has `round` /
        `stopped_before_family`.

    THE PAYLOAD SHAPE DOES NOT TELL YOU WHICH LOOP. `_pipeline_batch` is called for the SEED batch
    AND from inside Loop C for rewrite candidates, so a `skipped`-shaped stop does not mean the seed
    pipeline ran out of time. Measured on the corpus run `run-l3-43-20260909-015247`: it fired a
    `skipped: 1` stop at 13.51 h AND has 5 FAMILY_ROUND_RECORDED, with the last round landing in the
    same second as the stop. A first draft of this reader asserted "Loop C was never reached" from
    the payload shape alone and was wrong on the first real run it saw.

    So the loop is inferred from whether any round exists, and the stop site only refines the
    message. That distinction matters because it decides what a zero in check 1 means: with no
    rounds AND a budget stop, `NOT YET DECIDABLE` is wrong -- it IS decided, the answer is "the run
    never got there".

    RESUMES MAKE THE PAYLOAD'S OWN CLOCK UNDERSTATE THE TRUTH. `elapsed_hours` is `_elapsed_hours()`,
    read off `self.t0` set in `Orchestrator.__init__`, and `cmd_resume` builds a NEW orchestrator -- so
    a resumed run's reported hours count only from the resume. The wall clock actually spent is the
    event-timestamp span, which is measured here alongside. Live case: box 3 reported 7.5 h internally
    while its first event was 8.75 h old. This matters for arm parity above all -- comparing a resumed
    run's 12 h against an uninterrupted run's 12 h compares different amounts of compute, and the
    budget is the one thing a paired comparison cannot lose.
    """
    stops = []
    rounds = 0
    interrupts = 0
    t_first = None
    t_last = None
    for e in _events(run_dir):
        ts = e.get("ts")
        if isinstance(ts, (int, float)):
            t_first = ts if t_first is None else min(t_first, ts)
            t_last = ts if t_last is None else max(t_last, ts)
        if e.get("type") == "FAMILY_ROUND_RECORDED":
            rounds += 1
            continue
        if e.get("type") == "RUN_INTERRUPTED":
            interrupts += 1
            continue
        if e.get("type") != "WALL_CLOCK_REACHED":
            continue
        p = e.get("payload") or {}
        site = "candidate batch" if "skipped" in p else (
            "rewrite round" if "round" in p else "unknown site")
        stops.append({
            "site": site,
            "elapsed_hours": p.get("elapsed_hours"),
            "budget_hours": p.get("budget_hours"),
            "skipped_candidates": p.get("skipped"),
            "pipelined_candidates": p.get("pipelined"),
            "round": p.get("round"),
            "stopped_before_family": p.get("stopped_before_family"),
        })
    span = (t_last - t_first) / 3600.0 if (t_first is not None and t_last is not None) else None
    resume_note = ""
    if interrupts:
        resume_note = (" RESUMED %d time(s): the run's own `elapsed_hours` counts only from the last "
                       "resume, so use the event span %s h as the wall clock actually spent. A "
                       "resumed run's budget is NOT comparable with an uninterrupted arm's."
                       % (interrupts, "%.2f" % span if span is not None else "?"))
    if not stops:
        return {"stops": [], "rounds": rounds, "reached_loop_c": rounds > 0,
                "interrupts": interrupts, "event_span_hours": span,
                "verdict": ("no wall-clock stop recorded (event span %s h)"
                            % ("%.2f" % span if span is not None else "?")) + resume_note}

    first = stops[0]
    over = ""
    if isinstance(first["elapsed_hours"], (int, float)) and \
            isinstance(first["budget_hours"], (int, float)) and first["budget_hours"]:
        over = " (%.0f%% over)" % (
            100.0 * (first["elapsed_hours"] - first["budget_hours"]) / first["budget_hours"])
    head = "BUDGET STOPPED THE RUN at %s h of %s h%s, at %d site(s): %s." % (
        first["elapsed_hours"], first["budget_hours"], over, len(stops),
        ", ".join(sorted({s["site"] for s in stops})))

    if rounds == 0:
        tail = (" NO rewrite round ever ran, so a zero in check 1 means 'the run never got there' "
                "-- not 'nothing converted', and not 'not yet decidable'.")
        batch = [s for s in stops if s["site"] == "candidate batch"]
        if batch:
            tail += (" %s candidate(s) were tuned and %s skipped; the skipped ones stay registered, "
                     "so a resume with a larger budget picks them up." % (
                         batch[0]["pipelined_candidates"], batch[0]["skipped_candidates"]))
    else:
        tail = (" Loop C DID run (%d round(s)), so the round count is a floor set by the clock "
                "rather than by convergence -- do not read it as 'the search finished'." % rounds)
    return {"stops": stops, "rounds": rounds, "reached_loop_c": rounds > 0,
            "interrupts": interrupts, "event_span_hours": span,
            "verdict": head + tail + resume_note}


def report(run_dir: Path, label: str) -> dict:
    print("=" * 78)
    print("%s   %s" % (label, run_dir))
    print("=" * 78)
    conv = check_conversion(run_dir)
    stop = check_budget_stop(run_dir)
    print("\n[0] the wall clock -- did the budget cut this run short, and where?")
    print("    => %s" % stop["verdict"])

    print("\n[1] G27 -- conversion's first production evidence")
    print("    rewrite rounds recorded ....... %d" % conv["rounds"])
    print("    carrying `conversion` ......... %d" % conv["with_conversion"])
    print("    carrying `resource_deltas` .... %d" % conv["with_resource_deltas"])
    print("    verdict distribution .......... %s" % (conv["verdicts"] or "-"))
    print("    resources that improved ....... %s" % (conv["resources_improved"] or "none"))
    if conv["latency_gains_pct"]:
        gs = sorted(conv["latency_gains_pct"])
        print("    latency gain per round (%%) .... median %+.2f, range %+.2f..%+.2f" % (
            gs[len(gs) // 2], gs[0], gs[-1]))
    # `no_conversion` is G27's whole point: a resource moved and latency did not, which locates
    # where the limit is NOT. Print the notes, because the count alone loses the finding.
    for n in conv["no_conversion_notes"][:3]:
        print("      no_conversion: %s" % n)
    # A budget stop in the seed pipeline changes what a zero MEANS, so it must be said here and not
    # only in section 0 -- this is the line a reader quotes.
    if conv["rounds"] == 0 and stop["stops"]:
        print("    => NOT 'not yet decidable': the budget ended this run before any rewrite round, "
              "so it never had the chance to produce one. See [0].")
    else:
        print("    => %s" % conv["verdict"])

    fin = final_result(run_dir)
    print("\n[2] J2-5 -- the final result, on final_reeval_ms not tuned_ms")
    print("    final_reeval_ms (mean) ........ %s%s" % (
        "%.4f" % fin["final_reeval_ms"] if fin["final_reeval_ms"] else "NONE YET",
        "" if fin["final_reeval_ok"] is None else "   ok=%s" % fin["final_reeval_ok"]))
    print("    final_reeval_median_ms ........ %s" % (
        "%.4f" % fin["final_reeval_median_ms"]
        if isinstance(fin["final_reeval_median_ms"], (int, float)) else "-"))
    print("    tuned_ms (optimistic 1.5-6.7%%) . %s" % (
        "%.4f" % fin["tuned_ms"] if isinstance(fin["tuned_ms"], (int, float)) else "-"))
    print("    winning precision ............. %s" % (fin["precision"] or "-"))
    print("    best complete trial median .... %s" % (
        "%.4f" % fin["best_trial_median_ms"] if fin["best_trial_median_ms"] else "-"))
    print("    baselines ..................... %s" % (
        {k: round(v, 4) for k, v in fin["baselines"].items()} or "-"))
    if fin["excessive_speedup_flag"]:
        print("    !! excessive_speedup flagged -- the anti-cheat threshold fired, inspect before "
              "reporting any speedup")
    if fin["used_fallback"]:
        print("    !! no final re-eval in RUN_FINISHED yet: the trial median is NOT a substitute")

    s3 = check_s3_s4(run_dir)
    print("\n[3] S3 / S4' on real data")
    print("    dimension records ............. %d over %d diagnoses" % (
        s3["dimension_records"], s3["diagnoses"]))
    print("    compute ceiling names a precision %d of %d diagnoses%s" % (
        s3["diagnoses_whose_compute_ceiling_names_a_precision"], s3["diagnoses"],
        "" if s3["diagnoses"] else "   (no diagnosis yet, so 0 says nothing)"))
    print("    calibration identity .......... %s" % (
        "; ".join(s3["calibration_identities"]) or "-"))
    print("    per-record provenance w/ precision %d  (expected 0: occupancy/registers/shared are"
          % s3["per_record_provenance_naming_a_precision"])
    print("                                       definitional or hardware limits, no precision)")
    print("    precision_mismatch fired on ... %s" % (s3["precision_mismatch_fired_on"] or "nothing"))
    print("    unreachable ceilings named .... %s" % (s3["unreachable_ceilings"] or "none"))
    print("    prompt_mode in the events ..... %s" % (s3["prompt_modes"] or "-"))
    print("    complementary slackness ....... %s" % s3["complementary_slackness_state"])
    print("      dimensions judged slack ..... %s"
          % (", ".join(s3["dimensions_judged_slack"]) or "none"))
    print("      rounds with a verdict ....... %d" % s3["rounds_with_a_conversion_verdict"])
    print("    below-floor readings REPORTED . %d%s" % (
        s3["below_floor_readings_reported"],
        ("   %s" % s3["below_floor_dimensions"]) if s3["below_floor_dimensions"] else ""))
    print("      (candidate_aten_bytes below its floor means THE CANDIDATE IS FUSED: the reading is")
    print("       an aten-level lower bound and a fused kernel moves bytes where aten cannot see")
    print("       them, while the floor is measured on the reference. Reported, never clamped.)")
    print("    readings exactly on the floor . %d%s%s" % (
        s3["readings_sitting_exactly_on_the_floor"],
        ("   %s" % s3["clamp_suspicion_dimensions"])
        if s3["clamp_suspicion_dimensions"] else "",
        "   <- inspect: this is what a clamp looks like"
        if s3["readings_sitting_exactly_on_the_floor"] else ""))
    print("    dimensions seen ............... %s" % (s3["dimensions_seen"] or "-"))

    rec = check_reconciliation(run_dir)
    print("\n[3b] S2d -- the expectation ledger (journalled in BOTH arms; the switch gates only")
    print("     whether the rendered form reaches the rewriter's prompt)")
    print("    ledger entries ................ %d" % rec["entries"])
    print("    of which EMPTY (n_declared=0) .. %d" % rec["empty_entries"])
    print("    declarations reconciled ....... %d" % rec["declarations_reconciled"])
    print("    per-dimension rows ............ %d" % rec["per_dimension_rows"])
    print("    hit / miss / vacuous .......... %d / %d / %d   (vacuous = the agent DECLINED to "
          "predict, not a wrong call)" % (rec["hits"], rec["misses"], rec["vacuous"]))
    print("    moved but NOT predicted ....... %s" % (rec["dimensions_unpredicted"] or "none"))
    print("    declared but UNMEASURABLE ..... %s" % (rec["dimensions_unmeasured"] or "none"))
    print("    RECONCILE_FAILED .............. %d" % rec["reconcile_failed"])
    for c in rec["caveats"]:
        print("      caveat: %s" % c)
    print("    => %s" % rec["verdict"])

    # [3c] exists because [3b]'s PASS is about JOURNALLING and reads as if the treatment worked. The
    # ledger's only outlet is the next round's prompt for the SAME family, so a healthy [3b] beside an
    # unreached [3c] is the actual state of every run this project has produced.
    reach = check_ledger_reach(run_dir)
    print("\n[3c] S2d(c) -- did a ledger ever REACH a rewriter prompt? (`self.ledger` is keyed per")
    print("     FAMILY, so only a family's SECOND round can receive one)")
    print("    rounds per family ............. %s" % (reach["rounds_per_family"] or "none yet"))
    print("    reconciled per family ......... %s" % (reach["reconciled_per_family"] or "none"))
    print("    => %s" % reach["verdict"])
    print()
    return {"conversion": conv, "final": fin, "s3": s3, "reconciliation": rec,
            "ledger_reach": reach, "stop": stop}


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    # `--floor-from GLOB` widens the noise-floor sample to run directories the sibling scan cannot
    # reach. Needed in practice, not in theory: the arms live in `runs-v3/` and the finished corpus in
    # `runs-l3/`, so without it the floor comes back None and the verdict silently falls back to the
    # borrowed 2.35% -- below the 2.91% median of the measured spread, i.e. stricter than the
    # measurement supports. Repeatable.
    floor_globs: list[str] = []
    positional: list[str] = []
    i = 0
    while i < len(argv):
        if argv[i] == "--floor-from" and i + 1 < len(argv):
            floor_globs.append(argv[i + 1])
            i += 2
        elif argv[i].startswith("--floor-from="):
            floor_globs.append(argv[i].split("=", 1)[1])
            i += 1
        else:
            positional.append(argv[i])
            i += 1
    if not positional:
        print(__doc__)
        return 2
    argv = positional
    control = report(Path(argv[0]), "CONTROL ARM" if len(argv) > 1 else "RUN")
    if len(argv) > 1:
        treatment = report(Path(argv[1]), "TREATMENT ARM")
        # The L3:43 noise floor measured on both 4090s: 0.9765 / 0.9767 frac_within_tol, and the
        # latency-side floor used for the calibration gate is 2.35%.
        print("=" * 78)
        print("J2-5 / J2d-9 comparison")
        print("=" * 78)
        # PARITY FIRST. Every number below compares two runs on the premise that one thing changed
        # between them, so if that premise is false the comparison is not weakened -- it is answering
        # a different question. Measured on the live pair: a THIRD key differed (`wsl.venv`) and it
        # took a manual check of both interpreters to establish it was benign. That check now runs
        # here, keyed on the calibration identity rather than on my reading of two paths.
        parity = check_arm_parity(Path(argv[0]), Path(argv[1]))
        print("    arm parity: %s" % parity["verdict"])
        # Parity on the CONFIG is not parity on the CLOCK. A resumed run's own `elapsed_hours` counts
        # only from its resume, so two arms can both report "12 h" having spent different wall clocks.
        cs, ts_ = control["stop"], treatment["stop"]
        if cs.get("interrupts") or ts_.get("interrupts"):
            print("      !! BUDGET CLOCKS NOT COMPARABLE: control resumed %d time(s), treatment %d. "
                  "Event spans: control %s h, treatment %s h -- use THESE, not the runs' own "
                  "`elapsed_hours`, and state the interruption beside any wall-clock claim."
                  % (cs.get("interrupts") or 0, ts_.get("interrupts") or 0,
                     "%.2f" % cs["event_span_hours"] if cs.get("event_span_hours") else "?",
                     "%.2f" % ts_["event_span_hours"] if ts_.get("event_span_hours") else "?"))
        else:
            print("      budget clocks: no resume on either arm; event spans control %s h, "
                  "treatment %s h"
                  % ("%.2f" % cs["event_span_hours"] if cs.get("event_span_hours") else "?",
                     "%.2f" % ts_["event_span_hours"] if ts_.get("event_span_hours") else "?"))
        if parity["differing"]:
            print("      all differing config keys (%d):" % len(parity["differing"]))
            for k, (a, b) in list(parity["differing"].items())[:10]:
                print("        %-44s %r  vs  %r" % (k, a, b))
        # Equal configs do not imply equal search: a recorded pair matched on every config key while
        # one arm ran 1.76x the trials. Grouped BY SPACE, because a per-candidate count reads as a 2x
        # overrun on a correct run -- K-expansion publishes a second space for the same candidate.
        for name, d in (("control", Path(argv[0])), ("treatment", Path(argv[1]))):
            se = check_search_effort(d)
            print("      search effort (%s): %s" % (name, se["verdict"]))
            print("        spaces per candidate %s ; published %d, expanded %d"
                  % (se["spaces_per_candidate"], se["spaces_published"], se["spaces_expanded"]))
        print()
        # The tolerance: 2.35% is `1 - 0.9765`, a frac_within_tol (CORRECTNESS) figure reused as a
        # latency tolerance. Measure the latency floor from these runs too and use the WIDER of the
        # two, so the comparison is never stricter than the measurement supports -- and say which
        # one bound it, because "inside the noise floor" means nothing without naming the floor.
        BORROWED = 2.35
        measured, prov = latency_floor_from_runs(Path(argv[0]), Path(argv[1]),
                                                 extra_globs=tuple(floor_globs))
        tol = max(BORROWED, measured) if measured is not None else BORROWED
        print("    tolerance used: %.2f%%" % tol)
        print("      borrowed 2.35% = 1 - 0.9765, a frac_within_tol (CORRECTNESS) figure")
        print("      measured latency floor: %s" % (
            "%.2f%% (%s)" % (measured, prov) if measured is not None else prov))
        print("      => using the wider; a latency verdict must not be stricter than the "
              "latency measurement")
        print("    %s" % compare_arms(control["final"], treatment["final"], tol))
        print("\n    G27 in both arms: control %d/%d, treatment %d/%d rounds carry `conversion`" % (
            control["conversion"]["with_conversion"], control["conversion"]["rounds"],
            treatment["conversion"]["with_conversion"], treatment["conversion"]["rounds"]))
        print("    (the treatment arm is the one that must ALSO produce dimension records: "
              "control %d, treatment %d)" % (
                  control["s3"]["dimension_records"], treatment["s3"]["dimension_records"]))
        # The ledger is journalled in BOTH arms by design, so an asymmetry here is a defect rather
        # than the treatment working -- the opposite reading from the dimension records above.
        cl, tl = control["reconciliation"], treatment["reconciliation"]
        if cl["rounds"] and tl["rounds"] and bool(cl["entries"]) != bool(tl["entries"]):
            print("    !! LEDGER ASYMMETRY: control %d entries, treatment %d. It is journalled "
                  "unconditionally, so one arm having none is a defect, NOT the switch working"
                  % (cl["entries"], tl["entries"]))
        else:
            print("    ledger entries: control %d, treatment %d (expected in BOTH arms)"
                  % (cl["entries"], tl["entries"]))
        # S2d(c) is the only sub-claim the switch actually gates, and it is administered only if some
        # family got a second round. If NEITHER arm reached that, the comparison above cannot speak to
        # S2d(c) at all and must not be written up as a null result -- the treatment was never given.
        cr, tr = control["ledger_reach"], treatment["ledger_reach"]
        if not cr["families_revisited"] and not tr["families_revisited"]:
            print("    !! S2d(c) UNTESTED IN BOTH ARMS: no family in either run received a second")
            print("       rewrite round, so `ledger_entries` was [] on every rewriter call in BOTH")
            print("       arms. The arms are therefore IDENTICAL with respect to S2d(c); any")
            print("       latency difference between them is NOT evidence about it. Report S2d(c)")
            print("       as untested, not as tested-and-null. S2d(a)/(b) are unaffected -- their")
            print("       evidence is [3b] above.")
        elif bool(cr["families_revisited"]) != bool(tr["families_revisited"]):
            print("    !! S2d(c) REACH ASYMMETRY: control revisited %s, treatment %s. Only the"
                  % (cr["families_revisited"] or "none", tr["families_revisited"] or "none"))
            print("       treatment arm renders the ledger, but the ROTATION is arm-independent, so"
                  "  unequal reach means the arms diverged in search order -- check parity above.")
        else:
            print("    S2d(c) administered in both arms: control %s, treatment %s"
                  % (cr["families_revisited"], tr["families_revisited"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
