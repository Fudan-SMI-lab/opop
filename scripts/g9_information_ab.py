"""G9: does giving the rewriter MORE measured evidence actually produce better kernels?

WHY THIS EXISTS. Every value v3 claims rests on one premise: that a richer, measured account of what
limits a kernel makes the agent rewrite it better. There is a published precedent for the OPPOSITE.
The KernelBench paper measured few-shot optimization examples LOWERING fast_1 (o1 on level 1, 10% ->
6%), because the extra context made the model "attempt more aggressive optimization strategies" and
fail more often. KernelPro measured raw hardware-counter output making an LLM do WORSE than no
feedback at all (NoFeedback beat raw ncu, p=0.0007).

So this cannot be settled by design, only by experiment, and the experiment has to use REAL
operators, the REAL prompt and the REAL model -- a synthetic proxy would measure the proxy.

THE DESIGN. One variable: what evidence the rewriter is given. Everything else is held identical --
same task, same starting source, same prompt template, same model, same sandbox layout, same
temperature-free single call per arm. Three arms, chosen so a null result is still informative:

  none    the bottleneck report is replaced by a minimal stub that names no limiter. The floor:
          what the agent does on the kernel alone. (KernelBench's NoFeedback control.)
  verdict the harness's verdict and its numbers, as v3 ships them today.
  rich    the verdict PLUS the per-candidate cost measurements added for G1/G2/G4/G6 and the
          backend-reachable ceiling from G10 -- i.e. everything v3 can now say.

WHAT IS MEASURED, not judged. Each produced kernel is materialized, compiled, checked for
correctness against the reference, and TIMED on the GPU. The outcome per arm is: how many candidates
were syntactically usable, how many were correct, and the best latency achieved. My reading of the
code is not evidence; a kernel that runs faster is.

WHY A NULL RESULT MATTERS. If `rich` does not beat `verdict`, the honest conclusion is that the
extra measurement does not convert into better rewrites -- which is a finding about v3's central
claim, and per the register the retreat is to narrow the digestion layer, not to stop collecting.
The arms are therefore recorded whatever they say, and the sample size is stated with the result:
one call per arm per task is an ANECDOTE, and the script prints how many replicates it ran so the
number is never quoted as if it were a rate.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kernel_optimizer.store.read import latency_ms_of  # noqa: E402

# --- the three evidence arms -----------------------------------------------------------------
# WHAT THE REWRITER ACTUALLY READS, verified against a real run's events rather than assumed:
# `analysis/bottleneck.json` is a BottleneckReport = {summary: str, parameter_limits, hypotheses,
# suggested_action: Literal["tune_more","rewrite","stop"]}. It is the ANALYST AGENT's prose. The
# harness classifier's structured verdict (kind + evidence dict) is a SEPARATE event that reaches the
# rewriter only through that prose and through `docs/device.md`'s measured-ceilings block.
#
# So the variable is the analyst's account plus the ceilings block -- which is exactly v3's digestion
# layer, and therefore the right thing to vary. Named accordingly rather than pretending the arms
# swap a structured verdict the rewriter never sees.
ARMS = ("none", "verdict", "rich")


def _stub_report(report):
    """The `none` arm: the same schema with the analysis removed.

    A STUB, not an omission: the prompt names `analysis/bottleneck.json`, so deleting the file would
    confound "no evidence" with "a prompt pointing at a missing file". `suggested_action` must stay a
    legal value of its Literal -- "rewrite" is what the arm is being asked to do, so it carries no
    information beyond the task itself.
    """
    return report.model_copy(update={
        "summary": "No bottleneck analysis is available for this kernel.",
        "parameter_limits": [],
        "hypotheses": [],
        "suggested_action": "rewrite",
    })


def _rich_report(report, profile):
    """The `rich` arm: the shipped analysis PLUS the per-candidate measurements v3 now has.

    Appended to the summary rather than replacing anything, because the question is whether MORE
    information helps: the arm must be a strict superset of `verdict`, or a difference could be
    caused by something removed instead of something added.

    These are the quantities G4/G6/G2 added, and each is stated WITH ITS BASIS -- an unqualified
    `candidate_aten_bytes` invites reading a well-fused kernel as moving almost nothing, which is the
    opposite of true.
    """
    if profile is None:
        return report
    lines = []
    for f, unit in (("peak_alloc_bytes", "B"), ("peak_reserved_bytes", "B"),
                    ("candidate_aten_bytes", "B"), ("candidate_aten_ops", "ops"),
                    ("threads_launched", "threads")):
        v = getattr(profile, f, None)
        if v is not None:
            lines.append("  %s = %s %s" % (f, v, unit))
    if not lines:
        return report
    extra = (
        "\n\nPER-CANDIDATE MEASUREMENTS (measured on THIS kernel, not on the reference):\n"
        + "\n".join(lines)
        + "\n  Basis: candidate_aten_bytes is a LOWER bound -- it sees only aten-level "
          "materialization and is blind to traffic fused inside a kernel, so a well-fused kernel "
          "reads low. peak_alloc_bytes is the caching allocator's view, not device memory. "
          "candidate_aten_ops counts the ops NOT fused away, so a high value means work fell back "
          "to aten.")
    return report.model_copy(update={"summary": (report.summary or "") + extra})


def _apply_arm(report, profile, arm):
    if arm == "none":
        return _stub_report(report)
    if arm == "rich":
        return _rich_report(report, profile)
    return report


# --- measurement ------------------------------------------------------------------------------


def _evaluate(evaluator, task, source: str, workdir: Path, tag: str) -> dict:
    """Compile, check correctness and time one produced kernel THROUGH THE HARNESS'S OWN evaluator.

    Deliberately `quick_test` rather than a hand-built job: it applies the same correctness gate,
    the same dual-precision witness rules and the same timing method the harness applies to every
    candidate in a real run. A bespoke evaluation here would measure something the harness does not,
    and the arms would be compared on a yardstick nothing else uses.

    Returns a dict that always says what happened, including for failures: an arm that produces four
    kernels that do not compile has done worse than one producing a single correct kernel, and a
    summary counting only successes would hide that.
    """
    out: dict = {"usable": False, "correct": False, "ms": None, "failure": None}
    from kernel_optimizer.paramspace import materializer
    try:
        out["params"] = materializer.extract_defaults(source)
        out["usable"] = True
    except materializer.MaterializeError as exc:
        out["failure"] = "materialize:%s" % exc.kind
        return out

    path = workdir / ("%s.py" % tag)
    path.write_text(source, encoding="utf-8")
    try:
        res = evaluator.quick_test(task, path, tag)
    except Exception as exc:  # noqa: BLE001 -- one bad kernel must not lose the arm
        out["failure"] = "evaluator:%s" % type(exc).__name__
        return out

    out["correct"] = bool(res.get("ok"))
    if not out["correct"]:
        out["failure"] = str(res.get("failure_kind") or "unknown")
        out["log_tail"] = str(res.get("log_tail", ""))[:300]
        return out
    # G21, and this script was itself a fresh instance of it: `robust_ms`/`median_ms`/`mean_ms` are
    # names from the in-memory model, NOT the stored keys. The worker writes
    # {max, mean, median, min, n, samples, std} -- unsuffixed -- so all three guesses read None and
    # every candidate reported `ms: null` while the measurement sat correctly on disk. An A/B whose
    # arms all have a null objective cannot be ranked at all.
    #
    # `latency_ms_of` is the one accessor that knows the real keys, and it prefers median over mean
    # by measurement (rank-correctness 93.2% vs 64.8% at n=20). Its argument is anything carrying a
    # `latency_ms` dict, which the evaluator result is.
    out["ms"] = latency_ms_of(res)
    if out["ms"] is None:
        # Correct but untimed is a distinct outcome and must not be silently averaged away.
        out["failure"] = "no_latency_in_result:keys=%s" % sorted((res.get("latency_ms") or {}).keys())
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", required=True, help="e.g. level3:48")
    ap.add_argument("--config", required=True)
    ap.add_argument("--candidate", required=True,
                    help="the starting kernel: a real, already-tuned candidate source")
    ap.add_argument("--profile-json", default=None,
                    help="that candidate's ProfileRecord, for the rich arm's per-candidate numbers")
    ap.add_argument("--report-json", required=True,
                    help="that candidate's real BottleneckReport")
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--replicates", type=int, default=1,
                    help="calls per arm. 1 is an ANECDOTE and is labelled as such in the output.")
    ap.add_argument("--n-candidates", type=int, default=2)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

    from kernel_optimizer.agents.modules import RewriterInputs
    from kernel_optimizer.config import load_config
    from kernel_optimizer.models.core import ProfileRecord
    from kernel_optimizer.models.reports import BottleneckReport
    from kernel_optimizer.tasks.kernelbench import parse_task_arg
    from kernel_optimizer.wiring import Runtime, build_orchestrator, load_task

    cfg = load_config(Path(args.config))
    level, pid = parse_task_arg(args.task)
    task = load_task(cfg, level, pid)
    best_source = Path(args.candidate).read_text(encoding="utf-8")
    report = BottleneckReport.model_validate_json(
        Path(args.report_json).read_text(encoding="utf-8"))
    profile = None
    if args.profile_json:
        profile = ProfileRecord.model_validate_json(
            Path(args.profile_json).read_text(encoding="utf-8"))

    arms = [a for a in args.arms.split(",") if a]

    # GUARD AGAINST A VACUOUS ARM. The `rich` arm's whole content is the per-candidate fields added
    # for G4/G6/G2. A profile captured before those existed carries none of them, so `_rich_report`
    # returns the report unchanged and `rich` becomes a silent duplicate of `verdict` -- two arms
    # that differ in nothing, reported as if they were compared. That is the credible-no-op failure
    # this project keeps hitting, so it is refused rather than warned about.
    if "rich" in arms:
        present = [f for f in ("peak_alloc_bytes", "peak_reserved_bytes", "candidate_aten_bytes",
                               "candidate_aten_ops", "threads_launched")
                   if profile is not None and getattr(profile, f, None) is not None]
        if not present:
            print("REFUSING TO RUN: the `rich` arm would be identical to `verdict`.")
            print("  The profile supplied carries none of the per-candidate fields that arm exists")
            print("  to add (peak_alloc_bytes, candidate_aten_bytes, candidate_aten_ops,")
            print("  threads_launched). It was almost certainly captured before those were")
            print("  measured. Re-measure them on this candidate with")
            print("  scripts/probe_per_candidate_cost.py and pass the result via --profile-json,")
            print("  or run with --arms none,verdict and say so when reporting.")
            return 2
        print("rich arm adds: %s" % ", ".join(present))

    rows: list[dict] = []
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_path = Path(args.out or ("g9-%s-l%d-%d.json" % (stamp, level, pid)))

    from kernel_optimizer.store.run_store import RunStore

    runs_dir = Path(cfg.run.runs_dir)
    if not runs_dir.is_absolute():
        runs_dir = Path(__file__).resolve().parents[1] / runs_dir
    store = RunStore.create(runs_dir, "g9-%s" % stamp, {
        "task": task.name, "config": cfg.model_dump(mode="json"),
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "purpose": "G9 information A/B: does more measured evidence produce better rewrites",
    })
    workdir = store.run_dir / "candidates"

    with Runtime(cfg, log_dir=store.run_dir) as runtime:
        orch = build_orchestrator(cfg, store, task, runtime)
        # `eval_semantics` and `calibration` are populated by `orch.run()`, which is not being
        # called here -- so they must be established explicitly. Both are held CONSTANT across arms
        # (except the deliberate calibration removal in `none`): letting either differ would confound
        # the evidence variable with the evaluation rules or the ceilings.
        semantics = orch.deps.benchmarker.probe_semantics(task)
        calibration = getattr(orch, "calibration", None)
        if calibration is None:
            from kernel_optimizer.gpu.calibrate import ensure_calibration
            try:
                calibration = ensure_calibration(orch.deps.evaluator.worker, runs_dir, store=store)
            except Exception as exc:  # noqa: BLE001 -- a box without calibration must still run
                print("note: no calibration (%s); the ceilings block will be omitted from every "
                      "arm, which keeps the arms comparable" % type(exc).__name__)
        for rep in range(args.replicates):
            for arm in arms:
                t0 = time.time()
                rec: dict = {"arm": arm, "replicate": rep}
                try:
                    outcome = orch.deps.rewriter.invoke(RewriterInputs(
                        task=task, best_source=best_source,
                        report=_apply_arm(report, profile, arm),
                        failed_hypotheses=[], device=cfg.device,
                        n_candidates=args.n_candidates,
                        eval_semantics=semantics,
                        calibration=None if arm == "none" else calibration,
                    ))
                except Exception as exc:  # noqa: BLE001 -- one arm failing must not lose the rest
                    rec.update({"agent_error": "%s: %s" % (type(exc).__name__, str(exc)[:200])})
                    rows.append(rec)
                    print(json.dumps(rec))
                    continue
                rec["agent_s"] = round(time.time() - t0, 1)
                rec["cost"] = round(getattr(outcome, "cost", 0.0) or 0.0, 4)
                rec["n_produced"] = len(outcome.output.candidates)
                cands = []
                for i, c in enumerate(outcome.output.candidates):
                    try:
                        src = outcome.sandbox.read_output(c.file)
                    except Exception as exc:  # noqa: BLE001
                        cands.append({"file": c.file, "usable": False,
                                      "failure": "unreadable:%s" % type(exc).__name__})
                        continue
                    r = _evaluate(orch.deps.evaluator, task, src, workdir,
                                  "g9-%s-r%d-c%d" % (arm, rep, i))
                    r["file"] = c.file
                    r["hypothesis"] = (getattr(c, "hypothesis_id", "") or "")[:60]
                    r["change"] = (getattr(c, "change_summary", "") or "")[:160]
                    cands.append(r)
                rec["candidates"] = cands
                ok = [c["ms"] for c in cands if c.get("ms")]
                rec["n_usable"] = sum(1 for c in cands if c.get("usable"))
                rec["n_correct"] = sum(1 for c in cands if c.get("correct"))
                rec["best_ms"] = min(ok) if ok else None
                rows.append(rec)
                print(json.dumps({k: v for k, v in rec.items() if k != "candidates"}))
                out_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    out_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print()
    print("%-8s %9s %9s %9s %11s  %s" % ("arm", "produced", "usable", "correct", "best ms", "cost"))
    for arm in arms:
        sub = [r for r in rows if r["arm"] == arm]
        best = [r["best_ms"] for r in sub if r.get("best_ms")]
        print("%-8s %9s %9s %9s %11s  $%.4f" % (
            arm,
            sum(r.get("n_produced", 0) for r in sub),
            sum(r.get("n_usable", 0) for r in sub),
            sum(r.get("n_correct", 0) for r in sub),
            ("%.4f" % statistics.median(best)) if best else "-",
            sum(r.get("cost", 0.0) for r in sub)))
    print()
    n = args.replicates
    if n < 3:
        print("SAMPLE SIZE %d per arm. This is an ANECDOTE, not a rate: per-trial noise on these "
              "tasks reaches 16%% of the mean and near-tie bands span 5-9%%, so a single call per "
              "arm cannot separate arms that differ by less than that. Do not quote these as "
              "percentages." % n)
    else:
        print("SAMPLE SIZE %d per arm." % n)
    print("results: %s" % out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
