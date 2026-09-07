"""kernel-opt CLI: doctor, baseline, tune-file, agent-smoke, run, resume, report."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from kernel_optimizer.config import AppConfig, load_config


def _cfg(args) -> AppConfig:
    path = Path(args.config) if getattr(args, "config", None) else None
    return load_config(path, getattr(args, "override", None) or [])


def _new_store(cfg: AppConfig, prefix: str, task_repr: str):
    from kernel_optimizer.store.run_store import RunStore

    run_id = f"{prefix}-{time.strftime('%Y%m%d-%H%M%S')}"
    runs_dir = Path(cfg.run.runs_dir)
    if not runs_dir.is_absolute():
        runs_dir = Path(__file__).resolve().parents[2] / runs_dir
    return RunStore.create(runs_dir, run_id, {
        "task": task_repr,
        "config": cfg.model_dump(mode="json"),
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
    })


# ------------------------------------------------------------------ doctor


def cmd_doctor(args) -> int:
    cfg = _cfg(args)
    ok = True

    def check(label: str, passed: bool, detail: str = "") -> None:
        nonlocal ok
        mark = "OK " if passed else "FAIL"
        print(f"[{mark}] {label}" + (f" — {detail}" if detail else ""))
        ok = ok and passed

    # WSL
    try:
        out = subprocess.run(["wsl.exe", "-l", "-q"], capture_output=True, timeout=30)
        distros = out.stdout.decode("utf-16-le", errors="replace").split()
        check("WSL distro", cfg.wsl.distro in distros, f"found: {distros}")
    except (OSError, subprocess.TimeoutExpired) as exc:
        check("WSL distro", False, str(exc))

    # opencode CLI
    try:
        out = subprocess.run(["opencode", "--version"], capture_output=True,
                             timeout=30, shell=True)
        version = out.stdout.decode().strip()
        check("opencode CLI", bool(version), version)
    except (OSError, subprocess.TimeoutExpired) as exc:
        check("opencode CLI", False, str(exc))

    # KernelBench
    kb = Path(cfg.kernelbench_root)
    check("KernelBench problems", (kb / "KernelBench" / "level1").is_dir(), str(kb))
    check("KernelBench src", (kb / "src" / "kernelbench").is_dir())

    # WSL venv + GPU probe (real)
    from kernel_optimizer.gpu.jobs import make_env_probe_job
    from kernel_optimizer.gpu.worker_client import WslGpuWorker

    tmp_jobs = Path(cfg.run.runs_dir) / "_doctor-jobs"
    if not tmp_jobs.is_absolute():
        tmp_jobs = Path(__file__).resolve().parents[2] / tmp_jobs
    worker = WslGpuWorker(cfg.wsl, cfg.gpu.concurrency, jobs_dir=tmp_jobs)
    result = worker.run_job(make_env_probe_job(), timeout_s=120, tag="doctor",
                            lock_mode="shared")
    if result.get("ok"):
        check("WSL venv torch", True,
              f"torch {result.get('torch')}, cuda={result.get('cuda_available')}, "
              f"{result.get('device_name')}")
        check("WSL venv triton", result.get("triton") is not None,
              str(result.get("triton") or result.get("triton_error")))
        check("kernelbench importable in WSL", bool(result.get("kernelbench_importable")),
              str(result.get("kernelbench_error", "")))
        check("CUDA available", bool(result.get("cuda_available")))
        # INFORMATIONAL, deliberately not a check(): a Triton-only box is perfectly valid, and
        # every candidate produced across 20 runs so far has been Triton. But `load_inline`
        # (the whole `cuda` backend) needs nvcc AND ninja on PATH, and when ninja is missing
        # every CUDA candidate dies at the witness gate with a compile_error that reads like
        # the agent wrote bad code. Say up front which backends this box can build.
        if result.get("cuda_backend_buildable"):
            print(f"[info] cuda backend buildable — nvcc {result.get('nvcc')}, "
                  f"ninja {result.get('ninja')}")
        else:
            missing = [n for n in ("nvcc", "ninja") if not result.get(n)]
            print(f"[info] cuda backend NOT buildable — missing {', '.join(missing) or '?'}. "
                  f"Triton candidates are unaffected; a `backend: cuda` candidate would fail "
                  f"to compile at the witness gate. Fix: install ninja / put nvcc on PATH.")
    else:
        check("WSL venv probe", False,
              f"{result.get('failure_kind')}: {str(result.get('log_tail'))[:300]}")

    # --- step 4: which profiling TIER this box supports, and its measured ceilings ---------
    # Reported here because every bottleneck verdict in a run is a fraction of these numbers, and
    # an operator who learns at hour 9 of a 12-hour run that the box could not be calibrated has
    # learned it too late. Informational, not a check(): a box with no disassembler still runs
    # perfectly well, it just produces fewer signals -- and saying which signals are missing up
    # front is the whole point.
    print()
    print("[info] profiling capability tiers (hardware counters need a host-side kernel-module")
    print("       permission that cannot be set from inside a container, so Tier 3 is expected")
    print("       to be unavailable here):")
    from kernel_optimizer.evaluation.statics import find_cuda_tool

    nvdisasm = find_cuda_tool("nvdisasm")
    cuobjdump = find_cuda_tool("cuobjdump")
    ncu_path = find_cuda_tool("ncu")
    print(f"       Tier 0 (CUDA events, CPU-issue loop, FLOP/byte counts): available")
    print(f"       Tier 1 (SASS instruction mix, analytic occupancy): "
          f"{'available' if (nvdisasm or cuobjdump) else 'UNAVAILABLE'}"
          + (f" — {nvdisasm or cuobjdump}" if (nvdisasm or cuobjdump) else
             " — no nvdisasm/cuobjdump found; tensor-core use, spills, shared traffic and "
             "access widths will all be reported as unknown"))
    if ncu_path:
        probe = subprocess.run([ncu_path, "--version"], capture_output=True, timeout=60)
        print(f"       Tier 3 (hardware counters): ncu present at {ncu_path}, "
              f"{'runs' if probe.returncode == 0 else 'not usable'} — counters themselves are "
              f"gated by ERR_NVGPUCTRPERM in a container regardless")
    else:
        print("       Tier 3 (hardware counters): ncu not found (expected; not required)")

    cal = None
    try:
        from kernel_optimizer.evaluation.calibration import cache_path, load_cached
        from kernel_optimizer.gpu.calibrate import ensure_calibration

        runs_dir = Path(cfg.run.runs_dir)
        if not runs_dir.is_absolute():
            runs_dir = Path(__file__).resolve().parents[2] / runs_dir
        cal = load_cached(cache_path(runs_dir))
        if cal is None and getattr(args, "calibrate", False):
            cal = ensure_calibration(worker, runs_dir)
    except Exception as exc:  # noqa: BLE001 — doctor must report, never crash
        print(f"[info] calibration unreadable: {type(exc).__name__}: {exc}")

    if cal is not None:
        print()
        print(f"[info] cached calibration for {cal.device_name} (measured {cal.measured_at}):")
        print(f"       DRAM {cal.dram_tbs:.3f} TB/s, fp32 {cal.fp32_tflops:.1f} TFLOP/s"
              + (f", tf32 {cal.tf32_tflops:.1f} TFLOP/s" if cal.tf32_tflops else "")
              + f", launch floor {cal.empty_launch_floor_ms*1e3:.1f} us")
        if cal.thresholds:
            t = cal.thresholds
            print(f"       thresholds: dram>={t.dram_saturated_frac} "
                  f"compute>={t.compute_saturated_frac} idle<{t.idle_frac} "
                  f"cpu/gpu>={t.launch_bound_cpu_ratio}")
        for s in cal.suspect:
            print(f"       ⚠ SUSPECT: {s}")
        # A calibration measured on OTHER hardware is worse than none: it would be refused at
        # run time (the cache is identity-keyed) but an operator seeing it here would wrongly
        # believe the box is ready.
        probe_result = result if result.get("ok") else {}
        live_name = probe_result.get("device_name")
        if live_name and live_name != cal.device_name:
            check("cached calibration matches this GPU", False,
                  f"cache is for {cal.device_name}, this box is {live_name} — it will be "
                  f"refused and re-measured at run start")
    else:
        print()
        print("[info] no cached calibration for this box. The first `run` will measure it "
              "(~2 min of exclusive GPU time); `kernel-opt calibrate` does it now. Without one, "
              "bottleneck verdicts report `unknown` rather than comparing against a guessed "
              "ceiling.")

    print("\nall green" if ok else "\nsome checks FAILED", file=sys.stderr)
    return 0 if ok else 1


# ------------------------------------------------------------------ baseline


def cmd_baseline(args) -> int:
    cfg = _cfg(args)
    from kernel_optimizer.tasks.kernelbench import parse_task_arg
    from kernel_optimizer.wiring import build_gpu_stack, load_task

    level, pid = parse_task_arg(args.task)
    task = load_task(cfg, level, pid)
    store = _new_store(cfg, f"baseline-l{level}-{pid}", task.name)
    _, _, benchmarker, _ = build_gpu_stack(cfg, store)
    print(f"measuring baselines for {task.name} (level {level}) ...")
    for baseline in benchmarker.measure_baseline(task):
        store.append("BASELINE_DONE", {"baseline": baseline.model_dump()})
        lat = baseline.latency_ms
        note = f"  [{baseline.note}]" if baseline.note else ""
        print(f"  {baseline.kind:14s}: {lat.mean} ms (std {lat.std}, "
              f"min {lat.min}, max {lat.max}, n={lat.n_samples}){note}")
    print(f"run dir: {store.run_dir}")
    return 0


# ------------------------------------------------------------------ tune-file


def cmd_tune_file(args) -> int:
    cfg = _cfg(args)
    from kernel_optimizer.models.core import (
        Constraint,
        ParamDomain,
        ParameterSpace,
        ParamSet,
        TrialRecord,
        sha256_text,
    )
    from kernel_optimizer.paramspace import materializer
    from kernel_optimizer.paramspace.guard import check_config
    from kernel_optimizer.tasks.kernelbench import parse_task_arg
    from kernel_optimizer.tuning.stats import TuningStatsAnalyzer
    from kernel_optimizer.tuning.tpe import OptunaTPETuner
    from kernel_optimizer.wiring import build_gpu_stack, load_task
    from kernel_optimizer.evaluation.correctness import latency_from_result

    level, pid = parse_task_arg(args.task)
    task = load_task(cfg, level, pid)
    source = Path(args.candidate).read_text(encoding="utf-8")
    space_raw = json.loads(Path(args.space).read_text(encoding="utf-8"))

    store = _new_store(cfg, f"tunefile-l{level}-{pid}", task.name)
    _, evaluator, _, profiler = build_gpu_stack(cfg, store)

    defaults = materializer.extract_defaults(source)
    domains = [ParamDomain(**d) for d in space_raw["params"]]
    constraints = [Constraint(**c) for c in space_raw.get("constraints", [])]
    space = ParameterSpace(space_id="sp-manual", candidate_id="cand-manual",
                           source_sha=sha256_text(source), domains=domains,
                           constraints=constraints)
    tuner = OptunaTPETuner(
        space,
        guard_ok=lambda p: check_config(space, p, cfg.device) is None,
        budget=args.trials, seed=cfg.run.seed,
        anchors=(ParamSet(values=defaults),),
    )
    trials: list[TrialRecord] = []
    trials_dir = store.run_dir / "candidates" / "manual"
    trials_dir.mkdir(parents=True, exist_ok=True)
    while True:
        asked = tuner.ask()
        if asked is None:
            break
        trial_id, params = asked
        try:
            mat = materializer.materialize(source, params)
        except materializer.MaterializeError as exc:
            record = TrialRecord(trial_id=trial_id, candidate_id="cand-manual",
                                 space_id=space.space_id, params=params, status="fail",
                                 failure_kind="materialize_error",
                                 failure_detail=str(exc)[:300])
            tuner.tell(trial_id, record)
            trials.append(record)
            continue
        path = trials_dir / f"{trial_id}.py"
        path.write_text(mat, encoding="utf-8")
        result = evaluator.quick_test(task, path, tag=trial_id, backend=args.backend)
        lat = latency_from_result(result)
        if result.get("ok") and lat:
            record = TrialRecord(trial_id=trial_id, candidate_id="cand-manual",
                                 space_id=space.space_id, params=params,
                                 status="complete", latency_ms=lat,
                                 profile=profiler.extract(result))
            print(f"  {trial_id}: {lat.mean} ms  {params.values}")
        else:
            record = TrialRecord(trial_id=trial_id, candidate_id="cand-manual",
                                 space_id=space.space_id, params=params, status="fail",
                                 failure_kind=result.get("failure_kind") or "runtime_error",
                                 failure_detail=str(result.get("log_tail"))[:300])
            print(f"  {trial_id}: FAIL ({record.failure_kind})  {params.values}")
        tuner.tell(trial_id, record)
        trials.append(record)
        store.append("TRIAL_DONE", {"trial": record.model_dump()})

    best = tuner.best()
    stats = TuningStatsAnalyzer(cfg.device).analyze(space, trials)
    (store.run_dir / "report").mkdir(exist_ok=True)
    (store.run_dir / "report" / "tuning_stats.json").write_text(
        stats.model_dump_json(indent=2), encoding="utf-8")
    if best:
        print(f"\ntheta_best: {best.params.values} -> {best.latency_ms.mean} ms")
        if best.profile:
            print(f"profile: regs={best.profile.n_regs} spills={best.profile.n_spills} "
                  f"shared={best.profile.shared_bytes}B")
    else:
        print("\nno successful trial")
    print(f"stats: {store.run_dir / 'report' / 'tuning_stats.json'}")
    return 0


# ------------------------------------------------------------------ agent-smoke


def cmd_agent_smoke(args) -> int:
    cfg = _cfg(args)
    from kernel_optimizer.agents.modules import GeneratorInputs, ParameterizerInputs
    from kernel_optimizer.tasks.kernelbench import parse_task_arg
    from kernel_optimizer.wiring import Runtime, build_orchestrator, load_task

    level, pid = parse_task_arg(args.task)
    task = load_task(cfg, level, pid)
    store = _new_store(cfg, f"smoke-{args.module}-l{level}-{pid}", task.name)

    with Runtime(cfg, log_dir=store.run_dir) as runtime:
        orch = build_orchestrator(cfg, store, task, runtime)
        if args.module == "generator":
            outcome = orch.deps.generator.invoke(GeneratorInputs(
                task=task, ref_source=Path(task.ref_path).read_text(encoding="utf-8"),
                device=cfg.device, n_candidates=2))
            print(f"got {len(outcome.output.candidates)} candidates "
                  f"(attempts={outcome.attempts}, cost=${outcome.cost:.4f})")
            for c in outcome.output.candidates:
                src = outcome.sandbox.read_output(c.file)
                from kernel_optimizer.paramspace import materializer
                try:
                    defaults = materializer.extract_defaults(src)
                    print(f"  {c.file}: PARAMS ok {defaults} — {c.approach_summary[:80]}")
                except materializer.MaterializeError as exc:
                    print(f"  {c.file}: PARAMS INVALID ({exc.kind}) — {c.approach_summary[:80]}")
        elif args.module == "parameterizer":
            candidate_src = Path(args.candidate).read_text(encoding="utf-8")
            outcome = orch.deps.parameterizer.invoke(ParameterizerInputs(
                task=task, candidate_source=candidate_src, device=cfg.device))
            print(f"space: {[p.name for p in outcome.output.space.params]} "
                  f"(attempts={outcome.attempts}, cost=${outcome.cost:.4f})")
            print(json.dumps(outcome.output.space.model_dump(), indent=2)[:2000])
        else:
            print(f"unsupported module for smoke: {args.module}", file=sys.stderr)
            return 2
    print(f"run dir: {store.run_dir}")
    return 0


# ------------------------------------------------------------------ run / resume / report


def cmd_run(args) -> int:
    cfg = _cfg(args)
    from kernel_optimizer.reporting.report import ReportGenerator
    from kernel_optimizer.tasks.kernelbench import parse_task_arg
    from kernel_optimizer.wiring import Runtime, build_orchestrator, load_task

    level, pid = parse_task_arg(args.task)
    task = load_task(cfg, level, pid)
    store = _new_store(cfg, f"run-l{level}-{pid}", task.name)
    print(f"run dir: {store.run_dir}")

    with Runtime(cfg, log_dir=store.run_dir) as runtime:
        orch = build_orchestrator(cfg, store, task, runtime)
        summary = orch.run()
    report = ReportGenerator().generate(store)
    print(json.dumps(summary.get("best"), indent=2, ensure_ascii=False))
    print(f"report: {report}")
    return 0


def cmd_resume(args) -> int:
    cfg = _cfg(args)
    from kernel_optimizer.models.core import TaskSpec
    from kernel_optimizer.reporting.report import ReportGenerator
    from kernel_optimizer.store.run_store import RunStore
    from kernel_optimizer.wiring import Runtime, build_orchestrator

    run_dir = Path(args.run)
    store = RunStore.open(run_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    if getattr(args, "config", None) is None and manifest.get("config"):
        cfg = AppConfig.model_validate(manifest["config"])
    task_payload = None
    snapshot = run_dir / "state.json"
    if snapshot.exists():
        task_payload = json.loads(snapshot.read_text(encoding="utf-8")).get("task")
    if not task_payload:
        print("cannot recover task from run dir (no state.json)", file=sys.stderr)
        return 2
    task = TaskSpec.model_validate(task_payload)

    with Runtime(cfg, log_dir=store.run_dir) as runtime:
        orch = build_orchestrator(cfg, store, task, runtime)
        summary = orch.run()
    report = ReportGenerator().generate(store)
    print(json.dumps(summary.get("best"), indent=2, ensure_ascii=False))
    print(f"report: {report}")
    return 0


def cmd_report(args) -> int:
    from kernel_optimizer.reporting.report import ReportGenerator
    from kernel_optimizer.store.run_store import RunStore

    store = RunStore.open(Path(args.run))
    report = ReportGenerator().generate(store)
    print(f"report regenerated: {report}")
    return 0


def cmd_calibrate(args) -> int:
    """Measure this box's ceilings and derive the classification thresholds.

    Runnable on its own so an operator can see, and audit, the numbers every bottleneck verdict
    will be computed against -- before a 12-hour run rests on them.
    """
    cfg = _cfg(args)
    from kernel_optimizer.gpu.calibrate import ensure_calibration
    from kernel_optimizer.gpu.worker_client import WslGpuWorker

    runs_dir = Path(cfg.run.runs_dir)
    if not runs_dir.is_absolute():
        runs_dir = Path(__file__).resolve().parents[2] / runs_dir
    jobs = runs_dir / "_calibrate-jobs"
    worker = WslGpuWorker(cfg.wsl, cfg.gpu.concurrency, jobs_dir=jobs)

    cal = ensure_calibration(worker, runs_dir, recalibrate=args.recalibrate)
    if cal is None:
        print("calibration FAILED — the classifier will report `unknown` rather than guess",
              file=sys.stderr)
        return 1

    print(f"device      {cal.device_name} sm_{''.join(str(c) for c in cal.capability)}"
          f"  {cal.sm_count} SMs")
    print(f"measured at {cal.measured_at}")
    print()
    print("--- measured ceilings (what a kernel can actually get here, not datasheet) ---")
    print(f"  DRAM                 {cal.dram_tbs:.4f} TB/s"
          + (f"   ({cal.dram_tbs / cal.spec_dram_tbs * 100:.1f}% of derived spec "
             f"{cal.spec_dram_tbs:.3f})" if cal.spec_dram_tbs else ""))
    print(f"  fp32 (tf32 off)      {cal.fp32_tflops:.2f} TFLOP/s")
    print(f"  tf32 (tensor cores)  {cal.tf32_tflops:.2f} TFLOP/s")
    print(f"  empty-launch floor   {cal.empty_launch_floor_ms*1e3:.1f} us")
    print(f"  L2                   {cal.l2_bytes/1e6:.1f} MB")
    print(f"  roofline ridge       {cal.ridge_flop_per_byte:.1f} FLOP/byte (fp32)"
          + (f", {cal.tf32_ridge_flop_per_byte:.1f} (tf32)"
             if cal.tf32_ridge_flop_per_byte else ""))
    print()
    print("--- yardsticks (analytic truth on the left, what this box did on the right) ---")
    for y in cal.yardsticks:
        print(f"  {y.truth:12} {y.name:36} {y.gpu_ms:8.4f} ms  "
              f"cpu/gpu {y.cpu_over_gpu:6.3f}  "
              f"dram {y.pct_of_dram(cal.dram_tbs)*100:5.1f}%  "
              f"fp32 {y.pct_of_fp32(cal.fp32_tflops)*100:5.1f}%")
    print()
    if cal.thresholds:
        t = cal.thresholds
        print("--- derived thresholds (dimensionless, so they travel to the next card) ---")
        print(f"  dram_saturated_frac    {t.dram_saturated_frac}")
        print(f"  compute_saturated_frac {t.compute_saturated_frac}")
        print(f"  idle_frac              {t.idle_frac}")
        print(f"  launch_bound_cpu_ratio {t.launch_bound_cpu_ratio}")
        print()
        for key, why in t.derivation.items():
            print(f"  {key}:\n    {why}")
    if cal.suspect:
        print()
        for s in cal.suspect:
            print(f"[SUSPECT] {s}", file=sys.stderr)
    return 0


# ------------------------------------------------------------------ entry


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="kernel-opt")
    parser.add_argument("--config", help="YAML config path")
    parser.add_argument("-o", "--override", action="append",
                        help="dotted config override, e.g. budgets.trials_per_space=8")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("doctor", help="environment health check")
    p.add_argument("--calibrate", action="store_true",
                   help="also MEASURE this box's ceilings if no calibration is cached "
                        "(~2 min of exclusive GPU time). Without it doctor only reports whether "
                        "one exists.")

    p = sub.add_parser("baseline", help="measure reference baselines")
    p.add_argument("--task", required=True, help="e.g. level1:19")

    p = sub.add_parser("tune-file", help="tune a hand-written candidate (no agents)")
    p.add_argument("--task", required=True)
    p.add_argument("--candidate", required=True)
    p.add_argument("--space", required=True, help="JSON space file")
    p.add_argument("--trials", type=int, default=8)
    p.add_argument("--backend", default="triton")

    p = sub.add_parser("agent-smoke", help="single real agent call")
    p.add_argument("--module", required=True, choices=["generator", "parameterizer"])
    p.add_argument("--task", required=True)
    p.add_argument("--candidate", help="candidate file (for parameterizer)")

    p = sub.add_parser("run", help="full optimization run")
    p.add_argument("--task", required=True)

    p = sub.add_parser("resume", help="resume an interrupted run")
    p.add_argument("--run", required=True, help="run directory")

    p = sub.add_parser("calibrate",
                       help="measure this box's ceilings and derive classification thresholds")
    p.add_argument("--recalibrate", action="store_true",
                   help="ignore the cached calibration and measure again (use after a driver "
                        "change, or when a previous calibration was flagged SUSPECT)")

    p = sub.add_parser("report", help="regenerate report from events.jsonl")
    p.add_argument("--run", required=True)

    args = parser.parse_args(argv)
    commands = {
        "doctor": cmd_doctor,
        "baseline": cmd_baseline,
        "tune-file": cmd_tune_file,
        "agent-smoke": cmd_agent_smoke,
        "run": cmd_run,
        "resume": cmd_resume,
        "report": cmd_report,
        "calibrate": cmd_calibrate,
    }
    return commands[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
