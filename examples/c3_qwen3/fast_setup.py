"""Bounded original-reference A/A and operator-supplied setup controls; no kernel authoring."""

import json
from pathlib import Path
from statistics import median
from time import perf_counter

from pydantic import Field, JsonValue

from .fast_evaluation import FastEvaluator, proven_local
from .fast_records import Artifact, TargetSpec
from .manual_data import fingerprint
from .model_binding import load_bundle
from .runner_records import FrozenRecord, RunnerError


class ReferenceAA(FrozenRecord):
    valid: bool
    site_id: str
    fixture_ids: tuple[str, ...]
    reference_a_us: tuple[tuple[float, ...], ...] = ()
    reference_b_us: tuple[tuple[float, ...], ...] = ()
    noise_us: float | None = None
    device_identity: dict[str, JsonValue] = Field(default_factory=dict)
    detail: str | None = None
    forward_calls: int = 0
    wall_ms: float = 0


def measure_reference(engine: FastEvaluator) -> TargetSpec:
    engine.check_framework()
    runtime, budget = engine.runtime, engine.budget
    ordinal = budget.admit(engine.stage, "local")
    if ordinal is None:
        raise RunnerError("local A/A admission denied")
    started, before = perf_counter(), runtime.runner.backend.forward_calls
    aa, bb = [], []
    valid, detail, identity = False, None, {}
    try:
        runtime.runner.binding.restore()
        runtime.device.ready()
        identity = runtime.device.identity()
        if sum(f.resident_bytes for f in engine.fixtures) > 128 * 1024 * 1024:
            raise RunnerError("A/A fixture snapshots exceed unchanged 128 MiB cap")
        with runtime.device.execution():
            for fixture in engine.fixtures:
                original = runtime.runner.binding.originals[fixture.original.module_path]
                samples = {"a": [], "b": []}
                order = ("a", "b") if ordinal % 2 == 0 else ("b", "a")
                for iteration in range(23):
                    for side in order:
                        call = runtime.device.tools.clone(fixture.original.call)
                        def invoke():
                            return original(*call.args, **call.kwargs)
                        if iteration < 3:
                            runtime.device.tools.check(fixture.original.expected, invoke())
                        else:
                            samples[side].append(runtime.device.time_us(invoke))
                        runtime.device.tools.check_state(fixture.original.touched_state, runtime.device.tools.state(call))
                aa.append(tuple(samples["a"]))
                bb.append(tuple(samples["b"]))
        valid = bool(aa) and runtime.runner.backend.forward_calls == before
    except (OSError, ValueError, RuntimeError, TypeError, AssertionError) as exc:
        detail = str(exc)
    report = ReferenceAA(valid=valid, site_id=engine.target.brief.site_id,
        fixture_ids=tuple(f.identity for f in engine.fixtures), reference_a_us=tuple(aa), reference_b_us=tuple(bb),
        noise_us=max(abs(median(a) - median(b)) for a, b in zip(aa, bb, strict=True)) if valid else None,
        device_identity=identity, detail=detail, forward_calls=runtime.runner.backend.forward_calls - before,
        wall_ms=(perf_counter() - started) * 1000)
    path = engine.files.output / "reference-aa.json"
    path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    if not report.valid:
        raise RunnerError(report.detail or "reference A/A invalid")
    return engine.target.model_copy(update={"reference_noise_us": report.noise_us, "noise_evidence": path,
                                           "noise_evidence_sha256": fingerprint(path)})


def run_controls(engine: FastEvaluator, correct: Artifact, wrong: Artifact) -> Path:
    if engine.budget.spec.lane != "setup" or correct.author != "control" or wrong.author != "control":
        raise RunnerError("setup requires separately authored operator control artifacts, not candidates")
    engine.capture()
    engine.target = measure_reference(engine)
    a, b = engine.model(None), engine.model(None)
    engine.begin_version(0)
    local = engine.local(engine.request(correct))
    goal = next(g for g in engine.runtime.runner.prepared.contract.goals if g.id == engine.target.goal)
    from .fast_model import FastModelRequest, evaluate_fast_model
    from .fast_prepare import FastAccess
    model = evaluate_fast_model(FastModelRequest(stage=engine.stage,
        access=FastAccess(profile="c3_fast_device", phase="development"), bundle_path=correct.bundle,
        params=correct.params.values, goal=goal.id, prompt_ids=goal.search_prompt_ids,
        oracle_refs=engine.files.oracle, local_reports=(local,)), engine.runtime) if proven_local(local) else None
    engine.begin_version(1)
    rejected = engine.local(engine.request(wrong))
    engine.runtime.runner.bind(load_bundle(engine.files.baseline, {}))
    result = {"valid": bool(a.valid and b.valid and proven_local(local) and model and model.valid and not rejected.valid),
        "author": "control", "DEV_GO": False, "target": engine.target.model_dump(mode="json"),
        "baseline_a": a.model_dump(mode="json"), "baseline_b": b.model_dump(mode="json"),
        "correct_local": str(local.raw_path), "wrong_local": str(rejected.raw_path),
        "correct_model": model.model_dump(mode="json") if model else None,
        "budget": engine.budget.state.model_dump(mode="json")}
    path = engine.files.output / "control.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return path
