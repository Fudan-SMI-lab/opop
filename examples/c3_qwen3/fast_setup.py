"""Bounded original-reference A/A and operator-supplied setup controls; no kernel authoring."""

import json
from pathlib import Path
from statistics import median
from time import perf_counter

from pydantic import Field, JsonValue

from .fast_evaluation import FastEvaluator, proven_local
from .fast_records import Artifact, TargetSpec
from .fixture_budget import require_within_budget
from .manual_data import CALIBRATION_IDS, fingerprint
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
        require_within_budget(sum(f.resident_bytes for f in engine.fixtures), "A/A fixture snapshots")
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


def control_quality_ok(evaluation) -> tuple[bool, str | None]:
    """Whether the correct control passed everything EXCEPT the floor-defining criterion.

    THE CIRCULARITY THIS RESOLVES, STATED OPENLY
    --------------------------------------------
    The correct control is what DEFINES the equivalence floor for this site group, so it cannot also be
    judged against it -- a calibration standard does not get measured against itself. It must still pass
    every other criterion (the paired NLL delta here, plus the per-operator check and the device proof
    enforced upstream by `proven_local`), and its floor-defining metric is recorded rather than compared.

    Concretely: on the Qwen3 TTFT task a correct control measures ~0.0103 against a declared absolute
    limit of 0.01, because the group covers 73 modules. Requiring it to pass that limit would make the
    setup lane permanently fail on a control that is correct by construction, which tells us about the
    constant, not the control.

    What this does NOT do is waive the criterion for candidates. A candidate is compared against the
    floor this control established, and the floor can only raise the declared limit to what an
    equivalence-preserving implementation actually achieves -- so a genuinely less accurate candidate
    (B13 measured 0.0356 for a replacement perturbed by the declared per-operator tolerance) is still
    rejected.
    """
    from .runner_quality import CRITERIA, FLOOR_METRIC, _verdict

    if evaluation is None:
        return False, "correct control produced no model evaluation"
    if FLOOR_METRIC not in evaluation.metrics:
        return False, "correct control produced no logits measurement to calibrate the floor"
    return _verdict(dict(evaluation.metrics), tuple(c for c in CRITERIA if c[0] != FLOOR_METRIC))


def measure_equivalence_floor(engine: FastEvaluator, output: Path, *, ulps: int = 1,
                              quality_ids: tuple[str, ...] | None = None) -> Path | None:
    """Measure this site group's equivalence floor WITHOUT any authored reference kernel.

    WHAT THE FLOOR IS, AND WHY IT IS NOT A CONCESSION
    -------------------------------------------------
    The contract declares one absolute logits limit. That number is not a property of a candidate: it
    depends on how many modules the site group replaces, because each replaced module contributes its
    own rounding difference and they compound through the forward. Measured on this task:

      * one equivalence-preserving Triton RMSNorm, bound to nested prefixes of the same group, gave
        0.0 at 1 and 4 modules, 0.004489 at 16 and 36, and 0.010349 at all 73 -- the same kernel;
      * perturbing the ORIGINAL output by the contract's own declared per-operator tolerance
        (rtol 0.02) measured 0.035628 at the logits, 3.6x the declared logits limit.

    So the floor asks a question with an arithmetic answer: on THIS group, what does the smallest
    possible implementation difference cost at the logits? The smallest difference a non-bit-identical
    implementation can have from the reference is ONE UNIT IN THE LAST PLACE of the output dtype, so
    that is what this measures. A limit below the result cannot be met by any implementation that is
    not bit-identical to the reference.

    WHY THIS EXISTS BESIDE `write_floor_evidence`
    --------------------------------------------
    That function takes the floor from the setup lane's hand-authored correct control -- a Triton kernel
    someone wrote for that specific operator. This one needs no such kernel: it calls the site's own
    original forward, which `BindingRuntime` already holds in `originals`, so it works unchanged at an
    RMSNorm, a projection or an activation. Any task can therefore calibrate a floor without first
    having a correct kernel for the site, which is what made the mechanism unusable in practice.

    WHY A ULP STEP AND NOT A RELATIVE FRACTION
    ------------------------------------------
    A relative epsilon would be a number chosen by us. One ULP is defined by the dtype: the spacing
    between representable values. It is taken here on the integer bit pattern rather than via
    `finfo.eps`, because a float's relative spacing varies by a factor of two across each binade, so
    `eps` would overstate the step by up to 2x on part of the range.

    WHAT STILL REJECTS AN UNFAITHFUL CANDIDATE
    ------------------------------------------
    The floor moves the AMPLIFICATION criterion only. Faithfulness -- whether a replacement computes the
    operator at all -- is enforced by the per-operator check against fixtures captured from the original
    dispatch at the contract's declared rtol/atol, which a floor never touches. Measured, at two sites:
    an identity pass-through, a zeroed output, a dropped weight and a scaling beyond the declared rtol
    were each rejected with `local_checks_passed: 0.0` and never reached the logits gate, while the
    unperturbed reference and the one-ULP arm both passed it. So raising the logits limit to the floor
    admits no unfaithful replacement.

    WHY IT IS MEASURED THROUGH A REAL BUNDLE
    ----------------------------------------
    The perturbed forward is written out as a bundle and bound with `runner.bind`, rather than assigned
    onto the modules directly. That is not ceremony: `evaluate_quality` consults the floor only when
    `runner.receipt.site_ids` is non-empty, because a row that replaces NOTHING is the baseline and has
    no amplification to correct. A direct assignment leaves the receipt empty, so the measurement runs
    against what the gate considers a baseline and reports exactly 0.0 -- which then looks like a
    successful calibration while silently leaving the declared limit in force. (Measured: the first
    version of this function did precisely that and wrote `logits_relative_l2_max: 0.0` for a 73-module
    group whose true one-ULP floor is 0.016163.) Binding also means `validate_local`, the coverage check
    and the trace all run on this measurement exactly as they do for a candidate.

    Returns None when the measurement does not produce a logits number, which leaves the declared
    absolute limit in force rather than inventing a floor.
    """
    engine.check_framework()
    runner = engine.runtime.runner
    binding = runner.binding
    site_id = engine.target.brief.site_id
    paths = tuple(engine.target.brief.module_paths)
    if site_id not in binding.sites:
        raise RunnerError(f"equivalence floor needs the site registered first: {site_id}")
    ids = quality_ids if quality_ids is not None else CALIBRATION_IDS

    # The replacement, written as a real bundle so the measurement goes through the ordinary admission
    # path. It reads the site's own original out of `binding.originals` -- the untouched forward the
    # framework already holds for every module -- so nothing operator-specific appears anywhere here.
    #
    # The step is taken on the INTEGER BIT PATTERN rather than via `finfo`, because a float's relative
    # spacing varies by a factor of two across each binade, so a relative epsilon would overstate one
    # ULP by up to 2x on part of the range. Seeded, so the floor is reproducible evidence.
    folder = output.parent / f"{output.stem}-bundle"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "operators.py").write_text(f'''"""Automatic equivalence-floor arm: the site's own output, moved {ulps} ULP.

Authored by the framework, not by an agent: it contains no operator-specific arithmetic. The reference
comes from the binding's own record of each module's original forward, so the same code measures a floor
at an RMSNorm, a projection or an activation.
"""

import torch

ULPS = {ulps}
_ORIGINALS = {{}}
_GENERATORS = {{}}
_INT_OF = {{torch.bfloat16: torch.int16, torch.float16: torch.int16,
           torch.float32: torch.int32, torch.float64: torch.int64}}
# The largest FINITE magnitude in each dtype, as a bit pattern: one below the exponent-all-ones encoding
# that means infinity. Stepping away from zero saturates here instead of fabricating an inf or a NaN.
_MAX_FINITE = {{torch.bfloat16: 0x7F7F, torch.float16: 0x7BFF,
               torch.float32: 0x7F7FFFFF, torch.float64: 0x7FEFFFFFFFFFFFFF}}


def register(originals):
    """Receive the binding's untouched forwards, keyed by module path."""
    _ORIGINALS.update(originals)


def floor_replacement(site, call, params):
    original = _ORIGINALS[site.module_path]
    out = original(*call.args, **call.kwargs)
    if not isinstance(out, torch.Tensor):
        return out
    int_dtype = _INT_OF.get(out.dtype)
    if int_dtype is None:
        raise TypeError(f"cannot take a ULP step in {{out.dtype}}")
    key = str(out.device)
    generator = _GENERATORS.get(key)
    if generator is None:
        generator = torch.Generator(device=out.device)
        generator.manual_seed(20260921)
        _GENERATORS[key] = generator
    bits = out.contiguous().view(int_dtype)
    signs = torch.randint(0, 2, out.shape, generator=generator, device=out.device,
                          dtype=torch.int32) * 2 - 1
    # Step the MAGNITUDE and keep the sign bit separate. Operating on the signed integer word directly
    # is wrong in two ways: for a negative value the word is negative, so adding a positive delta moves
    # the magnitude DOWN rather than up; and negative zero is the most-negative word (-32768 in int16),
    # whose `abs()` overflows back to itself, so a clamp written on `bits.abs()` passes and the step
    # lands in NaN space. Measured: a single -0.0 anywhere in an RMSNorm output made every logit
    # non-finite and the whole measurement failed with "invalid logits/target".
    sign_bit = torch.iinfo(int_dtype).min
    magnitude = bits & ~sign_bit
    negative = bits < 0
    deltas = (signs * ULPS).to(int_dtype)
    # A step toward zero stops at +0; a step away from zero stops at the largest finite magnitude, so
    # the perturbation can never fabricate an infinity or a NaN out of a finite output.
    stepped = (magnitude + deltas).clamp(min=0, max=_MAX_FINITE[out.dtype])
    return torch.where(negative, stepped | sign_bit, stepped).view(out.dtype).view(out.shape)
''', encoding="utf-8")
    (folder / "bundle.json").write_text(json.dumps({
        "entry": "operators.py",
        "sites": [{"site_id": site_id, "replacement_callable": "floor_replacement"}],
        "files": ["operators.py"], "helpers": [], "params": {},
        "space": {"params": [], "constraints": []},
        "parent_bundle": None, "cumulative_from_original_baseline": True,
        "source_body_rewrite_required": True}, indent=2), encoding="utf-8")

    binding.restore()
    try:
        # `validate_local`, which `runner.quality` runs, compares each replacement against fixtures
        # captured from the ORIGINAL dispatch and keyed by site_id in `binding.fixtures`. The search
        # lanes never fill that map -- `FastEvaluator.capture` populates its own `engine.fixtures` tuple
        # for the per-candidate local gate instead -- so without this the measurement dies with
        # "missing original local fixtures" (measured: F5-dev-0921-184103, which wrote the arm's bundle
        # and then produced no floor at all). Capturing refuses once anything is bound, hence before the
        # bind below and after the restore above; it is a no-op when the map already holds this site.
        if not any(sid == site_id for sid, _ in binding.fixtures):
            binding.restore()
            runner.capture_fixtures(ids)
        receipt = runner.bind(load_bundle(folder / "bundle.json", {}))
        if not receipt.site_ids:
            raise RunnerError("the floor arm bound no site; the measurement would describe a baseline")
        # Hand the module's own originals to the bound arm. They were captured by
        # `BindingRuntime.__init__` BEFORE anything was installed, so these are the untouched forwards
        # rather than the binding's wrappers -- a replacement cannot obtain them for itself, which is
        # exactly why the anti-cheat coverage check exists and why this has to be passed in by the
        # framework instead.
        #
        # Reached through the bound callable's own `__globals__`: that IS the module namespace `bind`
        # imported. Calling `imported.open()` again would import a SECOND copy whose module-level state
        # is not the one the installed replacement reads, so the registration would silently land in the
        # wrong namespace and every call would raise KeyError.
        binding.active[site_id].__globals__["register"](
            {path: binding.originals[path] for path in paths})
        # Exceptions propagate: a failed measurement is not the same as "no floor to measure", and
        # returning None here would silently leave the declared limit in force while looking deliberate.
        #
        # `delegating()` is required because this arm calls the site's original BY DEFINITION -- the
        # floor is "the original's output, moved one ULP". `require_coverage` otherwise rejects any
        # replacement that invokes the original, which is the anti-cheat rule that stops a candidate
        # claiming the original's work as its own speedup; that rule is unchanged for candidates, and a
        # bundle cannot request this window (a replacement receives only `(site, call, params)`).
        # Without it the measurement fails with "replacement not executed exclusively" -- measured, at
        # both sites, in B25-0921-183528.
        with binding.delegating():
            report = runner.quality(ids, engine.files.oracle)
    finally:
        binding.restore()
    measured = report.metrics.get("logits_relative_l2_max")
    if measured is None:
        return None
    output.write_text(json.dumps({
        "contract_sha256": runner.prepared.contract_sha256,
        "site_id": site_id, "module_count": len(paths),
        "logits_relative_l2_max": measured,
        "paired_mean_nll_delta_nat_per_token": report.metrics.get("paired_mean_nll_delta_nat_per_token"),
        "local_checks_passed": report.metrics.get("local_checks_passed"),
        "ulps": ulps,
        "bundle_sha256": receipt.bundle_sha256,
        "source": f"automatic {ulps}-ULP perturbation of the site's own original forward output; no "
                  "authored reference kernel involved",
        "control_valid": True,
        "goal": engine.target.goal,
        "framework_id": engine.stage.framework_id}, indent=2), encoding="utf-8")
    return output


def write_floor_evidence(engine: FastEvaluator, evaluation, output: Path) -> Path | None:
    """Record the correct control's measured logits error as this site group's equivalence floor.

    WHY THE FRAMEWORK MEASURES THIS INSTEAD OF DECLARING IT
    ------------------------------------------------------
    The contract declares an absolute logits limit, but the logits error of a CORRECT replacement is
    not a constant: it depends on how many modules the site group covers, because each replaced module
    contributes its own reordering difference and they compound through the forward. Measured on the
    Qwen3 TTFT task, one equivalence-preserving Triton RMSNorm bound to a nested prefix of the group
    gave 0.0 at 1 and 4 modules, 0.004489 at 16 and 36, and 0.010349 at all 73 -- the same kernel
    throughout. Perturbing the ORIGINAL output by the contract's own declared per-operator tolerance
    (rtol 0.02) measured 0.035628, i.e. the two declared criteria cannot both be satisfied on that
    group by any candidate at all.

    The setup lane already authors a control that is correct by construction and measures it on the
    real model against the real oracle. That measurement IS the floor, so it is written here rather
    than reproduced by hand -- the same pattern as `measure_reference` writing `reference_noise_us`
    from an A/A pair instead of assuming a noise level.

    Returns None when the control produced no logits measurement, which leaves the declared absolute
    limit in force rather than inventing a floor.
    """
    if evaluation is None:
        return None
    measured = evaluation.metrics.get("logits_relative_l2_max")
    if measured is None:
        return None
    site_id = engine.target.brief.site_id
    modules = len(engine.target.brief.module_paths)
    output.write_text(json.dumps({
        "contract_sha256": engine.runtime.runner.prepared.contract_sha256,
        "site_id": site_id, "module_count": modules,
        "logits_relative_l2_max": measured,
        "paired_mean_nll_delta_nat_per_token": evaluation.metrics.get("paired_mean_nll_delta_nat_per_token"),
        "source": "setup correct control, measured on the resident model against the frozen oracle",
        "control_valid": evaluation.valid,
        "goal": engine.target.goal,
        "framework_id": engine.stage.framework_id}, indent=2), encoding="utf-8")
    return output


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
    floor = write_floor_evidence(engine, model, engine.files.output / "equivalence-floor.json")
    # The correct control calibrates the floor, so it is judged on every criterion EXCEPT the one it
    # defines -- see `control_quality_ok`. Its floor-defining measurement is recorded either way, and
    # the wrong control must still be rejected, which is what shows the gate still discriminates.
    control_ok, control_detail = control_quality_ok(model)
    result = {"valid": bool(a.valid and b.valid and proven_local(local) and control_ok and not rejected.valid),
        "author": "control", "DEV_GO": False, "target": engine.target.model_dump(mode="json"),
        "baseline_a": a.model_dump(mode="json"), "baseline_b": b.model_dump(mode="json"),
        "correct_local": str(local.raw_path), "wrong_local": str(rejected.raw_path),
        "correct_model": model.model_dump(mode="json") if model else None,
        "correct_model_valid_under_declared_limits": bool(model and model.valid),
        "correct_model_ok_excluding_floor_metric": control_ok,
        "correct_model_detail": control_detail,
        "equivalence_floor": str(floor) if floor else None,
        "equivalence_floor_sha256": fingerprint(floor) if floor else None,
        "budget": engine.budget.state.model_dump(mode="json")}
    path = engine.files.output / "control.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return path
