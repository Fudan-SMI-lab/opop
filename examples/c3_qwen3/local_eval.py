from collections.abc import Callable
import hashlib
from statistics import median
from time import perf_counter
from uuid import uuid4

from .device_gate import require_device_proof
from .device_records import FixtureResult, LocalInvocation, LocalReport, LocalRequest, LocalRuntime, PureFixture
from .model_binding import load_bundle
from .fast_prepare import require_feedback_inputs
from .operator_types import OperatorCall, Value
from .runner_records import RunnerError


def evaluate_local(request: LocalRequest, fixtures: tuple[PureFixture, ...], runtime: LocalRuntime) -> LocalReport:
    runner, device = runtime.runner, runtime.device
    started, before = perf_counter(), runner.backend.forward_calls
    counts = {"admitted": 0, "reference_calls": 0, "candidate_calls": 0, "warmup_calls": 0, "timed_calls": 0}
    rows: list[FixtureResult] = []
    valid, detail, stage = False, None, "admission"
    bundle_sha, params_sha, hashes = "", "", {}
    device_identity = {}
    stamp = runner.backend.weights_stamp()
    device_before = dict(device.counters)
    try:
        require_feedback_inputs(runner)
        if request.stage.stage == "final":
            raise RunnerError("fixture-only feedback is not available during sealed final evaluation")
        ordinal = runtime.admission.admit(request.stage, "local")
        if ordinal is None:
            raise RunnerError("stage budget denied local admission")
        counts["admitted"] = 1
        stage = "source_gate"
        bundle = load_bundle(request.bundle_path, request.params)
        bundle_sha, params_sha, hashes = bundle.bundle_sha256, bundle.params_sha256, dict(bundle.source_hashes)
        gate = request.source_gate
        if not gate.eligible or (gate.bundle_sha256, gate.params_sha256) != (bundle_sha, params_sha):
            raise RunnerError("matching SOURCE_ELIGIBLE receipt required before device work")
        if any(k.backend != "triton" for k in request.kernels):
            raise RunnerError("unsupported backend: fixture device proof currently supports Triton only")
        if not fixtures or any(f.description.site_id != request.site_id for f in fixtures):
            raise RunnerError("selected site fixtures are missing or mismatched")
        if sum(f.resident_bytes for f in fixtures) > 128 * 1024 * 1024:
            raise RunnerError("unsupported fixture: 128 MiB total cap exceeded")
        paths = runner.binding.sites.get(request.site_id, ())
        if not paths or any(f.original.module_path not in paths for f in fixtures):
            raise RunnerError("fixture paths must belong to the explicitly registered site group")
        proposed = dict(runner.binding.fixtures)
        for f in fixtures:
            proposed[(f.description.site_id, f.description.phase)] = f.original
        snapshot_bytes = sum(device.tools.size(f.call) + device.tools.size(OperatorCall((f.expected, f.touched_state), {}))
                             for f in proposed.values())
        working_bytes = max(device.tools.size(f.original.call) + device.tools.size(OperatorCall((f.original.expected,), {}))
                            for f in fixtures)
        if snapshot_bytes + working_bytes > 128 * 1024 * 1024:
            raise RunnerError("unsupported fixture: snapshots plus working input/output exceed 128 MiB")
        stage = "device_ready"
        device.ready()
        device_identity = device.identity()
        stage = "fixture_identity"
        for f in fixtures:
            if (f.description.contract_sha256 != runner.prepared.contract_sha256
                    or f.description.corpus_sha256 != runner.prepared.corpus_sha256):
                raise RunnerError("fixture contract/corpus identity mismatch")
            device.tools.require_pure(f.original.call)
            identity = hashlib.sha256((f.description.model_dump_json() + device.tools.digest(f.original.call)).encode()).hexdigest()
            if identity != f.identity or device.tools.describe_tensors(f.original.call) != f.description.tensors:
                raise RunnerError("fixture contents/shape/stride/dtype identity changed")
        stage = "bundle_import"
        runner.bind(bundle)
        if request.site_id not in runner.binding.active:
            raise RunnerError("requested site is not replaced by the cumulative bundle")
        for fixture in fixtures:
            rows.append(FixtureResult(fixture_identity=fixture.identity, description=fixture.description, quality_passed=False))
            path = fixture.original.module_path
            def candidate(call: OperatorCall) -> Value:
                counts["candidate_calls"] += 1
                return runner.binding.modules[path].forward(*call.args, **call.kwargs)
            def reference(call: OperatorCall) -> Value:
                counts["reference_calls"] += 1
                return runner.binding.originals[path](*call.args, **call.kwargs)
            stage = "compile_device_proof"
            invocation = LocalInvocation(bundle, fixture, candidate)
            evidence = tuple(device.prove(runner, declaration, invocation) for declaration in request.kernels)
            rows[-1] = rows[-1].model_copy(update={"device_evidence": evidence})
            for proof in evidence:
                if proof.compiled and proof.kernel_name in proof.cuda_names and proof.local_quality_passed is False:
                    stage = "local_correctness"
                require_device_proof(proof)
            stage = "local_correctness"
            with device.execution():
                for operation in (reference, candidate):
                    call = device.tools.clone(fixture.original.call)
                    result = operation(call)
                    device.tools.check(fixture.original.expected, result)
                    device.tools.check_state(fixture.original.touched_state, device.tools.state(call))
                    del result, call
                stage = "paired_microbenchmark"
                timings: dict[str, list[float]] = {"reference": [], "candidate": []}
                sides = ("reference", "candidate") if ordinal % 2 == 0 else ("candidate", "reference")
                operations: dict[str, Callable[[OperatorCall], Value]] = {"reference": reference, "candidate": candidate}
                for _ in range(3):
                    for side in sides:
                        operations[side](device.tools.clone(fixture.original.call))
                        counts["warmup_calls"] += 1
                for _ in range(20):
                    for side in sides:
                        call = device.tools.clone(fixture.original.call)
                        timings[side].append(device.time_us(lambda: operations[side](call)))
                        counts["timed_calls"] += 1
                rows[-1] = FixtureResult(fixture_identity=fixture.identity, description=fixture.description,
                    quality_passed=True, reference_us=tuple(timings["reference"]), candidate_us=tuple(timings["candidate"]),
                    pair_order=sides, device_evidence=evidence)
        for fixture in fixtures:
            key = (fixture.description.site_id, fixture.description.phase)
            runner.binding.fixtures[key] = fixture.original
        runner.binding.verify_installed()
        valid = True
    except (OSError, ValueError, RuntimeError, TypeError, SyntaxError, ImportError, ArithmeticError, AssertionError) as exc:
        detail = f"{type(exc).__name__}: {exc}"
    finally:
        runner.binding.restore()
        runner.quality_identity = None
        if runner.backend.weights_stamp() != stamp:
            runner.close()
            valid, detail = False, "resident weights changed; runner closed"
        counts["full_model_forwards"] = runner.backend.forward_calls - before
        counts.update({f"proof_{k}": v - device_before.get(k, 0) for k, v in device.counters.items()})
        if counts["full_model_forwards"]:
            valid, detail = False, "fixture-only operation unexpectedly executed a full model"
    runner.output_dir.mkdir(parents=True, exist_ok=True)
    path = runner.output_dir / f"fixture-only-{uuid4().hex}.json"
    report = LocalReport(stage=request.stage, valid=valid, failure_stage=None if valid else stage, detail=detail,
        latency_us=sum(median(r.candidate_us) for r in rows) / len(rows) if valid else None,
        bundle_sha256=bundle_sha, params_sha256=params_sha, source_hashes=hashes, fixtures=tuple(rows),
        declared_kernels=request.kernels,
        device_identity=device_identity,
        counts=counts, wall_ms=(perf_counter() - started) * 1000, raw_path=path.resolve())
    path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    return report
