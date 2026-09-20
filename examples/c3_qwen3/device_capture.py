import hashlib
import inspect
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from types import MappingProxyType

from .device_records import CaptureSpec, FixtureDescription, LocalRuntime, PureFixture, PureTools
from .device_scopes import replace_forward
from .fast_prepare import require_feedback_inputs
from .operator_types import ForwardModule, OperatorCall, OperatorFixture, TensorValue, Value
from .runner_records import BindingReceipt, RunnerError


@dataclass(frozen=True, slots=True)
class CaptureReport:
    valid: bool
    fixture: PureFixture | None
    raw_path: Path
    detail: str | None


def module_metadata(module: ForwardModule, tools: PureTools):
    parameters = tuple((name, value) for name, value in module.named_parameters() if isinstance(value, TensorValue))
    descriptors = tools.describe_tensors(OperatorCall(tuple(value for _, value in parameters), {}))
    weights = tuple(d.model_copy(update={"path": name}) for (name, _), d in zip(parameters, descriptors, strict=True))
    attributes = {name: value for name, value in vars(module).items()
                  if not name.startswith("_") and isinstance(value, (bool, int, float, str))}
    return weights, attributes


def capture_target_fixture(spec: CaptureSpec, runtime: LocalRuntime) -> CaptureReport:
    runner, tools = runtime.runner, runtime.device.tools
    started, before = perf_counter(), runner.backend.forward_calls
    fixture, detail = None, None
    current_step = 0
    captured = None
    stamp = runner.backend.weights_stamp()
    try:
        require_feedback_inputs(runner)
        if spec.stage.stage == "final":
            raise RunnerError("final inputs cannot be used for fixture feedback")
        if runtime.admission.admit(spec.stage, "capture") is None:
            raise RunnerError("stage budget denied fixture preparation")
        goal = next(g for g in runner.prepared.contract.goals if g.id == spec.goal)
        if spec.decode_step >= goal.output_tokens:
            raise RunnerError("capture position is outside the actual goal generation")
        runner.binding.restore()
        runner.receipt = BindingReceipt("baseline", MappingProxyType({}), "baseline", (), True)
        runner.quality_identity = None
        module = runner.binding.modules[spec.module_path]
        original = runner.binding.originals[spec.module_path]
        ids = goal.search_prompt_ids if spec.goal == "multi" else goal.search_prompt_ids[:1]
        rows = tuple(runner.prepared.prompts_by_id[p].input_ids for p in ids)
        if any(runner.prepared.prompts_by_id[p].split != "search" for p in ids):
            raise RunnerError("capture requires allowed search inputs")
        source = inspect.getsource(type(module))
        def capture(*args: Value, **kwargs: Value) -> Value:
            nonlocal captured
            if current_step != spec.decode_step or captured is not None:
                return original(*args, **kwargs)
            call = OperatorCall(args, kwargs)
            tools.require_pure(call)
            if tools.size(call) > 64 * 1024 * 1024:
                raise RunnerError("unsupported fixture: 64 MiB call cap")
            initial = tools.clone(call)
            result = original(*args, **kwargs)
            saved = tools.clone(OperatorCall((result, tools.state(call)), {}))
            size = tools.size(initial) + tools.size(saved)
            if size > 128 * 1024 * 1024:
                raise RunnerError("unsupported fixture: 128 MiB snapshot cap")
            captured = (OperatorFixture(spec.module_path, initial, saved.args[0], saved.args[1]), size)
            return result
        runtime.device.ready()
        runner.reset()
        with runtime.device.execution(), replace_forward(module, capture):
            runner.backend.begin(rows)
            current = rows
            for current_step in range(spec.decode_step + 1):
                logits = runner.backend.forward(current)
                tokens = runner.backend.greedy(logits)
                runner.backend.synchronize()
                if runner.backend.cache_length() != len(rows[0]) + current_step:
                    raise RunnerError("capture state did not reach the declared sequence position")
                current = tuple((token,) for token in tokens)
        if captured is None:
            raise RunnerError("selected module did not execute at the requested position")
        original_fixture, size = captured
        weights, attributes = module_metadata(module, tools)
        description = FixtureDescription(site_id=spec.site_id, module_path=spec.module_path, goal=spec.goal,
            phase="prefill" if spec.decode_step == 0 else "decode", decode_step=spec.decode_step,
            batch_size=len(rows), prompt_length=goal.input_tokens, tensors=tools.describe_tensors(original_fixture.call),
            weights=weights, module_attributes=attributes,
            source=source, source_sha256=hashlib.sha256(source.encode()).hexdigest(),
            contract_sha256=runner.prepared.contract_sha256, corpus_sha256=runner.prepared.corpus_sha256)
        identity = hashlib.sha256((description.model_dump_json() + tools.digest(original_fixture.call)).encode()).hexdigest()
        fixture = PureFixture(description, original_fixture, identity, size)
    except (OSError, ValueError, RuntimeError, TypeError, KeyError, AssertionError, StopIteration) as exc:
        detail = f"{type(exc).__name__}: {exc}"
    finally:
        runner.backend.reset()
        if runner.backend.weights_stamp() != stamp:
            runner.close()
            fixture, detail = None, "baseline weights changed; resident closed"
    path = runner._raw("target-fixture", {"stage": spec.stage.model_dump(mode="json"), "valid": fixture is not None,
        "detail": detail, "forward_calls": runner.backend.forward_calls - before,
        "wall_ms": (perf_counter() - started) * 1000,
        "fixture": fixture.description.model_dump(mode="json") if fixture else None,
        "fixture_identity": fixture.identity if fixture else None, "resident_bytes": fixture.resident_bytes if fixture else None})
    return CaptureReport(fixture is not None, fixture, path, detail)
