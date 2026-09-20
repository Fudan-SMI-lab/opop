import inspect
from collections.abc import Callable
from contextlib import ExitStack
from pathlib import Path
from time import perf_counter
from types import MappingProxyType
from typing import Protocol
from uuid import uuid4

from pydantic import JsonValue

from .cuda_trace import TorchTrace
from .device_records import LocalRuntime, StageIdentity
from .device_scopes import replace_forward
from .device_capture import module_metadata
from .fast_prepare import require_feedback_inputs
from .operator_types import CacheValue, OperatorCall, Value
from .runner_records import BindingReceipt, GoalId, RunnerError


class ProfileRequest:
    def __init__(self, stage: StageIdentity, goal: GoalId) -> None:
        self.stage, self.goal = stage, goal


def profile_goal(request: ProfileRequest, runtime: LocalRuntime, *, trace_factory=TorchTrace) -> Path:
    runner = runtime.runner
    started, before = perf_counter(), runner.backend.forward_calls
    rows_record: list[dict[str, JsonValue]] = []
    step, collecting = 0, False
    data: dict[str, JsonValue] = {}
    stamp = runner.backend.weights_stamp()
    startup_calls, observed_calls, windows = 0, 0, 0
    try:
        require_feedback_inputs(runner)
        if request.stage.stage == "final":
            raise RunnerError("sealed final phase cannot supply optimization profiles")
        ordinal = runtime.admission.admit(request.stage, "profile")
        if ordinal is None or not 0 <= ordinal < 3:
            raise RunnerError("profile admission denied; at most three short windows")
        goal = next(g for g in runner.prepared.contract.goals if g.id == request.goal)
        ids = goal.search_prompt_ids if request.goal == "multi" else goal.search_prompt_ids[:1]
        rows = tuple(runner.prepared.prompts_by_id[p].input_ids for p in ids)
        runner.binding.restore()
        runner.receipt = BindingReceipt("baseline", MappingProxyType({}), "baseline", (), True)
        runner.quality_identity = None
        runtime.device.ready()
        runner.reset()
        with runtime.device.execution():
            runner.backend.begin(rows)
            runner.backend.greedy(runner.backend.forward(rows))
            runner.backend.synchronize()
            startup_calls = 1
            runner.reset()
            runner.backend.begin(rows)
            current = rows
            trace = trace_factory()
            with trace, ExitStack() as stack:
                windows = 1
                def wrap(path: str):
                    original = runner.binding.originals[path]
                    def forward(*args: Value, **kwargs: Value) -> Value:
                        if not collecting:
                            return original(*args, **kwargs)
                        label = f"c3-site:{step}:{path}"
                        pure_kwargs = {k: v for k, v in kwargs.items() if not isinstance(v, CacheValue)}
                        call = OperatorCall(tuple(v for v in args if not isinstance(v, CacheValue)), pure_kwargs)
                        descriptions = runtime.device.tools.describe_tensors(call)
                        weights, attributes = module_metadata(runner.binding.modules[path], runtime.device.tools)
                        rows_record.append({"scope": label, "module_path": path,
                            "phase": "prefill" if step == 0 else "decode", "decode_step": step,
                            "tensors": [d.model_dump(mode="json") for d in descriptions],
                            "weights": [d.model_dump(mode="json") for d in weights], "module_attributes": attributes,
                            "source_file": inspect.getsourcefile(original), "source": inspect.getsource(original),
                            "module_source": inspect.getsource(type(runner.binding.modules[path]))})
                        with trace.region(label):
                            return original(*args, **kwargs)
                    return forward
                for path, module in runner.binding.modules.items():
                    if path:
                        stack.enter_context(replace_forward(module, wrap(path)))
                for step in range(goal.output_tokens):
                    collecting = step in (0, goal.output_tokens - 1)
                    trace.active(collecting)
                    tokens = runner.backend.greedy(runner.backend.forward(current))
                    runner.backend.synchronize()
                    if runner.backend.cache_length() != len(rows[0]) + step:
                        raise RunnerError("profile state did not reach the declared position")
                    observed_calls += int(collecting)
                    current = tuple((token,) for token in tokens)
            trace_path = runner.output_dir / f"cuda-profile-{uuid4().hex}.json"
            trace_path.parent.mkdir(parents=True, exist_ok=True)
            trace.export(trace_path)
            data = trace.records()
            data["chrome_trace"] = str(trace_path.resolve())
            data["valid"] = True
    except (OSError, ValueError, RuntimeError, TypeError, KeyError, StopIteration) as exc:
        data = {"valid": False, "detail": f"{type(exc).__name__}: {exc}"}
    finally:
        runner.backend.reset()
        if runner.backend.weights_stamp() != stamp:
            runner.close()
            data = {"valid": False, "detail": "weights changed; resident closed"}
    data.update(stage=request.stage.model_dump(mode="json"), goal=request.goal, calls=rows_record,
        forward_calls=runner.backend.forward_calls - before, startup_forwards=startup_calls,
        state_advance_forwards=max(0, runner.backend.forward_calls - before - startup_calls - observed_calls),
        profiled_forwards=observed_calls, trace_windows=windows, wall_ms=(perf_counter() - started) * 1000,
        critical_path_fraction=None, overlaps="CUDA intervals may overlap; per-site durations are not additive critical-path shares")
    return runner._raw("short-device-profile", data)
