"""Request-first probing with unique fresh attempts and checkpointable partial evidence."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field

from kernel_optimizer.conditional.task_response import TaskResponse, complete_response
from kernel_optimizer.control.direct_task import TaskSearch
from kernel_optimizer.evaluation.task_eval import TaskEvaluation
from kernel_optimizer.models.core import DeviceLimits, ParameterSpace, ParamSet, TrialRecord
from kernel_optimizer.models.reports import ProbeRequest
from kernel_optimizer.paramspace.guard import GuardRejection, check_config


class ProbeModel(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")


class ProbeObservation(ProbeModel):
    params: ParamSet
    evaluation: TaskEvaluation | None = None


class ProbeContrast(ProbeModel):
    request: ProbeRequest
    origin: Literal["request", "fallback"]
    fixed: ParamSet
    rejection: str | None = None
    response: TaskResponse | None = None


class ProbeEvidence(ProbeModel):
    contrasts: tuple[ProbeContrast, ...] = ()
    observations: tuple[ProbeObservation, ...] = Field(default=(), max_length=12)


@dataclass(frozen=True, slots=True)
class ProbeDomain:
    space: ParameterSpace
    selected: ParamSet
    device: DeviceLimits

    def rejection(self, params: ParamSet) -> GuardRejection | None:
        try:
            return check_config(self.space, params, self.device)
        except (ArithmeticError, TypeError) as exc:
            return GuardRejection(reason="constraint_invalid", detail=str(exc))


@dataclass(frozen=True, slots=True)
class ProbeRun[T]:
    requests: tuple[ProbeRequest, ...]
    context: Mapping[str, T]
    allowed: Callable[[], bool]
    checkpoint: Callable[[ProbeEvidence], None]


def admit_request(request: ProbeRequest, domain: ProbeDomain) -> ProbeContrast:
    values = {**domain.selected.values, **request.partners}
    fixed = ParamSet(values={k: v for k, v in values.items() if k != request.axis})
    contrast = ProbeContrast(request=request, origin="request", fixed=fixed)
    if request.axis not in domain.space.param_names():
        return contrast.model_copy(update={"rejection": "unknown_axis"})
    if request.a_value == request.b_value:
        return contrast.model_copy(update={"rejection": "identical_endpoints"})
    endpoints = [ParamSet(values={**values, request.axis: value}) for value in (request.a_value, request.b_value)]
    for params in endpoints:
        rejection = domain.rejection(params)
        if rejection:
            return contrast.model_copy(update={"rejection": rejection.reason})
    response = TaskResponse(candidate_id=domain.space.candidate_id, axis=request.axis,
                            a_params=endpoints[0], b_params=endpoints[1])
    return contrast.model_copy(update={"response": response})


def probe_plan(requests: tuple[ProbeRequest, ...], domain: ProbeDomain) -> tuple[ProbeContrast, ...]:
    contrasts = [admit_request(request, domain) for request in requests]
    for axis in domain.space.domains:
        legal = [value for value in axis.choices if domain.rejection(
            ParamSet(values={**domain.selected.values, axis.name: value})) is None]
        if len(legal) < 2:
            request = ProbeRequest(axis=axis.name, a_value=axis.choices[0], b_value=axis.choices[-1])
            contrast = admit_request(request, domain).model_copy(update={
                "rejection": "fewer_than_two_legal_endpoints", "response": None})
        else:
            contrast = admit_request(ProbeRequest(axis=axis.name, a_value=legal[0], b_value=legal[-1]), domain)
        contrasts.append(contrast.model_copy(update={"origin": "fallback"}))
    return tuple(contrasts)


def targeted_probes[T](search: TaskSearch, parent: TrialRecord, run: ProbeRun[T]) -> ProbeEvidence:
    space = search.spaces[parent.candidate_id]
    domain = ProbeDomain(ParameterSpace(space_id=parent.space_id, candidate_id=parent.candidate_id,
        source_sha="0" * 64, domains=space.params, constraints=space.constraints), parent.params, search.device)
    contrasts = list(probe_plan(run.requests, domain))
    observations: list[ProbeObservation] = []
    cutoff = False

    def checkpoint() -> ProbeEvidence:
        evidence = ProbeEvidence(contrasts=tuple(contrasts), observations=tuple(observations))
        run.checkpoint(evidence)
        return evidence

    checkpoint()
    for index, contrast in enumerate(contrasts):
        response = contrast.response
        if response is None:
            continue
        missing_reason = None
        for side, params in (("a", response.a_params), ("b", response.b_params)):
            if params is None:
                continue
            observation = next((o for o in observations if o.params == params), None)
            if observation is None:
                if search.probe_calls >= min(12, search.probe_budget):
                    missing_reason = "probe_budget"
                    continue
                cutoff = cutoff or not search.within_wall_budget() or not run.allowed()
                if cutoff:
                    missing_reason = "cutoff"
                    continue
                search.probe_calls += 1
                observations.append(ProbeObservation(params=params))
                checkpoint()
                try:
                    evaluation = search.evaluator.evaluate(search.paths[parent.candidate_id], params.values, run.context)
                    observation = ProbeObservation(params=params, evaluation=evaluation)
                    observations[-1] = observation
                finally:
                    checkpoint()
            response = response.model_copy(update={side: observation.evaluation})
            contrasts[index] = contrast.model_copy(update={"response": response})
        response = response.model_copy(update={"unknown_resources": list(search.resource_metrics)})
        numeric = next(d.kind != "str" for d in space.params if d.name == response.axis)
        response = complete_response(response, search.objective, numeric)
        if missing_reason:
            response = response.model_copy(update={"reason": missing_reason})
        contrasts[index] = contrast.model_copy(update={"response": response})
        checkpoint()
    search.responses.extend(c.response for c in contrasts if c.response is not None)
    return checkpoint()
