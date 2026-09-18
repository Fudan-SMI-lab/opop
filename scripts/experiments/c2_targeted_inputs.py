"""Explicit targeted envelope: response references do not count as new observations."""

from typing import Literal

from pydantic import Field

from kernel_optimizer.conditional.task_response import TaskResponse, complete_response
from kernel_optimizer.control.targeted_probes import ProbeDomain, ProbeEvidence, probe_plan
from kernel_optimizer.models.core import ParameterSpace, ParamSet, sha256_text
from kernel_optimizer.models.reports import BottleneckReport
from kernel_optimizer.tuning.objective import Objective
from scripts.experiments.c2_local_inputs import InputError, Responses, Shared


class TargetedResponses(Responses):
    strategy: Literal["targeted"] = "targeted"
    responses: list[TaskResponse] = Field(default_factory=list)
    source: str
    selected_params: ParamSet
    preliminary_report: BottleneckReport
    evidence: ProbeEvidence
    complete: bool = False

    def validate_for(self, shared: Shared) -> None:
        if self.shared_id != shared.identity() or self.source != shared.source or self.selected_params != shared.parent.params:
            raise InputError("targeted acquisition differs from its source or selected parent")
        if self.probe_calls != len(self.evidence.observations) or self.costs.worker_attempts != self.probe_calls:
            raise InputError("targeted unique attempt accounting mismatch")
        space = ParameterSpace(space_id=shared.parent.space_id, candidate_id=shared.parent.candidate_id,
            source_sha=sha256_text(shared.source), domains=shared.space.params, constraints=shared.space.constraints)
        domain = ProbeDomain(space, shared.parent.params, shared.device)
        planned = probe_plan(tuple(self.preliminary_report.probe_requests), domain)
        if len(planned) != len(self.evidence.contrasts):
            raise InputError("targeted request/fallback plan mismatch")
        endpoints: list[ParamSet] = []
        for expected, actual in zip(planned, self.evidence.contrasts, strict=True):
            if expected.model_dump(exclude={"response"}) != actual.model_dump(exclude={"response"}):
                raise InputError("targeted request admission or fixed partners mismatch")
            if expected.response is None:
                if actual.response is not None:
                    raise InputError("rejected request has a response")
                continue
            response = actual.response
            if response is None or (response.candidate_id, response.axis, response.a_params, response.b_params) != (
                    expected.response.candidate_id, expected.response.axis, expected.response.a_params, expected.response.b_params):
                raise InputError("targeted response endpoints differ from admitted request")
            for params, evaluation in ((response.a_params, response.a), (response.b_params, response.b)):
                if params is None:
                    continue
                endpoints.append(params)
                observation = next((o for o in self.evidence.observations if o.params == params), None)
                if evaluation != (observation.evaluation if observation else None):
                    raise InputError("response does not reference the unique fresh observation")
            numeric = next(d.kind != "str" for d in space.domains if d.name == response.axis)
            raw = TaskResponse(candidate_id=response.candidate_id, axis=response.axis,
                a_params=response.a_params, b_params=response.b_params, a=response.a, b=response.b,
                unknown_resources=list(shared.resource_metrics))
            recomputed = complete_response(raw, Objective(direction="minimize"), numeric)
            fields = {"delta_j", "gain", "parameter_slope", "resource_deltas", "resource_slopes", "resource_status"}
            if response.model_dump(include=fields) != recomputed.model_dump(include=fields):
                raise InputError("targeted response summary differs from fresh observations")
        seen: list[ParamSet] = []
        for observation in self.evidence.observations:
            if observation.params in seen or observation.params not in endpoints or domain.rejection(observation.params):
                raise InputError("targeted observation is duplicate, unrequested, or illegal")
            seen.append(observation.params)
        if self.responses != [c.response for c in self.evidence.contrasts if c.response is not None]:
            raise InputError("targeted response list differs from acquisition evidence")
