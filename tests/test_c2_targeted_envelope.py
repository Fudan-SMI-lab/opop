"""Targeted evidence validation must not weaken the provided-response contract."""

import pytest

from kernel_optimizer.models.reports import BottleneckReport, ProbeRequest
from scripts.experiments.c2_local_costs import Costs
from scripts.experiments.c2_local_inputs import InputError, Responses, Shared
from scripts.experiments.c2_targeted_inputs import TargetedResponses
from tests.test_c2_information_inputs import prepared
from tests.test_c2_targeted_probes import execute, probe_case


@pytest.fixture
def targeted(probe_case, prepared):
    original, _, cfg = prepared
    _, space, parent = probe_case
    shared = Shared.model_validate({**original.model_dump(), "space": space, "parent": parent,
        "source": "PARAMS={'early': 0, 'late': 1, 'mode': 'a'}\n", "trials": [parent]})
    request = ProbeRequest(axis="early", a_value=0, b_value=4, partners={"mode": "b"})
    evidence, calls, _ = execute(probe_case, [request, request], budget=2)
    envelope = TargetedResponses(shared_id=shared.identity(), source=shared.source, selected_params=parent.params,
        preliminary_report=BottleneckReport(summary="preliminary", probe_requests=[request, request]),
        evidence=evidence, responses=[c.response for c in evidence.contrasts if c.response is not None],
        probe_calls=len(calls), costs=Costs(worker_attempts=len(calls)), complete=True)
    return shared, envelope


def test_unique_attempts_validate_but_legacy_stays_strict(targeted):
    # Given: repeated references and a changed fixed partner are explicit in the targeted envelope.
    shared, envelope = targeted
    # When
    restored = TargetedResponses.model_validate_json(envelope.model_dump_json())
    restored.validate_for(shared)
    legacy = Responses.model_validate(envelope.model_dump(include={"shared_id", "responses", "probe_calls", "costs"}))
    # Then
    assert restored.probe_calls == 2 and len(restored.responses) > 2
    with pytest.raises(InputError):
        legacy.validate_for(shared)


def test_invalid_endpoint_cannot_carry_fabricated_zero_delta(targeted):
    # Given: the endpoint actually failed; both response views are tampered together.
    shared, envelope = targeted
    contrasts = list(envelope.evidence.contrasts)
    forged = contrasts[0].response.model_copy(update={"delta_j": 0.0, "gain": 0.0})
    contrasts[0] = contrasts[0].model_copy(update={"response": forged})
    changed = envelope.model_copy(update={"evidence": envelope.evidence.model_copy(update={"contrasts": tuple(contrasts)}),
        "responses": [c.response for c in contrasts if c.response is not None]})
    # When / Then
    with pytest.raises(InputError, match="summary"):
        changed.validate_for(shared)


@pytest.mark.parametrize("field", ["source", "selected", "count", "partners", "duplicate"])
def test_tampered_envelope_rejected(targeted, field):
    # Given
    shared, envelope = targeted
    changes = {
        "source": {"source": "unrelated"},
        "selected": {"selected_params": shared.parent.params.model_copy(update={"values": {"early": 0}})},
        "count": {"probe_calls": 1},
        "partners": {"evidence": envelope.evidence.model_copy(update={"contrasts": (
            envelope.evidence.contrasts[0].model_copy(update={"fixed": shared.parent.params}),
            *envelope.evidence.contrasts[1:])})},
        "duplicate": {"evidence": envelope.evidence.model_copy(update={"observations": (
            envelope.evidence.observations[0], envelope.evidence.observations[0])})},
    }
    changed = envelope.model_copy(update=changes[field])
    # When / Then
    with pytest.raises(InputError):
        changed.validate_for(shared)
