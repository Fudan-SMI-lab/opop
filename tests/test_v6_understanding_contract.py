import pytest
from pydantic import ValidationError

from kernel_optimizer.project.request import ProjectUnderstanding


def test_structured_state_protocol_is_rejected_in_agent_output_validation():
    value = {"goal": "minimize latency", "entrypoint": "main.py",
             "completion": "all work completed", "state_protocol": {"prepare": "load", "reset": "restore"},
             "invariants": ["preserve inputs"], "objective": {}, "workload": {}, "quality": {},
             "missing_information": [], "assumptions": []}
    with pytest.raises(ValidationError) as failure:
        ProjectUnderstanding.model_validate(value)
    assert failure.value.errors()[0]["loc"] == ("state_protocol",)
    value["state_protocol"] = "Load once; restore mutable state before each invocation."
    assert ProjectUnderstanding.model_validate(value).state_protocol == value["state_protocol"]
