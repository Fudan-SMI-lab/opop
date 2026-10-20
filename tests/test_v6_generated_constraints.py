import json
import pytest
from kernel_optimizer.project.generated import validate_proposal


@pytest.mark.parametrize("expression,error", [
    ("route == 'parent' implies n == 0", SyntaxError),
    ("route in ['parent']", ValueError),
    ("unknown == 1", ValueError),
    ("route != 'parent' or n == 0", None),
])
def test_real_constraint_failure_is_returned_before_tuning(tmp_path, expression, error):
    (tmp_path/"candidate.py").write_text("def build(context): return {}\n")
    (tmp_path/"proposal.json").write_text(json.dumps({"implementation":"distributed","replacement_calls":["call"],
        "rationale":"test","setup_work":"none","quality_risks":[],"parameter_space":{
            "params":[{"name":"route","kind":"str","choices":["parent","custom"]},
                      {"name":"n","kind":"int","choices":[0,1]}],
            "constraints":[{"expr":expression}]}}))
    if error:
        with pytest.raises(error): validate_proposal(tmp_path)
    else:
        validate_proposal(tmp_path)
