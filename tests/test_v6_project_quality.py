import torch
import pytest

from kernel_optimizer.project.evaluation import compare, corrupt, snapshot


def test_integer_tokens_stay_exact_with_float_tolerances():
    assert not compare(torch.tensor([100001]), torch.tensor([100000]), {"atol": 0.02, "rtol": 0.02})["passed"]


def test_full_nested_outputs_and_nan_are_checked():
    reference = {"logits": torch.arange(12.).reshape(3, 4), "tokens": (1, 2)}
    actual = snapshot(reference)
    assert compare(actual, reference, {"atol": 1e-4, "rtol": 1e-4})["passed"]
    actual["logits"][2, 3] = float("nan")
    assert not compare(actual, reference, {"atol": 1e-4, "rtol": 1e-4})["passed"]
    assert not compare({"logits": reference["logits"]}, reference, {})["passed"]


def test_negative_control_preserves_reference_and_exceeds_relative_tolerance():
    reference = {"state": [torch.tensor([1e8, 0.0])], "output": torch.tensor([5.])}
    wrong = corrupt(reference)
    assert reference["state"][0][0] == 1e8
    assert not compare(wrong, reference, {"atol": 0.02, "rtol": 0.02})["passed"]


@pytest.mark.parametrize("specs_mode", ["inline", "file"])
def test_installation_calls_do_not_satisfy_inherited_runtime_coverage(tmp_path, specs_mode):
    from kernel_optimizer.runtime.execution import ExecutionRequest, LocalExecutor
    entry = tmp_path/"entry.py"
    entry.write_text('''
from pathlib import Path
from kernel_optimizer.project.evaluation import worker_entry
from kernel_optimizer.project.discovery import tensor_description

def run(context):
    import torch
    import json
    specs_mode = context["parameters"]["specs_mode"]
    root=Path(context["output"])
    adapter=root/"adapter.py"
    adapter.write_text("import torch\\ndef prepare(c,r): return {'x':torch.tensor([1.,2.]),'m':torch.nn.Identity()}\\ndef reset(s): pass\\ndef run(s): return s['m'](s['x'])+1\\ndef observe(s,r): return r\\ndef metrics(s,r,t): return {'latency_ms':t}\\ndef models(s): return {'model':s['m']}\\n")
    request={"quality":{},"objective":{"name":"latency_ms"},"warmup":0,"repeats":3,"continuous_steps":2}
    context["parameters"]={"adapter":str(adapter),"request":request,"action":"controls"}
    assert worker_entry(context)["valid"]
    signature={"args":tensor_description((torch.tensor([1.,2.]),)),"kwargs":{}}
    first=root/"first.py";first.write_text("def build(c):\\n ref=c['references']['a']\\n return {'a':lambda x:ref(x)+0}\\n")
    second=root/"second.py";second.write_text("def build(c):\\n c['references']['b'](c['state'])\\n return {'b':lambda s:s['x']+1}\\n")
    context["parameters"].update(action="evaluate",bundles=[
        {"id":"first","file":str(first),"expected_calls":["a"],"specs":[{"id":"a","kind":"module","target":"model","phase":"run","signatures":[signature]}]},
        {"id":"second","file":str(second),"expected_calls":["b"],"specs":[{"id":"b","kind":"project","target":"project.invoke","phase":"run","signatures":[]}]}])
    if specs_mode == "file":
        for bundle in context["parameters"]["bundles"]:
            path = root/(bundle["id"]+".targets.json")
            path.write_text(json.dumps(bundle.pop("specs")))
            bundle["specs_path"] = str(path)
    result=worker_entry(context)
    assert all(check["passed"] for check in result["checks"])
    assert not result["valid"]
    assert result["coverage"]["first"]["counts"]=={}
    assert result["coverage"]["second"]["counts"]=={"b":2}
    return {"installation_cannot_supply_coverage":True}
''')
    receipt = LocalExecutor().run(ExecutionRequest(entry=entry, cwd=tmp_path, output=tmp_path/"run",
        lease_dir=tmp_path/"leases", worker_timeout_s=30, parameters={"specs_mode":specs_mode}))
    assert receipt["status"] == "passed", receipt
def test_failure_identifies_observation_without_claiming_first_execution_error():
    import torch
    from kernel_optimizer.project.evaluation import compare
    result = compare({"gradients": {"weight": torch.tensor([1., 2.])}},
                     {"gradients": {"weight": torch.tensor([1., 3.])}}, {"atol": 1e-6, "rtol": 1e-4})
    assert result["observation_path"] == "output.gradients.weight"
    assert result["atol"] == 1e-6
    assert "not earliest execution" in result["failure_scope"]
