from kernel_optimizer.project.evaluation import worker_entry
from kernel_optimizer.project.request import ProjectRequest


def test_invocation_measurement_releases_quality_and_prior_outputs(tmp_path):
    adapter = tmp_path/"adapter.py"
    adapter.write_text('''import torch,weakref
def prepare(context,request):return {'references':[],'mode':'quality'}
def reset(state):pass
def run(state):
    state['live_previous']=sum(ref() is not None for ref in state['references'])
    result=torch.ones(4)
    state['references'].append(weakref.ref(result))
    return result
def observe(state,result):return result
def metrics(state,result,elapsed_ms):return {'elapsed_ms':elapsed_ms,'live_previous':state['live_previous']}
def models(state):return {}
def set_mode(state,mode):state['mode']=mode
''')
    counts = {}
    for policy in ("legacy", "invocation"):
        output = tmp_path/policy
        output.mkdir()
        request = {"quality": {"atol": 0, "rtol": 0}, "objective": {"name": "elapsed_ms"},
                   "continuous_steps": 2, "warmup": 1, "repeats": 3, "measurement_lifetime": policy}
        result = worker_entry({"device": "cpu", "rank": 0, "world_size": 1, "output": str(output),
            "parameters": {"adapter": str(adapter), "request": request, "action": "controls"}})
        assert result["valid"]
        counts[policy] = [row["live_previous"] for row in result["samples"]]
    assert all(n > 0 for n in counts["legacy"])
    assert counts["invocation"] == [0, 0, 0]
    assert ProjectRequest.model_fields["measurement_lifetime"].default == "legacy"
