from pathlib import Path

from kernel_optimizer.runtime.execution import ExecutionRequest, LocalExecutor


def test_identical_functional_shapes_remain_distinct_sites_in_one_group(tmp_path):
    entry = tmp_path/"entry.py"
    entry.write_text('''
from pathlib import Path
from types import SimpleNamespace
import torch
from kernel_optimizer.project.discovery import capture
from kernel_optimizer.project.binding import BindingSession

def run(context):
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.a=torch.nn.ReLU(); self.b=torch.nn.ReLU()
        def forward(self, x):
            assert torch.isfinite(x).all()
            return self.a(x)+self.b(x)
    model=Model(); x=torch.tensor([-1.,2.])
    adapter=SimpleNamespace(models=lambda state:{"model":model}, reset=lambda state:None, run=lambda state:model(x))
    discovery=capture(adapter,None,Path(context["output"]),0)
    from kernel_optimizer.project.aggregation import merge_ranks
    from kernel_optimizer.models.project import Dependency
    assert all(Dependency(**edge) for edge in merge_ranks([discovery])["dependencies"])
    sites=[c for c in discovery["calls"] if c["target"]=="aten.relu.default"]
    assert len(sites)==2 and sites[0]["id"]!=sites[1]["id"]
    groups=[g for g in discovery["groups"] if sites[0]["id"] in g["calls"]]
    assert set(groups[0]["calls"])=={s["id"] for s in sites}
    candidate=Path(context["output"])/"candidate.py"
    candidate.write_text("def build(context):\\n key=next(iter(context['targets']))\\n reference=context['references'][key]\\n return {key:lambda x:reference(x)+7}\\n")
    binding=BindingSession(adapter,None,[sites[0]])
    binding.install(candidate,{})
    assert binding.invoke().tolist()==[7.,11.]
    assert dict(binding.counts)=={sites[0]["id"]:1}
    assert binding.path_telemetry()["reference_invocations"]=={sites[0]["id"]:1}
    binding.record_route(sites[0]["id"],"reference_plus_add")
    binding.tracking=False
    assert binding.invoke().tolist()==[7.,11.]
    binding.record_route(sites[0]["id"],"setup_or_timing_must_not_count")
    assert binding.path_telemetry()["reference_invocations"]=={sites[0]["id"]:1}
    assert binding.path_telemetry()["candidate_declared_routes"]==[{"call":sites[0]["id"],"route":"reference_plus_add","count":1}]
    binding.close()
    assert adapter.run(None).tolist()==[0.,4.]
    return {"sites":2,"group_size":2,"covered":dict(binding.counts),"restored":True}
''')
    receipt = LocalExecutor().run(ExecutionRequest(entry=entry, cwd=tmp_path, output=tmp_path/"run",
        lease_dir=tmp_path/"leases", worker_timeout_s=30))
    assert receipt["status"] == "passed", receipt
    assert receipt["ranks"][0]["result"]["restored"]


def test_one_module_has_independent_phase_replacements(tmp_path):
    entry = tmp_path/"phases.py"
    entry.write_text('''
from pathlib import Path
from types import SimpleNamespace
import torch
from kernel_optimizer.project.discovery import capture, phase
from kernel_optimizer.project.binding import BindingSession

def run(context):
    model=torch.nn.ReLU(); x=torch.tensor([-1.,2.])
    def workload(state):
        with phase("prefill"): a=model(x)
        with phase("decode"): b=model(x)
        return a+b
    adapter=SimpleNamespace(models=lambda state:{"model":model}, reset=lambda state:None, run=workload)
    observed=capture(adapter,None,Path(context["output"]),0)
    sites=[c for c in observed["calls"] if c["kind"]=="module"]
    assert {s["phase"] for s in sites}=={"prefill","decode"}
    file=Path(context["output"])/"candidate.py"
    file.write_text("def build(c):\\n return {key: (lambda x, ref=c['references'][key], n=3 if spec['phase']=='prefill' else 7: ref(x)+n) for key,spec in c['targets'].items()}\\n")
    binding=BindingSession(adapter,None,sites); binding.install(file,{})
    assert binding.invoke().tolist()==[10.,14.]
    assert set(binding.counts)=={s["id"] for s in sites}
    assert all(n==1 for n in binding.counts.values())
    binding.close(); assert workload(None).tolist()==[0.,4.]
    return {"phases":2,"restored":True}
''')
    receipt = LocalExecutor().run(ExecutionRequest(entry=entry, cwd=tmp_path, output=tmp_path/"run",
        lease_dir=tmp_path/"leases", worker_timeout_s=30))
    assert receipt["status"] == "passed", receipt


def test_passthrough_retains_parent_replacement_and_skips_candidate_build(tmp_path):
    entry = tmp_path/"passthrough.py"
    entry.write_text('''
from pathlib import Path
from types import SimpleNamespace
import torch
from kernel_optimizer.project.discovery import capture
from kernel_optimizer.project.binding import BindingSession

def run(context):
    model=torch.nn.ReLU(); x=torch.tensor([-1.,2.])
    adapter=SimpleNamespace(models=lambda state:{"model":model}, reset=lambda state:None, run=lambda state:model(x))
    observed=capture(adapter,None,Path(context["output"]),0)
    sites=[c for c in observed["calls"] if c["kind"]=="module"]
    parent_file=Path(context["output"])/"parent.py"
    parent_file.write_text("def build(c):\\n return {k:lambda x,ref=c['references'][k]:ref(x)+3 for k in c['targets']}\\n")
    parent=BindingSession(adapter,None,sites); parent.install(parent_file,{})
    unused=Path(context["output"])/"candidate.py"; unused.write_text("raise RuntimeError('must not load')\\n")
    child=BindingSession(adapter,None,sites); child.install(unused,{},diagnostic_passthrough=True)
    assert adapter.run(None).tolist()==[3.,5.]
    assert sum(parent.counts.values())==sum(child.counts.values())==1
    assert sum(child.reference_calls.values())==1
    child.close(); parent.close()
    assert adapter.run(None).tolist()==[0.,2.]
    return {"retained_parent":True}
''')
    receipt = LocalExecutor().run(ExecutionRequest(entry=entry, cwd=tmp_path, output=tmp_path/"run",
        lease_dir=tmp_path/"leases", worker_timeout_s=30))
    assert receipt["status"] == "passed", receipt
