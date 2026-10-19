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
            return self.a(x)+self.b(x)
    model=Model(); x=torch.tensor([-1.,2.])
    adapter=SimpleNamespace(models=lambda state:{"model":model}, reset=lambda state:None, run=lambda state:model(x))
    discovery=capture(adapter,None,Path(context["output"]),0)
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
    binding.close()
    assert adapter.run(None).tolist()==[0.,4.]
    return {"sites":2,"group_size":2,"covered":dict(binding.counts),"restored":True}
''')
    receipt = LocalExecutor().run(ExecutionRequest(entry=entry, cwd=tmp_path, output=tmp_path/"run",
        lease_dir=tmp_path/"leases", worker_timeout_s=30))
    assert receipt["status"] == "passed", receipt
    assert receipt["ranks"][0]["result"]["restored"]
