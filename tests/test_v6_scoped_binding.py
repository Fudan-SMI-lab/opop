from pathlib import Path

import pytest

from kernel_optimizer.runtime.execution import ExecutionRequest, LocalExecutor


@pytest.mark.parametrize("backend", ["legacy", "scoped"])
def test_scoped_binding_keeps_site_phase_shape_and_exception_semantics(tmp_path, backend):
    entry = tmp_path / "entry.py"
    entry.write_text('''
from pathlib import Path
from types import SimpleNamespace
import torch
from kernel_optimizer.project.binding import BindingSession
from kernel_optimizer.project.discovery import capture, phase

def run(context):
    class Leaf(torch.nn.Module):
        def forward(self, x):
            result = torch.relu(x)
            if state["fail"]:
                raise RuntimeError("intentional")
            return result
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.a, self.b = Leaf(), Leaf()
        def forward(self, x):
            return self.a(x) + self.b(x) + torch.relu(x)
    model = Model()
    state = {"x": torch.tensor([-1., 2.]), "fail": False}
    def workload(state):
        with phase("first"): a = model(state["x"])
        with phase("second"): b = model(state["x"])
        return a + b
    adapter = SimpleNamespace(models=lambda s: {"model": model}, run=workload, reset=lambda s: None)
    discovery = capture(adapter, state, Path(context["output"]), 0)
    modules = {c["id"]: c["target"] for c in discovery["calls"] if c["kind"] == "module"}
    sites = [c for c in discovery["calls"] if c["target"] == "aten.relu.default"
             and modules.get(c.get("site_parent")) == "model.a" and c["phase"] == "first"]
    assert len(sites) == 1
    file = Path(context["output"])/"candidate.py"
    file.write_text("def build(c):\\n return {k:lambda x,ref=c['references'][k]:ref(x)+7 for k in c['targets']}\\n")
    binding = BindingSession(adapter, state, sites, backend=context["parameters"]["backend"])
    binding.install(file, {})
    observed = []
    cls = type(binding._mode)
    original = cls.__torch_dispatch__
    def traced(self, func, types, args=(), kwargs=None):
        observed.append(str(func))
        return original(self, func, types, args, kwargs)
    cls.__torch_dispatch__ = traced
    assert binding.invoke().tolist() == [7., 19.]
    assert sum(binding.counts.values()) == 1
    relus = observed.count("aten.relu.default")
    assert relus == (1 if binding.scope_enabled else 6), (binding.execution_plan, observed)
    state["x"] = torch.tensor([-1., 2., 3.])
    assert binding.invoke().tolist() == [0., 12., 18.]
    assert sum(binding.counts.values()) == 1
    state["fail"] = True
    try:
        binding.invoke()
        raise AssertionError("exception missing")
    except RuntimeError as exc:
        assert str(exc) == "intentional"
    assert binding._scope_depth.get() == 0
    assert not binding._scope_frames.get()
    assert binding.active_modules.get() == ()
    state["fail"] = False
    state["x"] = torch.tensor([-1., 2.])
    assert binding.invoke().tolist() == [7., 19.]
    from concurrent.futures import ThreadPoolExecutor
    def threaded(_):
        result = binding.invoke().tolist()
        assert binding.active_modules.get() == ()
        assert binding._scope_depth.get() == 0
        assert not binding._scope_frames.get()
        return result
    with ThreadPoolExecutor(max_workers=2) as pool:
        assert list(pool.map(threaded, range(4))) == [[7., 19.]] * 4
    binding.close()
    assert adapter.run(state).tolist() == [0., 12.]
    return {"plan": binding.execution_plan, "relu_dispatches": relus, "restored": True}
''')
    receipt = LocalExecutor().run(ExecutionRequest(
        entry=entry, cwd=tmp_path, output=tmp_path / "run", lease_dir=tmp_path / "leases",
        worker_timeout_s=60, parameters={"backend": backend}))
    assert receipt["status"] == "passed", receipt


def test_unscoped_functionals_explicitly_retain_global_dispatch(tmp_path):
    entry = tmp_path / "entry.py"
    entry.write_text('''
from pathlib import Path
from types import SimpleNamespace
import torch
from kernel_optimizer.project.binding import BindingSession
from kernel_optimizer.project.discovery import capture

def run(context):
    adapter = SimpleNamespace(models=lambda s: {}, reset=lambda s: None, run=lambda s: torch.relu(s))
    state = torch.tensor([-1., 2.])
    observed = capture(adapter, state, Path(context["output"]), 0)
    sites = [c for c in observed["calls"] if c["target"] == "aten.relu.default"]
    file = Path(context["output"])/"candidate.py"
    file.write_text("def build(c):\\n return {k:lambda x,ref=c['references'][k]:ref(x)+1 for k in c['targets']}\\n")
    binding = BindingSession(adapter, state, sites, backend="scoped")
    binding.install(file, {})
    assert not binding.scope_enabled
    assert binding.execution_plan["effective"] == "legacy"
    assert binding.invoke().tolist() == [1., 3.]
    binding.close()
    return binding.execution_plan
''')
    receipt = LocalExecutor().run(ExecutionRequest(
        entry=entry, cwd=tmp_path, output=tmp_path / "run", lease_dir=tmp_path / "leases",
        worker_timeout_s=60))
    assert receipt["status"] == "passed", receipt
