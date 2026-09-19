"""Real invoke/check_output lifetime with provider-side edits; no candidate execution."""

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from pathlib import Path
from threading import Barrier

import pytest

from kernel_optimizer.agents.runtime import AgentCallError, OpencodeClient, PromptResult
from kernel_optimizer.agents.sandbox import Sandbox, SandboxFactory
from kernel_optimizer.agents.task_rewriter import TaskRewriteInputs, TaskRewriteResult, TaskRewriterAgent
from kernel_optimizer.config import AgentModuleConfig
from kernel_optimizer.control.direct_task import TaskSpace
from kernel_optimizer.models.core import ParamSet
from kernel_optimizer.store.run_store import RunStore
from kernel_optimizer.tuning.objective import Objective

PARENT = "PARAMS = {'x': 1}\ndef run(x): return x * 2\n"
CHILD = "PARAMS = {'x': 1}\ndef run(x): return x + x\n"
OTHER = "def run(x): return x - 99\n"
RAW = Path(__file__).parents[1] / "results/c3-qwen3-4b-operator-goals"


class EditingProvider(OpencodeClient):
    def __init__(self, edit):
        super().__init__("http://127.0.0.1:0")
        self.edit = edit
        self.calls = []

    def create_session(self, directory, title):
        return title

    def prompt(self, session_id, text, **kwargs):
        folder = kwargs["directory"]
        self.calls.append(folder)
        return PromptResult(text="", session_id=session_id, structured=self.edit(folder))


def request(root: Path, source=PARENT, *, bundle=False):
    root.mkdir()
    parent = root / "parent.py"
    parent.write_bytes(source.encode())
    return TaskRewriteInputs(project_root=root, candidate_id=root.name, candidate_path=parent,
        goal=root.name, context={}, objective=Objective(direction="minimize"), params=ParamSet(values={}),
        space=TaskSpace(), bundle_sources={"operators.py": source} if bundle else {},
        bundle_document={"entry": "operators.py", "files": ["operators.py"], "helpers": []} if bundle else None)


def agent_at(root: Path, provider):
    store = RunStore.create(root, "agent", {})
    return TaskRewriterAgent(provider, SandboxFactory(store.run_dir / "sandboxes"), store,
                            AgentModuleConfig(max_retries=0, max_transport_retries=0))


def emit(folder: Path, source: str, *, bundle=False, mirror=None):
    name = "operators.py" if bundle else "child.py"
    (folder / name).write_bytes(source.encode())
    if mirror is not None:
        (folder / "candidate/current.py").write_bytes(mirror.encode())
        (folder / "candidate/bundle-sources.json").write_text(json.dumps({"operators.py": mirror}))
        (folder / "analysis/task_response.json").write_text(json.dumps({"bundle_sources": {"operators.py": mirror}}))
    output = {"candidate_file": name}
    if bundle:
        (folder / "bundle.json").write_text(json.dumps({"entry": name, "files": [name], "helpers": []}))
        output["bundle_file"] = "bundle.json"
    return output


@pytest.mark.parametrize("bundle", [False, True])
@pytest.mark.parametrize("change", ["computation", "unchanged", "comments", "params"])
def test_provider_mirror_edits_cannot_change_original_parent_comparison(tmp_path, bundle, change):
    # Given: actual inputs exist before the provider gains control of the writable workspace.
    inputs = request(tmp_path / "parent", bundle=bundle)
    source = {"computation": CHILD, "unchanged": PARENT, "comments": PARENT + "# extra comment\n",
              "params": PARENT.replace("'x': 1", "'x': 8")}[change]
    mirror = source if change == "computation" else OTHER
    def edit(folder):
        inputs.candidate_path.write_bytes(mirror.encode())
        if bundle:
            inputs.bundle_sources["operators.py"] = mirror
        return emit(folder, source, bundle=bundle, mirror=mirror)
    with closing(EditingProvider(edit)) as provider:
        agent = agent_at(tmp_path, provider)
        # When / Then: a real change passes; true nonchanges still fail despite the fake mirror.
        if change == "computation":
            outcome = agent.invoke(inputs)
            assert outcome.sandbox.read_output(outcome.output.candidate_file) == source
        else:
            with pytest.raises(AgentCallError):
                agent.invoke(inputs)
        assert len(provider.calls) == 1
        assert (provider.calls[0] / "candidate/current.py").read_text() == mirror


def test_helper_only_change_passes_after_provider_replaces_bundle_mirror(tmp_path):
    # Given: unchanged entry, changed executed helper source, and no authoritative sandbox file.
    inputs = request(tmp_path / "parent", "from .helper import f\ndef run(x): return f(x)\n", bundle=True)
    inputs = inputs.model_copy(update={"bundle_sources": {"operators.py": inputs.candidate_path.read_text(),
        "helper.py": "def f(x): return x * 2\n"}, "bundle_document": {
            "entry": "operators.py", "files": ["operators.py"], "helpers": ["helper.py"]}})
    def edit(folder):
        sources = {**inputs.bundle_sources, "helper.py": "def f(x): return x + x\n"}
        for name, text in sources.items():
            (folder / name).write_bytes(text.encode())
        (folder / "bundle.json").write_text(json.dumps(inputs.bundle_document))
        (folder / "candidate/bundle-sources.json").write_text(json.dumps(sources))
        return {"candidate_file": "operators.py", "bundle_file": "bundle.json"}
    # When
    with closing(EditingProvider(edit)) as provider:
        outcome = agent_at(tmp_path, provider).invoke(inputs)
    # Then
    assert outcome.sandbox.read_output("helper.py") == "def f(x): return x + x\n"


@pytest.mark.parametrize("inner_fails", [False, True])
def test_nested_calls_restore_outer_snapshot_and_cleanup(tmp_path, inner_fails):
    # Given: inner parent equals outer child, exposing an overwritten instance snapshot.
    outer = request(tmp_path / "outer")
    inner = request(tmp_path / "inner", CHILD)
    def edit(folder):
        name = json.loads((folder / "analysis/task_response.json").read_text())["candidate_id"]
        if name == "outer":
            if inner_fails:
                with pytest.raises(AgentCallError):
                    agent.invoke(inner)
            else:
                agent.invoke(inner)
            return emit(folder, CHILD, mirror=CHILD)
        if inner_fails:
            raise AgentCallError("fixture provider failure")
        return emit(folder, PARENT, mirror=PARENT)
    with closing(EditingProvider(edit)) as provider:
        agent = agent_at(tmp_path, provider)
        # When
        outcome = agent.invoke(outer)
        # Then: standalone validation must not inherit the completed outer call's snapshot.
        assert outcome.output.candidate_file == "child.py" and len(provider.calls) == 2
        assert agent.check_output(outcome.output, outcome.sandbox) is not None


def test_concurrent_calls_on_one_agent_have_independent_snapshots(tmp_path):
    # Given: opposite parent/child pairs, both paused after seeding.
    first, second = request(tmp_path / "first"), request(tmp_path / "second", CHILD)
    barrier = Barrier(2)
    def edit(folder):
        name = json.loads((folder / "analysis/task_response.json").read_text())["candidate_id"]
        barrier.wait(timeout=10)
        source = CHILD if name == "first" else PARENT
        return emit(folder, source, mirror=source)
    with closing(EditingProvider(edit)) as provider:
        agent = agent_at(tmp_path, provider)
        # When
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(agent.invoke, inputs) for inputs in (first, second)]
            outcomes = [future.result(timeout=20) for future in futures]
        # Then
        assert [o.sandbox.read_output("child.py") for o in outcomes] == [CHILD, PARENT]


def test_direct_check_without_call_context_refuses_writable_parent(tmp_path):
    # Given: a standalone legacy check has no original host inputs.
    sandbox = Sandbox(tmp_path / "sandbox")
    sandbox.write_input("candidate/current.py", PARENT)
    sandbox.write_input("child.py", CHILD)
    with closing(EditingProvider(lambda folder: {})) as provider:
        agent = agent_at(tmp_path, provider)
        # When / Then
        assert agent.check_output(TaskRewriteResult(candidate_file="child.py"), sandbox) is not None


def test_failed_call_does_not_leave_authority_for_later_direct_checks(tmp_path):
    # Given: the provider wrote a real change but the call failed before returning it.
    inputs = request(tmp_path / "parent")
    def edit(folder):
        emit(folder, CHILD, mirror=PARENT)
        raise AgentCallError("fixture provider failure")
    with closing(EditingProvider(edit)) as provider:
        agent = agent_at(tmp_path, provider)
        # When
        with pytest.raises(AgentCallError):
            agent.invoke(inputs)
        # Then: a leaked parent snapshot would incorrectly validate this valid-looking artifact.
        sandbox = Sandbox(provider.calls[0])
        assert agent.check_output(TaskRewriteResult(candidate_file="child.py"), sandbox) is not None


@pytest.mark.parametrize("host,goal,call_id,prefix,number", [
    ("A", "single", "task_rewriter-b57855c5", "7863ec07", 1),
    ("A", "single", "task_rewriter-c4b71015", "ed5eb1b0", 2),
    ("B", "multi", "task_rewriter-a6b8a2bc", "c2a66001", 1),
])
def test_actual_returned_bundle_replay_uses_original_snapshot_without_changing_outcome(tmp_path, host, goal, call_id, prefix, number):
    # Given: exact exported final bytes and original request sources, read without candidate execution.
    root = RAW / f"main-af77534-host-{host}" / "main-af77534"
    exported = root / f"{goal}-agents/sandboxes/{call_id}"
    if not exported.is_dir():
        pytest.skip("local T7 export unavailable")
    request_path = exported / "analysis/task_response.json"
    bundle_path = exported / "candidate/bundle/bundle.json"
    document = json.loads(bundle_path.read_bytes())
    files = [bundle_path, request_path, exported / "candidate/bundle-sources.json", exported / "candidate/current.py",
             root / goal / "result.json", *(bundle_path.parent / n for n in (*document["files"], *document["helpers"]))]
    before = {path: path.read_bytes() for path in files}
    payload = json.loads(request_path.read_bytes())
    parent = tmp_path / "parent.py"
    parent.write_bytes(payload["bundle_sources"]["operators.py"].encode())
    assert hashlib.sha256(parent.read_bytes()).hexdigest() == "882e62a1165dc59b606568ef7c64f2994f0077b4b129a979e64cc62bfb490d7a"
    inputs = TaskRewriteInputs.model_validate({**payload, "candidate_path": parent, "project_root": tmp_path, "source_paths": []})
    def edit(folder):
        for path in files:
            if path.is_relative_to(exported) and path != request_path:
                target = folder / path.relative_to(exported)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(before[path])
        return {"candidate_file": "candidate/bundle/" + document["entry"], "bundle_file": "candidate/bundle/bundle.json"}
    # When: the real prospective invoke/validation loop sees the same mirror edits and final source.
    with closing(EditingProvider(edit)) as provider:
        outcome = agent_at(tmp_path, provider).invoke(inputs)
    # Then: only postprocessor correctness changes; old failed results and every original byte remain fixed.
    returned = (outcome.sandbox.root / outcome.output.candidate_file).read_bytes()
    assert returned == before[bundle_path.parent / document["entry"]]
    assert hashlib.sha256(returned).hexdigest().startswith(prefix)
    assert all(path.read_bytes() == data for path, data in before.items())
    old = json.loads((root / goal / "result.json").read_bytes())
    assert next(o for o in old["opportunities"] if o["number"] == number)["status"] == "failed"
