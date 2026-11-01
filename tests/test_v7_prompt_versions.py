import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from kernel_optimizer.config import AppConfig, load_config
from kernel_optimizer.project import prompt_versions


@pytest.mark.parametrize("role,filename", [
    ("project-intake", "intake_prompt.md"), ("project-evaluator", "adapter_prompt.md"),
    ("selection", "selection_prompt.md"), ("candidate", "candidate_prompt.md"),
])
def test_default_and_explicit_v6_preserve_original_text(role, filename):
    expected = Path(prompt_versions.__file__).with_name(filename).read_text(encoding="utf-8")
    config = AppConfig()
    assert prompt_versions.load_prompt(config, role) == expected
    config.project_defaults.prompt_versions[role] = "v6"
    assert prompt_versions.load_prompt(config, role) == expected


@pytest.mark.parametrize("version", ["", "../v6", "v7/missing", "unavailable-version"])
def test_unknown_version_never_silently_uses_another_prompt(version):
    config = AppConfig()
    config.project_defaults.prompt_versions["candidate"] = version
    with pytest.raises(ValueError, match="[Pp]rompt version"):
        prompt_versions.load_prompt(config, "candidate")


def test_unknown_role_is_not_silently_ignored():
    with pytest.raises(ValidationError):
        AppConfig.model_validate({"project_defaults": {"prompt_versions": {"canddiate": "v7"}}})


def test_cli_override_reaches_normal_generation_and_snapshot_and_can_revert(tmp_path):
    from kernel_optimizer.project.agents import ProjectAgents
    from kernel_optimizer.project.generation_snapshot import load
    from kernel_optimizer.project.runtime_config import project_agent_config
    from kernel_optimizer.store.run_store import RunStore

    config_path = tmp_path / "config.yaml"
    config_path.write_text("{}", encoding="utf-8")
    config = project_agent_config(load_config(config_path, ["project_defaults.prompt_versions.candidate=v7"]))
    calls = []

    class Client:
        def create_session(self, directory, title):
            return directory.name

        def prompt(self, session, prompt, *, directory, **kwargs):
            calls.append(prompt)
            (directory / "candidate.py").write_text("def build(context): return {}\n")
            return SimpleNamespace(error=None, message_id="message")

        def collect_usage(self, session, directory):
            return {}

    store = RunStore.create(tmp_path, "run", {})
    agents = ProjectAgents(SimpleNamespace(cfg=config, client=Client()), store)
    v6 = prompt_versions.load_prompt(AppConfig(), "candidate")
    v7 = prompt_versions.load_prompt(config, "candidate")
    assert v7 != v6
    assert "import the final candidate.py" in v7
    for version, expected in [("v7", v7), ("v6", v6)]:
        config.project_defaults.prompt_versions["candidate"] = version
        _, receipt = agents.generate("candidate", {"task.txt": "same facts"},
            prompt_versions.load_prompt(config, "candidate"), ("candidate.py",), feedback="same feedback")
        _, captured, files = load(Path(receipt["generation_snapshots"][0]))
        assert captured == calls[-1]
        assert expected in captured
        assert files["inputs/task.txt"] == b"same facts"
        assert files["inputs/previous-feedback.txt"] == b"same feedback"
    assert calls[0].replace(v7, v6) == calls[1]


def test_recovered_selection_uses_configured_prompt_for_new_candidate(tmp_path, monkeypatch):
    from contextlib import nullcontext
    from kernel_optimizer.project import recovery

    directory = tmp_path / "sandboxes" / "selection"
    inputs = directory / "inputs"
    inputs.mkdir(parents=True)
    (inputs / "project.json").write_text(json.dumps({"current_revision": "parent"}))
    (inputs / "search.json").write_text(json.dumps({"allowed_scopes": ["group"]}))
    (tmp_path / "discovery.json").write_text(json.dumps({"revision": "parent", "calls": []}))
    (directory / "selection.json").write_text(json.dumps({"affected_calls": []}))
    config = AppConfig()
    config.project_defaults.prompt_versions["candidate"] = "v7"
    event = SimpleNamespace(type="PROJECT_AGENT_STARTED", payload={
        "role": "selection", "session": "original-session", "sandbox": str(directory)})
    search = SimpleNamespace(output=tmp_path, config=config,
        store=SimpleNamespace(iter_events=lambda: [event]),
        revisions=SimpleNamespace(current=lambda: SimpleNamespace(id="parent")))

    class ReachedCandidate(Exception):
        pass

    class Agent:
        def _complete(self, role, sandbox, session, *args):
            assert role == "selection" and session == "original-session"
            return directory, {}

        def generate(self, role, inputs, prompt, *args, **kwargs):
            assert role == "candidate"
            assert prompt == prompt_versions.load_prompt(config, "candidate")
            raise ReachedCandidate

    monkeypatch.setattr(recovery, "Runtime", lambda *args, **kwargs: nullcontext(None))
    monkeypatch.setattr(recovery, "ProjectAgents", lambda *args: Agent())
    monkeypatch.setattr(recovery, "eligible_groups", lambda *args, **kwargs: [])
    with pytest.raises(ReachedCandidate):
        recovery.resume_generation(search, directory)
