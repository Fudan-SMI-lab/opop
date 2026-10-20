import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from kernel_optimizer.agents.sandbox import SandboxFactory
from kernel_optimizer.config import AppConfig
from kernel_optimizer.project.generation_snapshot import capture, edit_one, load
from kernel_optimizer.store.run_store import RunStore


def test_snapshot_preserves_original_inputs_after_generation_edits(tmp_path):
    store = RunStore.create(tmp_path, "run", {})
    sb = SandboxFactory(store.run_dir/"sandboxes").create("selection")
    sb.write_input("inputs/evidence.json", '{"measurement": 42}')
    path = capture(store, sb.root, role="selection", session="session-a", prompt="Read the inputs.",
                   outputs=("selection.json",), config=AppConfig(), fresh=True)
    sb.write_input("inputs/evidence.json", '{"measurement": 99}')
    saved, prompt, files = load(path)
    assert saved["fresh_session"] and prompt == "Read the inputs."
    assert json.loads(files["inputs/evidence.json"])["measurement"] == 42
    assert "selection.json" not in files


def test_comparison_changes_only_the_declared_input():
    files = {"inputs/a.txt": b"Measured time; interpretation unknown.", "inputs/b.txt": b"unchanged"}
    prompt, changed = edit_one("Read inputs.", files, {
        "path": "inputs/a.txt", "old": "interpretation unknown", "new": "not a transfer-time lower bound"})
    assert prompt == "Read inputs." and changed["inputs/b.txt"] == files["inputs/b.txt"]
    assert files["inputs/a.txt"] == b"Measured time; interpretation unknown."
    with pytest.raises(ValueError, match="occurrences"):
        edit_one(prompt, files, {"path": "inputs/a.txt", "old": "absent", "new": "replacement"})


def test_replay_uses_fresh_session_and_exact_initial_prompt(tmp_path):
    from kernel_optimizer.project.agents import ProjectAgents
    config = AppConfig()
    config.opencode.launch_cwd = str(tmp_path)
    calls = []

    class Client:
        def create_session(self, directory, title):
            return "session-" + directory.name

        def prompt(self, session, prompt, *, directory, **kwargs):
            calls.append((session, prompt))
            (directory/"answer.txt").write_text("generated")
            return SimpleNamespace(error=None, message_id="message")

        def collect_usage(self, session, directory):
            return {}

    store = RunStore.create(tmp_path, "run", {})
    agents = ProjectAgents(SimpleNamespace(cfg=config, client=Client()), store)
    _, receipt = agents.generate("probe", {"facts.txt": "source facts"}, "Write answer.txt", ("answer.txt",))
    snapshot = Path(receipt["generation_snapshots"][0])
    _, replayed = agents.replay_snapshot(snapshot)
    assert calls[0][0] != calls[1][0]
    assert calls[0][1] == calls[1][1]
    replay = load(Path(replayed["generation_snapshots"][0]))
    assert replay[0]["fresh_session"] and replay[0]["history_status"] == "empty"
    assert "answer.txt" not in replay[2]
