import os
from pathlib import Path
import sys

from kernel_optimizer.config import AppConfig
from kernel_optimizer.project.runtime_config import project_agent_config, worker_environment


def test_project_inspection_uses_worker_python_without_changing_legacy_config():
    original = AppConfig()
    original.opencode.server_env["PATH"] = "/unrelated/python"
    configured = project_agent_config(original)
    assert "permission" not in original.opencode.sandbox_extra_config
    assert configured.opencode.sandbox_extra_config["permission"] == {
        "edit": "allow", "bash": "allow", "webfetch": "allow", "external_directory": "allow"}
    assert project_agent_config(configured).opencode.sandbox_extra_config == configured.opencode.sandbox_extra_config
    assert original.opencode.server_env["PATH"] == "/unrelated/python"
    assert configured.opencode.server_env["PATH"].split(os.pathsep)[0] == str(Path(sys.executable).parent)
    assert configured.opencode.server_env["OPOP_WORKER_PYTHON"] == sys.executable
    assert project_agent_config(configured).opencode.server_env == configured.opencode.server_env
    facts = worker_environment()
    assert facts["python"] == sys.executable and "pydantic" in {p.lower() for p in facts["packages"]}
