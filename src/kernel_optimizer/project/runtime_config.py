"""Project role configuration; actual GPU work belongs to UUID-bound workers."""
import importlib.metadata
import os
from pathlib import Path
import platform
import sys


def worker_environment():
    """LocalExecutor uses this interpreter; do not inspect another system env."""
    return {"executor": "LocalExecutor", "python": sys.executable, "python_version": platform.python_version(),
            "prefix": sys.prefix, "platform": platform.platform(),
            "packages": dict(sorted((d.metadata["Name"], d.version) for d in importlib.metadata.distributions()
                                    if d.metadata.get("Name"))),
            "inspection_policy": "Use this Python for CPU/API/source inspection. GPU measurements remain in the framework's leased workers."}


def project_agent_config(config):
    result = config.model_copy(deep=True)
    result.opencode.server_env.setdefault("CUDA_VISIBLE_DEVICES", "")
    interpreter_dir = str(Path(sys.executable).parent)
    current_path = result.opencode.server_env.get("PATH", os.environ.get("PATH", ""))
    if current_path.split(os.pathsep)[0] != interpreter_dir:
        result.opencode.server_env["PATH"] = interpreter_dir + os.pathsep + current_path
    result.opencode.server_env["OPOP_WORKER_PYTHON"] = sys.executable
    if sys.prefix != sys.base_prefix:
        result.opencode.server_env["VIRTUAL_ENV"] = sys.prefix
    return result
