"""Explicit role prompt versions; omitted versions retain the original v6 text."""
from pathlib import Path
import re

from kernel_optimizer.config import AppConfig


_LEGACY_FILES = {
    "project-intake": "intake_prompt.md",
    "project-evaluator": "adapter_prompt.md",
    "selection": "selection_prompt.md",
    "candidate": "candidate_prompt.md",
}


def load_prompt(config: AppConfig, role: str) -> str:
    if role not in _LEGACY_FILES:
        raise ValueError(f"No versioned project prompt for role {role!r}")
    version = config.project_defaults.prompt_versions.get(role, "v6")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", version):
        raise ValueError(f"Invalid prompt version for {role}: {version!r}")
    root = Path(__file__).parent
    path = root / _LEGACY_FILES[role] if version == "v6" else root / "prompts" / role / f"{version}.md"
    if not path.is_file():
        raise ValueError(f"Prompt version {version!r} is unavailable for {role}")
    return path.read_text(encoding="utf-8")
