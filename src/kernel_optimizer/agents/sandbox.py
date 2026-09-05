"""Per-call agent sandboxes + SSE permission auto-responder fallback."""

from __future__ import annotations

import json
import subprocess
import threading
from pathlib import Path

import httpx


class Sandbox:
    def __init__(self, root: Path):
        self.root = root

    def write_input(self, rel: str, content: str | bytes) -> Path:
        path = self._resolve(rel)
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, bytes):
            path.write_bytes(content)
        else:
            path.write_text(content, encoding="utf-8")
        return path

    def read_output(self, rel: str) -> str:
        return self._resolve(rel).read_text(encoding="utf-8")

    def exists(self, rel: str) -> bool:
        try:
            return self._resolve(rel).is_file()
        except ValueError:
            return False

    def _resolve(self, rel: str) -> Path:
        path = (self.root / rel).resolve()
        if not path.is_relative_to(self.root.resolve()):
            raise ValueError(f"path escapes sandbox: {rel}")
        return path


PERMISSION_CONFIG = {
    "$schema": "https://opencode.ai/config.json",
    "permission": {"edit": "allow", "bash": "allow", "webfetch": "deny"},
}


class SandboxFactory:
    """Creates one throwaway directory per agent call.

    `extra_config` is merged into the sandbox's `opencode.json` on top of the permission
    block. It exists because that file makes the sandbox a project root for opencode, which
    STOPS its upward search for configuration: a provider declared in an ancestor directory
    is invisible from inside a sandbox. Providers that also appear in the user's global
    `~/.config/opencode` config still resolve (that one is always loaded), which is why
    `openai/gpt-5.6-sol` has always worked while a repo-only provider does not --
    `zhipuai/glm-5.3` failed with `ProviderModelNotFoundError: Model not found:
    zhipuai/glm-5.3` on every attempt until its provider block was written here.

    Passing the block through config rather than hardcoding it keeps the mechanism general:
    any model whose provider is defined per-project works without touching this module, and
    nothing model-specific lives in the harness.

    SECURITY: a provider block normally carries `options.apiKey`, so every sandbox under the
    run directory ends up holding a plaintext credential. Run directories are gitignored, but
    they are also archived and copied around — treat a run dir of an arm using this mechanism
    as secret-bearing, and rotate the key if one is shared. The gpt arm has no such copies
    because its provider comes from the user's global config instead.
    """

    def __init__(self, sandboxes_dir: Path, extra_config: dict | None = None):
        self.sandboxes_dir = sandboxes_dir
        self.extra_config = extra_config or {}

    def create(self, call_id: str) -> Sandbox:
        root = self.sandboxes_dir / call_id
        root.mkdir(parents=True, exist_ok=False)
        config = {**PERMISSION_CONFIG, **self.extra_config}
        (root / "opencode.json").write_text(
            json.dumps(config, indent=2), encoding="utf-8"
        )
        # git init helps opencode's diff tracking; best effort.
        try:
            subprocess.run(["git", "init", "-q"], cwd=root, capture_output=True, timeout=30)
        except (subprocess.TimeoutExpired, OSError):
            pass
        return Sandbox(root)


class PermissionAutoResponder:
    """SSE fallback: auto-approve permission requests for tracked sessions."""

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self.tracked: set[str] = set()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def track(self, session_id: str) -> None:
        self.tracked.add(session_id)

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                with httpx.stream("GET", f"{self.base_url}/event", timeout=None) as resp:
                    for line in resp.iter_lines():
                        if self._stop.is_set():
                            return
                        if not line.startswith("data:"):
                            continue
                        try:
                            event = json.loads(line[5:].strip())
                        except ValueError:
                            continue
                        self._handle(event)
            except httpx.HTTPError:
                if self._stop.wait(2.0):
                    return

    def _handle(self, event: dict) -> None:
        if event.get("type") not in ("permission.updated", "permission.asked"):
            return
        props = event.get("properties", {})
        session_id = props.get("sessionID")
        permission_id = props.get("id") or props.get("permissionID")
        if not session_id or not permission_id or session_id not in self.tracked:
            return
        try:
            httpx.post(
                f"{self.base_url}/session/{session_id}/permissions/{permission_id}",
                json={"response": "always"},
                timeout=10.0,
            )
        except httpx.HTTPError:
            pass
