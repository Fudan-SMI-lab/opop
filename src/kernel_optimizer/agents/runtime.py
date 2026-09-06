"""Opencode server lifecycle + REST client (v1 routes, blocking prompt)."""

from __future__ import annotations

import json
import re
import socket
import subprocess
import time
from pathlib import Path
from typing import Any

import httpx
from pydantic import BaseModel

from kernel_optimizer.config import OpencodeConfig


class AgentCallError(Exception):
    pass


class PromptResult(BaseModel):
    text: str
    structured: dict | None = None
    tokens: dict = {}
    cost: float = 0.0
    session_id: str
    message_id: str | None = None
    error: str | None = None
    # Why the model stopped, verbatim from the server ("stop", "tool-calls", "length", ...).
    # `length` means the answer was CUT OFF, which is a different failure from a badly
    # formatted answer and needs different feedback -- see AgentModule.invoke.
    finish: str | None = None


def _free_port(preferred: int) -> int:
    with socket.socket() as s:
        try:
            s.bind(("127.0.0.1", preferred))
            return preferred
        except OSError:
            s.bind(("127.0.0.1", 0))
            return s.getsockname()[1]


class OpencodeServer:
    def __init__(self, cfg: OpencodeConfig, log_path: Path | None = None):
        self.cfg = cfg
        self.proc: subprocess.Popen | None = None
        self.base_url: str | None = cfg.server_url
        self.log_path = log_path or Path("opencode-server.log")
        self._log_handle = None

    def start(self) -> str:
        if self.cfg.server_url:
            self.base_url = self.cfg.server_url.rstrip("/")
            self._wait_healthy()
            return self.base_url
        port = _free_port(self.cfg.port)
        self.base_url = f"http://{self.cfg.host}:{port}"
        self._log_handle = self.log_path.open("ab")
        self.proc = subprocess.Popen(
            ["opencode", "serve", "--hostname", self.cfg.host, "--port", str(port)],
            cwd=str(self.cfg.launch_cwd),
            stdout=self._log_handle,
            stderr=subprocess.STDOUT,
            shell=True,  # opencode is a .cmd shim on Windows
        )
        self._wait_healthy()
        return self.base_url

    def _wait_healthy(self) -> None:
        deadline = time.monotonic() + self.cfg.startup_timeout_s
        last_err: Exception | None = None
        while time.monotonic() < deadline:
            if self.proc is not None and self.proc.poll() is not None:
                tail = ""
                try:
                    tail = self.log_path.read_text(encoding="utf-8",
                                                   errors="replace")[-2000:]
                except OSError:
                    pass
                raise AgentCallError(
                    f"opencode serve exited rc={self.proc.returncode}: {tail}"
                )
            try:
                resp = httpx.get(f"{self.base_url}/config", timeout=3.0)
                if resp.status_code == 200:
                    return
            except httpx.HTTPError as exc:
                last_err = exc
            time.sleep(0.5)
        raise AgentCallError(f"opencode server not healthy at {self.base_url}: {last_err}")

    def version(self) -> str | None:
        try:
            out = subprocess.run(["opencode", "--version"], capture_output=True,
                                 timeout=30, shell=True)
            return out.stdout.decode().strip() or None
        except (subprocess.TimeoutExpired, OSError):
            return None

    def stop(self) -> None:
        if self.proc is None:
            return
        try:
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(self.proc.pid)],
                capture_output=True, timeout=30,
            )
        except (subprocess.TimeoutExpired, OSError):
            self.proc.kill()
        self.proc = None
        if self._log_handle is not None:
            self._log_handle.close()
            self._log_handle = None


_FENCED_JSON = re.compile(r"```(?:json)?\s*\n(.*?)```", re.DOTALL)


def extract_fenced_json(text: str) -> dict | None:
    """Last parseable fenced JSON block in the text, else last bare {...} attempt."""
    for match in reversed(_FENCED_JSON.findall(text)):
        try:
            data = json.loads(match)
            if isinstance(data, dict):
                return data
        except ValueError:
            continue
    # Bare-object fallback: outermost braces.
    start = text.find("{")
    end = text.rfind("}")
    if 0 <= start < end:
        try:
            data = json.loads(text[start : end + 1])
            if isinstance(data, dict):
                return data
        except ValueError:
            pass
    return None


class OpencodeClient:
    def __init__(self, base_url: str, timeout_s: float = 1200.0):
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self._http = httpx.Client(base_url=self.base_url, timeout=httpx.Timeout(timeout_s))

    def close(self) -> None:
        self._http.close()

    def create_session(self, directory: Path, title: str) -> str:
        resp = self._http.post(
            "/session",
            params={"directory": str(directory)},
            json={"title": title},
        )
        if resp.status_code != 200:
            raise AgentCallError(f"session create failed {resp.status_code}: {resp.text[:500]}")
        return resp.json()["id"]

    def prompt(
        self,
        session_id: str,
        text: str,
        *,
        model: str,
        agent: str = "build",
        schema: dict | None = None,
        directory: Path | None = None,
        system: str | None = None,
    ) -> PromptResult:
        provider_id, _, model_id = model.partition("/")
        body: dict[str, Any] = {
            "model": {"providerID": provider_id, "modelID": model_id},
            "agent": agent,
            "parts": [{"type": "text", "text": text}],
        }
        if system:
            body["system"] = system
        if schema is not None:
            # Server-side structured output (honored by newer CLIs); harmless otherwise —
            # extract_fenced_json remains the fallback.
            body["format"] = {"type": "json_schema", "schema": schema, "retryCount": 2}
        params = {"directory": str(directory)} if directory else None

        try:
            resp = self._http.post(f"/session/{session_id}/message", json=body, params=params)
        except httpx.HTTPError as exc:
            # A hung/slow agent call (ReadTimeout past self.timeout_s) or any
            # transport failure must NOT crash the whole run. Abort the stuck
            # session and surface a typed error the AgentModule retry loop
            # handles (retry, then drop the candidate) — see plan risk #8.
            self.abort(session_id)
            raise AgentCallError(
                f"prompt transport error ({type(exc).__name__}): {str(exc)[:400]}"
            ) from exc
        if resp.status_code != 200:
            raise AgentCallError(f"prompt failed {resp.status_code}: {resp.text[:800]}")
        data = resp.json()
        info = data.get("info", {})
        parts = data.get("parts", [])
        text_out = "\n".join(
            p.get("text", "") for p in parts if p.get("type") == "text"
        )
        structured = info.get("structured")
        if not isinstance(structured, dict):
            structured = extract_fenced_json(text_out)
        error = info.get("error")
        return PromptResult(
            text=text_out,
            structured=structured,
            tokens=info.get("tokens", {}),
            cost=float(info.get("cost", 0.0)),
            session_id=session_id,
            message_id=info.get("id"),
            error=json.dumps(error)[:500] if error else None,
            finish=info.get("finish") if isinstance(info.get("finish"), str) else None,
        )

    def abort(self, session_id: str) -> None:
        try:
            self._http.post(f"/session/{session_id}/abort", timeout=10.0)
        except httpx.HTTPError:
            pass

    def respond_permission(self, session_id: str, permission_id: str,
                           response: str = "always") -> None:
        self._http.post(
            f"/session/{session_id}/permissions/{permission_id}",
            json={"response": response},
        )
