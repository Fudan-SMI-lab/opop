"""Opencode server lifecycle + REST client (v1 routes, blocking prompt)."""

from __future__ import annotations

import json
import os
import re
import signal
import shutil
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



def resolve_opencode() -> str:
    """Absolute path to the opencode binary.

    Removing shell=True (so --hostname/--port are not swallowed by /bin/sh) also removed
    the shell's PATH lookup, and on this box opencode lives in /root/miniconda3/bin, which
    only a LOGIN shell adds to PATH. Launching a run from tmux, cron or systemd therefore
    died with a bare `FileNotFoundError: 'opencode'` raised from inside Popen, naming no
    candidate path. Resolve explicitly, and on failure report the search list and PATH.
    """
    found = shutil.which("opencode")
    if found:
        return found
    fallbacks = [
        Path.home() / "miniconda3" / "bin" / "opencode",
        Path.home() / ".opencode" / "bin" / "opencode",
        Path("/usr/local/bin/opencode"),
    ]
    for cand in fallbacks:
        if cand.is_file() and os.access(cand, os.X_OK):
            return str(cand)
    raise AgentCallError(
        "opencode binary not found. PATH lookup failed and none of these exist: "
        + ", ".join(str(f) for f in fallbacks)
        + f". PATH={os.environ.get('PATH', '')!r}"
    )


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
        # Inherit our environment and layer `server_env` on top: a few opencode settings
        # (notably the per-turn output-token ceiling) exist only as env vars, with no
        # config-file route -- see OpencodeConfig.server_env.
        env = {**os.environ, **{k: str(v) for k, v in self.cfg.server_env.items()}}
        # No shell=True here. On POSIX, shell=True with a LIST argv runs
        # `/bin/sh -c "opencode"` and passes the rest as $0,$1,... to the shell, so
        # --hostname/--port are silently discarded: the server binds its default port
        # and _wait_healthy() then times out against a URL nothing is listening on --
        # which reads exactly like a network problem. (The Windows reason for the flag
        # was that opencode is a .cmd shim there; on Linux it is a real executable.)
        # start_new_session puts the server in its own process group so stop() can
        # signal the whole tree instead of leaking an orphaned server per run.
        self.proc = subprocess.Popen(
            [resolve_opencode(), "serve", "--hostname", self.cfg.host,
             "--port", str(port)],
            cwd=str(self.cfg.launch_cwd),
            stdout=self._log_handle,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
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
            out = subprocess.run([resolve_opencode(), "--version"], capture_output=True,
                                 timeout=30)
            return out.stdout.decode().strip() or None
        except (subprocess.TimeoutExpired, OSError):
            return None

    def stop(self) -> None:
        if self.proc is None:
            return
        # SIGTERM the whole group first: opencode holds a sqlite session store and
        # SIGKILL can leave it mid-write. Escalate only if it does not exit.
        # (taskkill raised OSError here, and the fallback proc.kill() killed only the
        # direct child -- under the old shell=True that was /bin/sh, leaving the real
        # server orphaned: a leaked process and port per run.)
        try:
            pgid = os.getpgid(self.proc.pid)
        except OSError:
            pgid = None
        try:
            if pgid is not None:
                os.killpg(pgid, signal.SIGTERM)
            else:
                self.proc.terminate()
        except OSError:
            pass
        try:
            self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            try:
                if pgid is not None:
                    os.killpg(pgid, signal.SIGKILL)
                else:
                    self.proc.kill()
            except OSError:
                pass
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass
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


def _memory_pressure() -> tuple[float, float] | None:
    """(bytes_used, bytes_limit) for this container, or None when unknowable.

    Reads the cgroup rather than /proc/meminfo on purpose: the runaway that motivated this was
    inside a container whose limit (128 GB) was far below the host's, so host-level free memory
    looked fine while the cgroup was at 97.5% and throttling every process in it -- including
    the orchestrator, which went into D-state on mem_cgroup_handle_over_high and merely LOOKED
    like a stuck network call.

    Returns None rather than guessing when there is no cgroup (Windows, macOS, an unlimited
    cgroup): the caller then enforces nothing, which is correct -- a bound we cannot measure
    must not be approximated.
    """
    # cgroup v2
    try:
        with open("/sys/fs/cgroup/memory.current", encoding="utf-8") as fh:
            used = float(fh.read().strip())
        with open("/sys/fs/cgroup/memory.max", encoding="utf-8") as fh:
            raw = fh.read().strip()
        if raw != "max":
            limit = float(raw)
            if limit > 0:
                return used, limit
    except (OSError, ValueError):
        pass
    # cgroup v1
    try:
        base = "/sys/fs/cgroup/memory"
        with open(f"{base}/memory.usage_in_bytes", encoding="utf-8") as fh:
            used = float(fh.read().strip())
        with open(f"{base}/memory.limit_in_bytes", encoding="utf-8") as fh:
            limit = float(fh.read().strip())
        # v1 reports an absurd sentinel (~2^63) when unlimited; treat that as no limit.
        if 0 < limit < 2**62:
            return used, limit
    except (OSError, ValueError):
        pass
    return None


class OpencodeClient:
    def __init__(self, base_url: str, timeout_s: float = 1200.0,
                 memory_abort_frac: float = 0.92, resource_poll_s: float = 20.0):
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        # Fraction of the container's memory limit at which an in-flight agent call is aborted.
        # Deliberately NOT a time budget: an agent that runs an hour compiling and benchmarking
        # is working, and cutting it discards that work. What must be bounded is an agent
        # subprocess taking the machine down -- see the watchdog in `prompt`. 0.92 leaves room
        # to act before the kernel's own OOM killer picks a victim for us (the observed runaway
        # sat at 0.975 of the limit while ptxas kept growing).
        self.memory_abort_frac = memory_abort_frac
        self.resource_poll_s = resource_poll_s
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

        # A watchdog, but NOT a time budget on the agent's thinking. An agent legitimately runs
        # long: it compiles kernels, launches them, reads results. Cutting a call at a wall-clock
        # deadline destroys exactly the work it was doing -- measured on the 4057s call whose
        # `rw_1.py` was already complete on disk 8 minutes before it returned, and previously on
        # a 1500s ceiling that discarded a finished rewrite. So duration alone must never end a
        # call.
        #
        # What DID need bounding is different: the agent's own subprocesses consuming the box.
        # One agent-written sweep produced a 272,341-line PTX, ptxas reached 111 GiB resident,
        # the container hit its memory cgroup limit (125.5 GB against a 124.5 GB memory.high,
        # 16.0M throttle events) and the ORCHESTRATOR was throttled into D-state on
        # mem_cgroup_handle_over_high -- 3 GB from a hard OOM where the kernel, not us, chooses
        # the victim. That is a resource condition, is directly observable, and is what this
        # watchdog checks. A long call using nothing is left alone; a call about to take the
        # machine down is stopped however briefly it has run.
        done = threading.Event()
        fired = threading.Event()
        fired_reason: list[str] = []

        def _watch() -> None:
            if self.memory_abort_frac <= 0:
                return  # explicitly disabled
            while not done.wait(self.resource_poll_s):
                pressure = _memory_pressure()
                if pressure is None:
                    return  # no cgroup to read (not Linux, or v1): nothing to enforce
                used, limit = pressure
                if used / limit < self.memory_abort_frac:
                    continue
                fired.set()
                fired_reason.append(
                    f"container memory at {used / 1e9:.1f} GB of {limit / 1e9:.1f} GB "
                    f"({used / limit * 100:.0f}% >= {self.memory_abort_frac * 100:.0f}%)"
                )
                # Abort ends the turn AND its subprocesses (verified: afterwards opencode has
                # zero children and the GPU returns to 0 MiB). Closing the transport is also
                # required -- abort alone returns 200 while the already-streaming POST never
                # returns, leaving this thread blocked forever.
                self.abort(session_id)
                try:
                    self._http.close()
                except Exception:
                    pass
                return

        watchdog = threading.Thread(target=_watch, daemon=True,
                                    name=f"agent-resource-watch-{session_id[:12]}")
        watchdog.start()
        try:
            resp = self._http.post(f"/session/{session_id}/message", json=body, params=params)
        except httpx.HTTPError as exc:
            # A hung/slow agent call (ReadTimeout past self.timeout_s) or any
            # transport failure must NOT crash the whole run. Abort the stuck
            # session and surface a typed error the AgentModule retry loop
            # handles (retry, then drop the candidate) — see plan risk #8.
            hit_resource_abort = fired.is_set()
            done.set()
            if hit_resource_abort:
                # The watchdog closed the transport to break the blocked read; rebuild it so
                # the retry the module is about to attempt has a working client. Skipped for
                # ordinary transport errors, where the client is still usable.
                self._http = httpx.Client(
                    base_url=self.base_url, timeout=httpx.Timeout(self.timeout_s)
                )
                # Name the real cause. A log saying only "ReadTimeout" sends the next reader
                # looking for a slow endpoint, when what happened is that the agent's own
                # subprocesses were about to OOM the machine.
                raise AgentCallError(
                    f"agent call aborted to protect the machine: "
                    f"{fired_reason[0] if fired_reason else 'resource limit reached'}. "
                    f"The agent's own subprocesses were consuming the container; the call was "
                    f"NOT stopped for taking too long."
                ) from exc
            self.abort(session_id)
            raise AgentCallError(
                f"prompt transport error ({type(exc).__name__}): {str(exc)[:400]}"
            ) from exc
        finally:
            done.set()  # release the watchdog on every path
        if fired.is_set():
            # The abort made the blocked POST return instead of raising. Whatever came back is
            # a truncated turn, so report the abort rather than parsing a partial body.
            raise AgentCallError(
                f"agent call aborted to protect the machine: "
                f"{fired_reason[0] if fired_reason else 'resource limit reached'}. "
                f"The agent's own subprocesses were consuming the container; the call was NOT "
                f"stopped for taking too long."
            )
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
