"""Opencode server lifecycle + REST client (v1 routes, blocking prompt)."""

from __future__ import annotations

import json
import os
import re
import shutil
import signal
import socket
import subprocess
import threading
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


def _free_port(preferred: int) -> tuple[int, socket.socket]:
    """Pick a port AND keep it reserved until the caller has handed it to the server.

    Returns the port and the socket still holding it. The caller MUST close that socket immediately
    before spawning, and the window between the close and the server's own bind is the only race
    left -- microseconds, against the whole of server startup previously.

    WHY THE SOCKET IS RETURNED INSTEAD OF CLOSED HERE. The earlier version closed it and returned
    only the number, which makes the reservation meaningless: a closed socket's port is immediately
    rebindable (measured), so two orchestrators starting together both probed the default 4096, both
    saw it free, and both were told to use it. The loser's server exited rc=1 with a bare
    `ServeError` from `opencode`, surfacing as `AgentCallError: opencode serve exited rc=1` -- which
    names no port and reads like a broken install rather than a collision.

    That is not hypothetical: it is exactly how the first attempt to launch two co-resident arms on
    one box failed, with arm 2 taking 4096 and arm 3 dying. It could not happen while the project's
    rule was one run per box, and it became reachable the moment two arms shared a machine.

    An OPEN socket does refuse a second bind (measured, OSError), so holding it is what makes the
    probe mean something.
    """
    # `preferred` 0 means "any port" to bind(), which SUCCEEDS -- so the preferred path would return
    # the literal 0 as the port while the socket holds a real one, giving a URL of :0 that nothing
    # can reach. Only the ephemeral path below may name a port, so 0 skips straight to it.
    if preferred:
        s = socket.socket()
        try:
            s.bind(("127.0.0.1", preferred))
        except OSError:
            s.close()
        else:
            return preferred, s
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    return s.getsockname()[1], s


def resolve_opencode() -> str:
    """Absolute path to the opencode binary, on either platform.

    Not platform-specific, and not merely a convenience. Dropping `shell=True` on POSIX (needed, see
    `_shell_for_opencode`) also drops the shell's PATH lookup, and on a rented Linux box opencode
    typically lives somewhere only a LOGIN shell puts on PATH (e.g. /root/miniconda3/bin). A run
    started from tmux, cron or systemd then died with a bare `FileNotFoundError: 'opencode'` raised
    from inside Popen, naming no candidate path and reading like a missing install.

    So resolve explicitly and, on failure, report the search list AND the PATH -- the two facts
    needed to tell "not installed" from "installed where this shell cannot see it". Windows benefits
    from the same diagnostic, which is why this is unconditional.
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


def _shell_for_opencode() -> bool:
    """Whether launching opencode needs a shell.

    Windows: YES -- opencode is installed as a `.cmd` shim there, which CreateProcess cannot exec
    directly.

    POSIX: NO, and passing it would be a bug. `shell=True` with a LIST argv runs `/bin/sh -c
    "opencode"` and hands the remaining elements to the shell as $0, $1, ... -- so `--hostname` and
    `--port` are silently DISCARDED. The server then binds its default port, `_wait_healthy()`
    polls the port we intended, and the timeout that follows reads exactly like a network problem.
    """
    return os.name == "nt"


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
        port, reservation = _free_port(self.cfg.port)
        self.base_url = f"http://{self.cfg.host}:{port}"
        self._log_handle = self.log_path.open("ab")
        # Inherit our environment and layer `server_env` on top: a few opencode settings
        # (notably the per-turn output-token ceiling) exist only as env vars, with no
        # config-file route -- see OpencodeConfig.server_env.
        env = {**os.environ, **{k: str(v) for k, v in self.cfg.server_env.items()}}
        # start_new_session (POSIX) puts the server in its own process group so `stop()` can
        # signal the whole tree instead of leaking an orphaned server per run.
        popen_kw: dict = {}
        if os.name != "nt":
            popen_kw["start_new_session"] = True
        # Release the reservation as late as possible: while it is open no other orchestrator can
        # be handed this port, and the server cannot bind it either.
        reservation.close()
        self.proc = subprocess.Popen(
            [resolve_opencode(), "serve", "--hostname", self.cfg.host, "--port", str(port)],
            cwd=str(self.cfg.launch_cwd),
            stdout=self._log_handle,
            stderr=subprocess.STDOUT,
            shell=_shell_for_opencode(),
            env=env,
            **popen_kw,
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
                    # The URL is in the message because without it this exception names nothing
                    # actionable. `opencode` reports a port collision as a bare `ServeError` with
                    # rc=1 and no port, so the raised error read like a broken install -- and the
                    # actual cause was a second orchestrator on the same box holding this port.
                    f"opencode serve exited rc={self.proc.returncode} "
                    f"(was starting on {self.base_url}; a bare ServeError here usually means "
                    f"another process already holds that port): {tail}"
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
                                 timeout=30, shell=_shell_for_opencode())
            return out.stdout.decode().strip() or None
        except (subprocess.TimeoutExpired, OSError, AgentCallError):
            return None

    def stop(self) -> None:
        if self.proc is None:
            return
        # Stop the whole tree, not just the direct child. Under the old code `taskkill` raised
        # OSError on POSIX and the fallback `proc.kill()` killed only that child -- which with
        # shell=True was /bin/sh, leaving the real server orphaned: one leaked process and one held
        # port per run. SIGTERM first on POSIX, because opencode holds a sqlite session store that
        # SIGKILL can leave mid-write; escalate only if it does not exit.
        if os.name == "nt":
            try:
                subprocess.run(
                    ["taskkill", "/F", "/T", "/PID", str(self.proc.pid)],
                    capture_output=True, timeout=30,
                )
            except (subprocess.TimeoutExpired, OSError):
                try:
                    self.proc.kill()
                except OSError:
                    pass
        else:
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


def _output_fingerprint(directory: Path | None) -> tuple[int, int, float] | None:
    """(file count, total bytes, newest mtime) under the agent's sandbox, or None if unreadable.

    G20's productivity signal. An agent that is compiling, benchmarking and writing kernels changes
    this; an agent that has hung does not. Cheap enough to poll: a sandbox holds tens of files, and
    this walks metadata only -- no file is opened.

    Deliberately a fingerprint of the whole tree rather than a watch on the expected output file:
    the agent writes scratch kernels, logs and compile artifacts long before it writes its answer,
    and those are exactly the evidence that it is working. Waiting only for the final artifact would
    call a productive agent hung for most of its run.

    Returns None on any error, which the caller treats as "cannot tell" and therefore never as
    grounds to abort -- an unreadable sandbox must not look like a hung agent.
    """
    if directory is None:
        return None
    try:
        count = 0
        total = 0
        newest = 0.0
        for p in Path(directory).rglob("*"):
            try:
                st = p.stat()
            except OSError:
                continue
            if not p.is_file():
                continue
            count += 1
            total += st.st_size
            newest = max(newest, st.st_mtime)
        return count, total, newest
    except Exception:  # noqa: BLE001 -- a probe must never be the thing that fails a call
        return None


class OpencodeClient:
    def __init__(self, base_url: str, timeout_s: float = 1200.0,
                 memory_abort_frac: float = 0.92, resource_poll_s: float = 20.0,
                 idle_abort_frac: float = 0.5):
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
        # G20. How long a call may go WITHOUT PRODUCING ANYTHING before it is treated as hung,
        # expressed as a FRACTION OF `timeout_s` rather than as minutes. Deriving it means the two
        # cannot drift apart: raising the transport ceiling automatically raises how long a silent
        # agent is tolerated, and a config that shortens the ceiling shortens this too. A hardcoded
        # "20 minutes" would become either vacuous or trigger-happy the moment `request_timeout_s`
        # changed, which is precisely the kind of coupling this project keeps getting wrong.
        #
        # 0.5 means: half the transport ceiling with no new output at all. That is deliberately
        # generous -- the point is to catch a call that has stopped producing, not to race a slow
        # one -- and it still cuts a hung call in half the time the ceiling would, freeing the
        # budget for another sample instead of burning it on silence.
        self.idle_abort_frac = idle_abort_frac
        self._http = httpx.Client(base_url=self.base_url, timeout=httpx.Timeout(timeout_s))

    @property
    def idle_abort_s(self) -> float:
        """Seconds of zero output after which a call counts as hung. Derived, never hardcoded."""
        return max(60.0, self.timeout_s * self.idle_abort_frac)

    def _session_is_working(self, session_id: str) -> bool | None:
        """Is the server still working on this session's turn? True / False / None = cannot tell.

        The second half of G20's productivity signal, added after the sandbox-only version killed
        5 of 6 legitimate calls (G24). A reasoning model can think for many minutes without
        touching its sandbox, so "no local files changed" is not evidence of a hang. The server,
        however, knows whether the turn is still running.

        Three properties matter:

        1. It uses its OWN short-lived transport, NOT `self._http`. The watchdog runs while the
           main transport is blocked on the streaming POST, and issuing a request on it from
           another thread would either block behind that read or disturb it.
        2. Anything unexpected returns None, never False. None means "cannot tell" and the caller
           treats it exactly like an unreadable sandbox -- never grounds to abort. A probe failure
           must not be able to kill a working call; that inversion (reporting a diagnostic failure
           as an agent failure) is a shape this project has hit repeatedly.
        3. It is only consulted AFTER the sandbox has been silent for the whole window, so it costs
           one cheap request per abort decision, not one per poll.
        """
        try:
            with httpx.Client(base_url=self.base_url, timeout=httpx.Timeout(15.0)) as probe:
                resp = probe.get(f"/session/{session_id}")
                if resp.status_code != 200:
                    return None
                info = resp.json()
        except Exception:  # noqa: BLE001 -- a probe must never be the thing that fails a call
            return None
        if not isinstance(info, dict):
            return None
        # opencode has expressed "this turn is running" differently across versions, so accept any
        # of the known spellings and fall back to None (cannot tell) rather than guessing False.
        for key in ("working", "busy", "running", "isWorking", "isBusy"):
            v = info.get(key)
            if isinstance(v, bool):
                return v
        # A `revert`/`time.completed` style marker: a completed timestamp means not working.
        t = info.get("time")
        if isinstance(t, dict):
            if t.get("completed") is not None:
                return False
            if t.get("created") is not None:
                # Present but not completed -- the turn exists and has not finished. Report None
                # rather than True: absence of a completion marker is weaker evidence than an
                # explicit flag, and None is the safe direction (never aborts).
                return None
        return None

    def _abort_and_close(self, session_id: str) -> None:
        """End the turn and unblock the streaming POST.

        Abort ends the turn AND its subprocesses (verified: afterwards opencode has zero children
        and the GPU returns to 0 MiB). Closing the transport is also required -- abort alone
        returns 200 while the already-streaming POST never returns, leaving the watchdog thread
        blocked forever. Shared by both abort reasons so they cannot diverge.
        """
        self.abort(session_id)
        try:
            self._http.close()
        except Exception:  # noqa: BLE001
            pass

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

        # A transport closed by a PREVIOUS abort must not sink this call. `_abort_and_close` closes
        # `self._http` to unblock a stuck streaming POST, and while the abort path rebuilds it, an
        # abort raised on the watchdog thread can land between a caller's retries -- and httpx then
        # raises a bare `RuntimeError("Cannot send a request, as the client has been closed.")`,
        # which is NOT an AgentCallError and so escapes AgentModule's retry loop entirely.
        #
        # Measured, 2026-09-10 (G24): three G9 arms died exactly this way, losing calls that had
        # nothing wrong with them. Reopening here rather than at each abort site is deliberate --
        # it covers every path that can close the transport, including future ones, instead of
        # relying on each of them to remember.
        # `getattr` because `_http` is substituted by a fake in tests and by any future transport
        # wrapper; a missing attribute means "not a closed httpx client", so carry on rather than
        # crash. A guard that breaks callers who inject a transport is worse than the bug it fixes.
        if getattr(self._http, "is_closed", False):
            self._http = httpx.Client(
                base_url=self.base_url, timeout=httpx.Timeout(self.timeout_s)
            )

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
        #
        # G20 adds the second half of that same principle. The rule was already "duration alone must
        # never end a call"; what was missing is the converse -- SILENCE should. Measured across 9
        # rewriter calls: median 18.9 min, and 33% ran into the transport ceiling. A call that has
        # stopped producing anything is not thinking, and waiting out the full ceiling on it costs a
        # sample that could have been another measurement. So the watchdog also aborts a call that
        # has produced NO new output for `idle_abort_s` -- a fraction of the transport ceiling, so
        # the two move together and neither is a hardcoded interval.
        done = threading.Event()
        fired = threading.Event()
        fired_reason: list[str] = []

        def _watch() -> None:
            if self.memory_abort_frac <= 0 and self.idle_abort_frac <= 0:
                return  # both checks explicitly disabled
            last_print = _output_fingerprint(directory)
            last_change = time.time()
            while not done.wait(self.resource_poll_s):
                # --- memory: stop a call that is about to take the box down -------------------
                if self.memory_abort_frac > 0:
                    pressure = _memory_pressure()
                    if pressure is not None:
                        used, limit = pressure
                        if used / limit >= self.memory_abort_frac:
                            fired.set()
                            fired_reason.append(
                                f"container memory at {used / 1e9:.1f} GB of {limit / 1e9:.1f} GB "
                                f"({used / limit * 100:.0f}% >= "
                                f"{self.memory_abort_frac * 100:.0f}%)"
                            )
                            self._abort_and_close(session_id)
                            return
                # --- productivity: stop a call that has stopped producing ---------------------
                # None means the sandbox could not be read, which is "cannot tell" and must never
                # be grounds to abort: an unreadable directory would otherwise look exactly like a
                # hung agent and cut every call.
                #
                # MEASURED FALSE POSITIVE, 2026-09-10 (G24). The sandbox fingerprint ALONE killed
                # 5 of 6 legitimate calls on L3:21 at 12.7 min, each reporting "25 files,
                # unchanged". Those 25 files were the SEEDED INPUTS (18 .git hook samples,
                # opencode.json, 6 input docs) -- the agent had written nothing yet, and no
                # `rewrites/` directory existed in any of them. The sixth call, identical in every
                # way, wrote its first kernel at 11.5 min and finished at 15.2 min with two valid
                # candidates. So the five aborts landed roughly a minute before their output
                # would have appeared.
                #
                # The premise "an agent that is working changes its sandbox" is FALSE for a
                # reasoning model: glm-5.3 does its thinking server-side and writes nothing local
                # until it emits the answer. A long silent think is indistinguishable from a hang
                # by files alone -- so files alone must not decide.
                #
                # The fix keeps the principle (a call that has genuinely stopped should end) and
                # replaces the evidence: a call counts as productive if EITHER its sandbox changed
                # OR the server still reports its session as working. The server-side check is the
                # authority on "is this turn still alive", which is precisely what the file tree
                # cannot see. Only when BOTH say nothing is happening does the call get cut.
                if self.idle_abort_frac > 0 and last_print is not None:
                    now_print = _output_fingerprint(directory)
                    if now_print is not None:
                        if now_print != last_print:
                            last_print, last_change = now_print, time.time()
                        elif time.time() - last_change >= self.idle_abort_s:
                            # Silent on disk for the whole window. Before cutting, ask the server
                            # whether the turn is still running. `None` = cannot tell, which is
                            # treated exactly like the unreadable-sandbox case: never grounds to
                            # abort.
                            still_working = self._session_is_working(session_id)
                            if still_working is not False:
                                # Productive (or unknown) -- restart the window rather than cut.
                                # Restarting is deliberate: it means a call that keeps reporting
                                # itself alive is never cut by this rule, and the transport
                                # ceiling remains the only hard stop for it. That is the correct
                                # trade: the ceiling costs one sample, a false abort costs the
                                # sample AND corrupts the experiment it was part of.
                                last_change = time.time()
                                continue
                            idle_min = (time.time() - last_change) / 60.0
                            fired.set()
                            fired_reason.append(
                                f"no new output for {idle_min:.1f} min "
                                f"({self.idle_abort_frac:.0%} of the {self.timeout_s:.0f}s "
                                f"transport ceiling): {now_print[0]} files, "
                                f"{now_print[1]} bytes, unchanged, AND the server reports the "
                                f"session is not working. A call that has stopped producing is "
                                f"not thinking, and waiting out the full ceiling costs a sample."
                            )
                            self._abort_and_close(session_id)
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
