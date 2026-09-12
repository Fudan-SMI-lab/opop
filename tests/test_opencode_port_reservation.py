"""Two orchestrators on one box must not be handed the same opencode port.

This is a co-residency bug, and it could not happen while the project's rule was one run per box. It
became reachable the moment two arms shared a machine, and it is exactly how the first attempt to
launch step 3's two treatment arms failed: arm 2 took the default 4096, arm 3's server exited rc=1
with a bare `ServeError`, and the harness surfaced that as `AgentCallError: opencode serve exited
rc=1` -- naming no port, reading like a broken install.

The cause was that `_free_port` bound a probe socket, CLOSED it, and returned only the number. A
closed socket's port is immediately rebindable, so the "check" guaranteed nothing across the whole of
server startup: both orchestrators probed 4096, both saw it free, both were told to use it.

Measured on the box before writing the fix:
    a closed probe socket's port is rebindable            -> True   (the race)
    an OPEN probe socket refuses a second bind (OSError)  -> True   (so holding it works)
"""

from __future__ import annotations

import socket

import pytest

from kernel_optimizer.agents.runtime import _free_port


def _an_actually_free_port() -> int:
    """A port number that is free right now.

    Not 0: `bind(("127.0.0.1", 0))` SUCCEEDS and means "any port", so passing 0 to `_free_port`
    exercises the ephemeral path and returns a port unrelated to the argument. An early version of
    these tests used 0 and failed with `assert 0 != 0` -- the tests were wrong, not the code, and the
    real defect it exposed was that `_free_port(0)` used to return the literal 0 as the port while
    its socket held a real one, i.e. a URL of :0 that nothing can reach.
    """
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def test_free_port_returns_the_socket_still_holding_the_port():
    """The reservation must come back OPEN, not closed.

    This is the whole fix: returning a bare int is indistinguishable from the buggy version at every
    call site, and a test that only checked the number would pass on the code that caused the outage.
    """
    port, sock = _free_port(_an_actually_free_port())
    try:
        assert isinstance(port, int) and port > 0
        assert isinstance(sock, socket.socket)
        assert sock.fileno() != -1, "the reservation socket came back closed -- it reserves nothing"
        assert sock.getsockname()[1] == port, "the socket does not hold the port that was returned"
    finally:
        sock.close()


def test_zero_is_never_returned_as_a_port():
    """`_free_port(0)` must name the port it actually holds, not echo the 0 back.

    bind() accepts 0 as "any port" and succeeds, so the preferred path would have returned 0 while
    holding a real port -- producing `http://127.0.0.1:0`, which every health check then polls in
    vain until the startup timeout, reading as a server that failed to come up.
    """
    port, sock = _free_port(0)
    try:
        assert port != 0, "a port of 0 was returned; the base URL would be unreachable"
        assert sock.getsockname()[1] == port
    finally:
        sock.close()


def test_a_held_reservation_blocks_a_second_caller():
    """While one caller holds its reservation, a second must not be handed the same port.

    Driven through `_free_port` twice rather than through raw sockets, because the property that
    matters is the function's, not the OS's.
    """
    p1, s1 = _free_port(_an_actually_free_port())
    try:
        p2, s2 = _free_port(p1)          # ask for exactly the port already held
        try:
            assert p2 != p1, (
                "two callers were handed the same port while the first still held it -- this is the "
                "collision that killed arm 3")
        finally:
            s2.close()
    finally:
        s1.close()


def test_the_preferred_port_is_still_used_when_it_is_free():
    """The fix must not silently move every run onto a random port.

    A fix that always returned an ephemeral port would also pass the collision test above while
    making every run's port unpredictable, which is a debugging regression.
    """
    wanted = _an_actually_free_port()
    port, sock = _free_port(wanted)
    try:
        assert port == wanted, "a free preferred port was not honoured"
    finally:
        sock.close()


def test_a_taken_preferred_port_falls_back_instead_of_raising():
    """When the preferred port is genuinely occupied, fall back rather than crash.

    The original code's fallback path rebound the SAME socket object after a failed bind, which is
    what the rewrite had to preserve; a naive rewrite that reuses a socket already in a failed state
    raises instead of falling back.
    """
    holder = socket.socket()
    holder.bind(("127.0.0.1", 0))
    taken = holder.getsockname()[1]
    holder.listen(1)
    try:
        port, sock = _free_port(taken)
        try:
            assert port != taken
            assert sock.fileno() != -1
        finally:
            sock.close()
    finally:
        holder.close()


def test_the_startup_error_names_the_url_it_was_binding():
    """A collision must be diagnosable from the exception alone.

    `opencode` reports it as a bare `ServeError` with no port, so if the harness does not add the URL
    the operator is left with "exited rc=1" and no reason to suspect a port at all -- which is what
    happened. Asserted on the source of the message rather than by starting a real server, because
    the failure needs a genuinely occupied port and a real spawn.
    """
    import ast
    import inspect

    from kernel_optimizer.agents import runtime

    src = inspect.getsource(runtime.OpencodeServer._wait_healthy)
    tree = ast.parse(src.lstrip())
    texts = [n.value for n in ast.walk(tree)
             if isinstance(n, ast.Constant) and isinstance(n.value, str)]
    joined = " ".join(texts)
    assert "exited rc=" in joined, "this test is anchored on the wrong message"
    assert "base_url" in src, "the failure message does not mention which URL it was binding"
    assert "port" in joined.lower(), (
        "the failure message does not mention a port, so a collision reads as a broken install")


def test_the_reservation_is_closed_before_the_server_is_spawned():
    """The reservation must be released before `Popen`, or the server cannot bind its own port.

    A fix that holds the socket too long is worse than the bug: every run would fail, not just a
    co-resident one. Checked structurally -- the `reservation.close()` must appear before the
    `subprocess.Popen` call in the method that spawns (`_start_once`, which `start` calls per attempt).
    """
    import inspect

    from kernel_optimizer.agents import runtime

    src = inspect.getsource(runtime.OpencodeServer._start_once)
    close_at = src.find("reservation.close()")
    popen_at = src.find("subprocess.Popen")
    assert close_at != -1, "the reservation is never released; the server could never bind"
    assert popen_at != -1, "this test is anchored on the wrong call"
    assert close_at < popen_at, (
        "the reservation is still held when the server is spawned, so the server cannot bind it")


# --------------------------------------------------------------------------------------------------
# Reserving is NOT sufficient: the harness must retry on a fresh port.
#
# The first version of this fix claimed the remaining window was "microseconds". It is not. The
# reservation has to be released before spawning -- the server binds the port itself -- so the window
# spans however long `opencode` takes to start, i.e. SECONDS. Two orchestrators launched together both
# land inside it, which was then measured a SECOND time on box 4: arm 2 took 4096 and arm 3 died on
# `ServeError` with the reservation already in place. So a collision is a transient to retry.
# --------------------------------------------------------------------------------------------------

class _FakeProc:
    def __init__(self, rc: int | None = 1) -> None:
        self._rc = rc
        self.terminated = False

    def poll(self):
        return self._rc

    def terminate(self):
        self.terminated = True

    def wait(self, timeout=None):
        return self._rc


def _server(tmp_path, **over):
    from kernel_optimizer.agents.runtime import OpencodeServer
    from kernel_optimizer.config import OpencodeConfig

    cfg = OpencodeConfig(**over)
    return OpencodeServer(cfg, log_path=tmp_path / "opencode-server.log")


def test_a_serve_error_is_retried_on_a_different_port(tmp_path, monkeypatch):
    """A losing race must be retried, and the retry must not ask for the same port again.

    Retrying the CONFIGURED port would lose the identical race: the winner holds it for the life of
    its run, so attempt 2 onward has to be ephemeral.
    """
    from kernel_optimizer.agents import runtime

    srv = _server(tmp_path, port=4096, port_attempts=3)
    asked: list[int] = []
    real_free_port = runtime._free_port

    def fake_free_port(preferred):
        asked.append(preferred)
        return real_free_port(0)

    calls = {"n": 0}

    def fake_wait(self):
        calls["n"] += 1
        if calls["n"] == 1:
            self.proc = _FakeProc(rc=1)
            raise runtime.AgentCallError(
                "opencode serve exited rc=1 (was starting on http://127.0.0.1:4096; a bare "
                "ServeError here usually means another process already holds that port): ServeError")

    monkeypatch.setattr(runtime, "_free_port", fake_free_port)
    monkeypatch.setattr(runtime, "resolve_opencode", lambda: "/bin/true")
    monkeypatch.setattr(runtime.subprocess, "Popen", lambda *a, **k: _FakeProc(rc=None))
    monkeypatch.setattr(runtime.OpencodeServer, "_wait_healthy", fake_wait)

    url = srv.start()
    assert calls["n"] == 2, "the collision was not retried"
    assert asked[0] == 4096, "the first attempt should honour the configured port"
    assert asked[1] == 0, "the retry must ask for an EPHEMERAL port, not the one that just lost"
    assert url.startswith("http://127.0.0.1:") and not url.endswith(":4096")


def test_a_non_port_failure_is_not_retried(tmp_path, monkeypatch):
    """An error another port would not fix must be raised on the first attempt.

    Otherwise a missing binary or a config opencode rejects becomes `port_attempts` copies of the
    same message, and the real cause is buried under retries of a hopeless start.
    """
    from kernel_optimizer.agents import runtime

    srv = _server(tmp_path, port_attempts=3)
    calls = {"n": 0}

    def fake_wait(self):
        calls["n"] += 1
        self.proc = _FakeProc(rc=127)
        raise runtime.AgentCallError(
            "opencode serve exited rc=127 (was starting on http://127.0.0.1:4096): "
            "command not found")

    monkeypatch.setattr(runtime, "resolve_opencode", lambda: "/bin/true")
    monkeypatch.setattr(runtime.subprocess, "Popen", lambda *a, **k: _FakeProc(rc=None))
    monkeypatch.setattr(runtime.OpencodeServer, "_wait_healthy", fake_wait)

    with pytest.raises(runtime.AgentCallError):
        srv.start()
    assert calls["n"] == 1, "a non-port failure was retried; the real cause gets buried"


def test_retries_are_bounded_and_the_last_error_is_raised(tmp_path, monkeypatch):
    """With every attempt colliding, the error must surface rather than loop.

    Two ephemeral picks colliding in a row is not a race any more, so more attempts would only delay
    the report.
    """
    from kernel_optimizer.agents import runtime

    srv = _server(tmp_path, port_attempts=3)
    calls = {"n": 0}

    def fake_wait(self):
        calls["n"] += 1
        self.proc = _FakeProc(rc=1)
        raise runtime.AgentCallError("opencode serve exited rc=1: ServeError")

    monkeypatch.setattr(runtime, "resolve_opencode", lambda: "/bin/true")
    monkeypatch.setattr(runtime.subprocess, "Popen", lambda *a, **k: _FakeProc(rc=None))
    monkeypatch.setattr(runtime.OpencodeServer, "_wait_healthy", fake_wait)

    with pytest.raises(runtime.AgentCallError):
        srv.start()
    assert calls["n"] == 3, f"expected exactly port_attempts tries, got {calls['n']}"


def test_the_opaque_serve_error_is_recognised_as_a_port_conflict():
    """`opencode` never says "address in use" -- it prints a bare `ServeError` and exits rc=1.

    If the discriminator only looked for the conventional strings, the retry would never fire on the
    one failure that actually happened, twice, on box 4.
    """
    from kernel_optimizer.agents.runtime import _looks_like_port_conflict

    assert _looks_like_port_conflict(Exception("opencode serve exited rc=1: ServeError"))
    assert _looks_like_port_conflict(Exception("opencode serve exited rc=1: EADDRINUSE"))
    assert _looks_like_port_conflict(Exception("opencode serve exited rc=1: address already in use"))
    # Not retryable: a different failure, and one no other port would fix.
    assert not _looks_like_port_conflict(Exception("opencode serve exited rc=127: not found"))
    # Not a startup failure at all -- a health-check timeout must not be retried as a collision.
    assert not _looks_like_port_conflict(Exception("opencode serve did not become healthy"))


def test_the_shipped_default_actually_retries(tmp_path, monkeypatch):
    """`port_attempts` must default above 1, on the config a run really loads.

    Every other retry test here passes `port_attempts` explicitly, so all of them would still pass
    with the shipped default set back to 1 -- i.e. with the retry switched off for every real run.
    The revert check caught exactly that.
    """
    from kernel_optimizer.agents import runtime
    from kernel_optimizer.config import OpencodeConfig

    assert OpencodeConfig().port_attempts > 1, (
        "the shipped default does not retry, so a co-resident launch still loses the race")

    # And prove the default is what `start` uses, rather than trusting the number.
    srv = _server(tmp_path)                     # no port_attempts override
    calls = {"n": 0}

    def fake_wait(self):
        calls["n"] += 1
        if calls["n"] == 1:
            self.proc = _FakeProc(rc=1)
            raise runtime.AgentCallError("opencode serve exited rc=1: ServeError")

    monkeypatch.setattr(runtime, "resolve_opencode", lambda: "/bin/true")
    monkeypatch.setattr(runtime.subprocess, "Popen", lambda *a, **k: _FakeProc(rc=None))
    monkeypatch.setattr(runtime.OpencodeServer, "_wait_healthy", fake_wait)
    srv.start()
    assert calls["n"] == 2, "the default config did not retry a collision"


def test_a_health_check_timeout_is_not_retried_as_a_collision(tmp_path, monkeypatch):
    """A server that started but never became healthy must not be retried on a new port.

    That failure means something else is wrong (a broken provider config, a hung server), and
    retrying it costs another full `startup_timeout_s` per attempt while hiding the real cause. The
    discriminator requires "exited rc=" precisely so a timeout does not qualify.

    THE MESSAGE HERE IS COPIED FROM `_wait_healthy`, not invented. An earlier version of this test
    wrote its own plausible wording ("did not become healthy"), which contains neither "exited rc="
    nor "serveerror" -- so it passed no matter what the discriminator did, and the revert check
    correctly reported the guard as unguarded. The real deadline message is
    `f"opencode server not healthy at {self.base_url}: {last_err}"`, and the nastier case is that
    `last_err` is an arbitrary exception whose text could itself contain "ServeError" -- e.g. the
    server logged one, then hung instead of exiting. That must still not be retried, because the
    process never exited and so nothing released a port.
    """
    from kernel_optimizer.agents import runtime

    real_deadline_msg = (
        "opencode server not healthy at http://127.0.0.1:4096: "
        "ConnectError('ServeError while connecting')")

    srv = _server(tmp_path, port_attempts=3)
    calls = {"n": 0}

    def fake_wait(self):
        calls["n"] += 1
        # Still running, never healthy -- the process did NOT exit, so no rc is reported.
        raise runtime.AgentCallError(real_deadline_msg)

    monkeypatch.setattr(runtime, "resolve_opencode", lambda: "/bin/true")
    monkeypatch.setattr(runtime.subprocess, "Popen", lambda *a, **k: _FakeProc(rc=None))
    monkeypatch.setattr(runtime.OpencodeServer, "_wait_healthy", fake_wait)

    with pytest.raises(runtime.AgentCallError):
        srv.start()
    assert calls["n"] == 1, (
        "a health-check timeout was retried as a port collision, costing a full startup timeout "
        "per attempt and burying the real cause")
    # And the discriminator itself, on the verbatim message, so the reason is pinned independently
    # of `start`'s control flow.
    assert not runtime._looks_like_port_conflict(Exception(real_deadline_msg)), (
        "the deadline message is being read as a port conflict; note it can carry 'ServeError' "
        "inside `last_err` without the process ever having exited")


def test_the_retry_path_reaps_the_failed_server(tmp_path, monkeypatch):
    """Between attempts, a server left running would be an orphan.

    Driven through `start` rather than by calling `_reap_failed_proc` directly, because the test
    below it already covers the helper in isolation -- and a helper that is never CALLED on the retry
    path protects nothing.
    """
    from kernel_optimizer.agents import runtime

    srv = _server(tmp_path, port_attempts=3)
    procs: list[_FakeProc] = []

    def fake_popen(*a, **k):
        p = _FakeProc(rc=None)               # still running when the attempt fails
        procs.append(p)
        return p

    calls = {"n": 0}

    def fake_wait(self):
        calls["n"] += 1
        if calls["n"] == 1:
            raise runtime.AgentCallError("opencode serve exited rc=1: ServeError")

    monkeypatch.setattr(runtime, "resolve_opencode", lambda: "/bin/true")
    monkeypatch.setattr(runtime.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(runtime.OpencodeServer, "_wait_healthy", fake_wait)

    srv.start()
    assert len(procs) == 2, "expected one failed attempt and one successful one"
    assert procs[0].terminated, (
        "the first attempt's server was left running when the retry started -- an orphaned "
        "`opencode serve`, which this project has already paid for once")


def test_a_still_running_failed_server_is_reaped_before_the_retry(tmp_path):
    """A server left running between attempts would be an orphan holding the log handle.

    This project has already paid for orphaned `opencode serve` processes, so the retry path must not
    create a new way to make them.
    """
    srv = _server(tmp_path)
    proc = _FakeProc(rc=None)            # still running
    srv.proc = proc
    srv._reap_failed_proc()
    assert proc.terminated, "a running failed server was left behind before the retry"
    assert srv.proc is None, "the dead handle was kept, so stop() would signal the wrong process"

