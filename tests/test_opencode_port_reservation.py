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
    `subprocess.Popen` call in `start`.
    """
    import inspect

    from kernel_optimizer.agents import runtime

    src = inspect.getsource(runtime.OpencodeServer.start)
    close_at = src.find("reservation.close()")
    popen_at = src.find("subprocess.Popen")
    assert close_at != -1, "the reservation is never released; the server could never bind"
    assert popen_at != -1, "this test is anchored on the wrong call"
    assert close_at < popen_at, (
        "the reservation is still held when the server is spawned, so the server cannot bind it")
