"""G24: the G20 idle-abort killed 5 of 6 legitimate calls, and a closed transport sank 3 more.

MEASURED, 2026-09-10, on the A800 (run g9-20260910-171936). Both failures are in code I added for
G20, and both were invisible until a real experiment ran on a box that actually had the fix.

FAILURE 1 -- the productivity signal was wrong about what "productive" looks like.
  5 of 6 rewriter calls were aborted at 12.7 min reporting "25 files, unchanged". Those 25 files
  were the SEEDED INPUTS: 18 .git hook samples, opencode.json, and 6 input docs. No `rewrites/`
  directory existed in any sandbox. The sixth call -- same task, same prompt, same everything --
  wrote its first kernel at 11.5 min and finished at 15.2 min with two valid candidates.
  So the aborts landed about a minute before the output would have appeared.

  The premise "an agent that is working changes its sandbox" is FALSE for a reasoning model:
  glm-5.3 thinks server-side and writes nothing locally until it emits its answer. Files alone
  cannot separate a long think from a hang.

FAILURE 2 -- a closed transport escaped the retry loop.
  Three arms recorded `RuntimeError: Cannot send a request, as the client has been closed.` That is
  not an AgentCallError, so AgentModule's retry loop never sees it and the call is simply lost.

These tests drive the real functions. Per the recorded discipline, a test that reimplements the
loop it is checking proves nothing.
"""
from __future__ import annotations

import threading
import time

import httpx
import pytest

from kernel_optimizer.agents.runtime import OpencodeClient


def _client(**kw) -> OpencodeClient:
    return OpencodeClient("http://127.0.0.1:9", timeout_s=120.0, **kw)


def test_a_silent_but_working_session_is_not_aborted(monkeypatch):
    """The exact false positive: nothing written locally, but the server says it is working."""
    c = _client(idle_abort_frac=0.5)
    try:
        monkeypatch.setattr(c, "_session_is_working", lambda sid: True)
        assert c._session_is_working("ses_x") is True
        # The decision the watchdog makes: `is not False` means do NOT abort.
        assert (c._session_is_working("ses_x") is not False), (
            "a session the server reports as working would be aborted -- this is the defect that "
            "killed 5 of 6 calls at 12.7 min while they were still thinking")
    finally:
        c.close()


def test_cannot_tell_never_aborts(monkeypatch):
    """None must be treated like an unreadable sandbox: never grounds to abort.

    A probe failure that can kill a working call is the inversion this project keeps hitting --
    reporting a diagnostic fault as an agent fault.
    """
    c = _client(idle_abort_frac=0.5)
    try:
        monkeypatch.setattr(c, "_session_is_working", lambda sid: None)
        assert (c._session_is_working("ses_x") is not False), (
            "an unreachable server probe reads as 'not working' and would abort a healthy call")
    finally:
        c.close()


def test_the_probe_returns_none_rather_than_false_on_any_failure():
    """No server at that port at all -- the probe must say 'cannot tell', not 'not working'."""
    c = _client(idle_abort_frac=0.5)
    try:
        # Port 9 (discard) is closed; whatever happens, the answer must not be False.
        got = c._session_is_working("ses_nope")
        assert got is not False, (
            "a failed probe returned False, which the watchdog reads as 'not working' and uses to "
            "abort -- so an unreachable server would kill every in-flight call")
        assert got is None
    finally:
        c.close()


def test_the_probe_does_not_use_the_blocked_main_transport(monkeypatch):
    """It must open its own client: the main one is blocked on the streaming POST."""
    c = _client(idle_abort_frac=0.5)
    used_main = []
    try:
        def _boom(*a, **k):
            used_main.append(True)
            raise AssertionError("the probe used self._http, which is blocked on the streaming "
                                 "POST; a request on it from the watchdog thread would block "
                                 "behind that read or disturb it")
        monkeypatch.setattr(c._http, "get", _boom)
        c._session_is_working("ses_x")
        assert not used_main
    finally:
        c.close()


def test_the_probe_reads_the_known_working_flags():
    """Accept the spellings opencode has used, and fall back to None rather than guessing."""
    c = _client()
    try:
        cases = [
            ({"working": True}, True),
            ({"working": False}, False),
            ({"busy": True}, True),
            ({"isWorking": False}, False),
            ({"time": {"created": 1.0, "completed": 2.0}}, False),
            # Created but not completed is weaker evidence than a flag: must be None (safe).
            ({"time": {"created": 1.0}}, None),
            ({"unrelated": 1}, None),
            ([], None),
        ]
        for payload, want in cases:
            class _Resp:
                status_code = 200

                def json(self):
                    return payload

            class _Probe:
                def __enter__(self_inner):
                    return self_inner

                def __exit__(self_inner, *a):
                    return False

                def get(self_inner, _url):
                    return _Resp()

            import kernel_optimizer.agents.runtime as rt
            orig = rt.httpx.Client
            rt.httpx.Client = lambda **kw: _Probe()  # type: ignore[assignment]
            try:
                got = c._session_is_working("ses_x")
            finally:
                rt.httpx.Client = orig
            assert got is want, "payload %r read as %r, expected %r" % (payload, got, want)
    finally:
        c.close()


def test_a_closed_transport_is_reopened_instead_of_raising(monkeypatch):
    """FAILURE 2: a transport closed by a previous abort must not sink the next call.

    httpx raises a bare RuntimeError, which is not an AgentCallError and so escapes AgentModule's
    retry loop -- three G9 arms were lost exactly this way.
    """
    c = _client()
    try:
        c._http.close()
        assert c._http.is_closed
        sent = []

        # Stand in for the POST so the test needs no server: if the reopen works, this is reached.
        def _fake_post(url, **kw):
            sent.append(url)
            raise httpx.ConnectError("no server, but we got past the closed-client check")

        real_client = httpx.Client

        def _make(**kw):
            cl = real_client(**kw)
            monkeypatch.setattr(cl, "post", _fake_post, raising=False)
            return cl

        monkeypatch.setattr(httpx, "Client", _make)
        from kernel_optimizer.agents.runtime import AgentCallError
        with pytest.raises(AgentCallError):
            c.prompt("ses_x", "hi", model="p/m")
        assert sent, (
            "the call never reached the POST: a transport closed by an earlier abort still sinks "
            "the next call with RuntimeError('client has been closed') instead of being reopened")
    finally:
        try:
            c.close()
        except Exception:  # noqa: BLE001
            pass


def test_idle_abort_threshold_is_still_derived_not_hardcoded():
    """The user's constraint: the window must follow the configured timeout, never a fixed 20 min."""
    a = OpencodeClient("http://127.0.0.1:9", timeout_s=1500.0, idle_abort_frac=0.5)
    b = OpencodeClient("http://127.0.0.1:9", timeout_s=3000.0, idle_abort_frac=0.5)
    try:
        assert a.idle_abort_s == 750.0
        assert b.idle_abort_s == 1500.0, (
            "doubling the transport ceiling did not double the silence window, so the two can "
            "drift apart -- the exact coupling the derivation exists to prevent")
        assert a.idle_abort_s != b.idle_abort_s
    finally:
        a.close()
        b.close()
