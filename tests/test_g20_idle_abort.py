"""G20: silence ends a call; duration does not. And the threshold is DERIVED, not written in.

THE MEASUREMENT. Across 9 rewriter calls in finished L3 runs: median 18.9 min, mean 19.0, range
9.5-30.0, quartiles 13.3 / 18.9 / 25.0, and 33% ran into the transport ceiling. Any A/B over agent
behaviour therefore costs ~2 hours for 6 calls, while per-trial noise reaches 16% of the mean -- so
wall clock, not budget, is what caps the sample size.

WHY RAISING THE CEILING IS THE WRONG FIX. The recorded root cause of the ceiling-hitting calls is an
agent writing its own 1500+-compile exhaustive self-check and consuming the whole budget with zero
output. Doubling the ceiling doubles the waste, and with max_transport_retries=2 one hung call
already costs three attempts. It also makes each sample slower, which is the opposite of what an
experiment short on samples needs.

THE FIX. Keep "duration alone never ends a call" -- an agent compiling and benchmarking is working --
and add its converse: a call that has produced NOTHING for a while is not thinking. The productivity
signal is the agent sandbox's file tree, which changes constantly while an agent works.

THE THRESHOLD IS A FRACTION OF THE TRANSPORT CEILING, never an absolute interval, so the two cannot
drift apart. These tests pin that specifically, because a hardcoded "20 minutes" is exactly what a
future edit would reach for.
"""
from __future__ import annotations

import time
from pathlib import Path

from kernel_optimizer.agents.runtime import OpencodeClient, _output_fingerprint


def _client(timeout_s: float, frac: float = 0.5) -> OpencodeClient:
    # No server is contacted: only the derived threshold and the fingerprint are exercised.
    return OpencodeClient("http://127.0.0.1:1", timeout_s=timeout_s, idle_abort_frac=frac)


def test_the_idle_threshold_tracks_the_configured_timeout():
    """The whole point: change the ceiling and the silence tolerance moves with it."""
    a = _client(1500.0)
    b = _client(3000.0)
    assert b.idle_abort_s == 2 * a.idle_abort_s, (
        "the idle threshold did not scale with request_timeout_s (%s vs %s), so it is effectively "
        "hardcoded and will become vacuous or trigger-happy as soon as the ceiling is changed"
        % (a.idle_abort_s, b.idle_abort_s))
    assert a.idle_abort_s == 750.0, a.idle_abort_s

    # It must also stay BELOW the ceiling, or it can never fire before the transport gives up --
    # which would make the whole mechanism dead code.
    for t in (600.0, 1500.0, 1800.0, 3600.0):
        c = _client(t)
        assert c.idle_abort_s < t, (
            "at timeout_s=%s the idle threshold (%s) is not below the ceiling, so it can never "
            "fire and the fix is inert" % (t, c.idle_abort_s))

    # A different fraction must be honoured, and a floor must prevent an absurdly small value on a
    # short ceiling (a 10s silence window would abort healthy calls).
    assert _client(1500.0, frac=0.25).idle_abort_s == 375.0
    assert _client(60.0, frac=0.01).idle_abort_s >= 60.0, (
        "no floor on the derived threshold: a tiny fraction of a short ceiling would abort agents "
        "that are working normally")


def test_disabling_the_check_is_possible():
    """0.0 must switch it off, so a box where the signal misleads can opt out without a code edit."""
    c = _client(1500.0, frac=0.0)
    assert c.idle_abort_frac == 0.0


def test_the_productivity_signal_changes_when_an_agent_writes(tmp_path):
    """A fingerprint that does not move while files appear would abort working agents."""
    d = tmp_path / "sandbox"
    (d / "sub").mkdir(parents=True)
    empty = _output_fingerprint(d)
    assert empty == (0, 0, 0.0), empty

    (d / "sub" / "kernel.py").write_text("x = 1", encoding="utf-8")
    one = _output_fingerprint(d)
    assert one != empty, "writing a file did not change the fingerprint"
    assert one[0] == 1 and one[1] > 0, one

    # Growing an existing file must count as progress too: an agent appending to a log or rewriting
    # a kernel in place is working, and a count-only signal would call that silence.
    (d / "sub" / "kernel.py").write_text("x = 1\ny = 2\n" * 100, encoding="utf-8")
    two = _output_fingerprint(d)
    assert two != one, (
        "growing a file did not change the fingerprint, so an agent editing in place would be "
        "aborted as hung")
    assert two[0] == one[0], "file count changed unexpectedly"

    # And a quiet tree must be STABLE across polls, or the check could never fire.
    assert _output_fingerprint(d) == two, (
        "the fingerprint of an unchanged tree differs between calls, so it can never detect "
        "silence")


def test_an_unreadable_sandbox_is_not_treated_as_hung(tmp_path):
    """'Cannot tell' must never be grounds to abort.

    None is returned for a missing or unreadable directory, and for a call given no directory at
    all. If that were treated as silence, every such call would be cut -- a diagnostic failure
    presented as an agent failure, the inversion these fixes exist to prevent.
    """
    assert _output_fingerprint(None) is None, "no directory must read as 'cannot tell', not as idle"
    missing = _output_fingerprint(tmp_path / "does-not-exist")
    assert missing in (None, (0, 0, 0.0)), missing


def test_the_watchdog_checks_productivity_and_says_why():
    """The abort reason must name the cause, or a killed call looks like a transport fault."""
    import inspect

    src = inspect.getsource(OpencodeClient.prompt)
    assert "_output_fingerprint" in src, (
        "the watchdog does not sample the sandbox, so a hung call still burns the full ceiling")
    assert "idle_abort_s" in src, "the watchdog does not use the derived threshold"
    assert "no new output for" in src, (
        "the abort reason does not say the call went silent; without that, an idle-abort is "
        "indistinguishable from the ReadTimeout it replaces")
    # The memory check must survive alongside it: it stops a call that is about to take the box
    # down, which is a different condition from silence and must not be replaced by it.
    assert "memory_abort_frac" in src, "the memory watchdog was lost while adding the idle check"
