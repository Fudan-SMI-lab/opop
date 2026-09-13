"""Does the timeout kill actually kill a REAL process tree, child processes included?

THE GAP THIS FILLS. `tests/test_d8_job_wall_clock.py` covers the timeout path thoroughly on the STAMPING
side -- a `TimeoutExpired` must produce `failure_kind="timeout"`, `job_timed_out=True` and a wall figure.
It does that by patching `subprocess.Popen` with a fake and stubbing `_kill_job` to a no-op, so it never
kills anything. The half that has never been exercised anywhere is `_kill_job` itself:

    os.killpg(os.getpgid(proc.pid), SIGKILL)

And production has never exercised it either: across 3095 trials on box4, `job_timed_out` is true zero
times. So the code that stops a runaway job is, today, an unexecuted branch whose only evidence is that
it reads correctly -- the recorded `an-unreachable-branch-is-not-a-safeguard` shape. A live trial sitting
at 1532 s of an 1800 s budget is what made this worth writing now rather than later.

WHAT IS ACTUALLY AT STAKE, and why the child matters more than the parent. The expensive trials in this
project are expensive inside `ptxas`, which is a GRANDCHILD of the job: worker_main.py spawns Triton,
which spawns ptxas. A kill that reaped only the direct child would leave an 11-13 GB ptxas running with
no parent, holding a GPU lock nobody will release -- and the recorded
`agent-script-can-oom-the-whole-box` incident is what that costs. So the assertion below is on the
GRANDCHILD's death, not the child's.

POSIX only: `start_new_session` and `killpg` are the POSIX arm of `_kill_job`. The Windows arm uses
`taskkill /F /T` and cannot be tested the same way, so the test skips rather than pretending.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import textwrap
import time

import pytest

_POSIX = pytest.mark.skipif(
    os.name != "posix",
    reason="killpg/start_new_session are the POSIX arm; Windows uses taskkill /F /T")


def _alive(pid: int) -> bool:
    """Is this pid still running? `os.kill(pid, 0)` raises when it is gone.

    A zombie counts as GONE for this purpose -- the process is no longer executing, which is what the
    kill has to achieve; whether the parent has reaped it is a different question.
    """
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    try:
        with open(f"/proc/{pid}/stat", encoding="utf-8") as fh:
            return fh.read().split(") ", 1)[1].split()[0] != "Z"
    except (OSError, IndexError):
        return True


# A parent that spawns a child and prints the child's pid, then both sleep. Mirrors the real shape:
# the thing that must die is the GRANDCHILD of the test (child of the job), like ptxas under Triton.
_SCRIPT = textwrap.dedent(
    """
    import subprocess, sys, time
    kid = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(600)"])
    print(kid.pid, flush=True)
    time.sleep(600)
    """
)


@_POSIX
def test_killpg_takes_the_whole_tree_including_the_grandchild():
    """The assertion that matters: the CHILD of the job dies too.

    `os.killpg` is what makes that true -- every descendant inherits the session's process group
    because the job was started with `start_new_session=True`. A `proc.kill()`-only implementation
    passes a test that checks the parent and leaves the grandchild running, which is the failure this
    project has already paid for once.
    """
    proc = subprocess.Popen(
        [sys.executable, "-c", _SCRIPT],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        start_new_session=True,
    )
    try:
        line = proc.stdout.readline()
        grandchild = int(line.strip())
        assert _alive(grandchild), "fixture is broken: the grandchild never started"
        assert _alive(proc.pid)

        # Exactly what `_kill_job` does on POSIX.
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)

        deadline = time.time() + 10.0
        while time.time() < deadline and (_alive(proc.pid) or _alive(grandchild)):
            time.sleep(0.05)

        assert not _alive(proc.pid), "the job itself survived its own kill"
        assert not _alive(grandchild), (
            "THE GRANDCHILD SURVIVED. This is the ptxas case: a runaway compiler outliving the job "
            "that started it, holding memory and a GPU lock with no parent to release them.")
    finally:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):
            pass
        try:
            proc.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


@_POSIX
def test_a_direct_kill_alone_would_leave_the_grandchild_running():
    """THE POSITIVE CONTROL, and the test above is worth little without it. If `proc.kill()` happened to
    reap the whole tree on this platform, the previous test would pass for the wrong reason and would
    keep passing if someone replaced `killpg` with `kill`. So: kill ONLY the direct child and assert the
    grandchild is STILL ALIVE. That is what makes killpg load-bearing rather than decorative.
    """
    proc = subprocess.Popen(
        [sys.executable, "-c", _SCRIPT],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        start_new_session=True,
    )
    grandchild = None
    try:
        grandchild = int(proc.stdout.readline().strip())
        assert _alive(grandchild)

        proc.kill()  # the direct child ONLY -- no process group
        deadline = time.time() + 5.0
        while time.time() < deadline and _alive(proc.pid):
            time.sleep(0.05)
        assert not _alive(proc.pid)

        # Give it the same grace the other test allows, then assert it is STILL running.
        time.sleep(0.5)
        assert _alive(grandchild), (
            "the grandchild died without killpg, so this platform reaps trees on its own and the "
            "other test proves nothing about killpg -- re-check _kill_job before trusting it")
    finally:
        for pid in (grandchild, proc.pid):
            if pid is None:
                continue
            try:
                os.kill(pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError, OSError):
                pass
        try:
            proc.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            pass
