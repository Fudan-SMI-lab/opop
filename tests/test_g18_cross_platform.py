"""G18: one worker-client / runtime that works on both topologies.

The Linux port lived only as uncommitted files on one box, so a fresh clone could not run on any
Linux machine. Merging it back exposed that most of the diff was NOT platform-specific: three of the
changes were bugs on Windows too, stranded on one machine because they arrived inside a "Linux port".

These tests pin the three cross-platform fixes and the two genuine platform branches. Each states
what breaks without it, because a test whose failure message does not name the defect gets deleted
by the next person who sees it go red.
"""
from __future__ import annotations

import inspect
import os
import subprocess
from pathlib import Path

from kernel_optimizer.config import GpuConcurrencyConfig, WslConfig
from kernel_optimizer.gpu import worker_client as wc


def _worker(tmp_path: Path, max_shared: int = 2) -> wc.WslGpuWorker:
    cfg = WslConfig(
        distro="Ubuntu",
        venv="~/kernel-opt-venv",
        kernelbench_src="~/kb/src",
        triton_cache_dir="~/.triton-cache",
    )
    conc = GpuConcurrencyConfig(enabled=True, max_shared_jobs=max_shared)
    return wc.WslGpuWorker(cfg, conc, tmp_path / "jobs")


def test_tilde_is_expanded_on_both_platforms(tmp_path):
    """`~` in venv / cache / PYTHONPATH must be expanded by US, not by a shell.

    The config defaults use `~`. Under the old string command `bash -lc` expanded it, so the bug was
    invisible on the WSL route; nothing expands it on the native route, and a literal `./~/...`
    interpreter path fails EVERY job as worker_crash. Since the expansion is correct on both, it is
    done unconditionally.
    """
    w = _worker(tmp_path)
    argv, env = w._build_command(tmp_path / "j.json", tmp_path / "j.out.json")
    blob = " ".join(argv) + " " + " ".join(f"{k}={v}" for k, v in env.items()
                                          if k in ("TRITON_CACHE_DIR", "PYTHONPATH"))
    assert "~" not in blob, (
        "a literal ~ survived into the worker command or its environment, so the interpreter and "
        "cache paths do not exist and every job would fail as worker_crash: %s" % blob)
    assert os.path.expanduser("~") in blob, "the expansion produced no home-relative path at all"


def test_a_timeout_kills_only_its_own_job(tmp_path):
    """Killing must target one job's tree, never every worker on the box.

    `pkill -f worker_main.py` matched all of them: with max_shared_jobs=2, a timeout in one
    shared-lane job also killed the OTHER, healthy job, which then reported worker_crash ("no result
    file") -- a real failure attributed to the wrong candidate. Wrong on both platforms.
    """
    # Assert the BEHAVIOUR, not the absence of a word. Both docstrings legitimately name `pkill`
    # while explaining why it was removed, and a text search flags the explanation as the defect it
    # documents -- a source-text assertion that fails on correct code. So: record what the kill
    # actually invokes, and require that it targets this process alone.
    calls: list[list[str]] = []

    class _FakeProc:
        pid = 424242

        def kill(self):
            calls.append(["proc.kill"])

    w = _worker(tmp_path)
    real_run = subprocess.run
    real_killpg = getattr(os, "killpg", None)
    real_getpgid = getattr(os, "getpgid", None)
    try:
        subprocess.run = lambda argv, **kw: calls.append(list(argv))  # type: ignore[assignment]
        if real_killpg is not None:
            # getpgid must be faked too: the pid is invented, so the real call raises
            # ProcessLookupError and the code correctly falls back to proc.kill() -- which would
            # make this test assert the FALLBACK path while believing it tested the primary one.
            os.getpgid = lambda pid: pid
            os.killpg = lambda pgid, sig: calls.append(["killpg", str(pgid), str(sig)])
        w._kill_job(_FakeProc())
    finally:
        subprocess.run = real_run  # type: ignore[assignment]
        if real_killpg is not None:
            os.killpg = real_killpg
        if real_getpgid is not None:
            os.getpgid = real_getpgid

    flat = " ".join(" ".join(c) for c in calls)
    assert flat, "the kill path invoked nothing at all, so a timed-out worker keeps running"
    assert "pkill" not in flat, (
        "the kill invoked pkill, which matches EVERY worker on the box: a timeout in one "
        "shared-lane job would also kill the other, healthy job, and that job's candidate would be "
        "blamed for a worker_crash it did not cause. Invoked: %s" % flat)
    assert str(_FakeProc.pid) in flat, (
        "the kill is not addressed by this job's own pid, so it cannot be job-scoped: %s" % flat)
    assert hasattr(wc.WslGpuWorker, "_kill_job"), "the per-job kill entry point is gone"

    kill_src = inspect.getsource(wc.WslGpuWorker._kill_job)
    # The kill must be addressed by THIS process's identity, not by a name pattern.
    assert "proc.pid" in kill_src, (
        "the kill is not addressed by the job's own pid, so it cannot be job-scoped")
    if os.name == "nt":
        assert "/T" in kill_src, (
            "taskkill without /T leaves the distro shell's child (the real worker) running")
    else:
        assert "killpg" in kill_src, (
            "no process-group kill on POSIX, so the worker survives its own timeout")


def test_the_native_route_needs_no_shell_and_survives_spaces(tmp_path):
    """On the native route the command is an argv list, so a path with a space cannot break it."""
    if os.name == "nt":
        return  # the WSL route must build a shell string; covered by the next test
    w = _worker(tmp_path / "dir with space")
    argv, env = w._build_command(tmp_path / "j.json", tmp_path / "j.out.json")
    assert isinstance(argv, list) and argv, "argv is not a list"
    assert "wsl.exe" not in argv[0], "the native route is invoking wsl.exe"
    assert argv[0].endswith("python"), "argv[0] is not the venv interpreter: %s" % argv[0]
    # The two variables must reach the child through its environment, since there is no shell to
    # apply a `VAR=x cmd` prefix.
    assert env.get("PYTHONPATH"), "PYTHONPATH is not in the child environment"
    assert env.get("TRITON_CACHE_DIR"), "TRITON_CACHE_DIR is not in the child environment"
    assert "--job" in argv and "--out" in argv, "the worker's own flags are missing"


def test_the_wsl_route_still_wraps_and_quotes(tmp_path):
    """On Windows the command must cross into the distro, with its arguments quoted for bash."""
    if os.name != "nt":
        # Exercise the quoting helper directly, so this half is still covered on Linux CI.
        assert wc._shq("/plain/path") == "/plain/path"
        assert wc._shq("/has a space") == "'/has a space'"
        assert wc._shq("it's") == "'it'\"'\"'s'", wc._shq("it's")
        return
    w = _worker(tmp_path)
    argv, _env = w._build_command(tmp_path / "j.json", tmp_path / "j.out.json")
    assert argv[0] == "wsl.exe" and "-lc" in argv, (
        "the WSL route no longer wraps the command, so the Linux interpreter is being exec'd "
        "directly from Windows")
    inner = argv[-1]
    assert "TRITON_CACHE_DIR=" in inner and "PYTHONPATH=" in inner, (
        "the variables are not in the shell command; they cannot cross into the distro through "
        "the Windows process environment")


def test_opencode_is_resolved_and_the_shell_flag_is_platform_correct():
    """shell=True is required on Windows and is a BUG on POSIX.

    With a LIST argv, `shell=True` on POSIX runs `/bin/sh -c "opencode"` and passes the rest as
    $0,$1,... -- so `--hostname` and `--port` are silently discarded. The server binds its default
    port, the health poll hits the intended port, and the timeout reads like a network fault.
    """
    from kernel_optimizer.agents import runtime as rt

    assert rt._shell_for_opencode() == (os.name == "nt"), (
        "the shell flag does not follow the platform: on POSIX it discards --hostname/--port, on "
        "Windows its absence cannot exec the .cmd shim")

    # The resolver must report both the candidates and PATH on failure -- the two facts that
    # separate "not installed" from "installed where this shell cannot see it".
    src = inspect.getsource(rt.resolve_opencode)
    assert "PATH=" in src, "the failure message does not include PATH, so a PATH problem reads as "\
                           "a missing install"
    assert "which" in src, "no PATH lookup at all"


def test_server_stop_targets_the_whole_tree():
    """A stop that kills only the direct child leaks a server and holds a port per run."""
    from kernel_optimizer.agents import runtime as rt

    src = inspect.getsource(rt.OpencodeServer.stop)
    if os.name == "nt":
        assert "/T" in src, "taskkill without /T leaves the shim's child server running"
    else:
        assert "killpg" in src, "no process-group signal, so the server is orphaned on stop"
        assert "SIGTERM" in src, (
            "SIGKILL without a SIGTERM first can leave opencode's sqlite session store mid-write")


def test_doctor_does_not_demand_wsl_on_a_native_box():
    """The distro probe must be conditional, or a healthy Linux box reports a failed check."""
    from kernel_optimizer import cli

    src = inspect.getsource(cli.cmd_doctor)
    assert "_wsl_hop_needed" in src, (
        "doctor checks for a WSL distro unconditionally; on a native-Linux box there is no distro "
        "to find and the check fails for a box that is fine")
    assert "nvidia-smi" in src, (
        "doctor never checks the GPU driver, which is the actual precondition on both topologies")
