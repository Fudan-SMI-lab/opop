"""Is an in-flight agent call working, or burning its deadline? Look in the sandbox.

WHY. events.jsonl goes quiet for the whole duration of an agent call, so a call at 8 minutes and
a call that will still be there at 30 look identical in the log. The deadline (1800 s) says only
when the harness gives up, not whether anything is happening -- and this project has measured
both failure modes: an agent writing an exhaustive self-check that ate the full 1800 s, and a
per-READ idle timeout that let a call sit for 4057 s against a 1500 s setting.

The sandbox is the evidence. A working agent edits files; the newest mtime under its sandbox
directory moves. Sampling that twice, a minute apart, separates "working" from "idle" without
guessing from the log.

THE ONE WAY THIS MISLEADS, and it bit on the first live run: a call that FINISHES during the
sampling window leaves a sandbox that stops moving, which looks exactly like idle. So the event
log is read before AND after the wait, and a call that closed in between is reported as done
rather than as suspicious. Reading the log once, before a 60 s sample, is how I nearly reported a
stall on a parameterizer that had already written AGENT_CALL_FINISHED.

A quiet sandbox is also normal for a few minutes after the agent's last edit while the harness
reaps the output and the witness gate runs on the GPU. `parameterized.py` written 8 minutes ago
with the call still open is not by itself a fault.

DOES NOT kill anything. A call inside its deadline is the harness's business, not a probe's.
"""
from __future__ import annotations

import os
import sys
import time


def _newest(root: str) -> tuple[float, str, int]:
    """(newest mtime, its path, file count) under root. 0.0 if nothing readable."""
    best, best_p, n = 0.0, "", 0
    for dirpath, dirnames, filenames in os.walk(root):
        # Skip caches that churn on their own: a moving .git or __pycache__ mtime would read as
        # the agent working when nothing it produces has changed.
        dirnames[:] = [d for d in dirnames
                       if d not in (".git", "__pycache__", "node_modules", ".venv")]
        for fn in filenames:
            p = os.path.join(dirpath, fn)
            try:
                m = os.path.getmtime(p)
            except OSError:
                continue
            n += 1
            if m > best:
                best, best_p = m, p
    return best, best_p, n


def _open_calls(events_path: str) -> tuple[int, dict[str, float]]:
    """(event count, {call_id: start ts}) for calls with no terminal event yet.

    Read BEFORE and AFTER the sampling window. Without this the probe reports "nothing changed"
    for a call that simply FINISHED during the wait -- the sandbox stops moving because the work
    is done, which is the opposite of the idle it looks like. That happened on the first live run
    of this probe: the log had already recorded AGENT_CALL_FINISHED before the 60 s sample began.
    """
    import json
    n = 0
    open_calls: dict[str, float] = {}
    try:
        with open(events_path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                n += 1
                try:
                    e = json.loads(line)
                except ValueError:
                    continue
                p = e.get("payload") or {}
                cid = p.get("call_id")
                if not cid:
                    continue
                t = e.get("type")
                if t == "AGENT_CALL_STARTED":
                    open_calls[str(cid)] = float(e.get("ts") or 0.0)
                elif t == "AGENT_CALL_FINISHED" or (t == "AGENT_CALL_FAILED" and p.get("final")):
                    open_calls.pop(str(cid), None)
    except OSError:
        return -1, {}
    return n, open_calls


def sample(sandbox_root: str, wait_s: float = 60.0) -> int:
    if not os.path.isdir(sandbox_root):
        print("NO SUCH DIRECTORY: %s" % sandbox_root)
        return 2
    events_path = os.path.join(os.path.dirname(sandbox_root.rstrip(os.sep)), "events.jsonl")
    n0, open0 = _open_calls(events_path)
    # Newest sandboxes first -- the in-flight call's is among them.
    subs = []
    for name in os.listdir(sandbox_root):
        p = os.path.join(sandbox_root, name)
        if os.path.isdir(p):
            try:
                subs.append((os.path.getmtime(p), name, p))
            except OSError:
                pass
    subs.sort(reverse=True)
    if not subs:
        print("no sandboxes under %s" % sandbox_root)
        return 2
    watch = subs[:3]
    now = time.time()
    first = {}
    print("BEFORE (now = %s)" % time.strftime("%H:%M:%S"))
    for mt, name, p in watch:
        m, mp, n = _newest(p)
        first[name] = (m, n)
        print("  %-34s %4d files  newest %6.1f s ago  %s"
              % (name[:34], n, now - m if m else -1, os.path.basename(mp)))
    time.sleep(wait_s)
    now = time.time()
    print("AFTER %.0f s" % wait_s)
    moved = False
    for mt, name, p in watch:
        m, mp, n = _newest(p)
        pm, pn = first[name]
        d_files = n - pn
        changed = (m > pm) or (d_files != 0)
        moved = moved or changed
        print("  %-34s %4d files (%+d)  newest %6.1f s ago  %s"
              % (name[:34], n, d_files, now - m if m else -1,
                 "CHANGED" if changed else "unchanged"))
    print()
    n1, open1 = _open_calls(events_path)
    if n0 >= 0:
        closed = set(open0) - set(open1)
        if closed:
            print("THE CALL FINISHED DURING THE WAIT: %s" % ", ".join(sorted(closed)))
            print("So a quiet sandbox here means the work is DONE, not idle. The log advanced")
            print("%d -> %d events. Nothing is wrong." % (n0, n1))
            return 0
        if n1 > n0:
            print("The log advanced %d -> %d events during the wait, so the run is progressing"
                  % (n0, n1))
        elif open1:
            print("Still in flight after the wait: %s (log unchanged at %d events)"
                  % (", ".join(sorted(open1)), n1))
        else:
            # Say this rather than nothing. Silence here would leave "the log was read and no
            # call is open" indistinguishable from "the log was never read", and the second
            # would mean every conclusion below is unguarded.
            print("Log read (%d events), no agent call in flight -- so a quiet sandbox says"
                  % n1)
            print("nothing about an agent. Whatever is not moving here is not a stalled call.")
    else:
        print("COULD NOT READ %s -- the finished-during-the-wait check did NOT run, so a quiet"
              % events_path)
        print("sandbox below may simply be a call that completed. Do not read it as idle.")
    if moved:
        print("SOMETHING IS BEING WRITTEN -- the agent is working, not idle. A long call is a")
        print("COST (it burns the deadline) but not a stall.")
    else:
        print("NOTHING CHANGED in %.0f s. That is consistent with an idle call, but it is NOT" % wait_s)
        print("proof: an agent can spend minutes thinking before its next edit, and a finished")
        print("agent's sandbox also stops moving while the harness reaps it and the witness gate")
        print("runs. Re-sample before concluding, and remember the read timeout is per-READ idle")
        print("(measured 4057 s against a 1500 s setting), so the harness may not rescue this.")
    return 0


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    root = args[0] if args else ("/root/autodl-tmp/opop-workspace/opop-glm/runs-v3/s7-treatment/"
                                 "run-l3-43-20260913-202332/sandboxes")
    wait = float(args[1]) if len(args) > 1 else 60.0
    raise SystemExit(sample(root, wait))
