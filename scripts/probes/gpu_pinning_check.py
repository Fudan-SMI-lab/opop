"""Are the two arms actually on DIFFERENT GPUs, or is one arm's timing polluting the other's?

WHY THIS IS URGENT IF WRONG. The pair's whole claim rests on the arms sharing a denominator without
sharing a device: same box, one calibration.json, but one RTX 4090 each. If both arms landed on GPU1,
every latency in both arms is contaminated by the neighbour's SM/L2/bandwidth contention -- MPS and
time-slicing make concurrent timing mutually polluting, which is why the harness times exclusively
within an arm. That would not be a defect to note; it would invalidate the comparison and require a
restart.

WHY ONE nvidia-smi SNAPSHOT CANNOT ANSWER IT. GPU work here is a one-shot subprocess per job, so an arm
between jobs shows zero processes and zero memory. A snapshot showing "GPU0 idle" is equally consistent
with "the treatment arm is on GPU0 and between jobs" and with "the treatment arm is on GPU1 too". The
discriminator is which GPU each arm's PROCESSES have been seen on over time, tied back to the arm by the
process's own environment (CUDA_VISIBLE_DEVICES) and cwd/cmdline, not by which GPU happens to be busy.

HOW IT IDENTIFIES AN ARM. Each orchestrator was launched with CUDA_VISIBLE_DEVICES set and a distinct
XDG_DATA_HOME (/root/autodl-tmp/xdg-s7t vs xdg-s7c), both readable from /proc/<pid>/environ. That is the
authoritative tie: the launch command's own environment, not an inference from GPU utilisation.

Note on CUDA_VISIBLE_DEVICES and nvidia-smi: inside a process with CVD=1, torch sees ONE device numbered
0, but nvidia-smi reports the PHYSICAL index. So a worker of the control arm shows up on physical GPU1
while believing it is on cuda:0. Comparing the arm's CVD against the physical index the process appears
on is therefore the check -- not comparing torch's device number against anything.

Run on box4:  /root/autodl-tmp/orch-venv/bin/python /root/probe-clean/gpu_pinning_check.py
"""

from __future__ import annotations

import os
import re
import subprocess
import time
from pathlib import Path


def _sh(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True).stdout.decode("utf-8", "replace")
    except OSError as exc:
        return "ERROR: %s" % exc


def _environ(pid: str) -> dict[str, str]:
    try:
        raw = Path("/proc/%s/environ" % pid).read_bytes()
    except OSError:
        return {}
    out = {}
    for item in raw.split(b"\0"):
        if b"=" in item:
            k, v = item.split(b"=", 1)
            out[k.decode("utf-8", "replace")] = v.decode("utf-8", "replace")
    return out


def _cmdline(pid: str) -> str:
    try:
        return Path("/proc/%s/cmdline" % pid).read_bytes().replace(b"\0", b" ").decode(
            "utf-8", "replace").strip()
    except OSError:
        return ""


def _arm_of(env: dict[str, str], cmd: str) -> str:
    """Which arm does this process belong to? XDG_DATA_HOME is the launch-time discriminator; the
    config path in the cmdline is the fallback for a child that did not inherit it.

    Returns "gone" when /proc gave us nothing at all. A GPU job here is a one-shot subprocess, so
    nvidia-smi can list a pid that has already exited by the time /proc is read -- both environ and
    cmdline then come back empty. That is a RACE, not an unattributable process, and conflating the two
    made this probe print "the check is incomplete" on a run that had in fact confirmed separation. An
    unattributable process that really exists is a different and much more serious thing (another
    tenant, or an arm launched without CUDA_VISIBLE_DEVICES), so the two must not share a bucket.
    """
    if not env and not cmd:
        return "gone"
    xdg = env.get("XDG_DATA_HOME", "")
    if "s7t" in xdg:
        return "treatment"
    if "s7c" in xdg:
        return "control"
    if "treatment" in cmd or "s7-treatment" in cmd:
        return "treatment"
    if "control" in cmd or "s7-control" in cmd:
        return "control"
    return "?"


def main() -> int:
    print("PHYSICAL GPUs:")
    print(_sh(["nvidia-smi", "--query-gpu=index,uuid,utilization.gpu,memory.used",
               "--format=csv,noheader"]).strip())
    uuid_to_index = {}
    for line in _sh(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"]).splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) == 2:
            uuid_to_index[parts[1]] = parts[0]

    # Sample repeatedly: a one-shot job is invisible between jobs.
    seen: dict[str, set[str]] = {}          # arm -> physical GPU indices its processes appeared on
    claimed: dict[str, set[str]] = {}       # arm -> CUDA_VISIBLE_DEVICES values it was launched with
    detail: list[tuple[str, str, str, str]] = []
    SAMPLES, GAP = 24, 5.0
    print()
    print("sampling compute processes for %.0f s ..." % (SAMPLES * GAP))
    for _ in range(SAMPLES):
        out = _sh(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_memory",
                   "--format=csv,noheader"])
        for line in out.splitlines():
            parts = [x.strip() for x in line.split(",")]
            if len(parts) < 2 or not parts[0].isdigit():
                continue
            pid, uuid = parts[0], parts[1]
            idx = uuid_to_index.get(uuid, "?")
            env = _environ(pid)
            cmd = _cmdline(pid)
            arm = _arm_of(env, cmd)
            cvd = env.get("CUDA_VISIBLE_DEVICES", "(unset)")
            seen.setdefault(arm, set()).add(idx)
            claimed.setdefault(arm, set()).add(cvd)
            key = (arm, idx, cvd, cmd[:70])
            if key not in detail:
                detail.append(key)
        time.sleep(GAP)

    print()
    print("%-11s %-14s %-22s %s" % ("arm", "physical GPU", "CUDA_VISIBLE_DEVICES", "cmdline (trunc)"))
    for arm, idx, cvd, cmd in detail:
        print("%-11s %-14s %-22s %s" % (arm, idx, cvd, cmd or "(pid already exited)"))

    print()
    if not detail:
        print("NO COMPUTE PROCESSES SEEN IN THE WHOLE WINDOW. That is possible -- both arms can be")
        print("inside agent calls, which use no GPU -- but it means this run of the probe ANSWERS")
        print("NOTHING. Re-run it; do not read the silence as 'the arms are separated'.")
        return 1

    bad = False
    gone = seen.pop("gone", set())
    if gone:
        print("(%d sample(s) caught a pid that had already exited -- a one-shot GPU job finishing "
              "between nvidia-smi and /proc. Not attributable, and not a finding.)"
              % len(gone))
    for arm in sorted(seen):
        if arm == "?":
            print("UNIDENTIFIED but LIVE process(es) on GPU %s -- readable in /proc yet tied to no arm."
                  % ", ".join(sorted(seen[arm])))
            print("  This is the serious case: another tenant, or an arm launched without")
            print("  CUDA_VISIBLE_DEVICES. Look at the cmdlines above before trusting the verdict.")
            bad = True
            continue
        print("%-11s appeared on physical GPU %-8s launched with CUDA_VISIBLE_DEVICES=%s"
              % (arm, ",".join(sorted(seen[arm])), ",".join(sorted(claimed[arm]))))
        if len(seen[arm]) > 1:
            print("  THIS ARM'S WORK APPEARED ON MORE THAN ONE PHYSICAL GPU.")
            bad = True

    overlap = set()
    arms = [a for a in seen if a != "?"]
    for i, a in enumerate(arms):
        for b in arms[i + 1:]:
            overlap |= (seen[a] & seen[b])
    if overlap:
        print()
        print("*** BOTH ARMS SHARE PHYSICAL GPU %s. Concurrent timing on one device is mutually"
              % ", ".join(sorted(overlap)))
        print("*** polluting (time-slicing shares SMs, L2 and bandwidth), so every latency in BOTH")
        print("*** arms is contaminated. This invalidates the comparison -- stop and restart, do not")
        print("*** note it as a caveat.")
        return 2
    if len(arms) < 2:
        print()
        print("Only %d arm was seen doing GPU work in this window, so SEPARATION IS NOT YET"
              % len(arms))
        print("CONFIRMED -- the other arm may have been inside an agent call. Re-run to confirm.")
        return 1
    print()
    print("SEPARATED: each arm's GPU work appeared only on its own physical device.")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
