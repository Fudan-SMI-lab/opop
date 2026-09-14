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
import sys
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


def _arm_labels(argv: list[str]) -> list[tuple[str, str]]:
    """(label, discriminator) per arm, DERIVED from the run dirs passed on the command line.

    WHY NOT HARDCODED, which is what this was: the discriminators were the literals `s7t` / `s7c` and
    `treatment` / `control`. Pointed at the step-4 pair -- whose arms are `xdg-s4a` / `xdg-s4b` under
    `runs-v3/s4-c2off` and `s4-c2on` -- EVERY legitimate worker fell through to "?" and was reported as
    "UNIDENTIFIED but LIVE ... another tenant, or an arm launched without CUDA_VISIBLE_DEVICES", the
    probe's most serious verdict. Measured live: pid 643407 was arm B's own worker (ppid = arm B's
    orchestrator, CVD=0, XDG_DATA_HOME=/root/autodl-tmp/xdg-s4b) and was flagged as foreign.

    A false "another tenant is on your GPU" is worse than no check: the response to it is to stop and
    restart a 12 h pair. So the labels come from the arms actually being examined. The discriminator is
    the run dir's PARENT name (`s4-c2on`, `s7-treatment`), which is `run.runs_dir`'s last component --
    the same string that must differ per arm for the pair to be isolated at all, and which appears in
    the orchestrator's cmdline via its config path.
    """
    out: list[tuple[str, str]] = []
    for a in argv:
        p = Path(a)
        parent = p.parent.name or p.name
        out.append((parent, parent))
    return out


def _tokens(label: str) -> list[str]:
    """Substrings that identify an arm in an environ value or a cmdline.

    `s4-c2on` has to match `xdg-s4b`? No -- and that is the point: it must NOT guess. It matches the
    dir name itself (which is in the cmdline via `--config .../experiments_s4_c2on_box4gpu0.yaml`) and
    the same name with separators removed, so `s7-treatment` also matches `s7treatment`. Anything
    cleverer would be inventing a mapping between two independently chosen names.
    """
    lab = label.lower()
    toks = {lab, lab.replace("-", "_"), lab.replace("-", ""), lab.replace("_", "")}
    # A trailing/leading run-family prefix like `s4-` or `s7-` on its own is too weak to match on --
    # both arms share it -- so only the full name and its punctuation variants are used.
    return [t for t in toks if len(t) >= 4]


def _arm_of(env: dict[str, str], cmd: str, arms: list[tuple[str, str]]) -> str:
    """Which arm does this process belong to?

    Returns "gone" when /proc gave us nothing at all. A GPU job here is a one-shot subprocess, so
    nvidia-smi can list a pid that has already exited by the time /proc is read -- both environ and
    cmdline then come back empty. That is a RACE, not an unattributable process, and conflating the two
    made this probe print "the check is incomplete" on a run that had in fact confirmed separation. An
    unattributable process that really exists is a different and much more serious thing (another
    tenant, or an arm launched without CUDA_VISIBLE_DEVICES), so the two must not share a bucket.

    Attribution order: the process's own environment first (XDG_DATA_HOME, then any env value naming
    the arm), then its cmdline, then its ANCESTRY. The last is required, not optional: a worker is
    spawned as a subprocess and inherits XDG_DATA_HOME but its cmdline names only
    `kernel_optimizer/gpu/worker_main.py`, so a pair whose XDG names do not textually contain the arm
    name is attributable ONLY through the parent that carries `--config`.
    """
    if not env and not cmd:
        return "gone"
    hay_env = " ".join(env.values()).lower()
    for label, _ in arms:
        for tok in _tokens(label):
            if tok in hay_env or tok in cmd.lower():
                return label
    return "?"


def _arm_via_ancestry(pid: str, arms: list[tuple[str, str]]) -> str:
    """Walk /proc parents looking for one whose cmdline names an arm.

    The measured need for this: arm B's worker had CVD=0 and XDG_DATA_HOME=xdg-s4b, but neither string
    contains `s4-c2on`, so environ/cmdline matching cannot see it -- only its parent's
    `--config configs/experiments_s4_c2on_box4gpu0.yaml` can. Without this the probe reports its own
    healthy arms as foreign processes.
    """
    seen = 0
    cur = pid
    while cur and cur not in ("0", "1") and seen < 12:
        cmd = _cmdline(cur)
        for label, _ in arms:
            for tok in _tokens(label):
                if tok in cmd.lower():
                    return label
        try:
            status = Path("/proc/%s/status" % cur).read_text("utf-8", "replace")
        except OSError:
            return "?"
        nxt = ""
        for line in status.splitlines():
            if line.startswith("PPid:"):
                nxt = line.split()[1].strip()
                break
        cur, seen = nxt, seen + 1
    return "?"


def main() -> int:
    # The arms come from the command line. Previously they were baked in as `s7t`/`s7c`, so this probe
    # silently only worked for one pair; passing it any other pair turned its healthy arms into the
    # "another tenant" verdict. The two run dirs are what every caller already passes.
    argv = [a for a in sys.argv[1:] if not a.startswith("-")]
    arm_specs = _arm_labels(argv)
    if len(arm_specs) < 2:
        print("USAGE: gpu_pinning_check.py <arm1_run_dir> <arm2_run_dir>")
        print("The arm labels are derived from each run dir's PARENT name (e.g. s4-c2on, s7-treatment),")
        print("which is `run.runs_dir`'s last component -- the string that must differ per arm anyway.")
        print("Refusing to guess: a hardcoded label set is what made this probe report a pair's own")
        print("workers as foreign processes.")
        return 2
    print("arms under examination: %s" % ", ".join(lbl for lbl, _ in arm_specs))
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
            arm = _arm_of(env, cmd, arm_specs)
            if arm == "?":
                # A worker inherits XDG_DATA_HOME but its own cmdline names only worker_main.py, so
                # ancestry is the only attribution left. Tried second, not first, because the process's
                # own environment is the authoritative tie when it carries one.
                arm = _arm_via_ancestry(pid, arm_specs)
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
