"""The formal campaign: three fresh goal lanes under one frozen F*, then the 4x3x3 final matrix.

This is the step that produces the published table. It runs only after a development epoch reached
DEV_GO, and it never revisits that decision: `freeze_framework` re-verifies the accepted candidate
with `prove_candidate` before it will write F*, and each lane re-derives its own kernel from the
ORIGINAL baseline rather than inheriting the development winner.

CLOCKS -- A WATCHDOG, NOT A RESEARCH BUDGET
-------------------------------------------
One shared campaign clock S governs all three lanes. The original plan pinned the search cutoff to
exactly S+9000s and the final deadline to S+10800s; those particular durations were a schedule, not
a correctness property, so they are now arguments. What `SearchClock` still enforces -- and what the
science actually depends on -- is the ORDERING: every lane must finish searching before the first
held-out block is measured, so no row can be re-chosen in response to its held-out score.

Each lane is clipped by the common cutoff for the same reason.

WHAT EACH LANE CAN DO
---------------------
Uncapped. `BudgetSpec.limits` is left unset, so a lane draws candidates until one is ACCEPTED or its
clock runs out; the counters still record what it spent. A lane's acceptance test is unchanged --
device proof, local correctness, full-model quality, then a confirmed repeat measurement -- so more
draws buy more chances, never a weaker standard. A lane that still accepts nothing leaves its row as
an explicit baseline fallback, and that is recorded as the real result it is.

THE MATRIX
----------
All four rows (baseline plus the three goal rows) are measured against all three goals, three blocks
each -- 36 cells -- on the sealed held-out corpus. The off-diagonal cells are the point: they are
what distinguishes a goal-specific optimisation from a general speedup any objective would have
found. Rows are frozen before the first block is measured.
"""

import json
import os
import sys
import time
import traceback
from pathlib import Path

ROOT = Path("/root/autodl-tmp/c3-fast-kernel-iteration")
DEV_EPOCH = Path(sys.argv[1]).resolve()   # the development output root that reached DEV_GO
OUT = Path(sys.argv[2]).resolve()
CODE = Path(sys.argv[3]).resolve()
# Optional C2 regression receipt. Absent is fine: C2 isolation is structural (every C3 relaxation is
# opt-in), so the presence of a file was never what protected it.
COMPATIBILITY = Path(sys.argv[4]) if len(sys.argv) > 4 and sys.argv[4] not in ("", "-") else None
# Hours for the search phase and for the held-out measurement phase after it. Generous defaults.
SEARCH_HOURS = float(sys.argv[5]) if len(sys.argv) > 5 else 6.0
FINAL_HOURS = float(sys.argv[6]) if len(sys.argv) > 6 else 4.0

from kernel_optimizer.config import load_config

from examples.c3_qwen3.device_records import StageIdentity
from examples.c3_qwen3.fast_artifacts import framework_id
from examples.c3_qwen3.fast_budget import BudgetSpec, StageBudget
from examples.c3_qwen3.fast_formal import FormalInputs, formal_lane_summary, freeze, run_formal_goal
from examples.c3_qwen3.fast_matrix import MatrixInputs, run_matrix
from examples.c3_qwen3.fast_prepare import FastAccess, prepare_fast
from examples.c3_qwen3.fast_records import Framework, ProbeResult, TargetSpec
from examples.c3_qwen3.fast_runtime import RunInputs
from examples.c3_qwen3.fast_sites import build_selected_target, select_sites
from examples.c3_qwen3.fast_target import require_target
from examples.c3_qwen3.fast_workflow import scoped_config
from examples.c3_qwen3.search_records import SearchClock

GOALS = ("ttft", "single", "multi")
# One native short CUDA profile per goal, produced during setup. Building a TargetSpec from an
# existing profile is CPU work and costs no profile budget; only new profiling would.
PROFILES = {
    "ttft": "short-device-profile-17cb1c7131914914bcb7bf55bb0fd2e8.json",
    "single": "short-device-profile-3c0828b8c34445c5b3eadbc4039a1b74.json",
    "multi": "short-device-profile-d558213f2f1d4467a4643ed187afda86.json",
}
# THE SITE IS NOW CHOSEN PER GOAL FROM MEASUREMENT, NOT DECLARED.
#
# `C3_SITE_MODULE` used to name one module path for all three goals, defaulting to a single RMSNorm
# instance. That made the search space a hand-picked guess and, worse, the SAME guess under three
# different objectives -- while the goals' own profiles disagree about which site is largest (the
# composite MLP carries 146,793 us for ttft against 17,258 us for single). `fast_sites.select_sites`
# ranks every site each goal's profile makes eligible by its measured associated CUDA time and takes
# the top one, then makes the three choices mutually registrable.
#
# Setting `C3_SITE_MODULE` still forces one path for every goal, which is how an earlier campaign is
# reproduced; the ranking it overrode is recorded either way.
SITE_MODULE = os.environ.get("C3_SITE_MODULE") or None
# Opt-in, off by default. A composite site is a WRAPPER module (one with observed children), so the
# replacement subsumes its whole subtree and a candidate may fuse across operator KINDS instead of only
# across one family's repeated stages. `build_target` still refuses a subtree that receives the KV
# cache, because such a site could never be captured as a fixture and so could never carry a device
# proof.
ALLOW_COMPOSITE = os.environ.get("C3_ALLOW_COMPOSITE", "") == "1"
# Which goals this process runs. The three lanes are independent observations -- each re-derives its
# own kernel from the ORIGINAL baseline -- so they may be split across machines and merged for the
# matrix afterwards. The matrix inherits each row's device proof by bundle/params/framework hash, not
# by device identity, so a row proven on one box is measurable on another.
LANE_GOALS = tuple(g for g in (os.environ.get("C3_LANES") or ",".join(GOALS)).split(",") if g)
# Skip the matrix in this process. Set on a machine that runs only some lanes; the matrix is then run
# once, after the lanes are merged, by `run_split_matrix.py`.
SKIP_MATRIX = os.environ.get("C3_SKIP_MATRIX", "") == "1"


def development_result_path(root: Path) -> Path:
    """Find the epoch result inside a development output root.

    The subdirectory is named after the epoch that produced it (`F2/`, `F3/`, ...), so this looks for
    the result rather than assuming one epoch's name. A hardcoded `F2/result.json` silently pointed
    at nothing when the next epoch ran.
    """
    candidates = sorted(root.glob("F*/result.json"))
    if not candidates:
        raise SystemExit(f"no F*/result.json under {root}")
    if len(candidates) > 1:
        raise SystemExit(f"ambiguous development results under {root}: {[str(p) for p in candidates]}")
    return candidates[0]


def framework_path(root: Path) -> Path:
    candidates = sorted(root.glob("framework-F*.json"))
    if len(candidates) != 1:
        raise SystemExit(f"expected exactly one framework-F*.json under {root}, found {candidates}")
    return candidates[0]


OUT.mkdir(parents=True, exist_ok=False)
record: dict = {"campaign": "C3-formal-three-goals", "started_unix_s": time.time(),
                "development_epoch": str(DEV_EPOCH), "lanes": {},
                "hours": {"search": SEARCH_HOURS, "final": FINAL_HOURS}}
phase = "development_check"
try:
    development_result = development_result_path(DEV_EPOCH)
    probe = ProbeResult.model_validate_json(development_result.read_bytes())
    if probe.status != "DEV_GO":
        raise SystemExit(f"development epoch is {probe.status}, not DEV_GO; formal admission refused")
    frame = probe.framework
    fid = framework_id(frame)
    record["framework_id"] = fid
    record["development_result"] = str(development_result)
    config_path = DEV_EPOCH / "agent-config.json"
    # PER-PROCESS ISOLATION for the agent transport and the GPU, so two lane processes can share one
    # machine. The dev epoch's config pins one port with `port_attempts: 1`, one
    # `CUDA_VISIBLE_DEVICES`, and one set of compile caches; a second process inheriting all three
    # would lose the port race and exit, time its kernels against the other lane's GPU work, and
    # race it in the triton cache. When any override is set, this campaign writes its OWN copy of
    # the config and the lanes use that -- the dev epoch's file is never modified.
    #
    # This does NOT touch `framework_id`: that hashes `request_timeout_s`, the module set, the
    # rewriter prompt and the contract -- none of which change here. Ports and cache paths are not
    # part of the framework identity, and must not be, or two machines could never run one campaign.
    overrides = {k: os.environ.get(k) for k in ("C3_AGENT_PORT", "C3_CUDA_INDEX", "C3_RUNTIME_ROOT")}
    if any(overrides.values()):
        source_cfg = load_config(config_path)
        if overrides["C3_AGENT_PORT"]:
            source_cfg.opencode.port = int(overrides["C3_AGENT_PORT"])
            source_cfg.opencode.port_attempts = max(3, source_cfg.opencode.port_attempts)
        env = dict(source_cfg.opencode.server_env or {})
        if overrides["C3_CUDA_INDEX"]:
            env["CUDA_VISIBLE_DEVICES"] = overrides["C3_CUDA_INDEX"]
        if overrides["C3_RUNTIME_ROOT"]:
            for key, sub in (("TRITON_CACHE_DIR", "triton-cache"), ("TORCH_EXTENSIONS_DIR", "torch-extensions"),
                             ("TORCH_HOME", "torch-home"), ("CUDA_CACHE_PATH", "cuda-cache"),
                             ("XDG_DATA_HOME", "xdg-data"), ("XDG_CACHE_HOME", "xdg-cache"),
                             ("XDG_STATE_HOME", "xdg-state"), ("XDG_RUNTIME_DIR", "xdg-runtime"),
                             ("TMPDIR", "tmp"), ("TMP", "tmp"), ("TEMP", "tmp")):
                path = Path(overrides["C3_RUNTIME_ROOT"]) / sub
                path.mkdir(parents=True, exist_ok=True)
                env[key] = str(path)
            # XDG_CONFIG_HOME is left alone: it holds the opencode provider config (the GLM
            # credential), which is read-only to the run and shared on purpose.
        source_cfg.opencode.server_env = env
        config_path = OUT / "agent-config.json"
        config_path.write_text(source_cfg.model_dump_json(indent=2), encoding="utf-8")
        config_path.chmod(0o600)
        record["config_overrides"] = {k: v for k, v in overrides.items() if v}
    cfg = scoped_config(load_config(config_path))
    record["config"] = {"path": str(config_path), "request_timeout_s": cfg.opencode.request_timeout_s,
                        "port": cfg.opencode.port, "port_attempts": cfg.opencode.port_attempts,
                        "server_cuda_visible_devices": (cfg.opencode.server_env or {}).get("CUDA_VISIBLE_DEVICES")}
    print(json.dumps({"event": "CONFIG_SCOPED", **record["config"]}), flush=True)
    prepared = prepare_fast(ROOT / "setup/contract.json", ROOT / "setup/preflight.json",
                            FastAccess(profile="c3_fast_device", phase="development"))

    # Each goal gets a target built from ITS OWN profile. Re-labelling the TTFT target would be a
    # different task wearing the same name: the decode goals require two representatives
    # (steps 0 and 127) that a prefill-only profile cannot supply.
    phase = "targets"
    raw = ROOT / "setup-native-ad0694a/raw"
    targets: dict[str, TargetSpec] = {}
    source = TargetSpec.model_validate_json((DEV_EPOCH / "target.json").read_bytes())
    # Rank each goal's eligible sites from ITS OWN profile and take the best. `select_sites` also makes
    # the three choices mutually registrable: `BindingRuntime.register_site` refuses a module that
    # already belongs to another group, so two goals may share one site but may not hold overlapping
    # different ones -- and the matrix registers all of them in one binding.
    choices = select_sites({g: raw / PROFILES[g] for g in GOALS},
                           allow_composite=ALLOW_COMPOSITE, forced=SITE_MODULE)
    record["site_selection"] = {g: {"chosen": c.chosen.model_dump(mode="json"),
                                    "promoted_from": c.promoted_from,
                                    "forced": SITE_MODULE,
                                    "considered": [o.model_dump(mode="json") for o in c.considered[:8]]}
                                for g, c in choices.items()}
    print(json.dumps({"event": "SITES_SELECTED", "forced": SITE_MODULE,
                      "allow_composite": ALLOW_COMPOSITE,
                      "chosen": {g: {"module": c.chosen.module_path,
                                     "assoc_cuda_us": round(c.chosen.associated_cuda_us, 1),
                                     "modules": len(c.chosen.module_paths),
                                     "composite": c.chosen.composite,
                                     "promoted_from": c.promoted_from}
                                 for g, c in choices.items()}}), flush=True)
    for goal in GOALS:
        target = build_selected_target(raw / PROFILES[goal], choices[goal],
                                       allow_composite=ALLOW_COMPOSITE)
        if target.goal != goal:
            raise SystemExit(f"profile for {goal} declares goal {target.goal}")
        # Carry over the measured A/A noise from the development target, but ONLY for the goal it was
        # measured on: the noise floor is a property of that goal's fixture, and copying it across
        # goals would put a number next to a measurement it never described.
        #
        # It is also a property of THAT SITE's fixture, so the site has to match too. Without this
        # second condition, changing the site would silently attach the RMSNorm site's A/A noise to
        # an MLP site, which is a number measured on a different operator entirely. When the site
        # differs the noise is left unset and the lane measures its own (`measure_reference`), which is
        # what `require_target(noise=True)` demands before any candidate is admitted.
        if goal == source.goal and target.brief.site_id == source.brief.site_id:
            target = target.model_copy(update={"reference_noise_us": source.reference_noise_us,
                "noise_evidence": source.noise_evidence,
                "noise_evidence_sha256": source.noise_evidence_sha256})
        require_target(target, noise=False)
        path = OUT / f"target-{goal}.json"
        path.write_text(target.model_dump_json(indent=2), encoding="utf-8")
        targets[goal] = target
        record.setdefault("targets", {})[goal] = {"site_id": target.brief.site_id,
            "module": choices[goal].chosen.module_path,
            "representatives": [list(r) for r in target.representatives],
            "associated_cuda_us": target.associated_cuda_us,
            "modules": len(target.brief.module_paths),
            "noise_us": target.reference_noise_us}
    record["site"] = {"forced_module": SITE_MODULE, "allow_composite": ALLOW_COMPOSITE,
                      "per_goal": {g: choices[g].chosen.module_path for g in GOALS}}
    print(json.dumps({"event": "TARGETS_BUILT", "goals": list(targets),
                      "allow_composite": ALLOW_COMPOSITE,
                      "modules_per_site": {g: len(t.brief.module_paths) for g, t in targets.items()},
                      "associated_cuda_us": {g: round(t.associated_cuda_us, 1) for g, t in targets.items()},
                      "noise_carried_for": source.goal,
                      "sites": {g: t.brief.site_id[:20] for g, t in targets.items()}}), flush=True)

    phase = "freeze"
    frozen_path = OUT / "frozen-framework.json"
    frozen = freeze(development_result, targets, COMPATIBILITY,
                    ROOT / "setup/contract.json", ROOT / "setup/preflight.json", frozen_path)
    print(json.dumps({"event": "FROZEN", "framework_id": framework_id(frozen.framework),
                      "goals": sorted(frozen.targets),
                      "compatibility_receipt": str(COMPATIBILITY) if COMPATIBILITY else None}), flush=True)

    phase = "clock"
    S = time.time()
    cutoff = S + SEARCH_HOURS * 3600
    clock = SearchClock(campaign_started_unix_s=S, search_cutoff_unix_s=cutoff,
                        final_deadline_unix_s=cutoff + FINAL_HOURS * 3600)
    (OUT / "S-clock.json").write_text(clock.model_dump_json(indent=2), encoding="utf-8")
    record["clock"] = clock.model_dump(mode="json")
    original_run = RunInputs.model_validate_json((DEV_EPOCH / "run.json").read_bytes())
    dev_framework = framework_path(DEV_EPOCH)

    phase = "lanes"
    results: dict[str, Path] = {}
    # Only the goals assigned to this process, and each gets the WHOLE remaining window rather than a
    # share of it. When all three run here the shares are still needed -- one lane must not consume the
    # campaign and censor the others -- but when the lanes are split across machines, slicing a
    # single-lane process into thirds would throw away two thirds of its time for nothing.
    lanes = [g for g in GOALS if g in LANE_GOALS]
    record["lane_goals"] = lanes
    print(json.dumps({"event": "LANES_ASSIGNED", "goals": lanes, "skip_matrix": SKIP_MATRIX}), flush=True)
    for index, goal in enumerate(lanes):
        lane_start = time.time()
        if lane_start >= clock.search_cutoff_unix_s:
            record["lanes"][goal] = {"status": "censored", "reason": "common search cutoff reached"}
            print(json.dumps({"event": "LANE_CENSORED", "goal": goal}), flush=True)
            continue
        share = (clock.search_cutoff_unix_s - lane_start) / (len(lanes) - index)
        lane_deadline = min(lane_start + share, clock.search_cutoff_unix_s)
        lane_out = OUT / f"lane-{goal}"
        stage = StageIdentity(profile="c3_fast_device", stage="formal_search", framework_id=fid)
        # limits unset => the lane draws until acceptance or its clock expires.
        StageBudget(OUT / f"budget-{goal}.json", BudgetSpec(lane="formal_search", stage=stage,
            goal=goal, started_unix_s=lane_start, deadline_unix_s=lane_deadline))
        inputs = original_run.model_copy(update={
            "framework": dev_framework, "target": OUT / f"target-{goal}.json",
            # `agent_config` too: `original_run` names the DEV epoch's config, so without this a
            # per-lane port/GPU override would be written and then silently ignored.
            "agent_config": config_path,
            "output": lane_out, "budget": OUT / f"budget-{goal}.json",
            "development_state": None, "frozen_framework": frozen_path,
            "started_unix_s": lane_start, "deadline_unix_s": lane_deadline, "clock": clock})
        (OUT / f"run-{goal}.json").write_text(inputs.model_dump_json(indent=2), encoding="utf-8")
        print(json.dumps({"event": "LANE_START", "goal": goal, "deadline": lane_deadline,
                          "window_s": round(lane_deadline - lane_start),
                          "remaining_to_cutoff_s": round(clock.search_cutoff_unix_s - lane_start)}), flush=True)
        try:
            result = run_formal_goal(FormalInputs(inputs=inputs, frozen=frozen_path, clock=clock))
            results[goal] = lane_out / "formal-result.json"
            record["lanes"][goal] = {"status": result.status,
                "accepted": result.status == "accepted" and result.selected is not None,
                "draws": len(result.candidates),
                "candidate_status": [c.status for c in result.candidates],
                "result": str(results[goal])}
        except BaseException as exc:  # noqa: BLE001 - a failed lane is a real outcome, not a stop
            record["lanes"][goal] = {"status": "error", "error": f"{type(exc).__name__}: {exc}",
                                     "traceback": traceback.format_exc()[-2000:]}
            traceback.print_exc()
        (OUT / "campaign-record.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
        print(json.dumps({"event": "LANE_DONE", "goal": goal, **{k: v for k, v in
              record["lanes"][goal].items() if k != "traceback"}}), flush=True)

    phase = "lane_summary"
    record["selection"] = formal_lane_summary(frozen_path, {g: p for g, p in results.items()},
                                              OUT / "lane-selection.json")
    print(json.dumps({"event": "SELECTION_FROZEN", "accepted": record["selection"]["accepted_goals"]}), flush=True)

    if SKIP_MATRIX:
        # This process ran a SUBSET of the lanes. Measuring a matrix now would fill the goals it did
        # not run with baseline fallbacks and publish that as the table, which is exactly the way a
        # missing lane turns into a false "no improvement" row. The matrix runs once, after the lanes
        # from every machine are merged (`run_split_matrix.py`).
        #
        # Not `raise SystemExit(0)`: the `except BaseException` below would catch it and record this
        # successful lanes-only run as BLOCKED.
        record["status"] = "LANES_ONLY"
        print(json.dumps({"event": "MATRIX_SKIPPED", "reason": "lanes only; merge before the matrix",
                          "lanes": lanes, "accepted": record["selection"]["accepted_goals"]}), flush=True)
    else:
        phase = "matrix"
        matrix_start = time.time()
        if matrix_start >= clock.final_deadline_unix_s:
            raise SystemExit("final deadline passed before the matrix could start")
        report = run_matrix(MatrixInputs(contract=ROOT / "setup/contract.json",
            assets_manifest=ROOT / "setup/preflight.json", frozen_framework=frozen_path,
            formal_results={g: p for g, p in results.items()}, budget=OUT / "final-budget.json",
            output=OUT / "matrix", started_unix_s=matrix_start,
            deadline_unix_s=clock.final_deadline_unix_s, device="cuda:0"))
        record["matrix"] = {"complete": report.complete, "cells": len(report.cells),
            "valid": sum(1 for c in report.cells if c.status == "valid"),
            "device_proven_rows": sorted(r.row for r in report.rows if r.device_proven),
            "summary": report.summary}
        record["status"] = "COMPLETE" if report.complete else "PARTIAL"
        print(json.dumps({"event": "MATRIX_DONE", **record["matrix"]}, indent=2), flush=True)
except BaseException as exc:  # noqa: BLE001
    record.update(status="BLOCKED", failure_stage=phase, error=f"{type(exc).__name__}: {exc}",
                  traceback=traceback.format_exc())
    traceback.print_exc()
finally:
    record["finished_unix_s"] = time.time()
    (OUT / "campaign-record.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps({"event": "CAMPAIGN_DONE", "status": record.get("status"),
                      "failure_stage": record.get("failure_stage"), "error": record.get("error")}), flush=True)
