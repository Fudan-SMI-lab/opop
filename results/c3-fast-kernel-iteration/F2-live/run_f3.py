"""Run real development epochs on the current code until DEV_GO: seal F3, then develop.

CONTEXT -- WHAT THE PROBES ESTABLISHED
--------------------------------------
Six falsification probes ran before this, each on the real GPU, and every reconstructable cause of
the `cuda_names=[]` rejection was ruled out:

  B1  opencode server running vs not      -> byte-identical reads, `cuda_names` populated both ways.
  B2  F0 kernel bytes vs F1 kernel bytes  -> BOTH pass; the agent-written kernel was never at fault.
  B3  HelperService RPC, in-process and   -> BOTH pass, including a real `search_helper` child
      via the real child process              process connecting while the parent profiles.
  B4  several proves through ONE evaluator-> 3/3 passed (its other reads were refused by the
                                             params-choices gate and the budget, not by the device).
  B5  a fresh profiler per session        -> 5/5 fresh sessions collected CUDA activity.
  B6  reading events() inside vs outside  -> inside the context: 0 events, 12/12 sessions. A real
      the profiler context                   defect, but NOT this one: prove() already read outside
                                             and the failing reads carried 11-12 CPU events.
  B7  each layer of prove()'s context     -> all six arms collected in every session.
      stack, added one at a time

What remains is an asynchronous activity flush that sometimes misses its window, so `prove` now
executes-and-profiles up to `trace_attempts` times and records every attempt in `trace_attempts`.
This run is where that gets exercised against a live agent.

NO ARTIFICIAL CEILINGS
----------------------
The earlier protocol capped this lane at D+5400 seconds, 6 agent calls, 12 local evaluations, 2
source versions per epoch and 3 epochs. None of those decided whether a measurement is CORRECT --
they decided how much experimenting was allowed -- so they are no longer imposed (`BudgetSpec`
leaves `limits` unset, `develop`/`run_probe` take `max_epochs=None`/`max_versions=None`). Counters
are still kept, because the published table has to report what it actually spent.

What is NOT relaxed: device proof, quality limits, held-out isolation, and selecting a row before
measuring it. `--deadline-hours` exists only so a hung transport cannot wedge the box forever.
"""

import json
import os
import sys
import time
import traceback
from pathlib import Path

ROOT = Path("/root/autodl-tmp/c3-fast-kernel-iteration")
DEV = ROOT / "development-996318d"
OUT = Path(sys.argv[1]).resolve()
CODE = Path(sys.argv[2]).resolve()
REVISION = sys.argv[3]
CONFIG = Path(sys.argv[4])
# Wall-clock ceiling for the whole lane, in hours. Generous by default: it is a watchdog against a
# wedged transport, not a research budget.
HOURS = float(sys.argv[5]) if len(sys.argv) > 5 else 12.0

from kernel_optimizer.config import AppConfig, load_config

from examples.c3_qwen3.device_records import StageIdentity
from examples.c3_qwen3.fast_artifacts import framework_id, identify_framework
from examples.c3_qwen3.fast_budget import BudgetSpec, StageBudget
from examples.c3_qwen3.fast_cli import execute_development
from examples.c3_qwen3.fast_prepare import FastAccess, prepare_fast
from examples.c3_qwen3.fast_records import TargetSpec
from examples.c3_qwen3.fast_target import build_target
from examples.c3_qwen3.fast_runtime import RunInputs
from examples.c3_qwen3.fast_workflow import scoped_config

OUT.mkdir(parents=True, exist_ok=False)
record: dict = {"epoch": "F3-live-agent-unbounded", "started_unix_s": time.time(), "code_revision": REVISION}
phase = "config"
try:
    # Own copy of the approved config, with the one field that cannot accommodate a real rewriter
    # call raised to the project default. Recorded explicitly so the change is auditable.
    source = load_config(CONFIG)
    record["timeout_change"] = {"file_value": source.opencode.request_timeout_s,
                                "used": AppConfig().opencode.request_timeout_s,
                                "reason": "480s is below the measured rewriter median (519.2/736.2s)"}
    source.opencode.request_timeout_s = AppConfig().opencode.request_timeout_s
    # PER-PROCESS ISOLATION, so two lanes can share one machine.
    #
    # The approved config pins `port` with `port_attempts: 1` (fail-fast), names ONE
    # `CUDA_VISIBLE_DEVICES`, and points the triton/XDG caches at one directory. All three are
    # single-occupancy: a second process on the same box would lose the port race and exit with a
    # bare `ServeError`, time its kernels against the other lane's GPU work, and race it in the
    # compile cache. Each is overridable here so a lane can be given its own.
    #
    # `C3_AGENT_PORT` also raises `port_attempts` to 3: if two lanes are started together the
    # loser must fall back to an ephemeral port rather than dying, which is the behaviour every
    # other multi-arm launch in this project relies on.
    port = os.environ.get("C3_AGENT_PORT")
    if port:
        source.opencode.port = int(port)
        source.opencode.port_attempts = max(3, source.opencode.port_attempts)
    device_index = os.environ.get("C3_CUDA_INDEX")
    runtime_root = os.environ.get("C3_RUNTIME_ROOT")
    env = dict(source.opencode.server_env or {})
    if device_index:
        env["CUDA_VISIBLE_DEVICES"] = device_index
    if runtime_root:
        # Every cache the agent's own compiles touch, moved under this lane's own root. Sharing
        # `TRITON_CACHE_DIR` between two live lanes means one lane's compile can be served from the
        # other's entry, which is a correctness hazard and not merely a speed one.
        for key, sub in (("TRITON_CACHE_DIR", "triton-cache"), ("TORCH_EXTENSIONS_DIR", "torch-extensions"),
                         ("TORCH_HOME", "torch-home"), ("CUDA_CACHE_PATH", "cuda-cache"),
                         ("XDG_DATA_HOME", "xdg-data"), ("XDG_CACHE_HOME", "xdg-cache"),
                         ("XDG_STATE_HOME", "xdg-state"), ("XDG_RUNTIME_DIR", "xdg-runtime"),
                         ("TMPDIR", "tmp"), ("TMP", "tmp"), ("TEMP", "tmp")):
            path = Path(runtime_root) / sub
            path.mkdir(parents=True, exist_ok=True)
            env[key] = str(path)
        # XDG_CONFIG_HOME is deliberately NOT moved: it is where the opencode provider config
        # (and therefore the GLM credential) lives, and it is read-only to the run.
    source.opencode.server_env = env
    config_path = OUT / "agent-config.json"
    config_path.write_text(source.model_dump_json(indent=2), encoding="utf-8")
    config_path.chmod(0o600)
    cfg = scoped_config(load_config(config_path))
    record["config"] = {"request_timeout_s": cfg.opencode.request_timeout_s,
                        "idle_abort_s": max(60.0, cfg.opencode.request_timeout_s * cfg.opencode.idle_abort_frac),
                        "strip_cuda_env_from_server": cfg.opencode.strip_cuda_env_from_server,
                        "model": cfg.agents.default_model,
                        "port": cfg.opencode.port, "port_attempts": cfg.opencode.port_attempts,
                        "server_cuda_visible_devices": (cfg.opencode.server_env or {}).get("CUDA_VISIBLE_DEVICES"),
                        "runtime_root": runtime_root,
                        "rewriter_retries": (cfg.agents.rewriter.max_retries,
                                             cfg.agents.rewriter.max_transport_retries)}
    print(json.dumps({"event": "CONFIG_SCOPED", **record["config"]}), flush=True)

    phase = "seal"
    prepared = prepare_fast(ROOT / "setup/contract.json", ROOT / "setup/preflight.json",
                            FastAccess(profile="c3_fast_device", phase="development"))
    frame = identify_framework(REVISION, cfg, prepared)
    frame_path = OUT / "framework-F3.json"
    frame_path.write_text(frame.model_dump_json(indent=2), encoding="utf-8")
    fid = framework_id(frame)
    record["framework_id"] = fid
    print(json.dumps({"event": "SEALED", "framework_id": fid, "revision": REVISION}), flush=True)

    phase = "inputs"
    # The calibration oracle is reused: it is a frozen artifact of the setup stage, and rebuilding it
    # would spend capture budget to arrive at the same bytes.
    #
    # The TARGET is reused only when this lane is developing on the same site the setup stage picked.
    # `C3_SITE_MODULE` selects a different one, and `C3_ALLOW_COMPOSITE=1` additionally admits a
    # WRAPPER module so the replacement subsumes its subtree and a candidate may fuse across operator
    # KINDS. Rebuilding from a profile is CPU work and spends no profile budget. The A/A noise is NOT
    # carried across: it was measured on the setup site's fixture, so this lane measures its own
    # (`require_target(noise=True)` refuses a candidate until it has).
    original_run = RunInputs.model_validate_json((DEV / "run.json").read_bytes())
    assert original_run.oracle_refs is not None, "calibration oracle path required"
    site_module = os.environ.get("C3_SITE_MODULE")
    if site_module:
        profile_name = os.environ.get("C3_DEV_PROFILE",
                                      "short-device-profile-17cb1c7131914914bcb7bf55bb0fd2e8.json")
        target = build_target(ROOT / "setup-native-ad0694a/raw" / profile_name, site_module,
                              allow_composite=os.environ.get("C3_ALLOW_COMPOSITE", "") == "1")
        record["site"] = {"module": site_module, "profile": profile_name,
                          "allow_composite": os.environ.get("C3_ALLOW_COMPOSITE", "") == "1",
                          "site_id": target.brief.site_id,
                          "modules": len(target.brief.module_paths),
                          "associated_cuda_us": target.associated_cuda_us}
        print(json.dumps({"event": "SITE_REBUILT", **record["site"]}), flush=True)
    else:
        target = TargetSpec.model_validate_json((DEV / "target.json").read_bytes())
    target_path = OUT / "target.json"
    target_path.write_text(target.model_dump_json(indent=2), encoding="utf-8")

    D = time.time()
    deadline = D + HOURS * 3600
    (OUT / "D-clock.json").write_text(json.dumps({"D": D, "deadline_unix_s": deadline,
                                                  "hours": HOURS}, indent=2), encoding="utf-8")
    stage = StageIdentity(profile="c3_fast_device", stage="development", framework_id=fid)
    # limits unset => every operation uncapped; enforce_fixed_window unset => any deadline allowed.
    StageBudget(OUT / "dev-budget.json", BudgetSpec(lane="development", stage=stage, goal=target.goal,
        started_unix_s=D, deadline_unix_s=deadline))

    inputs = RunInputs(contract=ROOT / "setup/contract.json", assets_manifest=ROOT / "setup/preflight.json",
        agent_config=config_path, framework=frame_path, target=target_path, output=OUT / "F3",
        budget=OUT / "dev-budget.json", oracle_refs=original_run.oracle_refs,
        development_state=OUT / "dev-state.json", started_unix_s=D, deadline_unix_s=deadline,
        goal_text=original_run.goal_text, device="cuda:0")
    (OUT / "run.json").write_text(inputs.model_dump_json(indent=2), encoding="utf-8")
    print(json.dumps({"event": "F3_START", "D": D, "deadline": deadline, "hours": HOURS,
                      "timeout_s": cfg.opencode.request_timeout_s, "goal": target.goal}), flush=True)

    phase = "develop"
    result = execute_development(inputs)
    record["status"] = result.status
    record["result"] = str(OUT / "F3/result.json")
    # Surface the device evidence immediately, including every trace attempt, so a rejection can be
    # read as "collection was flaky" or "the kernel never ran" without reopening the receipts.
    evidence = [{"version": c.version, "status": c.status, "error": c.error,
                 "kernels": [{"kernel_name": e.kernel_name, "launches": e.launches,
                              "cuda_names": list(e.cuda_names), "compiled": e.compiled,
                              "trace_events": e.trace_events, "trace_cuda_events": e.trace_cuda_events,
                              "trace_sample": list(e.trace_sample),
                              "attempts": [a.model_dump(mode="json") for a in e.trace_attempts]}
                             for r in c.locals for f in r.fixtures for e in f.device_evidence]}
                for c in result.candidates]
    record["candidates"] = evidence
    print(json.dumps({"event": "F3_RESULT", "status": result.status, "candidates": evidence}), flush=True)
except BaseException as exc:  # noqa: BLE001
    record.update(status="BLOCKED", failure_stage=phase, error=f"{type(exc).__name__}: {exc}",
                  traceback=traceback.format_exc())
    traceback.print_exc()
finally:
    record["finished_unix_s"] = time.time()
    try:
        record["budget"] = json.loads((OUT / "dev-budget.json").read_text())["used"]
    except BaseException:  # noqa: BLE001
        pass
    (OUT / "f3-record.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps({"event": "F3_DONE", "status": record.get("status"),
                      "failure_stage": record.get("failure_stage"), "error": record.get("error"),
                      "budget": record.get("budget")}), flush=True)
