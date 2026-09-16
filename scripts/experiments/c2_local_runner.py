# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["kernel-optimizer"]
# ///
# Run with the existing v5 environment: python -m scripts.experiments.c2_local_runner --help
"""Remote-operator CLI; prepare/summarize are CPU-only, other modes use real workers."""

import argparse
import json
import os
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from statistics import median
from time import monotonic
from typing import Literal

from kernel_optimizer.config import AppConfig, load_config
from kernel_optimizer.agents.runtime import AgentCallError
from kernel_optimizer.models.core import TrialRecord
from kernel_optimizer.store.run_store import RunStore
from kernel_optimizer.tasks.kernelbench import parse_task_arg
from kernel_optimizer.wiring import Runtime, load_task
from scripts.experiments.c2_local_adapter import GpuAdapter
from scripts.experiments.c2_local_agents import Services
from scripts.experiments.c2_local_execution import ArmResult, acquire, run_arm, save_result
from scripts.experiments.c2_local_costs import costs
from scripts.experiments.c2_local_inputs import InputError, Responses, RunOptions, Selection, Shared, protocol, stage_inputs


class Arguments(argparse.Namespace):
    def __init__(self) -> None:
        super().__init__()
        self.mode: Literal["prepare", "acquire", "smoke-parent", "run-arm", "summarize", "schema"] = "prepare"
        self.config: Path | None = None
        self.set: list[str] = []
        self.selection: Path = Path()
        self.output: Path = Path()
        self.shared: Path = Path()
        self.run_dir: Path = Path()
        self.probe_budget: int = 0
        self.arm: Literal["A", "B"] = "A"
        self.path: Literal["direct", "legacy"] = "direct"
        self.responses: Path | None = None
        self.final_blocks: int = 3
        self.results: list[Path] = []


def isolated_config(cfg: AppConfig, root: Path) -> AppConfig:
    copy = cfg.model_copy(deep=True)
    if copy.evaluation.perf_trials != 100:
        raise InputError("this protocol requires evaluation.perf_trials=100")
    copy.budgets.trials_per_space = 40
    copy.agents.rewriter.n_candidates = 1
    copy.wsl.triton_cache_dir = str(root / "cache" / "triton")
    copy.wsl.extra_pythonpath = str(Path(__file__).resolve().parents[2] / "src")
    copy.opencode.launch_cwd = root
    env = dict(copy.opencode.server_env)
    for key, sub in (("XDG_DATA_HOME", "xdg-data"), ("XDG_CACHE_HOME", "xdg-cache"),
                     ("TMPDIR", "tmp"), ("TORCH_EXTENSIONS_DIR", "cache/torch"),
                     ("TRITON_CACHE_DIR", "cache/triton")):
        path = root / sub
        path.mkdir(parents=True, exist_ok=True)
        env[key] = str(path)
    copy.opencode.server_env = env
    return copy


@contextmanager
def worker_environment(cfg: AppConfig) -> Iterator[None]:
    keys = ("XDG_DATA_HOME", "XDG_CACHE_HOME", "TMPDIR", "TORCH_EXTENSIONS_DIR", "TRITON_CACHE_DIR")
    before = {key: os.environ.get(key) for key in keys}
    try:
        os.environ.update({key: cfg.opencode.server_env[key] for key in keys})
        yield
    finally:
        for key, value in before.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def summarize(paths: list[Path]) -> None:
    results = [ArmResult.model_validate_json(p.read_text(encoding="utf-8")) for p in paths]
    if len({r.shared_id for r in results}) > 1:
        raise InputError("cannot compare results with different shared inputs")
    if len(results) != 2 or {r.arm for r in results} != {"A", "B"} or len({r.path for r in results}) != 1:
        raise InputError("summarize requires exactly one A and one B of the same execution path")
    for result in results:
        def value(records: list[TrialRecord]) -> float | None:
            values = [t.latency_ms.robust_ms for t in records if t.status == "complete" and t.latency_ms]
            return round(median(values), 4) if values else None
        print(json.dumps({"arm": result.arm, "path": result.path, "state": result.state,
                          "selected_before_final": result.selected, "error": result.error,
                          "tuning_attempts": len(result.trials),
                          "fresh_parent_ms": value(result.fresh_parent), "fresh_child_ms": value(result.fresh_child),
                          "fresh_parent_valid": sum(t.status == "complete" for t in result.fresh_parent),
                          "fresh_child_valid": sum(t.status == "complete" for t in result.fresh_child),
                          "treatment_status": result.treatment_status, "costs": result.costs.model_dump(),
                          "shared_acquisition_costs": result.acquisition_costs.model_dump() if result.acquisition_costs else None,
                          "wall_s": round(result.wall_s, 2)}))
    print("Descriptive single-step information comparison only; no significance or package-net-benefit claim.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--set", action="append", default=[])
    sub = parser.add_subparsers(dest="mode", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--selection", type=Path, required=True)
    prep.add_argument("--output", type=Path, required=True)
    for name in ("acquire", "smoke-parent", "run-arm"):
        command = sub.add_parser(name)
        command.add_argument("--shared", type=Path, required=True)
        command.add_argument("--run-dir", type=Path, required=True)
        if name == "acquire":
            command.add_argument("--probe-budget", type=int, required=True)
        if name == "run-arm":
            command.add_argument("--arm", choices=("A", "B"), required=True)
            command.add_argument("--path", choices=("direct", "legacy"), required=True)
            command.add_argument("--responses", type=Path)
            command.add_argument("--final-blocks", type=int, default=3)
    summary = sub.add_parser("summarize")
    summary.add_argument("results", type=Path, nargs="+")
    schema = sub.add_parser("schema")
    schema.add_argument("--output", type=Path, required=True)
    args = Arguments()
    _ = parser.parse_args(argv, namespace=args)
    store = None
    shared = None
    started = monotonic()
    try:
        if args.mode == "summarize":
            summarize(args.results)
            return 0
        if args.mode == "schema":
            args.output.write_text(json.dumps(Selection.model_json_schema(), indent=2), encoding="utf-8")
            return 0
        if args.config is None:
            raise InputError("--config is required for prepare and worker modes")
        cfg = load_config(args.config, args.set)
        if args.mode == "prepare":
            selection = Selection.model_validate_json(args.selection.read_text(encoding="utf-8"))
            task = load_task(cfg, *parse_task_arg(selection.task))
            from scripts.experiments.c2_local_inputs import prepare
            shared = prepare(selection, cfg, task.ref_path)
            with args.output.open("x", encoding="utf-8") as output:
                output.write(shared.model_dump_json(indent=2))
            print(f"shared_id={shared.identity()}")
            return 0
        shared = Shared.model_validate_json(args.shared.read_text(encoding="utf-8"))
        if shared.evaluation != cfg.evaluation.model_dump(mode="json") or shared.device != cfg.device:
            raise InputError("worker evaluation/device config differs from prepared shared inputs")
        if shared.protocol_id != protocol(cfg):
            raise InputError("model/retries/seed/source differs from prepared protocol")
        root = args.run_dir.resolve()
        store = RunStore.create(root.parent, root.name, {"shared_id": shared.identity(), "mode": args.mode})
        cfg = isolated_config(cfg, root)
        with worker_environment(cfg):
            if args.mode == "smoke-parent":
                _ = stage_inputs(shared, root / "common", [])
                adapter = GpuAdapter(shared, cfg, store)
                adapter.full, adapter.phase = True, "parent_smoke"
                result = adapter.measure(root / "common" / "parent.py", shared.parent.params)
                (root / "smoke.json").write_text(result.model_dump_json(indent=2), encoding="utf-8")
                store.append("RUN_FINISHED", {"status": result.status})
                return 0 if result.status == "complete" else 1
            if args.mode == "acquire":
                if args.probe_budget < 2:
                    raise InputError("probe budget must admit at least two actual measurements")
                acquire(shared, (cfg, store), args.probe_budget)
                return 0
            baked = None
            if args.arm == "B":
                if args.responses is None:
                    raise InputError("B requires a baked --responses file")
                baked = Responses.model_validate_json(args.responses.read_text(encoding="utf-8"))
                baked.validate_for(shared)
            options = RunOptions(arm=args.arm, path=args.path, final_blocks=args.final_blocks, acquisition=baked)
            with Runtime(cfg, root) as runtime:
                run_arm(shared, options, Services(cfg, store, runtime))
            return 0
    except (AgentCallError, OSError, ValueError, RuntimeError) as exc:
        if store is not None:
            store.append("LOCAL_RUN_FAILED", {"error": f"{type(exc).__name__}: {exc}"})
            if args.mode == "run-arm" and shared is not None and not (store.run_dir / "result.json").exists():
                events = store.iter_events()
                frozen = next((e.payload for e in reversed(events) if e.type == "SELECTION_FROZEN"), {})
                result = ArmResult.model_validate({
                    "shared_id": shared.identity(), "arm": args.arm, "path": args.path, "state": shared.state,
                    "selected": frozen.get("selected", "parent"),
                    "selected_trial": frozen.get("trial", shared.parent.model_dump(mode="json")),
                    "trials": [e.payload["trial"] for e in events if e.type == "TRIAL_DONE"],
                    "error": f"{type(exc).__name__}: {exc}", "costs": costs(store),
                    "wall_s": monotonic() - started,
                })
                save_result(result, store)
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
