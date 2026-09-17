# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["kernel-optimizer"]
# ///
# Existing v5 environment: python -m scripts.experiments.c2_information --help
"""One legacy child opportunity under G0/G1/G2 information, followed by native retuning."""

import argparse
import sys
from contextlib import chdir
from pathlib import Path
from time import monotonic

from kernel_optimizer.agents.runtime import AgentCallError
from kernel_optimizer.config import load_config
from kernel_optimizer.models.core import ParameterSpace, sha256_text
from kernel_optimizer.paramspace.materializer import MaterializeError
from kernel_optimizer.store.run_store import RunStore
from kernel_optimizer.wiring import Runtime
from scripts.experiments.c2_information_inputs import Group, InformationResult, InformationRun, group_responses
from scripts.experiments.c2_local_adapter import GpuAdapter
from scripts.experiments.c2_local_agents import Services, legacy_child
from scripts.experiments.c2_local_costs import costs
from scripts.experiments.c2_local_inputs import InputError, Responses, Shared, Strict, stage_inputs
from scripts.experiments.c2_local_runner import isolated_config, worker_environment
from scripts.experiments.c2_retune import RetuneInputs, RetuneResult, retune


def run_information(shared: Shared, acquisition: Responses, run: InformationRun) -> InformationResult:
    started = monotonic()
    responses = group_responses(shared, acquisition, run.group)
    if shared.evaluation != run.cfg.evaluation.model_dump(mode="json") or shared.device != run.cfg.device:
        raise InputError("evaluation/device config differs from prepared shared inputs")
    if shared.parent.latency_ms is None:
        raise InputError("the prepared parent has no declared latency baseline")
    parent_baseline = shared.parent.latency_ms.robust_ms
    root = run.run_dir.resolve()
    store = RunStore.create(root.parent, root.name, {
        "group": run.group, "shared_id": shared.identity(), "state": shared.state,
        "parent_baseline_ms": parent_baseline, "sampler_seed": 0, "evaluation_seed": 0,
    })
    parent_store = RunStore.create(root, "parent", {"task": shared.task})
    generation_store = RunStore.create(root, "generation", {"task": shared.task})
    common = root / "common"
    inputs = stage_inputs(shared, common, responses)
    local = isolated_config(run.cfg, root)
    local.run.seed = 0
    local.opencode.launch_cwd = generation_store.run_dir
    child_result: RetuneResult | None = None
    error = None
    store.append("INFORMATION_BASELINE_DECLARED", {"parent_ms": parent_baseline,
                                                  "parent_params": shared.parent.params.model_dump(mode="json")})
    with worker_environment(local):
        parent = GpuAdapter(shared, local, parent_store)
        parent.full = True
        for block in range(3):
            parent.phase = f"parent_full_{block}"
            parent.measure(common / "parent.py", shared.parent.params)
        parent_store.append("RUN_FINISHED", {"valid_blocks": sum(t.status == "complete" for t in parent.records)})
        if all(t.status == "complete" for t in parent.records):
            try:
                with Runtime(local, generation_store.run_dir) as runtime:
                    with chdir(common):
                        child = legacy_child(shared, Services(local, generation_store, runtime), inputs)
                generation_store.append("RUN_FINISHED", {"status": "generated"})
                source = child.path.read_text(encoding="utf-8")
                space = ParameterSpace(space_id="information-child-space", candidate_id="information-child",
                                       source_sha=sha256_text(source), domains=child.space.params,
                                       constraints=child.space.constraints)
                space_path = root / "child-space.json"
                space_path.write_text(space.model_dump_json(indent=2), encoding="utf-8")
                spec = RetuneInputs.model_validate({
                    "task": shared.task, "source": child.path, "space": space_path,
                    "reference": common / "reference.py", "sampler_seed": 0, "evaluation_seed": 0,
                    "output": root / "retune", "backend": child.backend,
                })
                child_result = retune(spec, local)
                if child_result.rejection is not None:
                    error = f"retune_rejected: {child_result.rejection.reason}: {child_result.rejection.detail}"
                elif child_result.status != "complete":
                    error = f"retune_{child_result.status}"
            except (AgentCallError, MaterializeError, OSError, ValueError, RuntimeError) as exc:
                error = f"{type(exc).__name__}: {exc}"
                store.append("INFORMATION_OPPORTUNITY_FAILED", {"error": error})
        else:
            error = "parent_full_verification_failed"
    selected = "parent"
    artifact = "common/parent.py"
    child_ms = child_result.selected.latency_ms if child_result and child_result.selected else None
    if child_ms is not None and child_ms < parent_baseline:
        selected, artifact = "child", "retune/report/selected.py"
    result = InformationResult.model_validate({
        "group": run.group, "state": shared.state, "selected": selected, "selected_artifact": artifact,
        "parent_baseline_ms": parent_baseline, "parent_params": shared.parent.params,
        "parent_finals": parent.records, "child": child_result, "error": error,
        "generation_costs": costs(generation_store), "acquisition_costs": acquisition.costs,
        "parent_costs": costs(parent_store), "wall_s": monotonic() - started,
    })
    store.append("INFORMATION_SELECTED", {"selected": selected, "parent_baseline_ms": parent_baseline,
                                          "child_tuning_ms": child_ms, "artifact": artifact})
    (root / "result.json").write_text(result.model_dump_json(indent=2), encoding="utf-8")
    store.append("RUN_FINISHED", {"selected": selected, "error": error, "wall_s": result.wall_s})
    return result


class Options(Strict):
    config: Path
    shared: Path
    responses: Path
    group: Group
    run_dir: Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "shared", "responses", "run-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--group", choices=("G0", "G1", "G2"), required=True)
    options = Options.model_validate(vars(parser.parse_args(argv)))
    try:
        cfg = load_config(options.config)
        shared = Shared.model_validate_json(options.shared.read_text(encoding="utf-8"))
        acquisition = Responses.model_validate_json(options.responses.read_text(encoding="utf-8"))
        result = run_information(shared, acquisition, InformationRun(cfg, options.group, options.run_dir))
        print(f"selected={result.selected}; error={result.error}")
        return 1 if result.error else 0
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
