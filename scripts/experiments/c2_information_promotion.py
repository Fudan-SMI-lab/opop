"""Measure only requested confirmation pairs; the pure helper owns every score decision."""

from dataclasses import replace
from typing import Literal, assert_never

from kernel_optimizer.gpu.worker_client import to_wsl_path
from kernel_optimizer.models.core import Candidate, TrialRecord
from kernel_optimizer.store.run_store import RunStore
from scripts.experiments.c2_information_inputs import InformationResult, InformationRun
from scripts.experiments.c2_local_adapter import GpuAdapter
from scripts.experiments.c2_local_inputs import InputError, Shared
from scripts.experiments.c2_local_runner import isolated_config, worker_environment
from scripts.experiments.c2_method_protocol import SchedulingExpired
from scripts.experiments.c2_promotion_full import ConfirmationPair, PromotionDecision, PromotionEvidence, decide_promotion
from scripts.experiments.c2_retune_studies import StudyAccounting


def _retain(result: InformationResult, status: Literal["failed", "censored"], reason: str) -> InformationResult:
    return result.model_copy(update={"selected": "parent", "selected_artifact": "common/parent.py",
        "status": status, "error": result.error or reason, "promotion_decision": PromotionDecision("failed")})


def apply_full_promotion(shared: Shared, run: InformationRun, result: InformationResult) -> InformationResult:
    child = result.child
    if run.deadline and run.deadline.remaining() <= 0:
        return _retain(result, "censored", "promotion_deadline")
    if any(t.status == "fail" for t in result.parent_finals):
        return _retain(result, "failed", "promotion_parent_quality_failure")
    if child is None:
        return _retain(result, "failed" if result.status != "censored" else "censored", "promotion_child_unavailable")
    if child.status in {"rejected", "no_best"}:
        return _retain(result, "failed", "promotion_child_unavailable")
    if child.status == "final_failed" or any(t.status == "fail" for t in child.finals):
        return _retain(result, "failed", "promotion_quality_failure")
    if child.status != "complete" or child.selected is None or child.selected_trial is None:
        return _retain(result, "censored", "promotion_native_result_incomplete")
    if result.error is not None:
        return _retain(result, "censored", "promotion_upstream_error")
    accounting = StudyAccounting(spaces=child.spaces, studies=result.studies, expanded_count=result.expanded_count)
    if not accounting.b40_complete:
        return _retain(result, "censored", "promotion_incomplete_b40")
    evidence = PromotionEvidence(parent_screen=tuple(result.parent_finals), child_screen=tuple(child.finals),
        parent_quick_ms=result.parent_baseline_ms, child_quick_ms=child.selected.latency_ms)
    decision = decide_promotion(evidence)
    root = run.run_dir.resolve()
    if decision.outcome == "request_pair":
        try:
            candidates = [Candidate.model_validate(e.payload["candidate"]) for e in RunStore.open(root / "retune").iter_events()
                          if e.type == "CANDIDATE_REGISTERED"]
            candidate = next((c for c in candidates if c.candidate_id == child.selected.candidate_id), None)
            if candidate is None:
                raise InputError("confirmation child has no registered native backend")
            store = RunStore.create(root, "confirmation", {"max_pairs": 2, "performance_samples": 100, "correctness_trials": 5})
            local = isolated_config(run.cfg, store.run_dir)
            local.run.seed = 0
            base_pythonpath = local.wsl.extra_pythonpath
            targets = {
                "parent": (root / "common/parent.py", result.parent_params, shared.backend, root / "common/imports"),
                "child": (root / "retune/report/selected.py", child.selected.params, candidate.backend, root / "retune/inputs"),
            }
            with worker_environment(local):
                adapter = GpuAdapter(shared, local, store)
                adapter.full = True
                for index, order in enumerate((("parent", "child"), ("child", "parent")), 1):
                    if decision.outcome != "request_pair":
                        break
                    if decision.next_pair_index != index:
                        raise InputError("unexpected promotion pair request")
                    records: dict[str, TrialRecord | None] = {"parent": None, "child": None}
                    result = result.model_copy(update={"confirmation_pairs": [*result.confirmation_pairs, ConfirmationPair(None, None)]})
                    for arm in order:
                        if run.deadline:
                            run.deadline.check()
                        path, params, backend, imports = targets[arm]
                        adapter.backend, adapter.phase = backend, f"confirmation_{index}_{arm}"
                        adapter.correctness.worker.cfg.extra_pythonpath = f"{to_wsl_path(imports)}:{base_pythonpath}"
                        record = adapter.measure(path, params)
                        records[arm] = record
                        pair = ConfirmationPair(records["parent"], records["child"])
                        result = result.model_copy(update={"confirmation_pairs": [*result.confirmation_pairs[:-1], pair]})
                        if run.deadline:
                            run.deadline.check()
                        if record.status == "fail":
                            return _retain(result, "failed", "promotion_quality_failure")
                    evidence = replace(evidence, confirmations=tuple(result.confirmation_pairs))
                    decision = decide_promotion(evidence)
            store.append("RUN_FINISHED", {"decision": decision.outcome, "pairs": len(result.confirmation_pairs)})
        except SchedulingExpired as exc:
            return _retain(result, "censored", str(exc))
        except (OSError, ValueError, RuntimeError) as exc:
            return _retain(result, "censored", f"confirmation_error: {type(exc).__name__}: {exc}")
    match decision.outcome:
        case "parent" | "child" as selected:
            return result.model_copy(update={"selected": selected, "status": "valid", "error": None,
                "selected_artifact": "common/parent.py" if selected == "parent" else "retune/report/selected.py",
                "promotion_decision": decision})
        case "failed":
            return _retain(result, "censored", "promotion_incomplete_full_measurement")
        case "request_pair":
            return _retain(result, "censored", "promotion_pair_limit")
        case unreachable:
            assert_never(unreachable)
