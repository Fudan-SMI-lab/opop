"""Deterministic descriptive gates; block ranges are not confidence intervals."""

from math import isfinite
from statistics import median

from pydantic import Field

from kernel_optimizer.models.core import TrialRecord
from scripts.experiments.c2_local_inputs import InputError, Strict
from scripts.experiments.c2_method_protocol import BranchPlan, CellResult, ConditionalReadiness, SLOTS, Slot
from typing import Literal


class Comparison(Strict):
    status: Literal["pass", "fail", "inconclusive"]
    improvement: float | None = None
    baseline: tuple[float, ...] = ()
    treatment: tuple[float, ...] = ()


def compare_blocks(baseline: list[float], treatment: list[float]) -> Comparison:
    if len(baseline) != 3 or len(treatment) != 3 or not all(
        isfinite(x) and x > 0 for x in baseline + treatment
    ):
        return Comparison(status="inconclusive", baseline=tuple(baseline), treatment=tuple(treatment))
    base, treated = median(baseline), median(treatment)
    passes = treated <= base * 0.98 and max(treatment) < min(baseline)
    return Comparison(status="pass" if passes else "fail", improvement=1 - treated / base,
                      baseline=tuple(baseline), treatment=tuple(treatment))


def block_values(records: list[TrialRecord]) -> list[float]:
    values = [t.latency_ms.median for t in records if t.status == "complete" and t.latency_ms
              and t.latency_ms.n_samples == 100 and t.latency_ms.median is not None
              and isfinite(t.latency_ms.median) and t.latency_ms.median > 0]
    return values if len(records) == len(values) == 3 else []


class CellGate(Strict):
    slot: Slot
    task: str
    h_g0: Comparison
    p_l: Comparison
    h_p: Comparison


class GateSummary(Strict):
    status: Literal["pass", "fail", "inconclusive"]
    cells: list[CellGate] = Field(min_length=4, max_length=4)
    branch: BranchPlan


def summarize_core(results: list[CellResult], conditional: ConditionalReadiness | None = None) -> GateSummary:
    if len({r.slot for r in results}) != len(results):
        raise InputError("duplicate core slot")
    if any(len({r.shared_id for r in results if r.task == task}) > 1 for task in {r.task for r in results}):
        raise InputError("repetitions must use identical original Shared inputs")
    by_slot = {r.slot: r for r in results}
    rows: list[CellGate] = []
    eligible: list[Slot] = []
    conditional_slots: list[Slot] = []
    excluded: dict[Slot, str] = {}
    for slot, cell in SLOTS.items():
        result = by_slot.get(slot)
        comparisons: list[Comparison] = []
        arms = {r.arm: r for r in result.opportunities} if result else {}
        if result and (result.task != cell.task or result.rep != cell.rep or len(arms) != 4):
            raise InputError("core cell identity or opportunity count differs")
        for baseline, treatment in (("G0", "H"), ("L", "P"), ("P", "H")):
            a, b = arms.get(baseline), arms.get(treatment)
            comparison = compare_blocks(block_values(a.finals) if a else [], block_values(b.finals) if b else [])
            if a and b:
                if a.status == "censored" or b.status == "censored":
                    comparison = Comparison(status="inconclusive")
                elif b.status in {"invalid", "failed", "dependency_failed"}:
                    comparison = Comparison(status="fail", baseline=comparison.baseline, treatment=comparison.treatment)
            comparisons.append(comparison)
        h = arms.get("H")
        if h and h.status == "complete" and h.native_expansion_eligible:
            eligible.append(slot)
        else:
            excluded[slot] = "no native expansion trigger" if h and h.status == "complete" else "H absent, invalid or censored"
        if (h and h.status == "complete" and conditional and conditional.bridge_verified
                and conditional.worker_budget_verified and slot in conditional.planner_requested_slots):
            conditional_slots.append(slot)
        rows.append(CellGate(slot=slot, task=cell.task, h_g0=comparisons[0], p_l=comparisons[1], h_p=comparisons[2]))
    resolved = len(results) == 4 and all(row.h_g0.status != "inconclusive" for row in rows)
    resolved = resolved and all(len(block_values(result.parent_finals)) == 3 and all(
        arm.status != "censored" and len(block_values(arm.finals)) == 3
        for arm in result.opportunities) for result in results)
    winners = [row for row in rows if row.h_g0.status == "pass"]
    passed = resolved and len(winners) >= 3 and len({r.task for r in winners}) == 2
    status = "pass" if passed else "fail" if resolved else "inconclusive"
    if passed:
        branch = BranchPlan(phase="A", eligible_slots=tuple(SLOTS), study_opportunities=8, rounds=2,
                            reason="not_ready: wire existing closed-loop H feedback, seeds 1/2, ready cutoff and terminal finals")
    elif eligible:
        branch = BranchPlan(phase="B", eligible_slots=tuple(eligible), study_opportunities=2 * len(eligible),
                            not_applicable=excluded, reason="not_ready: paired native continuation requires post-core incumbent anchors")
    elif conditional_slots:
        branch = BranchPlan(phase="C", eligible_slots=tuple(conditional_slots), study_opportunities=2 * len(conditional_slots),
                            not_applicable={s: "native planner/bridge not eligible" for s in SLOTS if s not in conditional_slots},
                            reason="not_ready: implement selected native OFF/ACTIVE Q8 pairing using verified bridge")
    else:
        branch = BranchPlan(phase=None, not_applicable=excluded,
                            reason="stop: no native B eligibility; C planner/bridge verification not supplied")
    return GateSummary(status=status, cells=rows, branch=branch)
