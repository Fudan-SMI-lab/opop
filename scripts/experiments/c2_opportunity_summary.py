"""All eight opportunities and four fresh heldout pairs; descriptive gates, not significance."""

from statistics import median
from typing import Literal

from scripts.experiments.c2_local_inputs import InputError, Strict
from scripts.experiments.c2_method_gates import Comparison, block_values, compare_blocks
from scripts.experiments.c2_opportunity_inputs import CELLS, Arm, Slot, Wave
from scripts.experiments.c2_opportunity_records import HeldoutResult, OpportunityResult


class OpportunityRow(Strict):
    wave: Wave
    slot: Slot
    task: str
    arm: Arm
    status: Literal["complete", "failed", "censored"]
    asked: int | None = None
    expanded_count: int | None = None
    selected: Literal["parent", "child"] | None = None


class PairRow(Strict):
    wave: Wave
    slot: Literal["A0", "B0"]
    task: str
    g0_status: Literal["complete", "failed", "censored"]
    c2_status: Literal["complete", "failed", "censored"]
    comparison: Comparison
    parent_blocks: tuple[float, ...] = ()
    g0_ratio: float | None = None
    c2_ratio: float | None = None


class CampaignSummary(Strict):
    status: Literal["pass", "fail", "inconclusive"]
    opportunities: list[OpportunityRow]
    pairs: list[PairRow]
    wins: int


def summarize(opportunities: list[OpportunityResult], heldouts: list[HeldoutResult]) -> CampaignSummary:
    by_slot = {(r.wave, r.slot): r for r in opportunities}
    by_pair = {(r.wave, r.slot): r for r in heldouts}
    if len(by_slot) != len(opportunities) or len(by_pair) != len(heldouts):
        raise InputError("duplicate opportunity or heldout pair")
    deadlines = {r.deadline_unix_s for r in opportunities} | {r.deadline_unix_s for r in heldouts}
    if len(deadlines) > 1:
        raise InputError("campaign deadline changed between slots, waves or heldout")
    for task in ("level3:43", "level3:21"):
        if len({r.shared_id for r in opportunities if r.task == task}) > 1:
            raise InputError("original Shared identity changed between arms or repetitions")
    rows: list[OpportunityRow] = []
    for key, cell in CELLS.items():
        result = by_slot.get(key)
        if result and (result.task != cell.task or result.arm != cell.arm or result.rep != cell.rep or result.sampler_seed != cell.seed):
            raise InputError("opportunity labels differ from the fixed plan")
        info = result.information if result else None
        rows.append(OpportunityRow(wave=key[0], slot=key[1], task=cell.task, arm=cell.arm,
            status=result.status if result else "censored", asked=info.asked if info else None,
            expanded_count=info.expanded_count if info else None,
            selected=result.incumbent.selected if result and result.incumbent else None))
    pairs: list[PairRow] = []
    for wave in (1, 2):
        for slot in ("A0", "B0"):
            cell = CELLS[wave, slot]
            arms = {c.arm: by_slot.get(key) for key, c in CELLS.items() if key[0] == wave and key[1][0] == slot[0]}
            g0, c2 = arms["G0"], arms["C2"]
            measurement = by_pair.get((wave, slot))
            if measurement and (measurement.task != cell.task or any(
                    r is not None and r.shared_id != measurement.shared_id for r in (g0, c2))):
                raise InputError("heldout identity differs from its original-parent pair")
            comparison = Comparison(status="inconclusive")
            parent, g0_ratio, c2_ratio = [], None, None
            if measurement:
                parent = block_values(measurement.parent_finals)
                a, b = block_values(measurement.g0_finals), block_values(measurement.c2_finals)
                if parent and a and b:
                    g0_ratio, c2_ratio = median(a) / median(parent), median(b) / median(parent)
                if (measurement.status == "complete" and parent and g0 and c2
                        and g0.status != "censored" and c2.status != "censored"):
                    comparison = compare_blocks(a, b)
                    if c2.status == "failed" and comparison.status != "inconclusive":
                        comparison = Comparison(status="fail", baseline=tuple(a), treatment=tuple(b))
            pairs.append(PairRow(wave=wave, slot=slot, task=cell.task,
                g0_status=g0.status if g0 else "censored", c2_status=c2.status if c2 else "censored",
                comparison=comparison, parent_blocks=tuple(parent), g0_ratio=g0_ratio, c2_ratio=c2_ratio))
    winners = [p for p in pairs if p.comparison.status == "pass"]
    complete = all(p.comparison.status != "inconclusive" for p in pairs)
    passed = len(winners) >= 3 and len({p.task for p in winners}) == 2
    return CampaignSummary(status="inconclusive" if not complete else "pass" if passed else "fail",
                           opportunities=rows, pairs=pairs, wins=len(winners))
