"""Task cost model: how much arithmetic and how much traffic this task REQUIRES.

Measured on the REFERENCE, not on a candidate, and that is the whole design. These are properties
of the task -- the same numbers apply to every candidate optimizing it -- so they are the shared
denominator that makes "% of peak" comparable across candidates and across rewrite rounds. A
per-candidate count would instead measure what that candidate happens to do, which is exactly the
quantity being optimized and therefore useless as a yardstick.

THREE NUMBERS, and their differences are the signal:

  flop_count        Arithmetic the task requires, from torch's own FlopCounterMode. Not a
                    hand-derived formula: measured against analytic values for matmul, conv,
                    attention and depthwise it agrees to ratio 1.0000. It returns 0 for ops with
                    no multiply-accumulate (maxpool, elementwise), which is correct roofline
                    semantics -- such a kernel's ceiling is bandwidth, not FLOP/s.

  compulsory_bytes  The traffic no implementation can avoid: distinct inputs + parameters +
                    outputs, each counted once. Fusion can remove every intermediate but cannot
                    remove these. So flop_count / compulsory_bytes is the HIGHEST arithmetic
                    intensity any correct implementation of this task can reach, which is what
                    decides whether the task can ever be compute-bound on this card.

  reference_bytes   What the reference actually materializes, summed over every dispatched op.
                    Always >= compulsory_bytes, and the RATIO is the fusion headroom: a reference
                    moving 8x its compulsory traffic is telling the agent that most of its time
                    goes to writing and re-reading intermediates.

WHY A DISPATCH MODE rather than module hooks. Hooks fire on nn.Module boundaries, so a reference
written with torch.nn.functional calls (which most KernelBench references are) would report
almost nothing. `__torch_dispatch__` sees every aten op regardless of how the model is written --
the same mechanism FlopCounterMode uses, so the two counts come from the same view of the program.

WHAT THESE ARE NOT. `reference_bytes` is materialized-tensor traffic, not measured DRAM traffic:
a small intermediate that stays in L2 is counted here and never crosses the memory bus. So it is
an UPPER bound on the reference's traffic, and the honest reading of `reference_bytes /
compulsory_bytes` is "how much intermediate material the reference creates", not "how many bytes
the DRAM saw". Measuring the latter needs hardware counters, which these boxes do not have.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class TaskCost(BaseModel):
    """Arithmetic and traffic this task requires. A property of the task, not of a candidate."""

    model_config = ConfigDict(frozen=True)

    flop_count: int = 0
    compulsory_bytes: int = 0
    reference_bytes: int = 0
    op_count: int = 0

    # Populated when a count could not be taken, so a missing number is never mistaken for a
    # measured zero -- `flop_count = 0` is a legitimate result for maxpool and a failure result
    # for a matmul, and only this field distinguishes them.
    notes: list[str] = []

    @property
    def max_arithmetic_intensity(self) -> float:
        """The highest FLOP/byte any correct implementation of this task can reach.

        Uses compulsory traffic, so it is a CEILING on intensity: a candidate cannot exceed it,
        and comparing it against the card's roofline ridge answers a question no per-candidate
        measurement can -- whether this task is capable of being compute-bound at all.
        """
        if self.compulsory_bytes <= 0:
            return 0.0
        return self.flop_count / self.compulsory_bytes

    @property
    def reference_arithmetic_intensity(self) -> float:
        """The intensity the reference itself achieves, i.e. with its intermediates unfused."""
        if self.reference_bytes <= 0:
            return 0.0
        return self.flop_count / self.reference_bytes

    @property
    def fusion_headroom(self) -> float:
        """reference_bytes / compulsory_bytes: how much traffic fusion could in principle remove.

        1.0 means the reference already materializes only what it must. Large values mean most of
        the reference's traffic is intermediates, which is the case fusion addresses.

        NOT capped by op_count. A first attempt capped it there, reasoning that N ops cannot offer
        more than N ops' worth of fusable traffic -- which is false: a reference that re-reads the
        SAME tensor across many ops legitimately exceeds that ratio, and the cap dropped L3:43 from
        69.09x to 40x, destroying the very signal this measures. The actual defect it was papering
        over (a multi-output aten op inflating reference_bytes) is fixed in the worker's counter,
        where it belongs.
        """
        if self.compulsory_bytes <= 0:
            return 0.0
        return self.reference_bytes / self.compulsory_bytes

    def summary_line(self) -> str:
        """One line for the agent prompt and the report. Says what each number means, because a
        bare FLOP count invites the reader to compare it against the wrong denominator."""
        if self.flop_count <= 0 and self.compulsory_bytes <= 0:
            return "task cost: not measured"
        parts = []
        if self.flop_count > 0:
            parts.append(f"{self.flop_count/1e9:.3f} GFLOP required")
        else:
            parts.append("0 FLOP (no multiply-accumulate: this task's ceiling is bandwidth, "
                         "not FLOP/s)")
        if self.compulsory_bytes > 0:
            parts.append(f"{self.compulsory_bytes/1e6:.2f} MB unavoidable traffic "
                         f"(inputs+params+outputs)")
        if self.reference_bytes > 0 and self.compulsory_bytes > 0:
            parts.append(f"reference materializes {self.fusion_headroom:.1f}x that "
                         f"({self.reference_bytes/1e6:.2f} MB over {self.op_count} ops)")
        if self.flop_count > 0 and self.compulsory_bytes > 0:
            parts.append(f"max achievable intensity {self.max_arithmetic_intensity:.2f} FLOP/byte")
        return "task cost: " + "; ".join(parts)


def cost_from_worker(result: dict) -> TaskCost:
    """Build a TaskCost from a raw worker result. Pure, so a recorded result replays exactly."""
    cost = result.get("task_cost") or {}
    return TaskCost(
        flop_count=int(cost.get("flop_count", 0) or 0),
        compulsory_bytes=int(cost.get("compulsory_bytes", 0) or 0),
        reference_bytes=int(cost.get("reference_bytes", 0) or 0),
        op_count=int(cost.get("op_count", 0) or 0),
        notes=list(cost.get("notes", []) or []),
    )
