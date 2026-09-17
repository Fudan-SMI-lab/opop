"""Native witness conversion shared by initial tuning and space expansion."""

from pathlib import Path

from kernel_optimizer.evaluation.correctness import latency_from_result
from kernel_optimizer.evaluation.profilerx import LightProfiler
from kernel_optimizer.models.core import ParamSet, TrialRecord
from kernel_optimizer.paramspace import materializer
from kernel_optimizer.paramspace.validation import SpaceAccepted


def witness_inputs(
    accepted: SpaceAccepted, profiler: LightProfiler,
) -> tuple[tuple[ParamSet, ...], dict[str, TrialRecord]]:
    anchors = tuple(w.params for w in accepted.witnesses)
    cache = {
        w.params.key(): TrialRecord(
            trial_id=f"wit-{w.params.key()}",
            candidate_id=accepted.space.candidate_id,
            space_id=accepted.space.space_id,
            params=w.params, status="complete",
            latency_ms=latency_from_result(w.worker_result),
            profile=profiler.extract(w.worker_result),
        )
        for w in accepted.witnesses if w.latency_mean_ms is not None
    }
    return anchors, cache


def prior_source_matches(source: str, prior: TrialRecord, trials_dir: Path) -> bool:
    """Require the measured trial artifact, not merely legal parameters, for reuse."""
    try:
        measured = (trials_dir / f"{prior.trial_id}.py").read_text(encoding="utf-8")
        return measured == materializer.materialize(source, prior.params)
    except (OSError, materializer.MaterializeError):
        return False
