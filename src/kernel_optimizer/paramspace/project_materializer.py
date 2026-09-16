"""Generic project parameter materialization, separate from the legacy PARAMS API."""

from dataclasses import dataclass
from pathlib import Path

from kernel_optimizer.models.candidate_artifact import CandidateArtifact, ParameterApplicationRequest
from kernel_optimizer.store.run_store import RunStore
from kernel_optimizer.tasks.project_artifacts import apply_project_parameters


@dataclass(frozen=True, slots=True)
class ProjectMaterializer:
    store: RunStore
    workspace: Path

    def materialize(self, request: ParameterApplicationRequest) -> CandidateArtifact:
        return apply_project_parameters(self.store, request, self.workspace)
