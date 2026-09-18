"""Typed accounting projected from existing native publication, trial and tuning events."""

from collections import Counter

from pydantic import BaseModel, ConfigDict, Field

from kernel_optimizer.models.core import ParameterSpace, TrialRecord
from kernel_optimizer.store.run_store import RunStore
from scripts.experiments.c2_local_inputs import Strict


class RetuneStudy(Strict):
    candidate_id: str
    space_id: str
    asked: int | None = Field(default=None, ge=0)
    n_complete: int = Field(default=0, ge=0)
    n_fail: int = Field(default=0, ge=0)


class StudyAccounting(Strict):
    spaces: list[ParameterSpace]
    studies: list[RetuneStudy]
    expanded_count: int

    @property
    def asked_total(self) -> int | None:
        if any(study.asked is None for study in self.studies):
            return None
        return sum(study.asked for study in self.studies if study.asked is not None)

    @property
    def b40_complete(self) -> bool:
        published = {(space.candidate_id, space.space_id) for space in self.spaces}
        studied = {(study.candidate_id, study.space_id) for study in self.studies}
        return bool(self.studies) and published == studied and all(study.asked == 40 for study in self.studies)


class _Payload(BaseModel):
    model_config = ConfigDict(frozen=True)


class _Published(_Payload):
    space: ParameterSpace


class _Snapshot(_Payload):
    asked: int | None = Field(default=None, ge=0)


class _Tuned(_Payload):
    candidate_id: str
    space_id: str
    snapshot: _Snapshot = Field(default_factory=_Snapshot)


class _Trial(_Payload):
    trial: TrialRecord


def read_studies(store: RunStore) -> StudyAccounting:
    events = store.iter_events()
    spaces = [_Published.model_validate(e.payload).space for e in events if e.type == "SPACE_PUBLISHED"]
    tuned = [_Tuned.model_validate(e.payload) for e in events if e.type == "TUNING_DONE"]
    trials = [_Trial.model_validate(e.payload).trial for e in events if e.type == "TRIAL_DONE"]
    snapshots = {(row.candidate_id, row.space_id): row.snapshot for row in tuned}
    keys = dict.fromkeys([(s.candidate_id, s.space_id) for s in spaces]
                         + [(row.candidate_id, row.space_id) for row in tuned]
                         + [(t.candidate_id, t.space_id) for t in trials])
    counts = Counter((t.candidate_id, t.space_id, t.status) for t in trials)
    studies = [RetuneStudy(candidate_id=cid, space_id=sid,
        asked=snapshots[(cid, sid)].asked if (cid, sid) in snapshots else None,
        n_complete=counts[(cid, sid, "complete")], n_fail=counts[(cid, sid, "fail")]) for cid, sid in keys]
    return StudyAccounting(spaces=spaces, studies=studies,
                           expanded_count=sum(e.type == "SPACE_EXPANDED" for e in events))
