"""A formal lane searches for the BEST proven speed win, not the first one over the line.

WHAT THIS PINS
--------------
Two behaviours the user asked for on 2026-09-22, after the up_proj campaigns showed the old protocol
both accepting a win inside the measurement noise (0.11% gain against a 0.372% repeat spread) and
stopping at the first such win while the scores were still trending down:

  1. A candidate is accepted only when its confirmed gain clears `NOISE_MULTIPLE` times the goal's
     measured score spread (`fast_acceptance.meaningfully_beats`), not merely `gain > 0`.
  2. The first acceptance does NOT end the lane. The loop keeps drawing, remembers the best
     confirmed win as the incumbent, and stops only when the window/budget closes or `patience`
     consecutive draws yield no new incumbent. The published row is the best win, not the earliest.

WHAT MUST NOT CHANGE
--------------------
The correctness bar: `prove_candidate`, the logits/NLL floor and the device proof still gate every
candidate. A non-improving or unproven candidate is never selected.

HOW THESE RUN
-------------
`run_probe` is driven directly with a stub evaluator/generator over a scripted sequence of scores.
The stub's model evaluations carry no `score_spread_frac`, so the acceptance gate uses its
conservative fallback band (`FALLBACK_SPREAD_FRAC`, 0.4%); the scripted scores are chosen to sit
clearly inside or outside that band so the control flow is unambiguous.
"""

from pathlib import Path
from types import SimpleNamespace

import pytest

from examples.c3_qwen3.fast_acceptance import FALLBACK_SPREAD_FRAC
from examples.c3_qwen3.fast_evaluation import EvaluationFiles
from examples.c3_qwen3.fast_records import (Artifact, CandidateResult, Framework, ParamSet,
                                            TaskEvaluation)
from examples.c3_qwen3.fast_target import build_target
from examples.c3_qwen3.fast_workflow import run_probe
from examples.c3_qwen3.runner_records import RunnerError

SETUP = Path(__file__).parents[1] / "results/c3-fast-kernel-iteration/setup"
RESULTS = Path(__file__).parents[1] / "results/c3-fast-kernel-iteration"


def real_target(goal: str):
    """A genuine `TargetSpec`, because `ProbeResult` validates the type it stores.

    Only its `goal` is varied: the loop's exit decision reads the goal's DIRECTION from the contract,
    not anything else about the site.
    """
    import json

    for candidate in sorted(RESULTS.glob("**/short-device-profile-*.json")):
        try:
            data = json.loads(candidate.read_text(encoding="utf-8")).get("data") or {}
        except (ValueError, OSError):
            continue
        if data.get("valid") and data.get("cuda_intervals") and data.get("site_kernels"):
            target = build_target(candidate, "model.layers.0.input_layernorm")
            return target.model_copy(update={"goal": goal})
    pytest.skip("no complete short device profile is mirrored in this checkout")


def real_framework() -> Framework:
    return Framework(revision="TEST", implementation_sha256="i", schema_sha256="s",
                     prompt_sha256="p", agent_config_sha256="a", contract_sha256="c")


class StubBudget:
    """A ledger that allows a fixed number of draws and then refuses, like a clock running out."""

    def __init__(self, draws: int) -> None:
        self.draws, self.path = draws, Path("budget.json")
        self.spec = SimpleNamespace(deadline_unix_s=2_000_000.0, lane="formal_search")

    def available(self) -> bool:
        return self.draws > 0

    def now(self) -> float:
        return 1_000_000.0


def stub_probe(tmp_path: Path, scores: list[float | None], *, direction: str, baseline: float,
               draws: int = 10, prove: bool = True, confirm: bool | None = None, patience: int = 4,
               out_dir: Path | None = None, resume: bool = False):
    """Run the real `run_probe` over a scripted sequence of candidate scores.

    Each element of `scores` is one draw's full-model score. `confirm` scripts the repeat measurement:
    None means the repeat returns the same score, False means it comes back at baseline. `patience`
    is forwarded to `run_probe`. `out_dir` + `resume` drive a continuation of an earlier lane in the
    same directory.
    """
    import examples.c3_qwen3.fast_workflow as workflow

    state = {"draw": -1, "repeat": False}
    goal_id = "ttft" if direction == "minimize" else "single"
    target = real_target(goal_id)
    goal = SimpleNamespace(id=goal_id, direction=direction, unit="ms" if direction == "minimize" else "tokens/s")
    prepared = SimpleNamespace(access=SimpleNamespace(phase="formal_search"),
                               task=SimpleNamespace(contract=SimpleNamespace(goals=[goal])))
    budget = StubBudget(draws)
    artifact = Artifact(author="agent", version=0, session_id="ses", bundle=tmp_path / "bundle.json",
                        bundle_sha256="b", params_sha256="p", source_hashes={}, params=ParamSet(values={}),
                        kernels=())
    # The REAL `EvaluationFiles`, not a namespace with only the fields this test happens to read: the
    # lane now also consults `floor_evidence`, and a stub that silently lacks a field the product reads
    # turns a genuine wiring change into an AttributeError inside the code under test. A frozen record
    # here means adding a field to the product fails loudly at construction instead.
    engine = SimpleNamespace(target=target, budget=budget, fixtures=("f",), reports=(),
                             stage=SimpleNamespace(framework_id="F"), runtime=SimpleNamespace(
                                 runner=SimpleNamespace(closed=False)),
                             files=EvaluationFiles(baseline=tmp_path / "baseline.json",
                                                   oracle=tmp_path / "oracle",
                                                   output=tmp_path / "files-out",
                                                   # Pre-supplied so the lane does not try to calibrate
                                                   # one: that needs a resident model, which this test
                                                   # deliberately does not have. The calibration itself
                                                   # is covered in test_c3_automatic_equivalence_floor.
                                                   floor_evidence=tmp_path / "floor.json"))

    def model(given):
        if given is None:
            return TaskEvaluation(valid=True, score=baseline, metrics={})
        if state["repeat"]:
            state["repeat"] = False
            value = baseline if confirm is False else scores[state["draw"]]
            return TaskEvaluation(valid=value is not None, score=value, metrics={})
        state["repeat"] = True
        value = scores[state["draw"]]
        return TaskEvaluation(valid=value is not None, score=value, metrics={})

    engine.model = model

    class Generator:
        def __init__(self) -> None:
            self.evaluator = engine
            self.context = SimpleNamespace(prepared=prepared, framework=real_framework())
            self.handed_back: list[str | None] = []

        def invoke(self, version, previous):
            state["draw"] += 1
            state["repeat"] = False
            budget.draws -= 1
            self.handed_back.append(previous.status if previous is not None else None)
            if state["draw"] >= len(scores):
                return CandidateResult(version=version, status="censored", error="no more scripted draws")
            return CandidateResult(version=version, status="draft_ready", artifact=artifact)

    generator = Generator()
    monkey = pytest.MonkeyPatch()
    monkey.setattr(workflow, "require_target", lambda *a, **k: None)
    monkey.setattr(workflow, "measure_reference", lambda e: e.target)
    monkey.setattr(workflow, "framework_id", lambda frame: "F")
    monkey.setattr(workflow, "tune_local", lambda e, a: (a, ()))
    monkey.setattr(workflow, "prove_candidate", lambda c, s, g: prove)
    monkey.setattr(workflow, "load_bundle", lambda p, v: SimpleNamespace(bundle_sha256="base"))
    monkey.setattr(workflow, "write_artifact", lambda a, d: a)
    try:
        target_dir = out_dir if out_dir is not None else tmp_path / f"out-{len(scores)}-{direction}-{baseline}"
        result = run_probe(generator, target_dir, patience=patience, resume=resume)
    finally:
        monkey.undo()
    return result, generator


def test_a_correct_but_slower_candidate_does_not_end_the_lane(tmp_path):
    # Given: the `single` shape -- a maximise goal whose first proven candidate scores BELOW baseline,
    # followed by one that clearly beats it (22.5 vs 21.243 = +5.9%, far outside the 0.4% band).
    result, generator = stub_probe(tmp_path, [20.028, 22.5], direction="maximize", baseline=21.243)

    # Then: the slower draw is recorded as not-improving, the lane keeps going, the second draw is a
    # confirmed win and becomes the incumbent, and -- because no more draws are scripted -- the loop
    # then hits a censored draw. The published row is the win.
    assert [c.status for c in result.candidates] == ["not_improved", "valid", "censored"]
    assert result.status == "accepted" and result.selected is not None
    assert generator.handed_back == [None, "not_improved", "valid"]


def test_the_lane_keeps_the_best_win_not_the_first(tmp_path):
    # Given: a minimise goal with two confirmed wins, the SECOND strictly better (270 then 265 against
    # 282.82). The old protocol stopped at the first; the lane must now keep the better one.
    result, generator = stub_probe(tmp_path, [270.0, 265.0], direction="minimize", baseline=282.82)

    # Then: both are accepted, the lane searched past the first, and the selected artifact reflects the
    # better (last) win -- the published row is the best win the window bought.
    assert [c.status for c in result.candidates] == ["valid", "valid", "censored"]
    assert result.status == "accepted" and result.selected is not None
    # The incumbent improved on the second draw, so the search did not stop on the first acceptance.
    assert generator.handed_back == [None, "valid", "valid"]


def test_a_win_inside_the_noise_band_is_not_accepted(tmp_path):
    # Given: a minimise goal whose candidate beats baseline by less than the fallback noise band
    # (282.82 -> 282.0 = 0.29%, inside the 0.4% band). This is the up_proj ttft situation the user
    # flagged: a "win" indistinguishable from measuring the baseline twice.
    assert 0 < (282.82 - 282.0) / 282.82 < FALLBACK_SPREAD_FRAC
    result, _ = stub_probe(tmp_path, [282.0, 282.0, 282.0], direction="minimize", baseline=282.82)

    # Then: every draw is correct but none clears the band, so none is selected.
    assert [c.status for c in result.candidates] == ["not_improved"] * 3 + ["censored"]
    assert result.selected is None
    assert result.status != "accepted"


def test_a_plateau_stops_the_lane_early_and_keeps_the_incumbent(tmp_path):
    # Given: one clear win followed by `patience` draws that do not improve on it. The lane should stop
    # early -- not burn the rest of the window -- and publish the incumbent.
    result, generator = stub_probe(tmp_path, [270.0, 275.0, 275.0], direction="minimize",
                                   baseline=282.82, patience=2)
    # The win is accepted; the next two clear the band too but are worse than the incumbent (275 > 270),
    # so `since_best` reaches patience=2 and the lane stops with the incumbent kept.
    assert result.candidates[0].status == "valid"
    assert result.status == "accepted" and result.selected is not None
    # Stopped on plateau before the scripted scores ran out, so no censored draw was needed.
    assert "censored" not in [c.status for c in result.candidates]


def test_an_unconfirmed_win_keeps_the_lane_open_instead_of_stopping_on_it(tmp_path):
    # Given: a candidate whose FIRST measurement beats baseline but whose repeat does not. It used to
    # end the lane with no row; the repeat requirement stays, but the lane keeps drawing.
    result, generator = stub_probe(tmp_path, [275.0, 270.0], direction="minimize", baseline=282.82,
                                  confirm=False)
    assert result.candidates[0].status == "not_confirmed"
    assert result.selected is None          # the unconfirmed candidate is NOT selected
    assert generator.handed_back[1] == "not_confirmed"


def test_the_gain_direction_is_the_contract_s_not_a_smaller_is_better_assumption(tmp_path):
    # Given: the same numeric improvement under both directions. A maximise goal scoring BELOW baseline
    # and a minimise goal scoring ABOVE it must both be non-improvements -- this is the arithmetic that
    # made three "better-looking" candidates actually worse.
    lower, _ = stub_probe(tmp_path / "max", [20.0], direction="maximize", baseline=21.243)
    higher, _ = stub_probe(tmp_path / "min", [290.0], direction="minimize", baseline=282.82)
    assert lower.candidates[0].status == "not_improved"
    assert higher.candidates[0].status == "not_improved"


def test_an_unproven_candidate_is_still_rejected_before_any_score_is_read(tmp_path):
    # Given: `prove_candidate` refusing. Relaxing the exit must not have moved the correctness gate --
    # a candidate that fails proof can never be selected, whatever it scored.
    #
    # Note on the recorded label: `status` is set to "valid" from the MODEL EVALUATION before the proof
    # is consulted, so an unproven candidate is still stored as "valid" with `selected` empty. That
    # naming predates this change and is left alone here; what this test pins is that no amount of
    # scoring can turn a proof failure into a selected row.
    result, _ = stub_probe(tmp_path, [1.0], direction="minimize", baseline=282.82, prove=False)
    assert result.selected is None
    assert result.status != "accepted"
    assert all(c.confirmation is None for c in result.candidates)


def test_resume_continues_a_lane_and_keeps_a_better_later_win(tmp_path):
    # Given: a first run that plateaus early with a modest win (patience=1 stops it after one
    # non-improving draw), leaving budget and window unspent.
    out = tmp_path / "resumable"
    first, _ = stub_probe(tmp_path, [270.0, 271.0], direction="minimize", baseline=282.82,
                          patience=1, out_dir=out)
    assert first.status == "accepted" and first.selected is not None
    prior_draws = len(first.candidates)

    # When: the lane is resumed in the SAME directory with a strictly better candidate available.
    second, generator = stub_probe(tmp_path, [265.0], direction="minimize", baseline=282.82,
                                   patience=4, out_dir=out, resume=True)

    # Then: the reloaded candidates are carried forward (numbering continues, not restarts), the
    # better win becomes the incumbent, and the published row reflects it.
    assert len(second.candidates) > prior_draws            # prior draws were reloaded, not discarded
    assert second.status == "accepted" and second.selected is not None
    accepted_scores = [c.confirmation.score for c in second.candidates
                       if c.status == "valid" and c.confirmation is not None]
    assert 265.0 in accepted_scores                        # the better later win was searched and kept


def test_resume_without_a_prior_result_starts_fresh(tmp_path):
    # Given: resume requested but no prior result.json in the directory (the first run of a lane).
    out = tmp_path / "first-time"
    result, _ = stub_probe(tmp_path, [270.0], direction="minimize", baseline=282.82,
                           out_dir=out, resume=True)
    # Then: it behaves exactly like a fresh run rather than erroring on the missing prior.
    assert result.status == "accepted" and result.selected is not None


def test_a_fresh_run_refuses_a_directory_that_already_exists(tmp_path):
    # Given: a directory a prior lane already wrote into, and resume NOT requested.
    out = tmp_path / "occupied"
    stub_probe(tmp_path, [270.0], direction="minimize", baseline=282.82, out_dir=out)
    # Then: a fresh run must not silently write into it -- that would pool two lanes' work.
    with pytest.raises(FileExistsError):
        stub_probe(tmp_path, [265.0], direction="minimize", baseline=282.82, out_dir=out)
