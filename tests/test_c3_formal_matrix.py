"""The formal-to-matrix chain: three frozen rows, then 4 goals x 3 goals x 3 blocks measured.

These exercise the REAL drivers (`fast_formal`, `fast_matrix`) against the numerical CPU model
boundary from `tests.c3_device_fakes`. Only the device/provider edges are fake; the frozen
contract, the sealed held-out corpus, the budget lanes, quality evaluation and the measurement
arithmetic are the production ones.
"""

import json
from pathlib import Path
from sys import maxsize

import pytest

from examples.c3_qwen3 import local_eval
from examples.c3_qwen3.fast_matrix import MatrixCell, gain_pct, summarise
from examples.c3_qwen3.fast_records import TargetSpec
from examples.c3_qwen3.runner_records import RunnerError

RESULTS = Path(__file__).parents[1] / "results/c3-fast-kernel-iteration"
SETUP = RESULTS / "setup"


def native_target():
    """The `TargetSpec` produced natively on the GPU host, read back as-is.

    `build_target` is deliberately NOT re-run here: the locally mirrored raw profile is a TRUNCATED
    copy (753 KB against the 4.86 MB original), so parsing it would fail for a reason that has
    nothing to do with these drivers. `tests/test_c3_task_routing.py` already covers target
    construction from a whole profile.

    The file is SEARCHED for rather than named by one path. The same campaign tree is laid out
    differently on the GPU host (`development-<rev>/target.json`) and in the local mirror (under an
    extra `setup-native-*/host-A/` level), and a hardcoded path made this test pass on one machine
    and fail on the other for a reason unrelated to what it checks. Every match is a `TargetSpec`
    written by the same native `build_target`, so any one of them exercises these drivers equally.
    """
    for candidate in sorted(RESULTS.glob("**/target.json")):
        return TargetSpec.model_validate_json(candidate.read_bytes())
    raise AssertionError(f"no native target.json anywhere under {RESULTS}")


def cells(scores: dict[tuple[str, str], list[float]], status: str = "valid") -> tuple[MatrixCell, ...]:
    return tuple(MatrixCell(row=row, goal_id=goal, block=block, status=status, score=score)
                 for (row, goal), values in scores.items() for block, score in enumerate(values))


def test_gain_is_positive_for_improvement_in_both_directions() -> None:
    # Given: TTFT is minimised and the two throughput goals are maximised, so a single
    # "row minus baseline" would print the sign backwards for one of them.
    # When / Then: the published convention is that positive ALWAYS means better.
    assert gain_pct("ttft", 100.0, 90.0) == pytest.approx(10.0)
    assert gain_pct("ttft", 100.0, 110.0) == pytest.approx(-10.0)
    assert gain_pct("single", 100.0, 110.0) == pytest.approx(10.0)
    assert gain_pct("multi", 100.0, 90.0) == pytest.approx(-10.0)


def test_a_goal_column_is_summarised_only_when_all_three_blocks_are_valid() -> None:
    # Given: one row whose three blocks all measured, and one that lost a block. A partial column
    # must not be summarised -- otherwise a surviving block stands in for the pair that failed,
    # which is exactly how a lucky measurement becomes a published number.
    good = cells({("baseline", "single"): [100.0, 102.0, 101.0],
                  ("single", "single"): [110.0, 112.0, 111.0]})
    partial = cells({("multi", "single"): [999.0]})
    # When
    summary = summarise((*good, *partial))
    # Then
    assert summary["median_by_row_goal"]["baseline|single"] == pytest.approx(101.0)
    assert summary["median_by_row_goal"]["single|single"] == pytest.approx(111.0)
    assert "multi|single" not in summary["median_by_row_goal"]
    assert summary["gain_pct_vs_baseline"]["single|single"] == pytest.approx(100 * (111 / 101 - 1))
    assert "multi|single" not in summary["gain_pct_vs_baseline"]
    # ...and the baseline is never scored against itself.
    assert "baseline|single" not in summary["gain_pct_vs_baseline"]


def test_an_invalid_block_is_absent_rather_than_zero() -> None:
    # Given: a block that failed quality. Writing it as 0.0 would make a broken row look
    # infinitely fast on TTFT and infinitely slow on throughput.
    broken = cells({("ttft", "ttft"): [10.0, 10.0, 10.0]}, status="invalid")
    # When / Then
    summary = summarise(broken)
    assert summary["median_by_row_goal"] == {} and summary["gain_pct_vs_baseline"] == {}


def test_a_row_claiming_a_kernel_without_matching_device_evidence_is_rejected(tmp_path) -> None:
    # Given: an artifact that DECLARES a Triton kernel but carries no local report whose device
    # evidence matches that declaration. The published table's whole value is that each optimised
    # row is backed by an observed CUDA launch, so an unmatched declaration must fail loudly
    # rather than become a row with an empty evidence list.
    from kernel_optimizer.models.core import ParamSet
    from kernel_optimizer.models.device_operator import DeviceKernelDeclaration
    from examples.c3_qwen3.device_records import StageIdentity
    from examples.c3_qwen3.fast_formal import row_evidence
    from examples.c3_qwen3.fast_records import Artifact

    stage = StageIdentity(profile="c3_fast_device", stage="final", framework_id="F")
    artifact = Artifact(author="agent", version=0, session_id="s", bundle=tmp_path / "bundle.json",
        bundle_sha256="b", source_hashes={"operators.py": "h"}, params=ParamSet(values={}), params_sha256="p",
        kernels=(DeviceKernelDeclaration(backend="triton", source_file="operators.py", entry="kernel",
                                         output_arg_indices=(1,)),))
    # When / Then
    with pytest.raises(RunnerError, match="no matching device evidence"):
        row_evidence("single", artifact, (), None, stage, "base", None, None)


def test_the_baseline_row_is_recorded_as_not_device_proven(tmp_path) -> None:
    # Given: the baseline row is original model bytes with no authored kernel. It must never be
    # presented as a device-proven result, and its reason has to be on the record.
    from examples.c3_qwen3.device_records import StageIdentity
    from examples.c3_qwen3.fast_formal import row_evidence

    stage = StageIdentity(profile="c3_fast_device", stage="final", framework_id="F")
    # When
    row = row_evidence("baseline", None, (), None, stage, "base-sha", None, "original baseline bytes")
    # Then
    assert row.device_proven is False and row.kernels == ()
    assert row.fallback_reason == "original baseline bytes"
    assert row.original_baseline_sha256 == "base-sha"


def test_the_sealed_final_corpus_supplies_three_blocks_for_every_goal() -> None:
    # Given: the table is 4 rows x 3 goals x 3 blocks = 36 cells, and the blocks come from the
    # held-out corpus that was frozen before any candidate existed.
    from examples.c3_qwen3.fast_prepare import FastAccess, prepare_fast

    # When: the sealed phase is opened (it refuses to open without formal completion).
    final = prepare_fast(SETUP / "contract.json", SETUP / "preflight.json",
        FastAccess(profile="c3_fast_device", phase="final", formal_complete=True))
    # Then: every goal has exactly three blocks, each holding that goal's own request count.
    groups = {g.id: g.final_prompt_groups for g in final.task.contract.goals}
    assert set(groups) == {"ttft", "single", "multi"}
    for goal in final.task.contract.goals:
        assert len(goal.final_prompt_groups) == 3
        assert all(len(block) == goal.requests for block in goal.final_prompt_groups)
    # ...so the sealed corpus itself is what fixes the table: 4 rows (baseline plus one per goal) x
    # 3 goals x 3 blocks = 36 cells. That count comes from the CORPUS, and the final lane's
    # allowances are DERIVED from it rather than declared by hand, so they cannot drift out of step
    # with the table they are meant to cover.
    from examples.c3_qwen3.fast_matrix import ROWS, cell_plan, final_allowance
    from examples.c3_qwen3.fast_budget import BudgetSpec, StageBudget
    from examples.c3_qwen3.device_records import StageIdentity
    import tempfile

    plan = cell_plan(final)
    assert len(plan) == 9 and len(ROWS) * len(plan) == 36
    # Every (goal, block) appears exactly once, so no block is measured twice or skipped.
    assert len({(goal, block) for goal, block, _ in plan}) == 9
    allowance = final_allowance(final)
    assert allowance == {"agent": 0, "local": 0, "model": 36, "capture": 9, "profile": 0}

    stage = StageIdentity(profile="c3_fast_device", stage="final", framework_id="F")
    with tempfile.TemporaryDirectory() as directory:
        budget = StageBudget(Path(directory) / "final.json", BudgetSpec(lane="final", stage=stage,
            started_unix_s=1000.0, deadline_unix_s=2000.0, limits=allowance), now=lambda: 1001.0)
        assert budget.remaining("model") == 36 and budget.remaining("capture") == 9
        # The final lane may not run local feedback at all, which is why the matrix inherits the
        # formal lane's device proof instead of re-measuring it. This is NOT merely an allowance a
        # caller could raise: `evaluate_local` refuses `stage == "final"` outright, so the sealed
        # phase structurally cannot produce a fresh local measurement whatever the budget says.
        assert budget.remaining("local") == 0 and budget.remaining("agent") == 0
        assert 'raise RunnerError("fixture-only feedback is not available during sealed final evaluation")' in (
            Path(local_eval.__file__).read_text(encoding="utf-8"))
        # Without declared limits the same lane is uncapped -- the relaxation is a default, and the
        # structural refusal above is what actually protects the sealed phase.
        undeclared = StageBudget(Path(directory) / "open.json", BudgetSpec(lane="final", stage=stage,
            started_unix_s=1000.0, deadline_unix_s=2000.0), now=lambda: 1001.0)
        assert undeclared.remaining("model") == maxsize


def test_formal_lane_summary_records_acceptance_before_any_heldout_block(tmp_path) -> None:
    # Given: three lane results built from the REAL records -- one accepted, one that ended on the
    # baseline, one never run. The summary must be derived from the lane STATUS alone, so which
    # row represents a goal is fixed before the sealed corpus is ever touched.
    from kernel_optimizer.models.core import ParamSet
    from kernel_optimizer.models.device_operator import DeviceKernelDeclaration
    from examples.c3_qwen3.fast_artifacts import framework_id
    from examples.c3_qwen3.fast_formal import formal_lane_summary
    from examples.c3_qwen3.fast_prepare import FastAccess, prepare_fast
    from examples.c3_qwen3.fast_records import Artifact, Framework, FrozenFramework, ProbeResult, TargetSpec

    prepared = prepare_fast(SETUP / "contract.json", SETUP / "preflight.json",
                            FastAccess(profile="c3_fast_device", phase="development"))
    frame = Framework(revision="F9", implementation_sha256="i", schema_sha256="s", prompt_sha256="p",
                      agent_config_sha256="a", contract_sha256=prepared.task.contract_sha256)
    target = native_target()
    frozen_path = tmp_path / "frozen.json"
    frozen_path.write_text(FrozenFramework(framework=frame, development_result=tmp_path,
        development_result_sha256="d", targets={"ttft": target, "single": target, "multi": target},
        compatibility_evidence=tmp_path, compatibility_sha256="c").model_dump_json(), encoding="utf-8")

    artifact = Artifact(author="agent", version=0, session_id="sid", bundle=tmp_path / "bundle.json",
        bundle_sha256="bs", source_hashes={"operators.py": "h"}, params=ParamSet(values={}), params_sha256="ps",
        kernels=(DeviceKernelDeclaration(backend="triton", source_file="operators.py", entry="kernel",
                                         output_arg_indices=(1,)),))

    def lane(status, selected):
        return ProbeResult(phase="formal_search", epoch=0, framework=frame, target=target,
            original_baseline_sha256="base", started_unix_s=1.0, finished_unix_s=2.0, deadline_unix_s=3.0,
            drain_s=0.0, status=status, candidates=(), selected=selected, budget_path=tmp_path / "b.json")

    results = {}
    for goal, status, selected in (("ttft", "accepted", artifact), ("single", "baseline", None)):
        path = tmp_path / f"{goal}.json"
        path.write_text(lane(status, selected).model_dump_json(), encoding="utf-8")
        results[goal] = path
    # When: "multi" is deliberately absent, which is how a lane that never ran must appear.
    record = formal_lane_summary(frozen_path, results, tmp_path / "lanes.json")
    # Then
    assert record["accepted_goals"] == ["ttft"]
    assert record["framework_id"] == framework_id(frame)
    assert record["lanes"]["ttft"]["accepted"] is True
    assert record["lanes"]["ttft"]["bundle_sha256"] == "bs"
    assert record["lanes"]["ttft"]["framework_id_matches"] is True
    assert record["lanes"]["single"]["accepted"] is False and record["lanes"]["single"]["bundle_sha256"] is None
    assert record["lanes"]["multi"] == {"status": "not_run", "accepted": False}
    assert json.loads((tmp_path / "lanes.json").read_text())["accepted_goals"] == ["ttft"]


def test_the_matrix_refuses_a_row_whose_formal_lane_left_no_device_proof(tmp_path) -> None:
    # Given: an accepted lane whose stored candidates carry NO passing local report for the
    # selected artifact. That would leave an optimised row in the table with nothing showing its
    # kernel ever launched, so it must fail before the 14 GB model is loaded and before any cell
    # is measured -- not as an opaque rejection thirty-six cells later.
    from kernel_optimizer.models.core import ParamSet
    from kernel_optimizer.models.device_operator import DeviceKernelDeclaration
    from examples.c3_qwen3.fast_matrix import inherited_proof
    from examples.c3_qwen3.fast_prepare import FastAccess, prepare_fast
    from examples.c3_qwen3.fast_records import Artifact, Framework, ProbeResult, TargetSpec

    prepared = prepare_fast(SETUP / "contract.json", SETUP / "preflight.json",
                            FastAccess(profile="c3_fast_device", phase="development"))
    frame = Framework(revision="F9", implementation_sha256="i", schema_sha256="s", prompt_sha256="p",
                      agent_config_sha256="a", contract_sha256=prepared.task.contract_sha256)
    target = native_target()
    artifact = Artifact(author="agent", version=0, session_id="sid", bundle=tmp_path / "bundle.json",
        bundle_sha256="bs", source_hashes={}, params=ParamSet(values={}), params_sha256="ps",
        kernels=(DeviceKernelDeclaration(backend="triton", source_file="operators.py", entry="kernel",
                                         output_arg_indices=(1,)),))
    path = tmp_path / "ttft.json"
    path.write_text(ProbeResult(phase="formal_search", epoch=0, framework=frame, target=target,
        original_baseline_sha256="base", started_unix_s=1.0, finished_unix_s=2.0, deadline_unix_s=3.0,
        drain_s=0.0, status="accepted", candidates=(), selected=artifact,
        budget_path=tmp_path / "b.json").model_dump_json(), encoding="utf-8")
    # When / Then
    with pytest.raises(RunnerError, match="no passing device proof"):
        inherited_proof(path, artifact, "any-framework-id")

